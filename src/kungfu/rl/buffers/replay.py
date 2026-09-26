"""Frame-indexed replay buffer with n-step returns and prioritised sampling.

The memory bug in the original
-------------------------------
The old ``ReplayBuffer`` stored whole ``Experience`` tuples holding **both** the
stacked state and the stacked next state as full numpy arrays::

    3 frames x 74 x 240 uint8 = 53,280 B per state
    x2 (state + next_state)   = 106,560 B per transition
    x 200,000 capacity        = 21.3 GB

It could never fill; the process would be killed, or thrash, long before.
Sampling was worse: ``np.random.choice(sum(1 for x in self.buffer if x is not None), ...)``
rescanned all 200,000 slots on *every* sample call, an O(capacity) walk per
gradient step. And when the buffer was empty, ``sample()`` returned ``None``,
which the caller unpacked into five variables, raising a ``TypeError`` that the
bare ``except`` in the training loop swallowed in silence.

The fix is the standard one: store each frame **once** and rebuild stacks by
index. The same 200k transitions at 84x84 with a 4-frame stack cost ~1.4 GB.

Vectorised envs
---------------
Storage is ``(num_envs, steps_per_env, H, W)``. A naive flat timeline would
interleave frames from different envs, so a reconstructed "stack" would splice
together four unrelated games -- a silent, extremely hard to spot corruption.
Keeping each env on its own row makes stacking correct by construction.
"""

from __future__ import annotations

import numpy as np


class SumTree:
    """Fixed-size sum tree for O(log n) proportional sampling.

    The heap-style indexing (children of ``i`` at ``2i`` and ``2i+1``) is only
    valid on a *perfect* binary tree, so the leaf array is padded up to the next
    power of two. Sizing it to an arbitrary capacity -- 40,000, say -- makes the
    descent walk straight off the end of the array.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Next power of two >= capacity. Padding leaves stay at priority 0, so
        # they can never be selected.
        self.size = 1 << max(0, (capacity - 1).bit_length())
        self.tree = np.zeros(2 * self.size, dtype=np.float64)

    def set(self, idx: int, value: float) -> None:
        i = idx + self.size
        delta = value - self.tree[i]
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] += delta
            i //= 2

    def set_many(self, indices: np.ndarray, values: np.ndarray) -> None:
        for i, v in zip(indices, values, strict=True):
            self.set(int(i), float(v))

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def max_value(self) -> float:
        m = float(self.tree[self.size : self.size + self.capacity].max())
        return m if m > 0 else 1.0

    def leaf(self, idx: np.ndarray) -> np.ndarray:
        return self.tree[idx + self.size]

    def sample(self, targets: np.ndarray) -> np.ndarray:
        """Vectorised descent; ``targets`` are uniform draws in [0, total)."""
        idx = np.ones(len(targets), dtype=np.int64)
        targets = targets.copy()
        # Perfect tree: every index descends in lockstep, so one shared loop
        # bound is correct for the whole batch.
        while idx[0] < self.size:
            left = 2 * idx
            left_sum = self.tree[left]
            go_right = targets > left_sum
            targets = np.where(go_right, targets - left_sum, targets)
            idx = left + go_right.astype(np.int64)
        return np.clip(idx - self.size, 0, self.capacity - 1)


class FrameReplayBuffer:
    """Stores single frames per env row; reconstructs stacked observations on sample.

    At row ``e``, column ``t``: the newest frame of state ``s_t`` in env ``e``, the
    action taken from it, the reward received, whether it terminated, and whether
    it began an episode.
    """

    def __init__(
        self,
        capacity: int,
        frame_shape: tuple[int, int],
        stack: int = 4,
        n_step: int = 3,
        gamma: float = 0.99,
        prioritized: bool = True,
        alpha: float = 0.6,
        num_envs: int = 1,
    ) -> None:
        if n_step < 1:
            raise ValueError("n_step must be >= 1")
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        self.num_envs = num_envs
        self.steps_per_env = max(capacity // num_envs, stack + n_step + 2)
        self.capacity = self.num_envs * self.steps_per_env
        self.stack = stack
        self.n_step = n_step
        self.gamma = gamma
        self.prioritized = prioritized
        self.alpha = alpha

        h, w = frame_shape
        shape = (num_envs, self.steps_per_env)
        self.frames = np.zeros((*shape, h, w), dtype=np.uint8)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=bool)
        # Marks the first frame of an episode so a stack never reaches back
        # across a reset into a completely unrelated scene.
        self.starts = np.zeros(shape, dtype=bool)

        self._t = 0          # write column
        self._filled = 0     # columns written (per env), capped at steps_per_env
        self._tree = SumTree(self.capacity) if prioritized else None

    def __len__(self) -> int:
        return self._filled * self.num_envs

    @property
    def ready(self) -> bool:
        return self._filled > self.stack + self.n_step + 2

    # -- writing -----------------------------------------------------------
    def add(
        self,
        frames: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        episode_starts: np.ndarray,
    ) -> None:
        """Append one vectorised timestep. ``frames`` is (E, C, H, W) or (E, H, W)."""
        if frames.ndim == 4:
            frames = frames[:, -1]  # newest channel only
        t = self._t
        self.frames[:, t] = frames
        self.actions[:, t] = actions
        self.rewards[:, t] = rewards
        self.dones[:, t] = dones
        self.starts[:, t] = episode_starts

        if self._tree is not None:
            # New transitions enter at max priority so each is seen at least once.
            prio = self._tree.max_value()
            base = np.arange(self.num_envs) * self.steps_per_env + t
            self._tree.set_many(base, np.full(self.num_envs, prio))

        self._t = (t + 1) % self.steps_per_env
        self._filled = min(self._filled + 1, self.steps_per_env)

    # -- stacking ----------------------------------------------------------
    def _stack_at(self, env_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        """Gather (B, stack, H, W), zero-padding anything before an episode start."""
        b = len(t_idx)
        out = np.zeros((b, self.stack, *self.frames.shape[2:]), dtype=np.uint8)
        # offset 0 is the oldest frame in the stack; stack-1 is the newest.
        for offset in range(self.stack):
            back = self.stack - 1 - offset
            src = (t_idx - back) % self.steps_per_env
            out[:, offset] = self.frames[env_idx, src]
            if back == 0:
                continue
            # If any column strictly newer than src began an episode, then src
            # belongs to the previous episode and must not leak into this stack.
            crossed = np.zeros(b, dtype=bool)
            for probe in range(back):
                crossed |= self.starts[env_idx, (t_idx - probe) % self.steps_per_env]
            out[crossed, offset] = 0
        return out

    def _valid_columns(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Columns with a full stack behind and n steps ahead, avoiding the write head."""
        lo = self.stack - 1
        hi = self._filled - self.n_step - 1
        if hi <= lo:
            raise RuntimeError("buffer does not yet hold a full trajectory window")
        return rng.integers(lo, hi, size=count)

    def sample(self, batch_size: int, beta: float, rng: np.random.Generator) -> dict:
        if not self.ready:
            raise RuntimeError(f"replay buffer not ready ({len(self)} transitions)")

        lo = self.stack - 1
        hi = max(lo + 1, self._filled - self.n_step - 1)

        if self._tree is not None and self._tree.total > 0:
            flat = self._tree.sample(rng.random(batch_size) * self._tree.total)
            env_idx = flat // self.steps_per_env
            t_idx = np.clip(flat % self.steps_per_env, lo, hi - 1)
            leaf = self.tree_priorities(env_idx, t_idx)
            probs = np.maximum(leaf, 1e-8) / max(self._tree.total, 1e-8)
            weights = (len(self) * probs) ** (-beta)
            weights = (weights / weights.max()).astype(np.float32)
        else:
            env_idx = rng.integers(0, self.num_envs, size=batch_size)
            t_idx = self._valid_columns(batch_size, rng)
            weights = np.ones(batch_size, dtype=np.float32)

        # n-step return, stopping at the first terminal inside the window.
        n_reward = np.zeros(batch_size, dtype=np.float32)
        n_done = np.zeros(batch_size, dtype=bool)
        n_offset = np.full(batch_size, self.n_step, dtype=np.int64)
        discount = np.ones(batch_size, dtype=np.float32)
        alive = np.ones(batch_size, dtype=bool)
        for k in range(self.n_step):
            col = (t_idx + k) % self.steps_per_env
            n_reward += np.where(alive, discount * self.rewards[env_idx, col], 0.0)
            hit = alive & self.dones[env_idx, col]
            n_offset = np.where(hit, k + 1, n_offset)
            n_done |= hit
            alive &= ~self.dones[env_idx, col]
            discount = discount * self.gamma

        next_t = (t_idx + n_offset) % self.steps_per_env
        return {
            "states": self._stack_at(env_idx, t_idx),
            "actions": self.actions[env_idx, t_idx],
            "rewards": n_reward,
            "next_states": self._stack_at(env_idx, next_t),
            "dones": n_done,
            "discounts": (self.gamma**n_offset).astype(np.float32),
            "weights": weights,
            "env_indices": env_idx,
            "indices": t_idx,
        }

    # -- priorities --------------------------------------------------------
    def _flat(self, env_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        return env_idx * self.steps_per_env + t_idx

    def tree_priorities(self, env_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        assert self._tree is not None
        return self._tree.leaf(self._flat(env_idx, t_idx))

    def update_priorities(
        self, env_idx: np.ndarray, t_idx: np.ndarray, td_errors: np.ndarray
    ) -> None:
        if self._tree is None:
            return
        prios = (np.abs(td_errors) + 1e-6) ** self.alpha
        self._tree.set_many(self._flat(env_idx, t_idx), prios)

    def nbytes(self) -> int:
        return int(
            self.frames.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.dones.nbytes
            + self.starts.nbytes
        )
