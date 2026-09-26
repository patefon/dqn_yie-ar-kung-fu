"""Fixed-horizon rollout storage for on-policy algorithms.

Where the replay buffer keeps hundreds of thousands of old transitions and
samples them repeatedly, an on-policy buffer holds exactly one short segment
collected by the *current* policy, is consumed a handful of times, and is then
thrown away. PPO cannot learn from stale data: the importance ratio it corrects
with is only valid near the policy that generated the actions.

That difference is why adding PPO needed a second buffer type rather than a
flag on the existing one.

Layout is ``(T, E, ...)`` -- time-major, so a whole timestep across all envs is
one contiguous write, and the GAE recursion walks backwards over axis 0.
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = horizon
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cpu")

        shape = (horizon, num_envs)
        self.obs = np.zeros((*shape, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.logprobs = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        # `dones` marks that the step ENDED an episode, so the next state is a
        # fresh reset and must not be bootstrapped from.
        self.dones = np.zeros(shape, dtype=bool)

        self._t = 0

    def __len__(self) -> int:
        return self._t * self.num_envs

    @property
    def full(self) -> bool:
        return self._t >= self.horizon

    def reset(self) -> None:
        self._t = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        if self.full:
            raise RuntimeError("rollout buffer is full; compute returns and reset it")
        t = self._t
        self.obs[t] = obs
        self.actions[t] = actions
        self.logprobs[t] = logprobs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones
        self._t += 1

    def compute_returns(self, last_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation (Schulman et al. 2016).

        Walks backwards accumulating
        ``delta_t = r_t + gamma * V(s_{t+1}) * (1-done_t) - V(s_t)``
        into ``A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}``.

        ``lambda`` trades bias against variance: 1.0 is the plain Monte-Carlo
        return (unbiased, noisy), 0.0 is one-step TD (low variance, biased).

        The mask is ``dones[t]`` -- *this* step ended the episode -- which is
        the convention ``add`` stores. Published implementations often mask with
        ``dones[t+1]`` instead, because their buffers record "state t is a
        post-reset state"; mixing the two is an off-by-one that lets advantage
        leak backwards across an episode boundary, and nothing about it looks
        wrong on a training curve. ``test_terminal_stops_bootstrapping`` pins it.
        """
        advantages = np.zeros((self._t, self.num_envs), dtype=np.float32)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self._t)):
            nonterminal = 1.0 - self.dones[t].astype(np.float32)
            next_values = last_values if t == self._t - 1 else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_values * nonterminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[: self._t]
        return advantages, returns

    def batches(
        self, advantages: np.ndarray, returns: np.ndarray, minibatch_size: int, rng
    ):
        """Yield shuffled flat minibatches over the whole segment."""
        n = self._t * self.num_envs
        flat_obs = self.obs[: self._t].reshape(n, *self.obs_shape)
        flat = {
            "obs": flat_obs,
            "actions": self.actions[: self._t].reshape(n),
            "logprobs": self.logprobs[: self._t].reshape(n),
            "advantages": advantages.reshape(n),
            "returns": returns.reshape(n),
            "values": self.values[: self._t].reshape(n),
        }
        order = rng.permutation(n)
        for start in range(0, n, minibatch_size):
            idx = order[start : start + minibatch_size]
            yield {k: v[idx] for k, v in flat.items()}

    def nbytes(self) -> int:
        return int(
            self.obs.nbytes
            + self.actions.nbytes
            + self.logprobs.nbytes
            + self.rewards.nbytes
            + self.values.nbytes
            + self.dones.nbytes
        )
