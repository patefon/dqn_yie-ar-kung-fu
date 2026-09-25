"""Replay buffer correctness -- the component whose bugs are hardest to notice.

A broken replay buffer does not crash. It silently feeds the network slightly
wrong targets, and training just underperforms for reasons nobody can see. Each
test here pins one property the original got wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from kungfu.rl.buffers import FrameReplayBuffer, SumTree


def make_buffer(**kw):
    defaults = dict(
        capacity=512, frame_shape=(8, 8), stack=4, n_step=3, gamma=0.5,
        prioritized=False, num_envs=1,
    )
    defaults.update(kw)
    return FrameReplayBuffer(**defaults)


def fill(buf, n, *, reward=1.0, done_at=None, num_envs=1):
    for t in range(n):
        frame = np.full((num_envs, 8, 8), t % 256, dtype=np.uint8)
        done = np.array([done_at is not None and t == done_at] * num_envs)
        start = np.array([t == 0] * num_envs)
        buf.add(
            frame,
            np.zeros(num_envs, dtype=np.int64),
            np.full(num_envs, reward, dtype=np.float32),
            done,
            start,
        )


def test_memory_footprint_is_bounded():
    """200k transitions at 84x84 must cost ~1.4 GB, not the old 21 GB."""
    buf = FrameReplayBuffer(capacity=200_000, frame_shape=(84, 84), stack=4, num_envs=1)
    gb = buf.nbytes() / 1e9
    assert gb < 2.0, f"expected well under 2 GB, got {gb:.2f} GB"


def test_not_ready_raises_instead_of_returning_none():
    """The original returned None, which the caller unpacked into 5 values."""
    buf = make_buffer()
    fill(buf, 3)
    assert not buf.ready
    with pytest.raises(RuntimeError):
        buf.sample(4, beta=0.4, rng=np.random.default_rng(0))


def test_n_step_return_is_discounted_sum():
    buf = make_buffer(n_step=3, gamma=0.5)
    fill(buf, 64, reward=1.0)
    batch = buf.sample(16, beta=0.4, rng=np.random.default_rng(0))
    # 1 + 0.5 + 0.25 with gamma=0.5 over 3 steps.
    assert np.allclose(batch["rewards"], 1.75)
    assert np.allclose(batch["discounts"], 0.125)


def test_n_step_truncates_at_terminal():
    buf = make_buffer(n_step=3, gamma=0.5)
    fill(buf, 64, reward=1.0, done_at=20)
    rng = np.random.default_rng(0)
    batch = buf.sample(256, beta=0.4, rng=rng)
    idx = batch["indices"]
    # A window starting exactly on the terminal collects one reward only.
    on_terminal = idx == 20
    if on_terminal.any():
        assert np.allclose(batch["rewards"][on_terminal], 1.0)
        assert batch["dones"][on_terminal].all()
        assert np.allclose(batch["discounts"][on_terminal], 0.5)


def test_stack_does_not_leak_across_episode_start():
    """Frames from before a reset must be zeroed, not spliced in."""
    buf = make_buffer(stack=4, n_step=1)
    # Episode A, then a start marker at t=30 opening episode B.
    for t in range(64):
        buf.add(
            np.full((1, 8, 8), 100 + t, dtype=np.uint8),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.float32),
            np.array([False]),
            np.array([t in (0, 30)]),
        )
    states = buf._stack_at(np.array([0]), np.array([31]))[0]
    # Only t=30 and t=31 belong to episode B; the two older slots must be blank.
    assert states[0].sum() == 0
    assert states[1].sum() == 0
    assert states[2].max() == 130
    assert states[3].max() == 131


def test_envs_do_not_interleave():
    """Each env keeps its own row, so a stack never mixes two games."""
    buf = make_buffer(num_envs=4, stack=4, n_step=1)
    for t in range(64):
        frames = np.stack([np.full((8, 8), e * 50 + 1, dtype=np.uint8) for e in range(4)])
        buf.add(
            frames,
            np.zeros(4, dtype=np.int64),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=bool),
            np.array([t == 0] * 4),
        )
    for env_id in range(4):
        stack = buf._stack_at(np.array([env_id]), np.array([40]))[0]
        assert set(np.unique(stack)) == {env_id * 50 + 1}


def test_sampled_actions_match_their_slots():
    buf = make_buffer(n_step=1, num_envs=2)
    for t in range(64):
        buf.add(
            np.zeros((2, 8, 8), dtype=np.uint8),
            np.array([t % 7, (t + 3) % 7], dtype=np.int64),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=bool),
            np.array([t == 0, t == 0]),
        )
    batch = buf.sample(32, beta=0.4, rng=np.random.default_rng(1))
    expected = np.where(
        batch["env_indices"] == 0, batch["indices"] % 7, (batch["indices"] + 3) % 7
    )
    assert np.array_equal(batch["actions"], expected)


def test_prioritized_sampling_returns_valid_weights():
    buf = make_buffer(prioritized=True, n_step=1)
    fill(buf, 128)
    rng = np.random.default_rng(0)
    batch = buf.sample(32, beta=0.4, rng=rng)
    assert batch["weights"].shape == (32,)
    assert np.all(batch["weights"] > 0)
    assert np.isclose(batch["weights"].max(), 1.0)
    buf.update_priorities(batch["env_indices"], batch["indices"], np.abs(rng.normal(size=32)))


class TestSumTree:
    def test_total_tracks_updates(self):
        tree = SumTree(8)
        tree.set(0, 2.0)
        tree.set(5, 3.0)
        assert tree.total == pytest.approx(5.0)
        tree.set(0, 1.0)
        assert tree.total == pytest.approx(4.0)

    def test_sampling_respects_priority_mass(self):
        tree = SumTree(8)
        tree.set(3, 1.0)  # all mass on one leaf
        picks = tree.sample(np.random.default_rng(0).random(64) * tree.total)
        assert set(picks.tolist()) == {3}

    def test_non_power_of_two_capacity_does_not_overrun(self):
        """Heap indexing needs a perfect tree; an arbitrary capacity such as
        40,000 previously walked off the end of the array."""
        for cap in (3, 5, 1000, 40_000):
            tree = SumTree(cap)
            tree.set(cap - 1, 2.0)
            tree.set(0, 1.0)
            picks = tree.sample(np.random.default_rng(0).random(64) * tree.total)
            assert picks.min() >= 0 and picks.max() < cap
            assert set(picks.tolist()) <= {0, cap - 1}

    def test_padding_leaves_are_never_selected(self):
        tree = SumTree(5)  # padded to 8 leaves
        tree.set(4, 1.0)
        picks = tree.sample(np.random.default_rng(0).random(200) * tree.total)
        assert set(picks.tolist()) == {4}

    def test_sampling_is_proportional(self):
        tree = SumTree(4)
        tree.set(0, 1.0)
        tree.set(1, 3.0)
        picks = tree.sample(np.random.default_rng(0).random(4000) * tree.total)
        share = (picks == 1).mean()
        assert 0.70 < share < 0.80  # expect ~0.75


class TestResumeWarmup:
    """A resumed checkpoint must refill the replay before it learns again.

    `learn_start` used to be compared against the agent's global step counter.
    That is fine on a fresh run, where steps and buffer fill advance together,
    but a checkpoint resumed at 10M steps starts with an empty buffer and would
    begin updating immediately -- sampling batches of 32 from a handful of
    transitions, with PER concentrating on them.
    """

    def _agent(self, tmp_path, steps):
        import torch

        from kungfu.config import Config
        from kungfu.rl.algos import build_agent

        cfg = Config()
        cfg.train.num_envs = 2
        cfg.dqn.learn_start = 1000
        agent = build_agent(
            cfg, (4, 84, 84), 17, torch.device("cpu"), seed=0, num_envs=2
        )
        agent.steps = steps
        return agent

    def test_resumed_agent_waits_for_learn_start_not_the_step_counter(self, tmp_path):
        """The gap the old guard left open.

        `memory.ready` only requires stack + n_step + 2 columns, so an empty
        buffer was caught by luck. A buffer holding a couple of hundred
        transitions clears `ready` but is still 5x short of learn_start -- and
        with the old step-based guard a resumed agent trained on it.
        """
        import numpy as np

        from kungfu.train import Transition

        agent = self._agent(tmp_path, steps=10_000_000)
        rng = np.random.default_rng(0)
        for i in range(100):
            agent.observe(
                Transition(
                    obs=rng.integers(0, 255, (2, 84, 84), dtype=np.uint8),
                    actions=np.zeros(2, dtype=np.int64),
                    rewards=np.zeros(2, dtype=np.float32),
                    terminated=np.zeros(2, dtype=bool),
                    truncated=np.zeros(2, dtype=bool),
                    episode_starts=np.full(2, i == 0),
                    next_obs=rng.integers(0, 255, (2, 84, 84), dtype=np.uint8),
                )
            )
        assert agent.memory.ready, "precondition: the old guard would have let this pass"
        assert len(agent.memory) == 200 < agent.cfg.learn_start
        assert agent.learn() is None, "trained on 200 transitions after a resume"

    def test_learning_starts_once_the_buffer_fills(self, tmp_path):
        import numpy as np

        from kungfu.train import Transition

        agent = self._agent(tmp_path, steps=10_000_000)
        rng = np.random.default_rng(0)
        for i in range(700):
            agent.observe(
                Transition(
                    obs=rng.integers(0, 255, (2, 84, 84), dtype=np.uint8),
                    actions=np.zeros(2, dtype=np.int64),
                    rewards=np.zeros(2, dtype=np.float32),
                    terminated=np.zeros(2, dtype=bool),
                    truncated=np.zeros(2, dtype=bool),
                    episode_starts=np.full(2, i == 0),
                    next_obs=rng.integers(0, 255, (2, 84, 84), dtype=np.uint8),
                )
            )
        assert len(agent.memory) == 1400 >= 1000
        assert agent.learn() is not None, "never resumed learning after refilling"
