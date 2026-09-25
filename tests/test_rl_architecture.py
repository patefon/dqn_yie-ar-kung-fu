"""The refactor's contracts.

Splitting the network into encoder + head and the learner into a registry is
only safe if two things hold: the default DQN still computes exactly what it
computed before, and checkpoints trained before the split still load. Both are
pinned here, because a silent change to either would invalidate every result
already measured (stage 21, ~240k points) without anything failing loudly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from kungfu.config import Config, EncoderConfig, PPOConfig
from kungfu.rl.algos import ALGOS, build_agent
from kungfu.rl.algos.base import Transition
from kungfu.rl.buffers import RolloutBuffer
from kungfu.rl.encoders import ENCODERS, build_encoder
from kungfu.rl.networks import ActorCriticNetwork, QNetwork, migrate_legacy_state_dict
from kungfu.rl.registry import Registry

IN_C, N_ACT, HW = 4, 17, (84, 84)
CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# the bit-exactness guarantee
# --------------------------------------------------------------------------
def legacy_modules():
    """The pre-refactor network, written out explicitly as a reference."""
    torch.manual_seed(0)
    features = nn.Sequential(
        nn.Conv2d(IN_C, 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        nn.Flatten(),
    )
    flat = features(torch.zeros(1, IN_C, *HW)).shape[1]
    advantage = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, N_ACT))
    value = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, 1))
    return features, advantage, value


def legacy_state_dict(features, advantage, value) -> dict:
    """Keys exactly as they appear in checkpoints written before the split."""
    state = {}
    for i in (0, 2, 4):
        state[f"features.{i}.weight"] = features[i].weight.detach().clone()
        state[f"features.{i}.bias"] = features[i].bias.detach().clone()
    for name, seq in (("advantage", advantage), ("value", value)):
        for i in (0, 2):
            state[f"{name}.{i}.weight"] = seq[i].weight.detach().clone()
            state[f"{name}.{i}.bias"] = seq[i].bias.detach().clone()
    return state


class TestBitExactness:
    def test_default_network_matches_the_pre_refactor_maths(self):
        features, advantage, value = legacy_modules()
        net = QNetwork(IN_C, N_ACT, HW, dueling=True)
        net.load_state_dict(legacy_state_dict(features, advantage, value), strict=True)

        net.eval()
        for m in (features, advantage, value):
            m.eval()

        obs = torch.randint(0, 256, (5, IN_C, *HW), dtype=torch.uint8)
        with torch.no_grad():
            z = features(obs.clone().float() / 255.0)
            adv, val = advantage(z), value(z)
            expected = val + adv - adv.mean(dim=1, keepdim=True)
            actual = net(obs.clone())

        assert torch.equal(expected, actual), "the encoder/head split changed the maths"

    def test_flatten_width_is_still_3136(self):
        """Checkpoints on disk have advantage.0.weight of shape (512, 3136)."""
        enc = build_encoder("nature_cnn", IN_C, HW)
        assert enc.out_features == 3136

    def test_legacy_checkpoint_loads_without_missing_or_unexpected_keys(self):
        features, advantage, value = legacy_modules()
        net = QNetwork(IN_C, N_ACT, HW, dueling=True)
        missing, unexpected = net.load_state_dict(
            legacy_state_dict(features, advantage, value), strict=True
        )
        assert missing == [] and unexpected == []

    def test_migration_maps_every_prefix(self):
        migrated = migrate_legacy_state_dict(
            {
                "features.0.weight": torch.zeros(1),
                "advantage.2.bias": torch.zeros(1),
                "value.0.weight": torch.zeros(1),
            }
        )
        assert set(migrated) == {
            "encoder.net.0.weight",
            "head.advantage.2.bias",
            "head.value.0.weight",
        }

    def test_migration_is_a_noop_on_modern_keys(self):
        modern = {"encoder.net.0.weight": torch.zeros(1), "head.value.0.bias": torch.zeros(1)}
        assert migrate_legacy_state_dict(modern) is modern


# --------------------------------------------------------------------------
# registries
# --------------------------------------------------------------------------
class TestRegistry:
    def test_unknown_name_lists_what_is_available(self):
        reg: Registry[int] = Registry("widget")
        reg.register("real")(lambda: 1)
        with pytest.raises(KeyError, match="real"):
            reg.get("typo")

    def test_duplicate_registration_is_rejected(self):
        reg: Registry[int] = Registry("widget")
        reg.register("a")(lambda: 1)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("a")(lambda: 2)

    def test_shipped_registries_are_populated(self):
        assert {"nature_cnn", "impala"} <= set(ENCODERS.names())
        assert {"dqn", "ppo"} <= set(ALGOS.names())


# --------------------------------------------------------------------------
# encoders are interchangeable
# --------------------------------------------------------------------------
class TestEncoders:
    @pytest.mark.parametrize("name", ["nature_cnn", "impala"])
    def test_produces_a_flat_vector_of_the_advertised_width(self, name):
        enc = build_encoder(name, IN_C, HW)
        out = enc(torch.zeros(2, IN_C, *HW))
        assert out.shape == (2, enc.out_features)
        assert enc.out_features > 0

    @pytest.mark.parametrize("name", ["nature_cnn", "impala"])
    def test_any_encoder_drives_both_network_types(self, name):
        q = QNetwork(IN_C, N_ACT, HW, encoder=name)
        ac = ActorCriticNetwork(IN_C, N_ACT, HW, encoder=name)
        obs = torch.zeros(3, IN_C, *HW, dtype=torch.uint8)
        assert q(obs).shape == (3, N_ACT)
        logits, value = ac(obs)
        assert logits.shape == (3, N_ACT) and value.shape == (3,)

    def test_width_scales_capacity(self):
        small = build_encoder("nature_cnn", IN_C, HW, width=0.5)
        full = build_encoder("nature_cnn", IN_C, HW, width=1.0)
        assert small.out_features < full.out_features

    def test_out_features_adapts_to_input_size(self):
        """The 2021 net hard-coded nn.Linear(8320, 512), so changing the crop
        produced a shape error instead of a working network."""
        a = build_encoder("nature_cnn", IN_C, (84, 84)).out_features
        b = build_encoder("nature_cnn", IN_C, (96, 112)).out_features
        assert a != b


# --------------------------------------------------------------------------
# rollout buffer / GAE
# --------------------------------------------------------------------------
class TestRolloutBuffer:
    def make(self, horizon=4, envs=2, gamma=0.5, lam=1.0):
        return RolloutBuffer(horizon, envs, (1, 8, 8), gamma=gamma, gae_lambda=lam)

    def fill(self, buf, reward=1.0, value=0.0, done=False):
        for _ in range(buf.horizon):
            e = buf.num_envs
            buf.add(
                obs=np.zeros((e, 1, 8, 8), dtype=np.uint8),
                actions=np.zeros(e, dtype=np.int64),
                logprobs=np.zeros(e, dtype=np.float32),
                rewards=np.full(e, reward, dtype=np.float32),
                values=np.full(e, value, dtype=np.float32),
                dones=np.full(e, done, dtype=bool),
            )

    def test_fills_then_refuses_more(self):
        buf = self.make()
        self.fill(buf)
        assert buf.full
        with pytest.raises(RuntimeError, match="full"):
            self.fill(buf)

    def test_gae_with_lambda_one_is_the_discounted_return(self):
        """lambda=1, zero baseline -> advantage is just sum of discounted rewards."""
        buf = self.make(horizon=3, gamma=0.5, lam=1.0)
        self.fill(buf, reward=1.0, value=0.0)
        adv, ret = buf.compute_returns(np.zeros(buf.num_envs, dtype=np.float32))
        # last step 1.0; then 1 + 0.5*1 = 1.5; then 1 + 0.5*1.5 = 1.75
        assert adv[2] == pytest.approx(1.0)
        assert adv[1] == pytest.approx(1.5)
        assert adv[0] == pytest.approx(1.75)
        assert np.allclose(ret, adv)  # zero values -> returns == advantages

    def test_terminal_stops_bootstrapping(self):
        buf = self.make(horizon=2, gamma=0.5, lam=1.0)
        e = buf.num_envs
        for done in (True, False):
            buf.add(
                obs=np.zeros((e, 1, 8, 8), dtype=np.uint8),
                actions=np.zeros(e, dtype=np.int64),
                logprobs=np.zeros(e, dtype=np.float32),
                rewards=np.ones(e, dtype=np.float32),
                values=np.zeros(e, dtype=np.float32),
                dones=np.full(e, done, dtype=bool),
            )
        adv, _ = buf.compute_returns(np.full(buf.num_envs, 100.0, dtype=np.float32))
        # step 0 ends the episode, so step 1's value must not leak into it.
        assert adv[0] == pytest.approx(1.0)

    def test_reset_makes_it_reusable(self):
        buf = self.make()
        self.fill(buf)
        buf.reset()
        assert not buf.full and len(buf) == 0

    def test_batches_cover_every_sample_once(self):
        buf = self.make(horizon=4, envs=2)
        self.fill(buf)
        adv, ret = buf.compute_returns(np.zeros(2, np.float32))
        seen = sum(
            len(b["actions"])
            for b in buf.batches(adv, ret, minibatch_size=3, rng=np.random.default_rng(0))
        )
        assert seen == 4 * 2


# --------------------------------------------------------------------------
# algorithms behind one interface
# --------------------------------------------------------------------------
def make_cfg(algo: str, encoder: str = "nature_cnn", **train) -> Config:
    cfg = Config()
    cfg.algo = algo
    cfg.encoder = EncoderConfig(name=encoder)
    cfg.env.obs_height = cfg.env.obs_width = 84
    cfg.replay.capacity = 2000
    cfg.dqn.learn_start = 0
    cfg.ppo = PPOConfig(horizon=4, minibatch_size=4, epochs=1)
    for k, v in train.items():
        setattr(cfg.train, k, v)
    return cfg


def step_agent(agent, envs=2, steps=8, seed=0):
    obs_shape = (4, 84, 84)
    rng = np.random.default_rng(seed)
    for i in range(steps):
        obs = rng.integers(0, 256, (envs, *obs_shape), dtype=np.uint8)
        actions, extras = agent.act(obs)
        agent.observe(
            Transition(
                obs=obs,
                actions=actions,
                rewards=np.ones(envs, dtype=np.float32),
                terminated=np.zeros(envs, dtype=bool),
                truncated=np.zeros(envs, dtype=bool),
                next_obs=obs,
                episode_starts=np.full(envs, i == 0),
                extras=extras,
            )
        )
        yield agent.maybe_update()


class TestAlgorithms:
    @pytest.mark.parametrize("algo", ["dqn", "ppo"])
    def test_builds_acts_and_learns_through_the_shared_interface(self, algo):
        cfg = make_cfg(algo, num_envs=2)
        agent = build_agent(cfg, (4, 84, 84), N_ACT, CPU, seed=0, num_envs=2)
        assert agent.name == algo

        results = list(step_agent(agent, envs=2, steps=12))
        assert agent.steps == 24
        # Both must have produced at least one update in 12 steps.
        updates = [r for r in results if r is not None]
        assert updates, f"{algo} never updated"
        assert all(np.isfinite(u.loss) for u in updates)

    @pytest.mark.parametrize("algo", ["dqn", "ppo"])
    def test_actions_are_in_range(self, algo):
        cfg = make_cfg(algo, num_envs=3)
        agent = build_agent(cfg, (4, 84, 84), N_ACT, CPU, seed=0, num_envs=3)
        actions, _ = agent.act(np.zeros((3, 4, 84, 84), dtype=np.uint8))
        assert actions.shape == (3,)
        assert ((actions >= 0) & (actions < N_ACT)).all()

    def test_ppo_is_on_policy_and_dqn_is_not(self):
        dqn = build_agent(make_cfg("dqn"), (4, 84, 84), N_ACT, CPU)
        ppo = build_agent(make_cfg("ppo"), (4, 84, 84), N_ACT, CPU)
        assert dqn.on_policy is False and ppo.on_policy is True

    def test_ppo_only_updates_when_its_segment_is_full(self):
        cfg = make_cfg("ppo", num_envs=2)
        agent = build_agent(cfg, (4, 84, 84), N_ACT, CPU, seed=0, num_envs=2)
        results = list(step_agent(agent, envs=2, steps=cfg.ppo.horizon))
        assert all(r is None for r in results[:-1]), "PPO updated mid-segment"
        assert results[-1] is not None, "PPO did not update when the segment filled"

    @pytest.mark.parametrize("algo", ["dqn", "ppo"])
    def test_checkpoint_roundtrip(self, algo, tmp_path):
        cfg = make_cfg(algo, num_envs=2)
        a = build_agent(cfg, (4, 84, 84), N_ACT, CPU, seed=0, num_envs=2)
        list(step_agent(a, envs=2, steps=6))
        path = tmp_path / "ckpt.pt"
        a.save(path)

        b = build_agent(cfg, (4, 84, 84), N_ACT, CPU, seed=1, num_envs=2)
        b.load(path)
        assert b.steps == a.steps

        obs = np.zeros((2, 4, 84, 84), dtype=np.uint8)
        a.train_mode(False)
        b.train_mode(False)
        assert np.array_equal(a.act(obs, greedy=True)[0], b.act(obs, greedy=True)[0])

    def test_loading_another_algorithms_checkpoint_fails_clearly(self, tmp_path):
        ppo = build_agent(make_cfg("ppo"), (4, 84, 84), N_ACT, CPU)
        path = tmp_path / "ppo.pt"
        ppo.save(path)
        dqn = build_agent(make_cfg("dqn"), (4, 84, 84), N_ACT, CPU)
        with pytest.raises(ValueError, match="written by 'ppo'"):
            dqn.load(path)

    def test_any_algorithm_pairs_with_any_encoder(self):
        for algo in ("dqn", "ppo"):
            for enc in ("nature_cnn", "impala"):
                agent = build_agent(make_cfg(algo, enc), (4, 84, 84), N_ACT, CPU)
                actions, _ = agent.act(np.zeros((1, 4, 84, 84), dtype=np.uint8))
                assert actions.shape == (1,)

    def test_ppo_bootstraps_through_a_time_limit_truncation(self):
        """A truncation is not a terminal. Masking it as one teaches the agent
        the world ends at max_episode_steps -- which was 95% of episodes."""
        cfg = make_cfg("ppo", num_envs=1)
        agent = build_agent(cfg, (4, 84, 84), N_ACT, CPU, seed=0, num_envs=1)
        obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        # Zero obs through zero-initialised biases gives V = 0 exactly, which
        # would hide the bootstrap. Pin the value head so V(next) is known.
        with torch.no_grad():
            agent.net.head.value.bias.fill_(2.0)
        expected_v = 2.0
        actions, extras = agent.act(obs)

        def observe(terminated, truncated):
            agent.buffer.reset()
            agent.observe(
                Transition(
                    obs=obs, actions=actions,
                    rewards=np.ones(1, dtype=np.float32),
                    terminated=np.array([terminated]),
                    truncated=np.array([truncated]),
                    next_obs=obs,
                    episode_starts=np.array([False]),
                    extras=extras,
                )
            )
            return float(agent.buffer.rewards[0, 0]), bool(agent.buffer.dones[0, 0])

        r_trunc, done_trunc = observe(terminated=False, truncated=True)
        r_term, done_term = observe(terminated=True, truncated=False)

        # A real terminal is masked and its reward untouched.
        assert done_term is True
        assert r_term == pytest.approx(1.0)
        # A truncation is NOT masked, and carries the bootstrapped value.
        assert done_trunc is False, "truncation must not be masked as terminal"
        assert r_trunc == pytest.approx(1.0 + cfg.ppo.gamma * expected_v), (
            "truncation must fold gamma * V(next_obs) into the reward"
        )

    def test_unknown_algorithm_is_rejected_at_build_time(self):
        cfg = make_cfg("dqn")
        cfg.algo = "nope"
        with pytest.raises(KeyError, match="nope"):
            build_agent(cfg, (4, 84, 84), N_ACT, CPU)
