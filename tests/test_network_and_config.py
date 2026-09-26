"""Network shape handling and config validation."""

from __future__ import annotations

import pytest
import torch

from kungfu.config import Config, EnvConfig, Region, StartState
from kungfu.rl.networks import QNetwork


class TestQNetwork:
    @pytest.mark.parametrize("hw", [(84, 84), (96, 112), (64, 64)])
    def test_flatten_size_is_inferred(self, hw):
        """The original hard-coded nn.Linear(8320, 512), so any change to the
        crop broke the net with a shape error."""
        net = QNetwork(in_channels=4, n_actions=17, input_hw=hw)
        out = net(torch.zeros(2, 4, *hw, dtype=torch.uint8))
        assert out.shape == (2, 17)

    def test_dueling_and_plain_agree_on_shape(self):
        for dueling in (True, False):
            net = QNetwork(4, 17, (84, 84), dueling=dueling)
            assert net(torch.zeros(3, 4, 84, 84, dtype=torch.uint8)).shape == (3, 17)

    def test_uint8_input_is_not_mutated(self):
        net = QNetwork(4, 17, (84, 84))
        obs = torch.full((1, 4, 84, 84), 255, dtype=torch.uint8)
        net(obs)
        assert obs.max().item() == 255

    def test_noisy_layers_resample(self):
        net = QNetwork(4, 17, (84, 84), noisy=True)
        net.train()
        obs = torch.zeros(1, 4, 84, 84, dtype=torch.uint8)
        a = net(obs)
        net.reset_noise()
        b = net(obs)
        assert not torch.allclose(a, b)


class TestConfig:
    def test_defaults_validate(self):
        cfg = Config()
        assert cfg.env.frame_stack == 4
        assert cfg.dqn.double and cfg.dqn.dueling

    def test_region_rejects_out_of_bounds(self):
        with pytest.raises(ValueError):
            Region(x=240, y=0, w=64, h=8)

    def test_region_crop_shape(self):
        import numpy as np

        r = Region(x=10, y=20, w=32, h=8)
        assert r.crop(np.zeros((224, 240, 3), dtype=np.uint8)).shape == (8, 32, 3)

    def test_sticky_action_prob_bounded(self):
        with pytest.raises(ValueError):
            EnvConfig(sticky_action_prob=1.5)

    def test_checked_in_default_yaml_matches_code_defaults(self):
        """configs/default.yaml must not drift from Config().

        This bit once: the YAML was generated before the HUD coordinates were
        corrected, so training silently loaded the old wrong regions while the
        code defaults were right. The score then read as unreadable for every
        frame and the reward signal was flat -- with no error anywhere.
        """
        from pathlib import Path as P

        root = P(__file__).resolve().parents[1]
        shipped = root / "configs" / "default.yaml"
        assert shipped.exists(), "configs/default.yaml is missing"
        assert Config.load(shipped).model_dump() == Config().model_dump(), (
            "configs/default.yaml is stale; regenerate it with `make config`"
        )

    def test_start_states_default_to_none(self):
        """No curriculum configured must keep the original single-state path."""
        assert Config().env.start_states is None

    def test_start_state_weight_must_be_positive(self):
        with pytest.raises(ValueError):
            StartState(name="Stage10", weight=0)

    def test_start_states_survive_a_yaml_roundtrip(self, tmp_path):
        cfg = Config()
        cfg.env.start_states = [
            StartState(name="Level1", weight=1.0),
            StartState(name="Stage20", weight=2.5),
        ]
        p = tmp_path / "c.yaml"
        cfg.dump(p)
        loaded = Config.load(p)
        assert [s.name for s in loaded.env.start_states] == ["Level1", "Stage20"]
        assert loaded.env.start_states[1].weight == 2.5

    def test_yaml_roundtrip(self, tmp_path):
        cfg = Config()
        cfg.dqn.lr = 1e-4
        cfg.env.frame_skip = 6
        p = tmp_path / "c.yaml"
        cfg.dump(p)
        loaded = Config.load(p)
        assert loaded.dqn.lr == 1e-4
        assert loaded.env.frame_skip == 6
