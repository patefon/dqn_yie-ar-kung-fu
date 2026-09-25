"""Typed, validated configuration.

Replaces the old module-level ``consts.py``. Everything that used to be a magic
number lives here, is range-checked, and can be overridden from YAML or the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Region(BaseModel):
    """An axis-aligned crop in *native NES framebuffer* coordinates (240x224).

    The old code carried these as loose ints in ``consts.py`` and applied them to a
    scaled screenshot, so every value silently depended on the emulator window size.
    Native coordinates are resolution-independent and exact.
    """

    model_config = {"frozen": True}

    x: int = Field(ge=0, lt=256)
    y: int = Field(ge=0, lt=240)
    w: int = Field(gt=0, le=256)
    h: int = Field(gt=0, le=240)

    def crop(self, frame):
        return frame[self.y : self.y + self.h, self.x : self.x + self.w]

    @model_validator(mode="after")
    def _within_frame(self) -> Region:
        if self.x + self.w > 256 or self.y + self.h > 240:
            raise ValueError(f"region {self} extends past the NES framebuffer")
        return self


class HudLayout(BaseModel):
    """Where the HUD elements live on screen.

    These are **measured**, not guessed: every value below was read off the real
    240x224 framebuffer of the Japan Rev 1.4 ROM
    (sha1 d07be428cf7d198453f4942f5288a05fd55720dc) on stage 1. Re-verify with
    ``python -m tools.calibrate_vision dump`` if you swap ROM revision.

    Note ``score`` is the live SCORE field at x=16, *not* the HI field at x=88.
    They render identically, and reading HI by mistake gives a value that almost
    never changes -- a reward signal that is silently, permanently zero.
    """

    # "000100" -- six 8px digit tiles starting at x=16.
    score: Region = Region(x=16, y=40, w=48, h=8)
    # High score, kept only so calibration can tell the two fields apart.
    high_score: Region = Region(x=88, y=40, w=48, h=8)
    # The "01" at the end of "STAGE-01".
    stage: Region = Region(x=208, y=32, w=16, h=8)
    # Bars sit at the bottom either side of the central "KO" plate (x=104..135).
    player_health: Region = Region(x=40, y=202, w=64, h=4)
    enemy_health: Region = Region(x=136, y=202, w=64, h=4)
    # The small head icons for spare lives. Measured: exactly 37 lit pixels per
    # icon, so 74 / 37 / 0 px maps cleanly to 2 / 1 / 0 spare lives.
    lives: Region = Region(x=160, y=40, w=40, h=8)
    lives_pixels_per_icon: int = Field(default=37, gt=0)
    # The "GAME OVER" banner sits in the gap between the ceiling decorations
    # (which occupy x 16..77 and 162..223). Measured: 0 white pixels here across
    # 876 live frames, 232 when the banner is up.
    gameover_banner: Region = Region(x=64, y=114, w=128, h=12)
    gameover_banner_min_pixels: int = Field(default=40, gt=0)
    # Arena plus both health bars, excluding the score/stage text. Health is part
    # of the state a pixels-only agent legitimately needs; the score counter is
    # not, and leaving it in gives the convnet a monotone shortcut to memorise.
    playfield: Region = Region(x=0, y=62, w=240, h=150)

    # The bar green, measured as exactly (72, 220, 72). Listing colours beats the
    # old single magic constant (161) plus an unexplained fallback (98).
    health_fill_rgb: list[tuple[int, int, int]] = [(72, 220, 72)]
    health_match_tolerance: int = Field(default=24, ge=0, le=128)

    # A frame is an inter-stage card if the playfield holds fewer than this many
    # distinct colours. Measured separation on real frames: live gameplay shows
    # 7-10 distinct colours (the NES palette is tiny), stage cards show 1-2.
    # The value must sit in that gap -- an earlier threshold of 8 fell inside the
    # live distribution and silently suppressed reward on 17% of real frames.
    transition_max_colors: int = Field(default=4, ge=2, le=32)


class StartState(BaseModel):
    """One savestate the episode may begin from, and how often to pick it."""

    name: str
    weight: float = Field(default=1.0, gt=0.0)


class EnvConfig(BaseModel):
    """Environment dynamics. These knobs did not exist before -- the old loop ran
    "as fast as the queue allowed", which is why its transitions were not an MDP."""

    game: str = "YieArKungFu-Nes"
    state: str = "Level1"
    # Start-state curriculum. Every episode starting at stage 1 means ~96% of
    # experience is spent replaying stages the agent already beat -- measured:
    # of ~11,700 steps in an uncapped run, only ~500 are past stage 20. Seeding
    # some episodes further in puts the experience where the agent actually
    # fails. None keeps the single `state` above, which is the original
    # behaviour.
    start_states: list[StartState] | None = None

    # Each agent action is held for this many emulator frames. 4 is the Atari/DQN
    # standard and, critically, it is long enough for a NES attack animation to
    # register. The old code held keys indefinitely instead.
    frame_skip: int = Field(default=4, ge=1, le=16)
    # Max-pool the last two frames of each skip to kill sprite flicker.
    max_pool_frames: bool = True

    frame_stack: int = Field(default=4, ge=1, le=16)
    obs_height: int = Field(default=84, ge=32, le=240)
    obs_width: int = Field(default=84, ge=32, le=256)
    grayscale: bool = True

    # Sticky actions (Machado et al. 2018): with probability p the previous action
    # repeats, injecting stochasticity so the agent cannot memorise a fixed script.
    sticky_action_prob: float = Field(default=0.25, ge=0.0, lt=1.0)
    # Random no-ops at reset -- decorrelates parallel envs.
    noop_max: int = Field(default=30, ge=0, le=120)

    max_episode_steps: int = Field(default=6000, gt=0)
    # End the episode if the score has not moved for this many agent steps. Stops
    # the agent from farming a safe corner forever.
    stall_timeout: int = Field(default=600, gt=0)
    # Consecutive steps of the game-over signature required before terminating.
    # Debounced because both bars read empty for a frame or two during a normal
    # KO transition, which is NOT game over.
    gameover_patience: int = Field(default=8, ge=1, le=120)

    hud: HudLayout = HudLayout()


class RewardConfig(BaseModel):
    """Reward shaping, all derived from the *vision* extractor.

    The old scheme double-counted every event: it compared against the previous
    frame at ~100 fps, so one hit landed a -5 penalty on every frame the health bar
    spent animating downward. Here the signals are edge-triggered and normalised.
    """

    score_scale: float = Field(default=0.01, ge=0.0)
    # Fired once per detected transition, not once per frame.
    damage_dealt: float = 1.0
    damage_taken: float = -1.0
    stage_clear: float = 10.0
    death: float = -10.0
    # Small per-step cost so stalling is never optimal.
    time_penalty: float = -0.001
    # Clip the final per-step reward; keeps the TD target well-scaled.
    clip: float | None = Field(default=5.0, gt=0.0)


class ReplayConfig(BaseModel):
    capacity: int = Field(default=200_000, gt=0)
    # Prioritised experience replay (Schaul et al. 2016).
    prioritized: bool = True
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    beta_start: float = Field(default=0.4, ge=0.0, le=1.0)
    beta_frames: int = Field(default=1_000_000, gt=0)
    # Multi-step returns; 3 is the Rainbow default.
    n_step: int = Field(default=3, ge=1, le=10)


class EncoderConfig(BaseModel):
    """Which visual trunk to use. Independent of the algorithm.

    ``nature_cnn`` at width 1.0 is the Mnih et al. convnet and is bit-exact
    with the pre-refactor network, so it remains the baseline.
    """

    name: str = "nature_cnn"
    hidden: int = Field(default=512, gt=0)
    """Width of the head's hidden layer."""
    width: float = Field(default=1.0, gt=0.0)
    """Channel multiplier. 1.0 is the published architecture."""
    channels: list[int] | None = None
    """Per-stage channels; only used by encoders that take them (impala)."""

    @property
    def kwargs(self) -> dict:
        """Extra arguments for the encoder constructor, minus head-only keys."""
        out: dict = {"width": self.width}
        if self.channels is not None:
            out["channels"] = tuple(self.channels)
        return out


class PPOConfig(BaseModel):
    """On-policy actor-critic. Collects a segment, then does several epochs on it.

    Unlike DQN there is no replay and no epsilon: the policy is stochastic by
    construction and exploration comes from entropy regularisation.
    """

    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    """Bias/variance dial for advantage estimation. 1.0 = Monte Carlo, 0.0 = TD(0)."""
    lr: float = Field(default=2.5e-4, gt=0.0)
    adam_eps: float = Field(default=1e-5, gt=0.0)

    horizon: int = Field(default=128, gt=0)
    """Steps collected per env before each update."""
    epochs: int = Field(default=4, gt=0)
    minibatch_size: int = Field(default=256, gt=0)

    clip_coef: float = Field(default=0.1, gt=0.0)
    """The trust region. Larger moves faster and destabilises sooner."""
    value_coef: float = Field(default=0.5, ge=0.0)
    entropy_coef: float = Field(default=0.01, ge=0.0)
    """Exploration pressure. PPO's counterpart to epsilon."""
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    normalize_advantage: bool = True
    clip_value_loss: bool = True
    target_kl: float | None = Field(default=0.03, gt=0.0)
    """Stop the epoch loop early if the policy has moved this far. None disables."""


class DQNConfig(BaseModel):
    """Modern DQN. The original was vanilla 2013-era DQN with an MSE loss."""

    double: bool = True
    dueling: bool = True
    noisy: bool = False

    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    lr: float = Field(default=6.25e-5, gt=0.0)
    adam_eps: float = Field(default=1.5e-4, gt=0.0)
    batch_size: int = Field(default=32, gt=0)
    # Huber, not MSE -- bounded gradients on outlier TD errors.
    huber_delta: float = Field(default=1.0, gt=0.0)
    max_grad_norm: float = Field(default=10.0, gt=0.0)

    learn_start: int = Field(default=20_000, ge=0)
    train_every: int = Field(default=4, ge=1)
    target_sync_every: int = Field(default=8_000, ge=1)

    eps_start: float = Field(default=1.0, ge=0.0, le=1.0)
    eps_end: float = Field(default=0.01, ge=0.0, le=1.0)
    # Anneal over ENV STEPS, not over gradient updates. The old code decayed epsilon
    # inside learn(), so the schedule silently depended on train_every.
    eps_decay_steps: int = Field(default=500_000, gt=0)


class TrainConfig(BaseModel):
    total_steps: int = Field(default=10_000_000, gt=0)
    num_envs: int = Field(default=8, ge=1, le=64)
    seed: int = 0
    device: Literal["auto", "cuda", "cpu"] = "auto"
    torch_compile: bool = True
    amp: bool = True

    run_dir: Path = Path("runs")
    run_name: str | None = None
    checkpoint_every: int = Field(default=100_000, gt=0)
    eval_every: int = Field(default=250_000, gt=0)
    eval_episodes: int = Field(default=10, gt=0)
    record_eval_video: bool = True
    # Always resume from the latest checkpoint if one exists. The old project
    # defined load_models() and never called it, so every run started from zero.
    auto_resume: bool = True


class Config(BaseModel):
    # Which learner to run. The encoder is chosen independently, so any
    # algorithm can be paired with any trunk.
    algo: str = "dqn"
    encoder: EncoderConfig = EncoderConfig()

    env: EnvConfig = EnvConfig()
    reward: RewardConfig = RewardConfig()
    replay: ReplayConfig = ReplayConfig()
    dqn: DQNConfig = DQNConfig()
    ppo: PPOConfig = PPOConfig()
    train: TrainConfig = TrainConfig()

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
