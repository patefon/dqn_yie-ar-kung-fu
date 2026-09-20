"""Gymnasium environment: pixels in, vision-derived reward out.

The agent observes only the framebuffer. Emulator RAM is never read here.

What changed versus the old pipeline
------------------------------------
The old design ran the grabber, the detector and the agent as three independent
processes joined by bounded queues that *dropped frames when full*
(``put_nowait`` + ``except queue.Full: pass``). The consequence is subtle and
fatal: the frame the agent received after choosing action ``a`` was not
necessarily the frame that action produced -- it could predate the keypress, or
postdate it by an unknown number of frames. Q-learning assumes ``(s, a, r, s')``
is causally linked. It was not, so the agent was fitting a Bellman equation to
noise. That, more than any hyperparameter, is why it stalled around 5 levels.

Here ``step()`` is synchronous: press buttons, advance the emulator exactly
``frame_skip`` frames, read the resulting frame. The MDP is real.
"""

from __future__ import annotations

from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from kungfu.config import EnvConfig, RewardConfig
from kungfu.envs.actions import NUM_ACTIONS, build_action_table
from kungfu.envs.reward import RewardShaper
from kungfu.vision.atlas import DigitAtlas
from kungfu.vision.hud import HudReader


class YieArKungFuEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        atlas: DigitAtlas,
        render_mode: str | None = None,
        capture_audio: bool = False,
    ) -> None:
        super().__init__()
        try:
            import stable_retro as retro
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "stable-retro is required. Install with `uv pip install stable-retro`, "
                "then run `python -m tools.setup_integration` to register the ROM."
            ) from exc

        self.cfg = env_cfg
        self.render_mode = render_mode
        # Only the demo recorder needs sound; training never touches this.
        self.capture_audio = capture_audio
        self._audio: list[np.ndarray] = []

        self._retro = retro.make(
            game=env_cfg.game,
            state=env_cfg.state,
            use_restricted_actions=retro.Actions.ALL,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            render_mode=None,
        )
        self._action_table = build_action_table()
        self._hud = HudReader(env_cfg.hud, atlas)
        self._shaper = RewardShaper(reward_cfg)

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        channels = 1 if env_cfg.grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(channels, env_cfg.obs_height, env_cfg.obs_width),
            dtype=np.uint8,
        )

        self._last_action = 0
        self._elapsed = 0
        self._steps_since_score = 0
        self._best_score = 0
        self._gameover_streak = 0
        self._last_frame: np.ndarray | None = None

    # -- observation -------------------------------------------------------
    def _observe(self, frame: np.ndarray) -> np.ndarray:
        """Crop to the playfield, grayscale, downscale.

        Cropping away the HUD matters: the score digits are a fast-changing,
        high-contrast region that carries no control-relevant information, and
        leaving them in gives the convnet a spurious, easily-memorised signal.
        """
        play = self.cfg.hud.playfield.crop(frame)
        if self.cfg.grayscale:
            play = cv2.cvtColor(play, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            play,
            (self.cfg.obs_width, self.cfg.obs_height),
            interpolation=cv2.INTER_AREA,
        )
        if self.cfg.grayscale:
            return resized[None, :, :].astype(np.uint8)
        return np.transpose(resized, (2, 0, 1)).astype(np.uint8)

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        frame, _ = self._retro.reset(seed=seed)

        # Random no-ops decorrelate parallel envs so they do not all walk the
        # same trajectory, which would make a batch of 8 envs worth ~1 env.
        noops = int(self.np_random.integers(0, self.cfg.noop_max + 1)) if self.cfg.noop_max else 0
        for _ in range(noops):
            frame, _, term, trunc, _ = self._retro.step(self._action_table[0])
            if term or trunc:
                frame, _ = self._retro.reset()
                break

        self._shaper.reset()
        self._last_action = 0
        self._elapsed = 0
        self._steps_since_score = 0
        self._best_score = 0
        self._gameover_streak = 0
        self._last_frame = frame

        # Prime the shaper's baselines without emitting reward.
        self._shaper.step(self._hud.read(frame))
        return self._observe(frame), {}

    def step(self, action: int):
        # Sticky actions: with probability p the previous action repeats. Without
        # this the policy can lock onto a frame-perfect deterministic script that
        # does not generalise -- a known failure mode on retro benchmarks.
        if self.cfg.sticky_action_prob > 0 and self.np_random.random() < self.cfg.sticky_action_prob:
            action = self._last_action
        self._last_action = int(action)

        buttons = self._action_table[int(action)]

        frames: list[np.ndarray] = []
        terminated = False
        for i in range(self.cfg.frame_skip):
            frame, _, term, trunc, _ = self._retro.step(buttons)
            # Audio must be drained every emulator frame, not once per agent
            # step: the core clears its buffer on each step, so sampling only
            # the outer loop would keep 1 frame of sound in every frame_skip.
            if self.capture_audio:
                chunk = self._retro.em.get_audio()
                if chunk is not None and len(chunk):
                    self._audio.append(np.asarray(chunk, dtype=np.int16).copy())
            # Only the final two frames matter for max-pooling.
            if i >= self.cfg.frame_skip - 2:
                frames.append(frame)
            if term or trunc:
                terminated = True
                break

        # NES sprites flicker on alternating frames when many are on screen;
        # max-pooling the last two makes them reliably visible.
        if self.cfg.max_pool_frames and len(frames) >= 2:
            frame = np.maximum(frames[-1], frames[-2])
        else:
            frame = frames[-1]
        self._last_frame = frame

        stats = self._hud.read(frame)
        reward, breakdown = self._shaper.step(stats)

        self._elapsed += 1
        if stats.score is not None and stats.score > self._best_score:
            self._best_score = stats.score
            self._steps_since_score = 0
        else:
            self._steps_since_score += 1

        # Debounce: both bars read empty for a frame or two during a normal KO,
        # so a single hit of the signature is not game over.
        self._gameover_streak = self._gameover_streak + 1 if stats.is_gameover else 0
        terminated = terminated or self._gameover_streak >= self.cfg.gameover_patience
        truncated = (
            self._elapsed >= self.cfg.max_episode_steps
            or self._steps_since_score >= self.cfg.stall_timeout
        )

        # Gymnasium's vector envs aggregate `info` into typed arrays and cannot
        # hold None, so unreadable fields are reported as -1 rather than None.
        # `hud_readable` says whether the sentinels are in play.
        info: dict[str, Any] = {
            "score": -1 if stats.score is None else int(stats.score),
            "stage": -1 if stats.stage is None else int(stats.stage),
            "player_health": -1.0 if stats.player_health is None else float(stats.player_health),
            "enemy_health": -1.0 if stats.enemy_health is None else float(stats.enemy_health),
            "lives": -1 if stats.lives is None else int(stats.lives),
            "hud_readable": bool(stats.readable),
            "best_score": int(self._best_score),
            "misreads": int(self._shaper.misreads),
            "reward_terms": breakdown.as_dict(),
        }
        return self._observe(frame), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._last_frame
        return None

    # -- audio (demo recording only) ---------------------------------------
    @property
    def audio_rate(self) -> int:
        return int(self._retro.em.get_audio_rate())

    def pop_audio(self) -> np.ndarray:
        """Return and clear the audio captured since the last call."""
        if not self._audio:
            return np.zeros((0, 2), dtype=np.int16)
        out = np.concatenate(self._audio, axis=0)
        self._audio.clear()
        return out

    def close(self):
        self._retro.close()
