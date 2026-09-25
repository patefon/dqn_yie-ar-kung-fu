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

# Episode termination reasons, reported in info["end_reason"].
END_NONE = 0
END_GAMEOVER = 1
END_STALL = 2
END_TIME_LIMIT = 3
END_REASON_NAMES = {END_GAMEOVER: "gameover", END_STALL: "stall", END_TIME_LIMIT: "time_limit"}


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

        # Preload every start state once. reset() then only has to assign bytes
        # rather than hit the disk, so a curriculum costs nothing per episode.
        self._states: list[tuple[str, bytes]] = []
        self._state_weights: np.ndarray | None = None
        if env_cfg.start_states:
            import gzip

            for spec in env_cfg.start_states:
                path = retro.data.get_file_path(
                    env_cfg.game,
                    spec.name if spec.name.endswith(".state") else spec.name + ".state",
                    retro.data.Integrations.CUSTOM_ONLY,
                )
                if path is None:
                    raise FileNotFoundError(
                        f"start state {spec.name!r} not found for {env_cfg.game}. "
                        "Generate it with: python -m tools.make_stage_states"
                    )
                with gzip.open(path, "rb") as fh:
                    self._states.append((spec.name, fh.read()))
            w = np.array([s.weight for s in env_cfg.start_states], dtype=np.float64)
            self._state_weights = w / w.sum()
        self._current_state = 0

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
        if self._states:
            self._current_state = int(
                self.np_random.choice(len(self._states), p=self._state_weights)
            )
            self._retro.initial_state = self._states[self._current_state][1]
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
        hit_time_limit = self._elapsed >= self.cfg.max_episode_steps
        hit_stall = self._steps_since_score >= self.cfg.stall_timeout
        truncated = hit_time_limit or hit_stall

        # Why an episode ended is not recoverable after the fact, and the three
        # reasons call for completely different fixes: dying means the policy is
        # weak, stalling means it survives but cannot score, and the time limit
        # means it outlived the clock. Logged as an int because Gymnasium's
        # vector envs aggregate info into typed arrays and cannot hold strings.
        end_reason = END_NONE
        if terminated:
            end_reason = END_GAMEOVER
        elif hit_stall:
            end_reason = END_STALL
        elif hit_time_limit:
            end_reason = END_TIME_LIMIT

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
            "end_reason": int(end_reason),
            "start_state": int(self._current_state),
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
