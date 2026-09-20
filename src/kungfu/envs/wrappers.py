"""Observation wrappers.

``ChannelStack`` is not optional decoration. A single frame of a fighting game
cannot tell you whether the opponent is stepping in or backing off, whether a
leg is extending or retracting -- the state is not Markov without motion. The
original stacked 3 frames, but those frames came from a queue that dropped
entries under load, so the temporal spacing between them was unknown and
variable. Here the spacing is exactly ``frame_skip`` emulator frames, always.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ChannelStack(gym.ObservationWrapper):
    """Stack the last ``n`` observations along the channel axis.

    Input  (C, H, W) uint8
    Output (n*C, H, W) uint8, oldest first.
    """

    def __init__(self, env: gym.Env, n: int = 4) -> None:
        super().__init__(env)
        if n < 1:
            raise ValueError("stack size must be >= 1")
        self.n = n
        c, h, w = env.observation_space.shape
        self._frames: deque[np.ndarray] = deque(maxlen=n)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(n * c, h, w), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        # Repeat the first frame so the stack is full and implies zero motion,
        # rather than padding with black, which reads as a scene cut.
        for _ in range(self.n):
            self._frames.append(obs)
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if not self._frames:
            for _ in range(self.n):
                self._frames.append(obs)
        elif self._frames[-1] is not obs:
            self._frames.append(obs)
        return np.concatenate(list(self._frames), axis=0)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return np.concatenate(list(self._frames), axis=0), reward, terminated, truncated, info


class EpisodeStats(gym.Wrapper):
    """Attach per-episode totals to ``info`` on the terminal step.

    Gymnasium's own ``RecordEpisodeStatistics`` tracks the shaped reward only.
    Tracking game score separately matters here because the shaped return and
    the thing we actually want to maximise are not the same quantity.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._ret = 0.0
        self._len = 0
        self._score = 0

    def reset(self, **kwargs):
        self._ret, self._len, self._score = 0.0, 0, 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ret += float(reward)
        self._len += 1
        if int(info.get("score", -1)) >= 0:
            self._score = max(self._score, int(info["score"]))
        if terminated or truncated:
            info["episode"] = {"r": self._ret, "l": self._len, "score": self._score}
        return obs, reward, terminated, truncated, info
