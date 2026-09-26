"""The interface every algorithm implements.

The training loop is deliberately ignorant of *when* an algorithm learns. DQN
updates every few env steps from a replay buffer; PPO collects a fixed horizon
and then does several epochs over it. Baking either cadence into the loop
would mean the loop needs an if-chain per algorithm.

So the loop only ever does:

    actions, extras = agent.act(obs)
    next_obs, reward, term, trunc, info = envs.step(actions)
    agent.observe(Transition(...))
    stats = agent.maybe_update()     # the agent decides whether anything happens

and each algorithm owns its own schedule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass(slots=True)
class Transition:
    """One vectorised timestep. Arrays are (num_envs, ...)."""

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_obs: np.ndarray
    episode_starts: np.ndarray
    # Algorithm-specific values act() produced, e.g. PPO's logprob and value.
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def done(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)


@dataclass
class UpdateStats:
    """Scalars to log. `extra` keeps algorithm-specific metrics out of the loop."""

    loss: float
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {"loss": self.loss, **self.extra}


class Agent(ABC):
    """Base class for learners."""

    #: On-policy algorithms cannot resume mid-segment from a checkpoint and
    #: cannot reuse stale data; the loop and the docs both key off this.
    on_policy: bool = False
    #: Human-readable, used in log lines and run metadata.
    name: str = "agent"

    def __init__(self, device: torch.device, num_envs: int, seed: int = 0) -> None:
        self.device = device
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.updates = 0

    # -- acting ------------------------------------------------------------
    @abstractmethod
    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple[np.ndarray, dict]:
        """``obs`` is (E, C, H, W) uint8. Returns (actions, extras)."""

    # -- learning ----------------------------------------------------------
    @abstractmethod
    def observe(self, t: Transition) -> None:
        """Record a timestep. Must advance ``self.steps``."""

    @abstractmethod
    def maybe_update(self) -> UpdateStats | None:
        """Called every env step. Returns stats only when an update happened."""

    # -- introspection -----------------------------------------------------
    @property
    @abstractmethod
    def modules(self) -> dict[str, torch.nn.Module]:
        """Named modules, so the loop can compile or eval them generically."""

    def exploration(self) -> float:
        """A single number describing current exploration, for logging."""
        return 0.0

    def train_mode(self, training: bool = True) -> None:
        for m in self.modules.values():
            m.train(training)

    # -- checkpoints -------------------------------------------------------
    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        ...

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"algo": self.name, **self.state_dict()}, p)

    def load(self, path: str | Path, strict: bool = True) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        saved = ckpt.get("algo")
        # A PPO checkpoint loaded into DQN would fail deep inside load_state_dict
        # with an unhelpful shape error; say what actually went wrong.
        if saved is not None and saved != self.name:
            raise ValueError(
                f"checkpoint was written by {saved!r} but this run uses {self.name!r}"
            )
        self.load_state_dict(ckpt, strict=strict)
