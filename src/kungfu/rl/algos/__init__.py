"""Algorithm registry and the single place an agent gets constructed.

``build_agent`` is the only function the training loop, the evaluator, the
viewer and the demo recorder need. Adding an algorithm means writing the class
and registering it here -- no call site changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from kungfu.rl.algos.base import Agent, Transition, UpdateStats
from kungfu.rl.algos.dqn import DQNAgent
from kungfu.rl.algos.ppo import PPOAgent
from kungfu.rl.registry import Registry

if TYPE_CHECKING:
    from kungfu.config import Config

ALGOS: Registry[Agent] = Registry("algorithm")
ALGOS.register("dqn")(DQNAgent)
ALGOS.register("ppo")(PPOAgent)


def build_agent(
    cfg: Config,
    obs_shape: tuple[int, int, int],
    n_actions: int,
    device: torch.device,
    seed: int = 0,
    num_envs: int = 1,
) -> Agent:
    """Construct the agent named by ``cfg.algo``.

    Each algorithm takes its own config section, so PPO hyperparameters cannot
    silently leak into a DQN run or vice versa.
    """
    name = cfg.algo.lower()
    common = dict(
        obs_shape=obs_shape,
        n_actions=n_actions,
        device=device,
        seed=seed,
        num_envs=num_envs,
        encoder_cfg=cfg.encoder,
    )
    if name == "dqn":
        return ALGOS.build(name, dqn_cfg=cfg.dqn, replay_cfg=cfg.replay, **common)
    if name == "ppo":
        return ALGOS.build(name, ppo_cfg=cfg.ppo, **common)
    raise KeyError(f"unknown algorithm {cfg.algo!r}. Available: {', '.join(ALGOS.names())}")


__all__ = [
    "ALGOS",
    "Agent",
    "DQNAgent",
    "PPOAgent",
    "Transition",
    "UpdateStats",
    "build_agent",
]
