"""Double + Dueling + n-step DQN with prioritised replay.

Corrections to the original agent
----------------------------------
* **Target network was never synchronised at init.** ``q_eval`` and ``q_target``
  were two independently random networks, so the first 5,000 steps of bootstrapping
  aimed at pure noise.
* **No Double DQN.** Vanilla ``max_a Q_target(s', a)`` systematically
  over-estimates. Double DQN (select with online, evaluate with target) is a
  two-line change that was already standard practice in 2016.
* **MSE loss.** Replaced with Huber: a single bad TD error no longer produces a
  gradient spike large enough to wreck the network.
* **No gradient clipping.**
* **Epsilon decayed inside ``learn()``**, which only ran every 4th step and only
  after burn-in, so the exploration schedule was implicitly coupled to
  ``learn_every`` and to burn-in length. It is now a pure function of env steps.
* **``load_models()`` existed but was never called**, so every run restarted from
  scratch -- which makes the self-improvement cycle impossible. Checkpoints here
  round-trip the optimizer state and the step counter too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from kungfu.config import DQNConfig, ReplayConfig
from kungfu.rl.networks import QNetwork
from kungfu.rl.replay import FrameReplayBuffer


@dataclass
class LearnStats:
    loss: float
    q_mean: float
    td_error: float
    grad_norm: float


class DQNAgent:
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        dqn_cfg: DQNConfig,
        replay_cfg: ReplayConfig,
        device: torch.device,
        seed: int = 0,
        num_envs: int = 1,
    ) -> None:
        c, h, w = obs_shape
        self.cfg = dqn_cfg
        self.replay_cfg = replay_cfg
        self.device = device
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        self.online = QNetwork(c, n_actions, (h, w), dqn_cfg.dueling, dqn_cfg.noisy).to(device)
        self.target = QNetwork(c, n_actions, (h, w), dqn_cfg.dueling, dqn_cfg.noisy).to(device)
        # Synchronise immediately -- the thing the original forgot.
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=dqn_cfg.lr, eps=dqn_cfg.adam_eps
        )
        self.loss_fn = nn.HuberLoss(reduction="none", delta=dqn_cfg.huber_delta)

        self.memory = FrameReplayBuffer(
            capacity=replay_cfg.capacity,
            frame_shape=(h, w),
            stack=c,
            n_step=replay_cfg.n_step,
            gamma=dqn_cfg.gamma,
            prioritized=replay_cfg.prioritized,
            alpha=replay_cfg.alpha,
            num_envs=num_envs,
        )

        self.num_envs = num_envs
        self.steps = 0
        self.updates = 0
        self._amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp)

    # -- schedules ---------------------------------------------------------
    @property
    def epsilon(self) -> float:
        if self.cfg.noisy:
            return 0.0
        frac = min(1.0, self.steps / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @property
    def beta(self) -> float:
        frac = min(1.0, self.steps / self.replay_cfg.beta_frames)
        return self.replay_cfg.beta_start + frac * (1.0 - self.replay_cfg.beta_start)

    # -- acting ------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        """``obs`` is (B, C, H, W) uint8. Returns (B,) int64 actions."""
        batch = obs.shape[0]
        eps = 0.0 if greedy else self.epsilon
        actions = self.rng.integers(0, self.n_actions, size=batch)

        explore = self.rng.random(batch) < eps
        if explore.all():
            return actions

        t = torch.as_tensor(obs, device=self.device)
        self.online.eval()
        q = self.online(t)
        self.online.train()
        greedy_actions = q.argmax(dim=1).cpu().numpy()
        return np.where(explore, actions, greedy_actions)

    def remember(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        episode_starts: np.ndarray,
    ) -> None:
        """Record one vectorised timestep. ``obs`` is (E, C, H, W) uint8."""
        self.memory.add(obs, actions, rewards, dones, episode_starts)

    # -- learning ----------------------------------------------------------
    def learn(self) -> LearnStats | None:
        if self.steps < self.cfg.learn_start or not self.memory.ready:
            return None

        batch = self.memory.sample(self.cfg.batch_size, self.beta, self.rng)

        states = torch.as_tensor(batch["states"], device=self.device)
        next_states = torch.as_tensor(batch["next_states"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device).long()
        rewards = torch.as_tensor(batch["rewards"], device=self.device).float()
        dones = torch.as_tensor(batch["dones"], device=self.device).float()
        discounts = torch.as_tensor(batch["discounts"], device=self.device).float()
        weights = torch.as_tensor(batch["weights"], device=self.device).float()

        if self.cfg.noisy:
            self.online.reset_noise()
            self.target.reset_noise()

        with torch.amp.autocast("cuda", enabled=self._amp):
            q_pred = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if self.cfg.double:
                    # Select with the online net, evaluate with the target net.
                    next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                    q_next = self.target(next_states).gather(1, next_actions).squeeze(1)
                else:
                    q_next = self.target(next_states).max(dim=1).values
                target = rewards + discounts * q_next * (1.0 - dones)

            elementwise = self.loss_fn(q_pred, target)
            loss = (elementwise * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.cfg.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        td = (q_pred.detach() - target).abs().float().cpu().numpy()
        self.memory.update_priorities(batch["env_indices"], batch["indices"], td)

        self.updates += 1
        if self.updates % max(1, self.cfg.target_sync_every // self.cfg.train_every) == 0:
            self.target.load_state_dict(self.online.state_dict())

        return LearnStats(
            loss=float(loss.detach().cpu()),
            q_mean=float(q_pred.detach().mean().cpu()),
            td_error=float(td.mean()),
            grad_norm=float(grad_norm),
        )

    # -- checkpoints -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "steps": self.steps,
                "updates": self.updates,
                "config": self.cfg.model_dump(),
            },
            p,
        )

    def load(self, path: str | Path, strict: bool = True) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"], strict=strict)
        self.target.load_state_dict(ckpt["target"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.steps = int(ckpt.get("steps", 0))
        self.updates = int(ckpt.get("updates", 0))
