"""Double + Dueling + n-step DQN with prioritised replay.

The learning maths here is unchanged from the pre-refactor implementation --
same targets, same loss, same schedules, same update cadence. What moved is the
plumbing: the network is now an encoder + head pair, and the agent implements
the shared ``Agent`` interface so the training loop does not know it is DQN.

Corrections to the 2021 original are documented in REVIEW.md; the load-bearing
ones are Double Q targets, Huber loss with gradient clipping, a target network
synchronised at construction, and an epsilon schedule that is a pure function
of environment steps rather than of gradient updates.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from kungfu.config import DQNConfig, EncoderConfig, ReplayConfig
from kungfu.rl.algos.base import Agent, Transition, UpdateStats
from kungfu.rl.buffers import FrameReplayBuffer
from kungfu.rl.networks import QNetwork


class DQNAgent(Agent):
    on_policy = False
    name = "dqn"

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        dqn_cfg: DQNConfig,
        replay_cfg: ReplayConfig,
        device: torch.device,
        seed: int = 0,
        num_envs: int = 1,
        encoder_cfg: EncoderConfig | None = None,
    ) -> None:
        super().__init__(device=device, num_envs=num_envs, seed=seed)
        c, h, w = obs_shape
        self.cfg = dqn_cfg
        self.replay_cfg = replay_cfg
        self.n_actions = n_actions
        enc = encoder_cfg or EncoderConfig()

        def build() -> QNetwork:
            return QNetwork(
                c, n_actions, (h, w),
                dueling=dqn_cfg.dueling,
                noisy=dqn_cfg.noisy,
                hidden=enc.hidden,
                encoder=enc.name,
                encoder_kwargs=enc.kwargs,
            ).to(device)

        self.online = build()
        self.target = build()
        # Synchronise immediately -- the 2021 version left these independently
        # random, so early bootstrapping aimed at noise.
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

        self._amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp)

    @property
    def modules(self) -> dict[str, nn.Module]:
        return {"online": self.online, "target": self.target}

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

    def exploration(self) -> float:
        return self.epsilon

    # -- acting ------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple[np.ndarray, dict]:
        batch = obs.shape[0]
        eps = 0.0 if greedy else self.epsilon
        actions = self.rng.integers(0, self.n_actions, size=batch)

        explore = self.rng.random(batch) < eps
        if explore.all():
            return actions, {}

        t = torch.as_tensor(obs, device=self.device)
        self.online.eval()
        q = self.online(t)
        self.online.train()
        greedy_actions = q.argmax(dim=1).cpu().numpy()
        return np.where(explore, actions, greedy_actions), {}

    # -- learning ----------------------------------------------------------
    def observe(self, t: Transition) -> None:
        # Only a true terminal breaks bootstrapping; a time-limit truncation
        # must still bootstrap or the agent learns the clock is a cliff.
        self.memory.add(t.obs, t.actions, t.rewards, t.terminated, t.episode_starts)
        self.steps += self.num_envs

    def maybe_update(self) -> UpdateStats | None:
        if self.steps % self.cfg.train_every >= self.num_envs:
            return None
        return self.learn()

    def learn(self) -> UpdateStats | None:
        # Gate on how much is actually in the buffer, not on the global step.
        # They agree on a fresh run, but a resumed checkpoint arrives with
        # steps already in the millions and an *empty* replay: gating on steps
        # would start training instantly on ~10 transitions, and prioritised
        # sampling would grind those few frames into the weights.
        if len(self.memory) < self.cfg.learn_start or not self.memory.ready:
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

        return UpdateStats(
            loss=float(loss.detach().cpu()),
            extra={
                "q_mean": float(q_pred.detach().mean().cpu()),
                "td_error": float(td.mean()),
                "grad_norm": float(grad_norm),
                "beta": self.beta,
            },
        )

    # -- checkpoints -------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "steps": self.steps,
            "updates": self.updates,
            "config": self.cfg.model_dump(),
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        # QNetwork.load_state_dict migrates pre-refactor key names, so
        # checkpoints from before the encoder/head split still load.
        self.online.load_state_dict(state["online"], strict=strict)
        self.target.load_state_dict(state["target"], strict=strict)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        self.steps = int(state.get("steps", 0))
        self.updates = int(state.get("updates", 0))
