"""Proximal Policy Optimisation (Schulman et al. 2017).

The natural counterpart to DQN here, and the reason the algorithm layer exists:
PPO is on-policy, so it changes the *shape* of training, not just the loss.

* It stores one short segment collected by the current policy, not a replay of
  old experience, and discards it after a few epochs.
* It updates in bursts (``horizon`` steps of collection, then N epochs over
  that data) rather than a little at every step.
* It is stochastic by construction, so there is no epsilon schedule; entropy
  regularisation supplies the exploration instead.

On a long-horizon game like this one it tends to be steadier than DQN but
slower per unit of wall clock, because it cannot reuse experience. Worth racing
against the DQN baseline rather than assuming either wins.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from kungfu.config import EncoderConfig, PPOConfig
from kungfu.rl.algos.base import Agent, Transition, UpdateStats
from kungfu.rl.buffers import RolloutBuffer
from kungfu.rl.networks import ActorCriticNetwork


class PPOAgent(Agent):
    on_policy = True
    name = "ppo"

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        ppo_cfg: PPOConfig,
        device: torch.device,
        seed: int = 0,
        num_envs: int = 1,
        encoder_cfg: EncoderConfig | None = None,
    ) -> None:
        super().__init__(device=device, num_envs=num_envs, seed=seed)
        c, h, w = obs_shape
        self.cfg = ppo_cfg
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        enc = encoder_cfg or EncoderConfig()

        self.net = ActorCriticNetwork(
            c, n_actions, (h, w),
            hidden=enc.hidden,
            encoder=enc.name,
            encoder_kwargs=enc.kwargs,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=ppo_cfg.lr, eps=ppo_cfg.adam_eps
        )
        self.buffer = RolloutBuffer(
            horizon=ppo_cfg.horizon,
            num_envs=num_envs,
            obs_shape=obs_shape,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            device=device,
        )

        self._amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp)
        self._last_obs: np.ndarray | None = None
        self._last_dones = np.zeros(num_envs, dtype=bool)
        self._last_entropy = 0.0

    @property
    def modules(self) -> dict[str, nn.Module]:
        return {"net": self.net}

    def exploration(self) -> float:
        """Policy entropy: PPO's analogue of epsilon, but measured not scheduled."""
        return self._last_entropy

    # -- acting ------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple[np.ndarray, dict]:
        t = torch.as_tensor(obs, device=self.device)
        logits, value = self.net(t)
        if greedy:
            actions = logits.argmax(dim=1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        self._last_entropy = float(dist.entropy().mean().cpu())
        return (
            actions.cpu().numpy(),
            {
                "logprob": logprob.cpu().numpy().astype(np.float32),
                "value": value.cpu().numpy().astype(np.float32),
            },
        )

    # -- learning ----------------------------------------------------------
    def observe(self, t: Transition) -> None:
        if "logprob" not in t.extras:
            raise ValueError("PPO needs the logprob and value from act(); pass extras through")

        rewards = np.asarray(t.rewards, dtype=np.float32).copy()
        if t.truncated.any():
            # A time-limit truncation is NOT a terminal: the episode would have
            # continued, so the future still has value. Masking it as terminal
            # teaches the agent that the world ends at the cap -- and with the
            # 6000-step limit, that was 95% of episodes.
            #
            # The fix is to fold the bootstrapped value of the state we were cut
            # off in into that step's reward, so GAE can mask on `terminated`
            # alone. Gymnasium's NEXT_STEP autoreset returns the true final
            # observation on the terminating step, so next_obs is the right
            # state to evaluate.
            idx = np.flatnonzero(t.truncated)
            with torch.no_grad():
                v = self.net.value_only(
                    torch.as_tensor(t.next_obs[idx], device=self.device)
                )
            rewards[idx] += self.cfg.gamma * v.cpu().numpy().astype(np.float32)

        self.buffer.add(
            obs=t.obs,
            actions=t.actions,
            logprobs=t.extras["logprob"],
            rewards=rewards,
            values=t.extras["value"],
            dones=t.terminated,
        )
        self._last_obs = t.next_obs
        self._last_dones = t.terminated
        self.steps += self.num_envs

    def maybe_update(self) -> UpdateStats | None:
        if not self.buffer.full:
            return None
        return self.learn()

    def learn(self) -> UpdateStats | None:
        # Bootstrap the tail of the segment from the value of the state we
        # stopped at; without this the last steps get truncated returns.
        with torch.no_grad():
            last_value = (
                self.net.value_only(torch.as_tensor(self._last_obs, device=self.device))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        advantages, returns = self.buffer.compute_returns(last_value)

        losses, pg_losses, v_losses, entropies, clipfracs, kls = [], [], [], [], [], []
        grad_norm = 0.0

        for _ in range(self.cfg.epochs):
            for mb in self.buffer.batches(
                advantages, returns, self.cfg.minibatch_size, self.rng
            ):
                obs = torch.as_tensor(mb["obs"], device=self.device)
                acts = torch.as_tensor(mb["actions"], device=self.device).long()
                old_logp = torch.as_tensor(mb["logprobs"], device=self.device)
                adv = torch.as_tensor(mb["advantages"], device=self.device)
                ret = torch.as_tensor(mb["returns"], device=self.device)
                old_val = torch.as_tensor(mb["values"], device=self.device)

                if self.cfg.normalize_advantage and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                with torch.amp.autocast("cuda", enabled=self._amp):
                    logits, value = self.net(obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(acts)
                    entropy = dist.entropy().mean()

                    ratio = (logp - old_logp).exp()
                    # The clipped surrogate: take the pessimistic branch so an
                    # update that moves the policy far from the data-collecting
                    # policy gains nothing.
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef) * adv
                    pg_loss = -torch.min(unclipped, clipped).mean()

                    if self.cfg.clip_value_loss:
                        v_clipped = old_val + torch.clamp(
                            value - old_val, -self.cfg.clip_coef, self.cfg.clip_coef
                        )
                        v_loss = 0.5 * torch.max(
                            (value - ret) ** 2, (v_clipped - ret) ** 2
                        ).mean()
                    else:
                        v_loss = 0.5 * ((value - ret) ** 2).mean()

                    loss = (
                        pg_loss
                        + self.cfg.value_coef * v_loss
                        - self.cfg.entropy_coef * entropy
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                with torch.no_grad():
                    log_ratio = logp - old_logp
                    # Schulman's low-variance KL estimator; the usual early-stop
                    # signal that the policy has moved too far this update.
                    approx_kl = float(((ratio - 1) - log_ratio).mean().cpu())
                    clipfracs.append(
                        float(((ratio - 1).abs() > self.cfg.clip_coef).float().mean().cpu())
                    )
                losses.append(float(loss.detach().cpu()))
                pg_losses.append(float(pg_loss.detach().cpu()))
                v_losses.append(float(v_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                kls.append(approx_kl)

            if self.cfg.target_kl is not None and kls and kls[-1] > self.cfg.target_kl:
                break

        self.buffer.reset()
        self.updates += 1

        return UpdateStats(
            loss=float(np.mean(losses)),
            extra={
                "policy_loss": float(np.mean(pg_losses)),
                "value_loss": float(np.mean(v_losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": float(np.mean(kls)),
                "clip_fraction": float(np.mean(clipfracs)),
                "grad_norm": grad_norm,
            },
        )

    # -- checkpoints -------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "steps": self.steps,
            "updates": self.updates,
            "config": self.cfg.model_dump(),
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self.net.load_state_dict(state["net"], strict=strict)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        self.steps = int(state.get("steps", 0))
        self.updates = int(state.get("updates", 0))
