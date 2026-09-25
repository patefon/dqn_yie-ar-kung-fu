"""Output heads: encoder features in, algorithm-specific predictions out.

Splitting head from encoder is what makes one trunk serve several algorithms.
A Q-learner needs per-action values; an actor-critic needs logits plus a scalar
value. Both consume the same ``(B, out_features)`` vector.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from kungfu.rl.encoders import orthogonal_


class NoisyLinear(nn.Module):
    """Factorised noisy layer (Fortunato et al. 2018).

    An alternative to epsilon-greedy: exploration becomes a learned, state
    dependent property instead of a hand-tuned schedule.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma0 * bound)
        self.bias_sigma.data.fill_(self.sigma0 * bound)

    @staticmethod
    def _scale(size: int, device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scale(self.in_features, self.weight_mu.device)
        eps_out = self._scale(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return nn.functional.linear(x, w, b)


def make_linear(noisy: bool):
    """Layer factory so a head does not care which linear it is using."""
    if noisy:
        return lambda i, o: NoisyLinear(i, o)
    return lambda i, o: orthogonal_(nn.Linear(i, o))


def mlp(in_features: int, hidden: int, out_features: int, noisy: bool) -> nn.Sequential:
    linear = make_linear(noisy)
    return nn.Sequential(
        linear(in_features, hidden), nn.ReLU(inplace=True), linear(hidden, out_features)
    )


class QHead(nn.Module):
    """Plain Q-values, one per action."""

    def __init__(self, in_features: int, n_actions: int, hidden: int = 512, noisy: bool = False):
        super().__init__()
        self.advantage = mlp(in_features, hidden, n_actions, noisy)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.advantage(z)


class DuelingQHead(nn.Module):
    """Value/advantage decomposition (Wang et al. 2016).

    Helps where the action barely matters, which is most of a fighting game:
    positioning dominates, and only a few frames are decision points.

    The submodules are named ``advantage`` and ``value`` deliberately -- those
    names appear in every checkpoint trained before the refactor.
    """

    def __init__(self, in_features: int, n_actions: int, hidden: int = 512, noisy: bool = False):
        super().__init__()
        self.advantage = mlp(in_features, hidden, n_actions, noisy)
        self.value = mlp(in_features, hidden, 1, noisy)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        adv = self.advantage(z)
        val = self.value(z)
        # Mean-subtracted advantage keeps the decomposition identifiable.
        return val + adv - adv.mean(dim=1, keepdim=True)


class ActorCriticHead(nn.Module):
    """Policy logits plus a state value, for on-policy algorithms.

    Init gains follow the PPO convention: a small gain on the policy output so
    the initial policy is near-uniform (large logits early make the first
    updates enormous), and 1.0 on the value output.
    """

    def __init__(self, in_features: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.body = nn.Sequential(orthogonal_(nn.Linear(in_features, hidden)), nn.ReLU(inplace=True))
        self.policy = orthogonal_(nn.Linear(hidden, n_actions), gain=0.01)
        self.value = orthogonal_(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(z)
        return self.policy(h), self.value(h).squeeze(-1)
