"""Q-network.

Differences from the original ``SimpleNet``
--------------------------------------------
* **Normalisation moved out of ``forward``.** The old net did
  ``x = T.tensor(input).to(device).float(); x /= 255.0`` *inside* ``forward``,
  so the module silently owned host-to-device transfer and dtype conversion.
  That makes it impossible to batch efficiently, impossible to ``torch.compile``
  cleanly, and it re-uploaded a single observation to the GPU on every
  action selection. Tensors now arrive ready; the module only scales.
* **Dueling head** (Wang et al. 2016): separates state value from action
  advantage, which helps a lot in states where the action barely matters --
  most of a fighting game is positioning.
* **Orthogonal init.** The original relied on PyTorch defaults.
* **Hard-coded flatten size removed.** ``nn.Linear(8320, 512)`` silently
  encoded the exact input resolution; changing the crop broke the net with a
  shape error. The size is now inferred.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _orthogonal(module: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    if isinstance(module, nn.Conv2d | nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


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


class QNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        input_hw: tuple[int, int] = (84, 84),
        dueling: bool = True,
        noisy: bool = False,
        hidden: int = 512,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.dueling = dueling
        self.noisy = noisy

        self.features = nn.Sequential(
            _orthogonal(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            _orthogonal(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            _orthogonal(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Infer the flattened width instead of hard-coding it.
        with torch.no_grad():
            probe = torch.zeros(1, in_channels, *input_hw)
            flat = self.features(probe).shape[1]

        linear = (lambda i, o: NoisyLinear(i, o)) if noisy else (lambda i, o: _orthogonal(nn.Linear(i, o)))

        self.advantage = nn.Sequential(
            linear(flat, hidden), nn.ReLU(inplace=True), linear(hidden, n_actions)
        )
        if dueling:
            self.value = nn.Sequential(
                linear(flat, hidden), nn.ReLU(inplace=True), linear(hidden, 1)
            )

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs`` is uint8 or float in [0, 255], shape (B, C, H, W)."""
        x = obs.float().div_(255.0) if obs.dtype == torch.uint8 else obs / 255.0
        z = self.features(x)
        adv = self.advantage(z)
        if not self.dueling:
            return adv
        val = self.value(z)
        # Mean-subtracted advantage keeps the decomposition identifiable.
        return val + adv - adv.mean(dim=1, keepdim=True)
