"""Encoder interface: pixels in, a flat feature vector out.

An encoder is the visual trunk. It knows nothing about Q-values, policies or
advantages -- that is the head's job -- so the same encoder serves DQN and PPO
unchanged.

Contract
--------
* input  : ``(B, C, H, W)`` float already scaled to [0, 1]
* output : ``(B, out_features)``
* ``out_features`` is available before the first forward pass, because the
  heads need it to size their layers.

Normalisation lives in the network wrapper, not here, so every encoder sees the
same units and a new one cannot get the scaling subtly wrong.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


def orthogonal_(module: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """Orthogonal weights, zero bias -- the standard init for RL convnets."""
    if isinstance(module, nn.Conv2d | nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


class Encoder(nn.Module, ABC):
    """Base class for visual trunks."""

    def __init__(self, in_channels: int, input_hw: tuple[int, int]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.input_hw = input_hw

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Width of the flat vector this encoder produces."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` in [0, 1] -> ``(B, out_features)``."""

    def _infer_out_features(self, net: nn.Module) -> int:
        """Run a zero probe to measure the flattened width.

        Beats hard-coding it: the original implementation baked
        ``nn.Linear(8320, 512)`` into the net, so changing the observation crop
        produced a shape error rather than a working network.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            probe = torch.zeros(1, self.in_channels, *self.input_hw)
            width = int(net(probe).shape[1])
        self.train(was_training)
        return width
