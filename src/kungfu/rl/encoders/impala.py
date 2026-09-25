"""IMPALA residual encoder (Espeholt et al. 2018).

Deeper than NatureCNN and residual, which in practice generalises better across
visually varied levels. That is the interesting property here: Yie Ar Kung-Fu
changes backdrop and opponent sprite every stage, and the agent currently dies
at stage 21 — a plausible cause is a trunk that has overfitted to the early
stages' visuals rather than learned to read a fight.

Costs more per step than NatureCNN, so it trades throughput for capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kungfu.rl.encoders.base import Encoder, orthogonal_


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a skip connection; pre-activation ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = orthogonal_(nn.Conv2d(channels, channels, 3, padding=1))
        self.conv2 = orthogonal_(nn.Conv2d(channels, channels, 3, padding=1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-activation: ReLU before each conv, identity path left untouched
        # so gradients reach the early layers directly.
        out = self.conv1(self.relu(x))
        out = self.conv2(self.relu(out))
        return out + x


class ImpalaStage(nn.Module):
    """Conv -> max-pool -> two residual blocks. Halves the spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = orthogonal_(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res2(self.res1(self.pool(self.conv(x))))


class ImpalaCNN(Encoder):
    def __init__(
        self,
        in_channels: int,
        input_hw: tuple[int, int] = (84, 84),
        channels: tuple[int, ...] = (16, 32, 32),
        width: float = 1.0,
    ) -> None:
        super().__init__(in_channels, input_hw)
        if not channels:
            raise ValueError("channels must not be empty")
        if width <= 0:
            raise ValueError("width must be > 0")

        scaled = [max(1, int(round(c * width))) for c in channels]
        stages: list[nn.Module] = []
        prev = in_channels
        for c in scaled:
            stages.append(ImpalaStage(prev, c))
            prev = c

        self.net = nn.Sequential(*stages, nn.ReLU(inplace=True), nn.Flatten())
        self._out_features = self._infer_out_features(self.net)

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
