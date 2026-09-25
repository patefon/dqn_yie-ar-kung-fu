"""The DQN convnet from Mnih et al. 2015.

This is the default, and it is **bit-exact** with the pre-refactor
implementation: same three conv layers, same kernel sizes and strides, same
orthogonal init with gain sqrt(2), same inplace ReLUs, same ordering inside a
single ``nn.Sequential`` named ``net``.

That last detail is not cosmetic. Parameters are named by their module path, so
keeping the sequential named ``net`` gives keys ``encoder.net.0.weight``, which
``kungfu.rl.networks.migrate_legacy_state_dict`` maps the old ``features.0.weight``
onto. Rename it and every checkpoint trained before the refactor stops loading.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kungfu.rl.encoders.base import Encoder, orthogonal_


class NatureCNN(Encoder):
    """84x84 -> 3136 features. Cheap, well understood, hard to beat on Atari-likes."""

    def __init__(
        self,
        in_channels: int,
        input_hw: tuple[int, int] = (84, 84),
        width: float = 1.0,
    ) -> None:
        super().__init__(in_channels, input_hw)
        if width <= 0:
            raise ValueError("width must be > 0")

        # width scales the channel counts; 1.0 reproduces the paper exactly.
        c1, c2, c3 = (max(1, int(round(c * width))) for c in (32, 64, 64))

        self.net = nn.Sequential(
            orthogonal_(nn.Conv2d(in_channels, c1, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            orthogonal_(nn.Conv2d(c1, c2, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            orthogonal_(nn.Conv2d(c2, c3, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self._out_features = self._infer_out_features(self.net)

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
