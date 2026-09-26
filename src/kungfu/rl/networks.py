"""Networks: an encoder plus a head.

Before the refactor this file held one hard-wired convnet with the dueling
head baked in. Now it composes a registered encoder with a head chosen by the
algorithm, so the trunk and the learner vary independently.

The default (``nature_cnn`` + dueling head) is bit-exact with the previous
implementation, and ``migrate_legacy_state_dict`` keeps older checkpoints
loadable -- see ``tests/test_networks_refactor.py``, which pins both.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kungfu.rl.encoders import Encoder, build_encoder
from kungfu.rl.heads import ActorCriticHead, DuelingQHead, NoisyLinear, QHead

# Parameter paths changed when the monolithic net was split into encoder+head.
# The math did not, so old weights remain valid -- they just need renaming.
LEGACY_PREFIX_MAP = {
    "features.": "encoder.net.",
    "advantage.": "head.advantage.",
    "value.": "head.value.",
}


def migrate_legacy_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename pre-refactor keys onto the encoder/head layout.

    Returns the dict unchanged if it is already in the new layout, so calling
    this on a modern checkpoint is a no-op.
    """
    if not any(k.startswith(("features.", "advantage.", "value.")) for k in state):
        return state

    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for old, new in LEGACY_PREFIX_MAP.items():
            if key.startswith(old):
                out[new + key[len(old) :]] = value
                break
        else:
            out[key] = value
    return out


def _normalise(obs: torch.Tensor) -> torch.Tensor:
    """uint8 or float in [0, 255] -> float in [0, 1].

    Kept in the network rather than the encoder so every trunk receives the
    same units; an encoder cannot get the scaling subtly wrong.
    """
    return obs.float().div_(255.0) if obs.dtype == torch.uint8 else obs / 255.0


class QNetwork(nn.Module):
    """Encoder + Q head, for value-based algorithms."""

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        input_hw: tuple[int, int] = (84, 84),
        dueling: bool = True,
        noisy: bool = False,
        hidden: int = 512,
        encoder: str | Encoder = "nature_cnn",
        encoder_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.dueling = dueling
        self.noisy = noisy

        self.encoder = (
            encoder
            if isinstance(encoder, Encoder)
            else build_encoder(encoder, in_channels, input_hw, **(encoder_kwargs or {}))
        )
        head_cls = DuelingQHead if dueling else QHead
        self.head = head_cls(self.encoder.out_features, n_actions, hidden=hidden, noisy=noisy)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs`` is uint8 or float in [0, 255], shape (B, C, H, W)."""
        return self.head(self.encoder(_normalise(obs)))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(
            migrate_legacy_state_dict(state_dict), strict=strict, assign=assign
        )


class ActorCriticNetwork(nn.Module):
    """Encoder + actor-critic head, for on-policy algorithms."""

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        input_hw: tuple[int, int] = (84, 84),
        hidden: int = 512,
        encoder: str | Encoder = "nature_cnn",
        encoder_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.encoder = (
            encoder
            if isinstance(encoder, Encoder)
            else build_encoder(encoder, in_channels, input_hw, **(encoder_kwargs or {}))
        )
        self.head = ActorCriticHead(self.encoder.out_features, n_actions, hidden=hidden)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(logits, value)``."""
        return self.head(self.encoder(_normalise(obs)))

    def value_only(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)[1]
