"""Encoder registry.

Add a trunk by writing the class and registering it here; nothing in the
training loop changes.
"""

from __future__ import annotations

from kungfu.rl.encoders.base import Encoder, orthogonal_
from kungfu.rl.encoders.impala import ImpalaCNN
from kungfu.rl.encoders.nature_cnn import NatureCNN
from kungfu.rl.registry import Registry

ENCODERS: Registry[Encoder] = Registry("encoder")
ENCODERS.register("nature_cnn")(NatureCNN)
ENCODERS.register("impala")(ImpalaCNN)


def build_encoder(name: str, in_channels: int, input_hw: tuple[int, int], **kwargs) -> Encoder:
    return ENCODERS.build(name, in_channels=in_channels, input_hw=input_hw, **kwargs)


__all__ = ["ENCODERS", "Encoder", "ImpalaCNN", "NatureCNN", "build_encoder", "orthogonal_"]
