"""Ground-truth RAM reader -- CALIBRATION AND TESTING ONLY.

Nothing in the training path may import this module. The agent is pixels-only by
design; this exists purely so the vision extractor can be *checked* against
something that cannot be wrong.

Why bother
----------
The single biggest blind spot in the original project was that there was no way
to tell whether the CV was reading the game correctly. If ``read_text_stat``
misread the score, the reward silently became garbage and training silently
failed -- indistinguishable from "the agent has not learned yet". Here you can
run ``tools/calibrate_vision.py`` and get a hard number: what fraction of frames
does the vision pipeline agree with RAM on.

The address map is *discovered*, not copied from a wiki, so it is verified
against the exact ROM revision in use. See ``tools/discover_ram.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MAP_PATH = Path("configs/ram_map.json")


@dataclass(frozen=True)
class RamField:
    """One game variable located in NES work RAM."""

    address: int
    length: int = 1
    encoding: str = "u8"  # "u8" | "bcd" (big-endian BCD across `length` bytes)
    scale: float = 1.0

    def read(self, ram: np.ndarray) -> float:
        raw = ram[self.address : self.address + self.length]
        if self.encoding == "bcd":
            value = 0
            for byte in raw:
                value = value * 100 + (byte >> 4) * 10 + (byte & 0x0F)
        else:
            value = 0
            for byte in raw:
                value = value * 256 + int(byte)
        return value * self.scale


class RamOracle:
    """Reads named fields out of the emulator work RAM."""

    def __init__(self, fields: dict[str, RamField]) -> None:
        self.fields = fields

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MAP_PATH) -> RamOracle:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"no RAM map at {p}. Generate one with: python -m tools.discover_ram"
            )
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls({name: RamField(**spec) for name, spec in raw.items()})

    def save(self, path: str | Path = DEFAULT_MAP_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({k: vars(v) for k, v in self.fields.items()}, indent=2),
            encoding="utf-8",
        )

    def read(self, ram: np.ndarray) -> dict[str, float]:
        return {name: field.read(ram) for name, field in self.fields.items()}


@dataclass
class AgreementReport:
    """How often the vision pipeline matches ground truth."""

    frames: int
    agreements: dict[str, int]
    disagreements: dict[str, list[tuple[int, object, object]]]

    def rate(self, field: str) -> float:
        if self.frames == 0:
            return 0.0
        return self.agreements.get(field, 0) / self.frames

    def summary(self) -> str:
        lines = [f"compared {self.frames} frames"]
        for field in sorted(self.agreements):
            rate = self.rate(field)
            flag = "OK " if rate >= 0.99 else "BAD"
            lines.append(f"  [{flag}] {field:<16} agreement {rate:6.2%}")
            for frame_i, got, want in self.disagreements.get(field, [])[:3]:
                lines.append(f"         frame {frame_i}: vision={got!r} ram={want!r}")
        return "\n".join(lines)
