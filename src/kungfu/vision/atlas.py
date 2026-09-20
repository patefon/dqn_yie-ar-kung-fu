"""Digit OCR for the NES HUD.

Why this is a rewrite rather than a port
----------------------------------------
The old ``read_text_stat`` ran ``cv2.findContours`` over an upscaled screenshot and
then compared every contour against every glyph with ``np.array_equal`` -- an
O(contours x glyphs) loop per frame, on interpolated pixels, where a single
resampling artifact broke equality. Worse, it ended with ``int(result or 0)``: an
unreadable score silently became **0**, which the reward function then read as a
gigantic negative score delta.

At the native 240x224 framebuffer there is no scaling, and NES text is drawn from
8x8 tiles on an 8-pixel grid. So matching is exact and O(1) per tile via a hash,
and an unreadable tile returns ``None`` instead of lying.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TILE = 8


def binarize(tile: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Collapse a tile to a boolean foreground mask."""
    if tile.ndim == 3:
        tile = tile.max(axis=2)
    return tile >= threshold


def tile_key(mask: np.ndarray) -> str:
    """Stable, compact key for an 8x8 boolean mask (16 hex chars)."""
    return np.packbits(mask.astype(np.uint8)).tobytes().hex()


class DigitAtlas:
    """Maps 8x8 glyph bitmaps to the digit they render.

    A digit may have several bitmaps (different palette rows, shadowed variants),
    so the mapping is many-to-one -- exactly what the old pickle's ``1_2``/``0_2``
    style keys were groping toward, but explicit.
    """

    def __init__(self, mapping: dict[str, int] | None = None) -> None:
        self._map: dict[str, int] = dict(mapping or {})

    def __len__(self) -> int:
        return len(self._map)

    @property
    def digits(self) -> set[int]:
        return set(self._map.values())

    def add(self, tile: np.ndarray, digit: int) -> None:
        if not 0 <= digit <= 9:
            raise ValueError(f"digit must be 0-9, got {digit}")
        mask = binarize(tile)
        if mask.shape != (TILE, TILE):
            raise ValueError(f"expected an {TILE}x{TILE} tile, got {mask.shape}")
        key = tile_key(mask)
        existing = self._map.get(key)
        if existing is not None and existing != digit:
            raise ValueError(
                f"glyph collision: bitmap already maps to {existing}, cannot remap to {digit}"
            )
        self._map[key] = digit

    def lookup(self, tile: np.ndarray) -> int | None:
        return self._map.get(tile_key(binarize(tile)))

    def is_blank(self, tile: np.ndarray) -> bool:
        return not binarize(tile).any()

    def read_number(self, region: np.ndarray, *, strict: bool = True) -> int | None:
        """Read a left-to-right run of digit tiles.

        Returns ``None`` when any tile is unrecognised (``strict``) -- never a
        silent zero. Leading/trailing blanks are skipped, which is how the NES
        renders a score that has not filled its field yet.
        """
        if region.shape[0] < TILE:
            return None
        row = region[:TILE]
        n_tiles = row.shape[1] // TILE

        digits: list[int] = []
        for i in range(n_tiles):
            tile = row[:, i * TILE : (i + 1) * TILE]
            if self.is_blank(tile):
                continue
            d = self.lookup(tile)
            if d is None:
                if strict:
                    return None
                continue
            digits.append(d)

        if not digits:
            # An all-blank field legitimately means zero.
            return 0
        return int("".join(str(d) for d in digits))

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._map, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DigitAtlas:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"no digit atlas at {p}. Build one with: python -m tools.calibrate_vision --command harvest"
            )
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls({k: int(v) for k, v in raw.items()})
