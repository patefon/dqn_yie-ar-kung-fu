"""Read game state out of the HUD pixels.

Everything the reward function needs is derived here, from the frame alone. No
emulator RAM is touched -- see ``kungfu.vision.oracle`` for the offline-only
ground-truth reader used to *verify* this module.

Design rule: every field is ``Optional``. A failed reading reports itself as
``None`` and the caller holds the last known good value. The old pipeline instead
coerced failures to 0, turning a missed OCR into a phantom reward spike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kungfu.config import HudLayout
from kungfu.vision.atlas import DigitAtlas


@dataclass(frozen=True, slots=True)
class GameStats:
    score: int | None
    stage: int | None
    player_health: float | None
    enemy_health: float | None
    is_gameover: bool
    # True on the between-stage / demo screens, where the HUD is not meaningful
    # and no reward should be emitted.
    is_transition: bool
    # Spare lives remaining, read from the head icons. None when unreadable.
    lives: int | None = None

    @property
    def readable(self) -> bool:
        return (
            self.score is not None
            and self.player_health is not None
            and self.enemy_health is not None
        )


class HudReader:
    def __init__(self, layout: HudLayout, atlas: DigitAtlas) -> None:
        self.layout = layout
        self.atlas = atlas
        self._fill = np.asarray(layout.health_fill_rgb, dtype=np.int16)
        self._tol = layout.health_match_tolerance

    # -- health ------------------------------------------------------------
    def _health_fraction(self, frame: np.ndarray, region) -> float | None:
        """Fraction of the bar that is still filled.

        Counts *columns*, not raw pixels. A health bar is one-dimensional: what
        matters is how far along it is filled, and column counting is immune to
        the bar's height and to a stray antialiased pixel. The old code divided a
        raw pixel count by the ROI area, which conflated the two.
        """
        crop = region.crop(frame).astype(np.int16)
        if crop.size == 0:
            return None
        # (h, w, 3) vs (k, 3) -> (h, w, k) distance, then any-match.
        diff = np.abs(crop[:, :, None, :] - self._fill[None, None, :, :]).max(axis=-1)
        matched = (diff <= self._tol).any(axis=-1)  # (h, w)
        columns_filled = matched.any(axis=0).sum()
        return float(columns_filled) / float(region.w)

    # -- lives -------------------------------------------------------------
    def _lives(self, frame: np.ndarray) -> int | None:
        """Count the spare-life head icons.

        Each icon lights exactly ``lives_pixels_per_icon`` pixels, so this is a
        division rather than a template match.
        """
        crop = self.layout.lives.crop(frame)
        if crop.size == 0:
            return None
        lit = int((crop.max(axis=2) > 40).sum())
        per = self.layout.lives_pixels_per_icon
        count = round(lit / per)
        # Reject anything that is not close to a whole number of icons: it means
        # something else is being drawn there and the reading cannot be trusted.
        if abs(lit - count * per) > per // 3:
            return None
        return int(count)

    # -- game over banner --------------------------------------------------
    def _banner_present(self, frame: np.ndarray) -> bool:
        """White text in the centre gap between the ceiling decorations."""
        crop = self.layout.gameover_banner.crop(frame)
        if crop.size == 0:
            return False
        white = (crop[:, :, 0] > 200) & (crop[:, :, 1] > 200) & (crop[:, :, 2] > 200)
        return int(white.sum()) >= self.layout.gameover_banner_min_pixels

    # -- transition --------------------------------------------------------
    def _is_transition(self, frame: np.ndarray) -> bool:
        """Between-stage and title screens are almost entirely flat colour.

        Same idea as the old ``__is_level_split_screen``, but measured on the
        playfield's colour diversity rather than a nonzero-pixel ratio, so it does
        not fire on a legitimately dark stage. The threshold is calibrated
        against measured frames -- see ``HudLayout.transition_max_colors``.
        """
        play = self.layout.playfield.crop(frame)
        if play.size == 0:
            return False
        sample = play[::4, ::4].reshape(-1, play.shape[-1])
        unique = np.unique(sample, axis=0).shape[0]
        return unique < self.layout.transition_max_colors

    # -- main entry point --------------------------------------------------
    def read(self, frame: np.ndarray) -> GameStats:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"expected an HxWx3 RGB frame, got {frame.shape}")

        if self._is_transition(frame):
            return GameStats(None, None, None, None, False, True, None)

        player = self._health_fraction(frame, self.layout.player_health)
        enemy = self._health_fraction(frame, self.layout.enemy_health)
        lives = self._lives(frame)

        # Game-over detection, all measured on real frames.
        #
        # Two distinct screens follow the last life, and they look different:
        #
        #   1. the GAME OVER banner, which can appear with the health bars still
        #      full, and
        #   2. the attract "DEMO" loop, where both bars read empty and the
        #      controller does nothing at all.
        #
        # Detecting only (2) means the episode runs on through the banner and
        # collects transitions in which actions have no effect -- actively
        # teaching the agent that actions do not matter. So either signature
        # counts, but both are gated on having no spare lives, which keeps a
        # stray white pixel on some later stage from ending a live episode.
        no_lives = lives == 0
        bars_empty = (
            player is not None and enemy is not None and player <= 0.0 and enemy <= 0.0
        )
        gameover = no_lives and (self._banner_present(frame) or bars_empty)

        return GameStats(
            score=self.atlas.read_number(self.layout.score.crop(frame)),
            stage=self.atlas.read_number(self.layout.stage.crop(frame)),
            player_health=player,
            enemy_health=enemy,
            is_gameover=bool(gameover),
            is_transition=False,
            lives=lives,
        )
