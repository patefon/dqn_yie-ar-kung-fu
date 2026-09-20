"""Vision OCR and reward shaping.

These pin the two failure modes that made the original reward signal unusable:
an unreadable HUD silently becoming ``0``, and a single hit being paid out once
per animation frame instead of once.
"""

from __future__ import annotations

import numpy as np
import pytest

from kungfu.config import RewardConfig
from kungfu.envs.actions import ACTION_COMBOS, NES_BUTTONS, build_action_table
from kungfu.envs.reward import RewardShaper
from kungfu.vision.atlas import TILE, DigitAtlas
from kungfu.vision.hud import GameStats


def glyph(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tile = (rng.random((TILE, TILE)) > 0.5).astype(np.uint8) * 255
    tile[0, 0] = 255  # never blank
    return tile


def atlas_with(digits: list[int]) -> tuple[DigitAtlas, dict[int, np.ndarray]]:
    atlas = DigitAtlas()
    tiles = {}
    for d in digits:
        t = glyph(d)
        tiles[d] = t
        atlas.add(t, d)
    return atlas, tiles


def row_of(tiles: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(tiles, axis=1)


class TestDigitAtlas:
    def test_reads_a_multi_digit_number(self):
        atlas, tiles = atlas_with([1, 2, 3])
        assert atlas.read_number(row_of([tiles[1], tiles[2], tiles[3]])) == 123

    def test_unknown_glyph_returns_none_not_zero(self):
        """The original coerced a failed read to 0, which the reward read as a
        massive score delta."""
        atlas, tiles = atlas_with([1, 2])
        region = row_of([tiles[1], glyph(99), tiles[2]])
        assert atlas.read_number(region) is None

    def test_blank_field_is_zero(self):
        atlas, _ = atlas_with([0])
        assert atlas.read_number(np.zeros((TILE, TILE * 3), dtype=np.uint8)) == 0

    def test_blanks_are_skipped_between_digits(self):
        atlas, tiles = atlas_with([4, 7])
        blank = np.zeros((TILE, TILE), dtype=np.uint8)
        assert atlas.read_number(row_of([blank, tiles[4], tiles[7]])) == 47

    def test_conflicting_label_is_rejected(self):
        atlas, tiles = atlas_with([5])
        with pytest.raises(ValueError, match="collision"):
            atlas.add(tiles[5], 6)

    def test_roundtrip(self, tmp_path):
        atlas, tiles = atlas_with([3, 8])
        p = tmp_path / "atlas.json"
        atlas.save(p)
        assert DigitAtlas.load(p).read_number(row_of([tiles[3], tiles[8]])) == 38

    def test_missing_file_has_actionable_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="calibrate_vision"):
            DigitAtlas.load(tmp_path / "nope.json")


def stats(score=None, stage=1, php=1.0, ehp=1.0, gameover=False, transition=False, lives=2):
    return GameStats(score, stage, php, ehp, gameover, transition, lives)


class TestRewardShaper:
    def cfg(self, **kw):
        base = dict(
            score_scale=0.01, damage_dealt=1.0, damage_taken=-1.0,
            stage_clear=10.0, death=-10.0, time_penalty=0.0, clip=None,
        )
        base.update(kw)
        return RewardConfig(**base)

    def test_damage_is_not_double_counted_over_an_animation(self):
        """The core bug: the old code paid a flat -5 on every frame the bar was
        still draining. Proportional terms must telescope to one payout."""
        shaper = RewardShaper(self.cfg())
        shaper.step(stats(score=0, php=1.0))
        # A single hit animating down over 10 frames.
        total = 0.0
        for i in range(1, 11):
            r, _ = shaper.step(stats(score=0, php=1.0 - 0.05 * i))
            total += r
        assert total == pytest.approx(-0.5, abs=1e-6)

    def test_sampling_rate_does_not_change_the_payout(self):
        """Same damage observed in 1 step or 20 must be worth the same."""
        coarse = RewardShaper(self.cfg())
        coarse.step(stats(php=1.0))
        r_coarse, _ = coarse.step(stats(php=0.5))

        fine = RewardShaper(self.cfg())
        fine.step(stats(php=1.0))
        r_fine = sum(fine.step(stats(php=1.0 - 0.025 * i))[0] for i in range(1, 21))
        assert r_coarse == pytest.approx(r_fine, abs=1e-6)

    def test_enemy_respawn_is_not_a_penalty(self):
        shaper = RewardShaper(self.cfg())
        shaper.step(stats(ehp=0.2))
        r, b = shaper.step(stats(ehp=1.0))  # fresh opponent
        assert b.damage_dealt == 0.0
        assert r == pytest.approx(0.0)

    def test_score_increase_is_rewarded(self):
        shaper = RewardShaper(self.cfg())
        shaper.step(stats(score=100))
        _, b = shaper.step(stats(score=400))
        assert b.score == pytest.approx(3.0)

    def test_implausible_score_jump_is_treated_as_a_misread(self):
        shaper = RewardShaper(self.cfg())
        shaper.step(stats(score=100))
        _, b = shaper.step(stats(score=9_999_999))
        assert b.score == 0.0
        assert shaper.misreads == 1

    def test_unreadable_score_does_not_emit_reward(self):
        shaper = RewardShaper(self.cfg())
        shaper.step(stats(score=500))
        _, b = shaper.step(stats(score=None))
        assert b.score == 0.0

    def test_baseline_survives_an_unreadable_frame(self):
        """Hold the last good value so one dropped read does not fabricate a spike."""
        shaper = RewardShaper(self.cfg())
        shaper.step(stats(score=500))
        shaper.step(stats(score=None))
        _, b = shaper.step(stats(score=600))
        assert b.score == pytest.approx(1.0)

    def test_transition_screen_emits_nothing_and_holds_baselines(self):
        shaper = RewardShaper(self.cfg())
        shaper.step(stats(score=500, php=1.0))
        r, b = shaper.step(stats(transition=True))
        assert r == 0.0 and b.total == 0.0
        _, b2 = shaper.step(stats(score=500, php=1.0))
        assert b2.damage_taken == 0.0

    def test_death_fires_once(self):
        shaper = RewardShaper(self.cfg())
        shaper.step(stats(php=0.1))
        _, first = shaper.step(stats(php=0.0))
        _, second = shaper.step(stats(php=0.0))
        assert first.death == -10.0
        assert second.death == 0.0

    def test_stage_clear_rewarded(self):
        shaper = RewardShaper(self.cfg())
        shaper.step(stats(stage=1))
        _, b = shaper.step(stats(stage=2))
        assert b.stage_clear == 10.0

    def test_clipping_bounds_the_step_reward(self):
        shaper = RewardShaper(self.cfg(clip=1.0))
        shaper.step(stats(score=0))
        r, _ = shaper.step(stats(score=5000, stage=1))
        assert r == pytest.approx(1.0)


class TestGameOverDetection:
    """Both post-death screens must terminate the episode.

    The viewer caught this: the GAME OVER banner can show with the health bars
    still full, so a bars-empty-only check let the episode run on into the
    attract loop, where the controller does nothing.
    """

    def reader(self):
        from kungfu.config import HudLayout
        from kungfu.vision.hud import HudReader

        return HudReader(HudLayout(), DigitAtlas())

    def frame(self, *, lives_icons=2, banner=False, bars_full=True):
        from kungfu.config import HudLayout

        lay = HudLayout()
        f = np.zeros((224, 240, 3), dtype=np.uint8)
        # Keep the playfield colourful so it is not read as a transition screen.
        rng = np.random.default_rng(0)
        pf = lay.playfield
        f[pf.y:pf.y + pf.h, pf.x:pf.x + pf.w] = rng.integers(
            0, 200, size=(pf.h, pf.w, 3), dtype=np.uint8
        )
        # Lives icons: exactly lives_pixels_per_icon lit pixels each.
        lv = lay.lives
        f[lv.y:lv.y + lv.h, lv.x:lv.x + lv.w] = 0
        flat = f[lv.y:lv.y + lv.h, lv.x:lv.x + lv.w].reshape(-1, 3)
        flat[: lives_icons * lay.lives_pixels_per_icon] = 255
        f[lv.y:lv.y + lv.h, lv.x:lv.x + lv.w] = flat.reshape(lv.h, lv.w, 3)
        # Health bars.
        for r in (lay.player_health, lay.enemy_health):
            f[r.y:r.y + r.h, r.x:r.x + r.w] = (
                lay.health_fill_rgb[0] if bars_full else (0, 0, 0)
            )
        # Banner.
        b = lay.gameover_banner
        f[b.y:b.y + b.h, b.x:b.x + b.w] = 0
        if banner:
            f[b.y + 2:b.y + 8, b.x + 4:b.x + 60] = 255
        return f

    def test_live_play_is_not_game_over(self):
        s = self.reader().read(self.frame(lives_icons=2, banner=False, bars_full=True))
        assert s.lives == 2
        assert not s.is_gameover

    def test_banner_with_full_bars_is_game_over(self):
        s = self.reader().read(self.frame(lives_icons=0, banner=True, bars_full=True))
        assert s.lives == 0
        assert s.is_gameover

    def test_demo_mode_empty_bars_is_game_over(self):
        s = self.reader().read(self.frame(lives_icons=0, banner=False, bars_full=False))
        assert s.is_gameover

    def test_banner_while_lives_remain_is_not_game_over(self):
        """Gated on lives, so a stray white pixel on a later stage cannot end a run."""
        s = self.reader().read(self.frame(lives_icons=2, banner=True, bars_full=True))
        assert not s.is_gameover

    def test_lives_are_counted(self):
        for n in (0, 1, 2):
            assert self.reader().read(self.frame(lives_icons=n)).lives == n


class TestActions:
    def test_punch_exists(self):
        """The original action set had no A button at all, so the agent could
        never punch -- half the move set was unreachable."""
        assert any("A" in combo for combo in ACTION_COMBOS)

    def test_kick_exists(self):
        assert any("B" in combo for combo in ACTION_COMBOS)

    def test_table_shape_and_encoding(self):
        table = build_action_table()
        assert table.shape == (len(ACTION_COMBOS), len(NES_BUTTONS))
        assert table[0].sum() == 0  # no-op presses nothing
        punch = ACTION_COMBOS.index(("A",))
        assert table[punch][NES_BUTTONS.index("A")] == 1
        assert table[punch].sum() == 1

    def test_directional_attacks_are_reachable(self):
        for combo in (("DOWN", "B"), ("UP", "B"), ("DOWN", "A"), ("UP", "A")):
            assert combo in ACTION_COMBOS
