"""Turn a stream of vision readings into a well-scaled reward signal.

What went wrong before
----------------------
The old ``__get_reward`` compared the current frame against the immediately
previous frame at ~100 fps and applied a **flat** ``-5`` whenever player health
had dropped. A single hit takes many frames to animate the bar downward, so one
hit produced ``-5`` repeatedly -- tens of times. Meanwhile landing a hit was a
flat ``+1``, also repeated. The relative value of attacking versus not getting hit
therefore depended on animation length, not on the game.

The fix is to make every term **proportional to the change**, so it telescopes:
draining a full enemy bar is worth exactly ``damage_dealt``, no matter how many
frames the drain is spread across or how often we sample it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kungfu.config import RewardConfig
from kungfu.vision.hud import GameStats

# A score jump larger than this in one agent step is not physically possible in
# this game; treat it as a misread rather than a jackpot.
MAX_PLAUSIBLE_SCORE_DELTA = 5_000


@dataclass
class RewardBreakdown:
    """Per-term accounting, logged to TensorBoard so shaping stays debuggable."""

    score: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    stage_clear: float = 0.0
    death: float = 0.0
    time: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.score
            + self.damage_dealt
            + self.damage_taken
            + self.stage_clear
            + self.death
            + self.time
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "score": self.score,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "stage_clear": self.stage_clear,
            "death": self.death,
            "time": self.time,
        }


@dataclass
class RewardShaper:
    cfg: RewardConfig

    # Last *successfully read* values. Holding these across unreadable frames is
    # what makes the signal robust to a transient OCR failure.
    _score: int | None = None
    _stage: int | None = None
    _player: float | None = None
    _enemy: float | None = None
    _was_dead: bool = False
    _misreads: int = field(default=0, repr=False)

    def reset(self) -> None:
        self._score = None
        self._stage = None
        self._player = None
        self._enemy = None
        self._was_dead = False

    @property
    def misreads(self) -> int:
        return self._misreads

    def step(self, stats: GameStats) -> tuple[float, RewardBreakdown]:
        b = RewardBreakdown()

        # Between-stage screens carry no meaningful HUD: emit nothing and, crucially,
        # do NOT update the baselines, so the bars resetting does not read as damage.
        if stats.is_transition:
            return 0.0, b

        b.time = self.cfg.time_penalty

        # -- score ---------------------------------------------------------
        if stats.score is not None:
            if self._score is not None:
                delta = stats.score - self._score
                if 0 < delta <= MAX_PLAUSIBLE_SCORE_DELTA:
                    b.score = delta * self.cfg.score_scale
                elif delta > MAX_PLAUSIBLE_SCORE_DELTA:
                    self._misreads += 1
            self._score = stats.score
        else:
            self._misreads += 1

        # -- stage ---------------------------------------------------------
        if stats.stage is not None:
            if self._stage is not None and stats.stage > self._stage:
                b.stage_clear = self.cfg.stage_clear * (stats.stage - self._stage)
            self._stage = stats.stage

        # -- damage dealt --------------------------------------------------
        # Proportional, so it telescopes over the drain animation. An *increase*
        # means a fresh opponent spawned; that is not a penalty.
        if stats.enemy_health is not None:
            if self._enemy is not None:
                drained = self._enemy - stats.enemy_health
                if drained > 0:
                    b.damage_dealt = self.cfg.damage_dealt * drained
            self._enemy = stats.enemy_health

        # -- damage taken / death -------------------------------------------
        if stats.player_health is not None:
            if self._player is not None:
                lost = self._player - stats.player_health
                if lost > 0:
                    b.damage_taken = self.cfg.damage_taken * lost
            is_dead = stats.player_health <= 0.0
            if is_dead and not self._was_dead:
                b.death = self.cfg.death
            self._was_dead = is_dead
            self._player = stats.player_health

        total = b.total
        if self.cfg.clip is not None:
            total = max(-self.cfg.clip, min(self.cfg.clip, total))
        return total, b
