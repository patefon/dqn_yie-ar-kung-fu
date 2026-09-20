"""Find the RAM addresses for score / health / stage empirically.

This exists so the calibration oracle is grounded in *this* ROM revision rather
than in addresses copied from a wiki that may describe a different dump. Nothing
it produces is ever read by the agent -- see ``kungfu.vision.oracle``.

Method: roll out with random actions while recording work RAM alongside the
vision readings, then rank candidates.

* **score**   -- BCD runs that only ever increase, correlated against the OCR.
* **health**  -- bytes correlated with the measured health-bar fill fraction.
* **stage**   -- small-valued bytes that step upward when the stage changes.

Usage::

    python -m tools.discover_ram --frames 4000
    # inspect the ranked report, then accept the top candidates:
    python -m tools.discover_ram --frames 4000 --write
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from rich.console import Console
from rich.table import Table

from kungfu.config import Config
from kungfu.vision.atlas import DigitAtlas
from kungfu.vision.hud import HudReader
from kungfu.vision.oracle import RamField, RamOracle

console = Console()


@dataclass
class Args:
    config: Path | None = None
    frames: int = 4000
    seed: int = 0
    write: bool = False
    """Write configs/ram_map.json from the top-ranked candidates."""
    top: int = 5


def rollout(cfg: Config, args: Args):
    """Collect (ram, stats) pairs over a random rollout."""
    import stable_retro as retro

    from kungfu.emulator.integration import GAME_NAME, INTEGRATION_DIR, register
    from kungfu.envs.actions import build_action_table

    register(INTEGRATION_DIR)
    env = retro.make(
        game=GAME_NAME,
        state=cfg.env.state,
        use_restricted_actions=retro.Actions.ALL,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=None,
    )

    atlas = None
    try:
        atlas = DigitAtlas.load("configs/digit_atlas.json")
    except FileNotFoundError:
        console.print("[yellow]no digit atlas yet; score correlation will be skipped[/yellow]")
    reader = HudReader(cfg.env.hud, atlas) if atlas else None

    table = build_action_table()
    rng = np.random.default_rng(args.seed)

    rams: list[np.ndarray] = []
    scores: list[float] = []
    p_health: list[float] = []
    e_health: list[float] = []

    try:
        obs, _ = env.reset()
        for _ in range(args.frames):
            action = table[rng.integers(0, len(table))]
            for _ in range(cfg.env.frame_skip):
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    obs, _ = env.reset()
            rams.append(np.asarray(env.get_ram(), dtype=np.uint8).copy())
            if reader is not None:
                s = reader.read(obs)
                scores.append(np.nan if s.score is None else float(s.score))
                p_health.append(np.nan if s.player_health is None else s.player_health)
                e_health.append(np.nan if s.enemy_health is None else s.enemy_health)
    finally:
        env.close()

    return (
        np.stack(rams),
        np.array(scores) if scores else None,
        np.array(p_health) if p_health else None,
        np.array(e_health) if e_health else None,
    )


def _corr(series: np.ndarray, signal: np.ndarray) -> float:
    ok = ~np.isnan(signal)
    if ok.sum() < 32:
        return 0.0
    a = series[ok].astype(np.float64)
    b = signal[ok].astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def rank_health(ram: np.ndarray, signal: np.ndarray | None) -> list[tuple[int, float]]:
    if signal is None:
        return []
    scored = [(addr, _corr(ram[:, addr], signal)) for addr in range(ram.shape[1])]
    return sorted(scored, key=lambda x: -x[1])


def _decode_bcd(block: np.ndarray) -> np.ndarray:
    value = np.zeros(block.shape[0], dtype=np.float64)
    for col in range(block.shape[1]):
        byte = block[:, col].astype(np.float64)
        value = value * 100 + (byte // 16) * 10 + (byte % 16)
    return value


def rank_score(ram: np.ndarray, signal: np.ndarray | None, lengths=(2, 3, 4)):
    """BCD runs that never decrease; ranked by correlation with the OCR score."""
    out = []
    n_addr = ram.shape[1]
    for length in lengths:
        for addr in range(n_addr - length):
            block = ram[:, addr : addr + length]
            # Cheap reject: any nibble above 9 cannot be BCD.
            if ((block // 16) > 9).any() or ((block % 16) > 9).any():
                continue
            value = _decode_bcd(block)
            if value.max() == value.min():
                continue
            decreases = int((np.diff(value) < 0).sum())
            # A reset legitimately drops the score, so allow a few.
            if decreases > 5:
                continue
            score = _corr(value, signal) if signal is not None else 0.0
            out.append((addr, length, score, decreases, float(value.max())))
    return sorted(out, key=lambda x: (-x[2], x[3]))


def main() -> None:
    args = tyro.cli(Args)
    cfg = Config.load(args.config)

    console.print(f"rolling out {args.frames} agent steps...")
    ram, scores, php, ehp = rollout(cfg, args)
    console.print(f"captured RAM {ram.shape} ({ram.shape[1]} addresses)")

    score_c = rank_score(ram, scores)
    php_c = rank_health(ram, php)
    ehp_c = rank_health(ram, ehp)

    t = Table(title="score candidates (BCD, monotonic)")
    for col in ("address", "len", "corr", "decreases", "max"):
        t.add_column(col, justify="right")
    for addr, length, corr, dec, mx in score_c[: args.top]:
        t.add_row(f"0x{addr:04X}", str(length), f"{corr:.3f}", str(dec), f"{mx:,.0f}")
    console.print(t)

    for title, cands in (("player health", php_c), ("enemy health", ehp_c)):
        t = Table(title=f"{title} candidates (correlation with bar fill)")
        t.add_column("address", justify="right")
        t.add_column("corr", justify="right")
        for addr, corr in cands[: args.top]:
            t.add_row(f"0x{addr:04X}", f"{corr:.3f}")
        console.print(t)

    if not args.write:
        console.print("[dim]re-run with --write to save the top candidates[/dim]")
        return

    fields: dict[str, RamField] = {}
    if score_c and score_c[0][2] > 0.9:
        addr, length, *_ = score_c[0]
        fields["score"] = RamField(address=addr, length=length, encoding="bcd")
    if php_c and php_c[0][1] > 0.9:
        fields["player_health"] = RamField(address=php_c[0][0])
    if ehp_c and ehp_c[0][1] > 0.9:
        fields["enemy_health"] = RamField(address=ehp_c[0][0])

    if not fields:
        raise SystemExit(
            "no candidate cleared the 0.9 correlation bar. Either the digit atlas "
            "is not built yet, or the HUD regions are wrong -- run "
            "`python -m tools.calibrate_vision --command dump` and check them first."
        )

    RamOracle(fields).save()
    console.print(f"wrote configs/ram_map.json with {sorted(fields)}")


if __name__ == "__main__":
    main()
