"""Calibrate and verify the vision pipeline.

Run this once after dropping in the ROM, and any time you change HUD regions.

    # 1. See what the extractor is actually looking at.
    python -m tools.calibrate_vision --command dump --frames 300

    # 2. Harvest every distinct glyph and label them (only ~10 to do, once).
    python -m tools.calibrate_vision --command harvest
    #    -> fills out/glyphs/labels.json, then:
    python -m tools.calibrate_vision --command build-atlas

    # 3. Prove it works.
    python -m tools.calibrate_vision --command check --frames 1000

Step 3 is the step the original project never had. Without it, a misread score
is indistinguishable from an agent that has not learned yet, and you can burn
weeks of GPU time on a reward signal that is quietly noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tyro
from rich.console import Console

from kungfu.config import Config
from kungfu.vision.atlas import TILE, DigitAtlas, binarize, tile_key

console = Console()
OUT = Path("out")


@dataclass
class Args:
    command: Literal["dump", "harvest", "build-atlas", "check"]
    config: Path | None = None
    frames: int = 300
    out: Path = OUT
    seed: int = 0


def _make_raw_env(cfg: Config):
    """A bare stable-retro env -- no vision wrapper, since that is what we are testing."""
    import stable_retro as retro

    from kungfu.emulator.integration import GAME_NAME, INTEGRATION_DIR, register

    register(INTEGRATION_DIR)
    return retro.make(
        game=GAME_NAME,
        state=cfg.env.state,
        use_restricted_actions=retro.Actions.ALL,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=None,
    )


def collect_frames(cfg: Config, n: int, seed: int) -> list[np.ndarray]:
    from kungfu.envs.actions import build_action_table

    env = _make_raw_env(cfg)
    table = build_action_table()
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    try:
        obs, _ = env.reset()
        for _ in range(n):
            action = table[rng.integers(0, len(table))]
            for _ in range(cfg.env.frame_skip):
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    obs, _ = env.reset()
            frames.append(obs.copy())
    finally:
        env.close()
    return frames


def cmd_dump(cfg: Config, args: Args) -> None:
    """Save annotated frames so you can eyeball whether the regions are right."""
    frames = collect_frames(cfg, args.frames, args.seed)
    d = args.out / "dump"
    d.mkdir(parents=True, exist_ok=True)

    hud = cfg.env.hud
    regions = {
        "score": (hud.score, (255, 0, 0)),
        "stage": (hud.stage, (0, 255, 0)),
        "player_health": (hud.player_health, (0, 128, 255)),
        "enemy_health": (hud.enemy_health, (255, 0, 255)),
        "playfield": (hud.playfield, (255, 255, 0)),
    }

    for i, frame in enumerate(frames[:: max(1, len(frames) // 20)]):
        canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
        for name, (r, colour) in regions.items():
            cv2.rectangle(canvas, (r.x, r.y), (r.x + r.w, r.y + r.h), colour, 1)
            cv2.putText(
                canvas, name[:4], (r.x, max(6, r.y - 2)),
                cv2.FONT_HERSHEY_PLAIN, 0.4, colour, 1,
            )
        big = cv2.resize(canvas, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(d / f"frame_{i:03d}.png"), big)

    # Also dump each region on its own, upscaled, for pixel-level checking.
    for name, (r, _) in regions.items():
        crop = r.crop(frames[len(frames) // 2])
        cv2.imwrite(
            str(d / f"region_{name}.png"),
            cv2.resize(
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), None,
                fx=8, fy=8, interpolation=cv2.INTER_NEAREST,
            ),
        )
    console.print(f"wrote annotated frames and region crops to {d}")
    console.print("[yellow]Check region_*.png: each HUD box must tightly frame its element.[/yellow]")


def cmd_harvest(cfg: Config, args: Args) -> None:
    """Collect every distinct glyph bitmap in the numeric HUD fields."""
    frames = collect_frames(cfg, args.frames, args.seed)
    d = args.out / "glyphs"
    d.mkdir(parents=True, exist_ok=True)

    seen: dict[str, np.ndarray] = {}
    for frame in frames:
        for region in (cfg.env.hud.score, cfg.env.hud.stage):
            crop = region.crop(frame)
            if crop.shape[0] < TILE:
                continue
            row = crop[:TILE]
            for i in range(row.shape[1] // TILE):
                tile = row[:, i * TILE : (i + 1) * TILE]
                mask = binarize(tile)
                if not mask.any():
                    continue
                seen.setdefault(tile_key(mask), mask)

    for key, mask in seen.items():
        img = (mask.astype(np.uint8) * 255)
        cv2.imwrite(
            str(d / f"{key}.png"),
            cv2.resize(img, None, fx=16, fy=16, interpolation=cv2.INTER_NEAREST),
        )

    labels_path = d / "labels.json"
    existing = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    # Merge, never replace: a glyph labelled in an earlier run must survive a run
    # in which that digit simply did not happen to appear on screen.
    labels = dict(existing)
    for k in seen:
        labels.setdefault(k, None)
    labels = {k: labels[k] for k in sorted(labels)}
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    new_count = sum(1 for k in seen if existing.get(k) is None)
    console.print(f"harvested {len(seen)} glyphs ({new_count} unlabelled) -> {d}")
    console.print(
        f"[bold]Now open {labels_path} and set each key to the digit its PNG shows "
        f"(0-9), then run: python -m tools.calibrate_vision --command build-atlas[/bold]"
    )


def cmd_build_atlas(cfg: Config, args: Args) -> None:
    labels_path = args.out / "glyphs" / "labels.json"
    if not labels_path.exists():
        raise SystemExit(f"{labels_path} not found -- run the harvest command first")

    labels = json.loads(labels_path.read_text())
    unlabelled = [k for k, v in labels.items() if v is None]
    if unlabelled:
        raise SystemExit(
            f"{len(unlabelled)} glyph(s) still unlabelled in {labels_path}; "
            "every glyph must be labelled or the OCR will silently fail on it"
        )

    atlas = DigitAtlas({k: int(v) for k, v in labels.items()})
    out = Path("configs/digit_atlas.json")
    atlas.save(out)
    console.print(f"wrote {out} with {len(atlas)} glyph(s) covering digits {sorted(atlas.digits)}")
    missing = set(range(10)) - atlas.digits
    if missing:
        console.print(
            f"[yellow]digits {sorted(missing)} never appeared -- play longer "
            f"(higher score) and re-harvest before trusting the OCR[/yellow]"
        )


def cmd_check(cfg: Config, args: Args) -> None:
    """Report how readable the HUD is, and cross-check against RAM if available."""
    from kungfu.vision.hud import HudReader

    atlas = DigitAtlas.load("configs/digit_atlas.json")
    reader = HudReader(cfg.env.hud, atlas)

    oracle = None
    try:
        from kungfu.vision.oracle import RamOracle

        oracle = RamOracle.load()
        console.print("RAM oracle loaded -- cross-checking vision against ground truth")
    except FileNotFoundError:
        console.print("[dim]no RAM map; reporting readability only[/dim]")

    env = _make_raw_env(cfg)
    from kungfu.envs.actions import build_action_table

    table = build_action_table()
    rng = np.random.default_rng(args.seed)

    n = 0
    readable = 0
    transitions = 0
    score_match = 0
    score_compared = 0
    try:
        obs, _ = env.reset()
        for _ in range(args.frames):
            action = table[rng.integers(0, len(table))]
            for _ in range(cfg.env.frame_skip):
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    obs, _ = env.reset()
            stats = reader.read(obs)
            n += 1
            if stats.is_transition:
                transitions += 1
                continue
            if stats.readable:
                readable += 1
            if oracle is not None and "score" in oracle.fields:
                truth = int(oracle.read(env.get_ram())["score"])
                score_compared += 1
                score_match += int(stats.score == truth)
    finally:
        env.close()

    live = n - transitions
    console.print(f"frames            : {n}")
    console.print(f"transition screens: {transitions} ({transitions / max(n, 1):.1%})")
    console.print(
        f"readable HUD      : {readable}/{live} ({readable / max(live, 1):.1%})"
    )
    if score_compared:
        rate = score_match / score_compared
        verdict = "[green]OK[/green]" if rate >= 0.99 else "[red]TOO LOW[/red]"
        console.print(f"score vs RAM      : {rate:.2%} {verdict}")

    if readable / max(live, 1) < 0.95:
        console.print(
            "[red]Readability below 95%. Fix the HUD regions in configs/default.yaml "
            "before training -- a noisy reward signal will not produce a good agent.[/red]"
        )


def main() -> None:
    args = tyro.cli(Args)
    cfg = Config.load(args.config)
    {
        "dump": cmd_dump,
        "harvest": cmd_harvest,
        "build-atlas": cmd_build_atlas,
        "check": cmd_check,
    }[args.command](cfg, args)


if __name__ == "__main__":
    main()
