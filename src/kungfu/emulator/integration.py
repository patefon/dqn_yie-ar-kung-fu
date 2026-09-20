"""Register the Yie Ar Kung-Fu ROM as a stable-retro integration.

stable-retro ships 1,006 integrations, 298 of them NES, but Yie Ar Kung-Fu is
**not** among them (the closest are ``KungFu-Nes`` -- a different Irem game --
and ``KungFuHeroes-Nes``). So we build our own integration directory.

An integration is four files plus a savestate::

    YieArKungFu-Nes/
      rom.nes            the ROM itself (never committed -- supply your own)
      rom.sha            sha1 of the ROM, so a mismatched dump fails loudly
      data.json          RAM variable declarations
      scenario.json      emulator-side reward / done -- deliberately empty here
      Level1.state       gzipped savestate at the first fight

``scenario.json`` is intentionally inert. Reward and termination are computed in
Python from pixels (``kungfu.envs.reward``), which is the whole point of the
project. stable-retro is used only as a synchronous, headless frame source.

``data.json`` likewise declares nothing by default. It is populated only if you
run ``tools/discover_ram.py``, and even then the values feed the *calibration
oracle*, never the agent.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

GAME_NAME = "YieArKungFu-Nes"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROM_DIR = PROJECT_ROOT / "roms"
INTEGRATION_DIR = PROJECT_ROOT / "integrations"

INES_MAGIC = b"NES\x1a"


def find_rom(rom_dir: Path = ROM_DIR) -> Path:
    """Locate a NES ROM in ``roms/`` and sanity-check its header."""
    candidates = sorted(p for p in rom_dir.glob("*.nes") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"no .nes ROM found in {rom_dir}.\n"
            "Place your own legally-obtained Yie Ar Kung-Fu (USA/Japan) dump there. "
            "ROMs are not distributed with this project and are gitignored."
        )
    if len(candidates) > 1:
        raise RuntimeError(f"multiple ROMs in {rom_dir}: {[p.name for p in candidates]}")

    rom = candidates[0]
    header = rom.read_bytes()[:4]
    if header != INES_MAGIC:
        raise ValueError(
            f"{rom.name} is not an iNES ROM (header {header!r}). "
            "If this is a .zip or .7z, extract it first."
        )
    return rom


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_integration(rom: Path, out_dir: Path = INTEGRATION_DIR) -> Path:
    """Create the integration directory; returns the game directory."""
    game_dir = out_dir / GAME_NAME
    game_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(rom, game_dir / "rom.nes")
    (game_dir / "rom.sha").write_text(sha1_of(rom) + "\n", encoding="utf-8")

    # No emulator-side variables: the agent is pixels-only.
    data_path = game_dir / "data.json"
    if not data_path.exists():
        data_path.write_text(json.dumps({"info": {}}, indent=2), encoding="utf-8")

    # No emulator-side reward or done: Python owns both.
    (game_dir / "scenario.json").write_text(
        json.dumps({"done": {"variables": {}}, "reward": {"variables": {}}}, indent=2),
        encoding="utf-8",
    )
    (game_dir / "metadata.json").write_text(
        json.dumps({"default_state": "Level1"}, indent=2), encoding="utf-8"
    )
    return game_dir


def register(out_dir: Path = INTEGRATION_DIR) -> None:
    """Make stable-retro aware of our custom integration directory."""
    import stable_retro as retro

    retro.data.Integrations.add_custom_path(str(out_dir.resolve()))


def make_start_state(
    game_dir: Path,
    state_name: str = "Level1",
    skip_frames: int = 600,
    start_presses: int = 6,
) -> Path:
    """Boot the ROM, tap START through the title screen, and snapshot.

    stable-retro savestates are gzipped raw emulator states. The frame counts
    here are heuristics -- confirm the resulting state lands on a live fight by
    running ``python -m tools.calibrate_vision --command dump``.
    """
    import numpy as np
    import stable_retro as retro

    register(game_dir.parent)

    from kungfu.envs.actions import NES_BUTTONS

    env = retro.make(
        game=GAME_NAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=None,
    )
    try:
        env.reset()
        start_idx = NES_BUTTONS.index("START")
        noop = np.zeros(len(NES_BUTTONS), dtype=np.uint8)
        press = noop.copy()
        press[start_idx] = 1

        for _ in range(skip_frames):
            env.step(noop)
        for _ in range(start_presses):
            for _ in range(4):
                env.step(press)
            for _ in range(20):
                env.step(noop)
        for _ in range(120):
            env.step(noop)

        raw = env.em.get_state()
    finally:
        env.close()

    path = game_dir / f"{state_name}.state"
    with gzip.open(path, "wb") as f:
        f.write(raw)
    return path


def main() -> None:
    """Entry point for ``kungfu-setup``."""
    rom = find_rom()
    print(f"ROM      : {rom.name}")
    print(f"sha1     : {sha1_of(rom)}")

    game_dir = build_integration(rom)
    print(f"integration: {game_dir}")

    state = make_start_state(game_dir)
    print(f"savestate  : {state} ({state.stat().st_size} bytes)")
    print(
        "\nNext:\n"
        "  1. python -m tools.calibrate_vision --command dump   # confirm HUD regions\n"
        "  2. python -m tools.discover_ram              # optional: build the oracle\n"
        "  3. kungfu-train"
    )


if __name__ == "__main__":
    main()
