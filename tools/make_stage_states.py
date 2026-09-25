"""Generate savestates at later stages, for a start-state curriculum.

The problem this solves
-----------------------
Every episode begins at stage 1, so almost all experience is spent replaying
stages the agent already beats. Measured on an uncapped DQN run: ~11,700 steps
per episode, of which roughly 500 are past stage 20 -- about 4%. Reaching a
useful amount of practice at the frontier would take ~23M steps of training.

Seeding episodes further in fixes that directly. This tool drives an already
trained checkpoint through the game and snapshots the emulator whenever it
first reaches a target stage, writing the states into the integration
directory so `env.start_states` can sample them.

    python -m tools.make_stage_states --checkpoint runs/BIG-ML-DQN/final.pt \\
        --stages 5 10 15 20

Then add to a config:

    env:
      start_states:
        - {name: Level1,  weight: 1.0}
        - {name: Stage10, weight: 1.0}
        - {name: Stage15, weight: 1.5}
        - {name: Stage20, weight: 2.0}
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from rich.console import Console

from kungfu.config import Config

console = Console()


@dataclass
class Args:
    checkpoint: Path
    """A trained checkpoint good enough to reach the target stages."""
    stages: list[int] = field(default_factory=lambda: [5, 10, 15, 20])
    """Stages to snapshot on first arrival."""
    config: Path | None = None
    """Defaults to <checkpoint dir>/config.yaml."""
    max_steps: int = 40000
    """Give up after this many agent steps."""
    episodes: int = 6
    """Retries. A single run may die before reaching the deepest stage."""
    seed: int = 0
    epsilon: float = 0.01
    prefix: str = "Stage"


def main() -> None:
    args = tyro.cli(Args)

    from kungfu.emulator.integration import GAME_NAME, INTEGRATION_DIR, register
    from kungfu.envs.kungfu import YieArKungFuEnv
    from kungfu.envs.wrappers import ChannelStack
    from kungfu.rl.algos import build_agent
    from kungfu.train import ATLAS_PATH
    from kungfu.vision.atlas import DigitAtlas

    cfg_path = args.config or (args.checkpoint.parent / "config.yaml")
    cfg = Config.load(cfg_path if Path(cfg_path).exists() else None)

    # Snapshotting needs long episodes, and a curriculum on the *source* run
    # would defeat the purpose -- always start these probes from the beginning.
    cfg.env.max_episode_steps = args.max_steps
    cfg.env.stall_timeout = args.max_steps
    cfg.env.start_states = None

    register(INTEGRATION_DIR)
    base = YieArKungFuEnv(cfg.env, cfg.reward, DigitAtlas.load(ATLAS_PATH))
    env = ChannelStack(base, cfg.env.frame_stack)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = build_agent(
        cfg, env.observation_space.shape, int(env.action_space.n), device, seed=args.seed
    )
    agent.load(args.checkpoint)
    agent.train_mode(False)
    console.print(f"driving [green]{args.checkpoint.name}[/green] @ step {agent.steps:,}")
    console.print(f"targets: {args.stages}\n")

    out_dir = INTEGRATION_DIR / GAME_NAME
    wanted = sorted(args.stages)
    captured: dict[int, Path] = {}
    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        if not set(wanted) - set(captured):
            break
        obs, _ = env.reset(seed=args.seed + ep)
        best_stage = 0
        for step in range(args.max_steps):
            if rng.random() < args.epsilon:
                action = int(rng.integers(0, int(env.action_space.n)))
            else:
                action = int(agent.act(obs[None, ...], greedy=True)[0][0])
            obs, _, term, trunc, info = env.step(action)

            stage = int(info.get("stage", -1))
            if stage > best_stage:
                best_stage = stage
            # Snapshot on first arrival at a target we do not have yet. The
            # emulator state is taken now, mid-fight, so the resumed episode
            # begins exactly here.
            if stage in wanted and stage not in captured:
                path = out_dir / f"{args.prefix}{stage}.state"
                with gzip.open(path, "wb") as fh:
                    fh.write(base._retro.em.get_state())
                captured[stage] = path
                console.print(
                    f"  [green]captured[/green] stage {stage:>2} at step {step:>6,} "
                    f"(score {info.get('score', -1):,}) -> {path.name}"
                )
            if term or trunc:
                break
        console.print(
            f"episode {ep + 1}: reached stage {best_stage}, "
            f"{len(captured)}/{len(wanted)} targets captured"
        )

    env.close()

    missing = sorted(set(wanted) - set(captured))
    console.print(f"\n[bold]captured {len(captured)}/{len(wanted)}[/bold] -> {out_dir}")
    if missing:
        console.print(
            f"[yellow]never reached: {missing}. Use a stronger checkpoint, raise "
            f"--episodes, or drop those targets.[/yellow]"
        )

    if captured:
        console.print("\nAdd to your config:\n")
        console.print("env:")
        console.print("  start_states:")
        console.print("    - {name: Level1, weight: 1.0}")
        for stage in sorted(captured):
            console.print(f"    - {{name: {args.prefix}{stage}, weight: 1.0}}")


if __name__ == "__main__":
    main()
