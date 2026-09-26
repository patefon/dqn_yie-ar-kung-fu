"""Greedy evaluation and video capture.

The old project had no evaluation path at all. It reported an "average score"
computed from a list it only appended to when the episode *improved*, so the
average was a running mean of a monotonically-increasing subsequence -- it could
only go up, it never reflected real performance, and because checkpoints were
saved on that same condition, saving effectively stopped once the biased average
outran what the agent could actually do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from rich.console import Console
from rich.table import Table

from kungfu.config import Config
from kungfu.rl.algos import Agent, build_agent

console = Console()


@dataclass
class Args:
    checkpoint: Path
    """Checkpoint to evaluate."""
    config: Path | None = None
    episodes: int = 10
    seed: int = 12345
    video: Path | None = None
    """Write an mp4 of the best episode here."""
    epsilon: float = 0.01
    """A little noise avoids deterministic soft-locks in a stuck state."""
    max_steps: int | None = None
    """Override the episode length cap (config default 6000 agent steps)."""
    stall_steps: int | None = None
    """Override the no-score-progress timeout (config default 600 steps)."""
    endless: bool = False
    """Remove both caps: measure how far the agent gets before it actually loses."""


@dataclass
class EpisodeResult:
    ret: float
    score: int
    stage: int
    steps: int


def run_episode(
    env,
    agent: Agent,
    epsilon: float,
    capture: bool,
    rng: np.random.Generator | None = None,
) -> tuple[EpisodeResult, list]:
    rng = rng or np.random.default_rng()
    obs, _ = env.reset()
    total, steps = 0.0, 0
    score, stage = 0, 0
    frames: list = []

    while True:
        batch = obs[None, ...]
        if rng.random() < epsilon:
            action = int(rng.integers(0, agent.n_actions))
        else:
            action = int(agent.act(batch, greedy=True)[0][0])

        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
        if int(info.get("score", -1)) >= 0:
            score = max(score, int(info["score"]))
        if int(info.get("stage", -1)) >= 0:
            stage = max(stage, int(info["stage"]))
        if capture:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        if terminated or truncated:
            break

    return EpisodeResult(total, score, stage, steps), frames


def apply_limit_overrides(cfg: Config, args: Args) -> None:
    """The episode caps live in the env, so evaluation inherits them too.

    A capped eval measures "score per 6000 steps", which is an endurance number,
    not a skill ceiling. `--endless` measures how far the agent actually gets.
    """
    if args.endless:
        cfg.env.max_episode_steps = 10_000_000
        cfg.env.stall_timeout = 10_000_000
    if args.max_steps is not None:
        cfg.env.max_episode_steps = args.max_steps
    if args.stall_steps is not None:
        cfg.env.stall_timeout = args.stall_steps


def evaluate(cfg: Config, args: Args) -> list[EpisodeResult]:
    from kungfu.emulator.integration import INTEGRATION_DIR, register
    from kungfu.envs.kungfu import YieArKungFuEnv
    from kungfu.envs.wrappers import ChannelStack
    from kungfu.vision.atlas import DigitAtlas

    apply_limit_overrides(cfg, args)
    console.print(
        f"episode cap: [dim]{cfg.env.max_episode_steps:,} steps, "
        f"stall {cfg.env.stall_timeout:,}[/dim]"
    )
    register(INTEGRATION_DIR)
    atlas = DigitAtlas.load("configs/digit_atlas.json")
    # Same reason as periodic_eval: a curriculum belongs to training only.
    if cfg.env.start_states:
        console.print("[dim]ignoring training start_states; evaluating from Level1[/dim]")
    eval_env_cfg = cfg.env.model_copy(update={"start_states": None})
    env = ChannelStack(
        YieArKungFuEnv(eval_env_cfg, cfg.reward, atlas, render_mode="rgb_array"),
        cfg.env.frame_stack,
    )
    env.reset(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = build_agent(
        cfg, env.observation_space.shape, int(env.action_space.n),
        device, seed=args.seed, num_envs=1,
    )
    agent.load(args.checkpoint)
    agent.train_mode(False)

    rng = np.random.default_rng(args.seed)
    results: list[EpisodeResult] = []
    best_frames: list = []
    best_score = -1

    for ep in range(args.episodes):
        result, frames = run_episode(
            env, agent, args.epsilon, capture=args.video is not None, rng=rng
        )
        results.append(result)
        if result.score > best_score:
            best_score, best_frames = result.score, frames
        console.print(
            f"episode {ep + 1:>3}: return {result.ret:8.2f} | "
            f"score {result.score:>7,} | stage {result.stage:>2} | steps {result.steps:,}"
        )

    env.close()

    if args.video and best_frames:
        import imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, best_frames, fps=60, quality=8)
        console.print(f"wrote {args.video} ({len(best_frames)} frames)")

    table = Table(title=f"{args.episodes} greedy episodes")
    table.add_column("metric")
    table.add_column("value", justify="right")
    scores = [r.score for r in results]
    rets = [r.ret for r in results]
    stages = [r.stage for r in results]
    table.add_row("score mean", f"{np.mean(scores):,.0f}")
    table.add_row("score max", f"{np.max(scores):,}")
    table.add_row("score median", f"{np.median(scores):,.0f}")
    table.add_row("return mean", f"{np.mean(rets):.2f}")
    table.add_row("stage max", f"{np.max(stages)}")
    table.add_row("stage mean", f"{np.mean(stages):.1f}")
    console.print(table)
    return results


def main() -> None:
    args = tyro.cli(Args)
    cfg = Config.load(args.config)
    evaluate(cfg, args)


if __name__ == "__main__":
    main()
