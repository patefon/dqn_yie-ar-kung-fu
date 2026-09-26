"""Training entry point.

Structural differences from the old ``app.py``
-----------------------------------------------
The old runner spread the grabber, detector, agent and GUI across a 16-process
``multiprocessing.Pool`` joined by lossy queues, then busy-waited in the parent
on ``proc.ready()`` -- a spin loop that burned a core doing nothing. Frames were
dropped whenever a queue was full, so the agent trained on a corrupted MDP.

Here there is one loop. Envs are vectorised (each is a real, independent
emulator), the step is synchronous, and nothing is ever dropped.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tyro
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter

from kungfu.config import Config
from kungfu.envs.actions import ACTION_NAMES
from kungfu.envs.kungfu import END_REASON_NAMES
from kungfu.rl.algos import Agent, Transition, build_agent

console = Console()

# Resolved absolute so spawned worker processes do not depend on their cwd.
ATLAS_PATH = str(Path(__file__).resolve().parents[2] / "configs" / "digit_atlas.json")


@dataclass
class Args:
    config: Path | None = None
    """YAML config to load; omit for defaults."""
    run_name: str | None = None
    """Name for this run directory; defaults to a timestamp."""
    total_steps: int | None = None
    """Override config.train.total_steps."""
    num_envs: int | None = None
    """Override config.train.num_envs."""
    seed: int | None = None
    resume: Path | None = None
    """Explicit checkpoint to resume from."""


def resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


class EnvFactory:
    """Picklable env constructor.

    Must be a module-level class, not a closure: ``AsyncVectorEnv`` with the
    "spawn" context pickles each factory to send it to the worker process, and a
    locally-defined function cannot be pickled.
    """

    def __init__(self, cfg: Config, seed: int, idx: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.idx = idx

    def __call__(self):
        from kungfu.emulator.integration import INTEGRATION_DIR, register
        from kungfu.envs.kungfu import YieArKungFuEnv
        from kungfu.envs.wrappers import ChannelStack, EpisodeStats
        from kungfu.vision.atlas import DigitAtlas

        register(INTEGRATION_DIR)
        atlas = DigitAtlas.load(ATLAS_PATH)
        env = YieArKungFuEnv(self.cfg.env, self.cfg.reward, atlas)
        env = ChannelStack(env, self.cfg.env.frame_stack)
        env = EpisodeStats(env)
        env.reset(seed=self.seed + self.idx)
        env.action_space.seed(self.seed + self.idx)
        return env


def build_vector_env(cfg: Config, seed: int) -> gym.vector.VectorEnv:
    fns = [EnvFactory(cfg, seed, i) for i in range(cfg.train.num_envs)]
    # Separate processes: each emulator is a C extension holding its own state,
    # and this is real parallelism rather than the old pool of frame-droppers.
    if cfg.train.num_envs == 1:
        return gym.vector.SyncVectorEnv(fns)
    return gym.vector.AsyncVectorEnv(fns, context="spawn")


def find_c_compiler() -> str | None:
    """Locate a C compiler, which Triton needs to build kernels for torch.compile."""
    explicit = os.environ.get("CC")
    if explicit:
        return shutil.which(explicit) or (explicit if Path(explicit).exists() else None)
    for name in ("cc", "gcc", "clang"):
        found = shutil.which(name)
        if found:
            return found
    return None


def maybe_compile(agent: Agent, cfg: Config, device: torch.device, obs_shape) -> bool:
    """Enable torch.compile only if it actually works, and prove it before training.

    torch.compile is lazy: it returns a wrapper immediately and Triton does not
    build a kernel until the first forward pass. So wrapping the `torch.compile`
    call in try/except catches nothing -- the real failure (for example a missing
    C compiler) lands thousands of steps later, mid-run, and kills training.
    Here we force compilation with a probe batch up front and fall back to eager
    on any failure.
    """
    if not cfg.train.torch_compile:
        return False
    if device.type != "cuda":
        console.print("[dim]torch.compile skipped (CPU device)[/dim]")
        return False

    compiler = find_c_compiler()
    if compiler is None:
        console.print(
            "[yellow]torch.compile disabled: no C compiler on PATH.[/yellow]\n"
            "[dim]  Triton builds its kernels with one. Install it for the speedup:\n"
            "    sudo apt install build-essential\n"
            "  Or set train.torch_compile: false in your config to silence this.[/dim]"
        )
        return False

    # Compile whatever modules this algorithm happens to have -- DQN exposes
    # online/target, PPO exposes a single actor-critic net.
    eager = dict(agent.modules)
    try:
        for attr, module in eager.items():
            setattr(agent, attr, torch.compile(module))
        # Force the kernel build now, while we can still recover from it.
        probe = torch.zeros((cfg.train.num_envs, *obs_shape), dtype=torch.uint8, device=device)
        with torch.no_grad():
            for module in agent.modules.values():
                module(probe)
    except Exception as exc:
        for attr, module in eager.items():
            setattr(agent, attr, module)
        console.print(
            f"[yellow]torch.compile failed, continuing in eager mode:[/yellow] "
            f"[dim]{type(exc).__name__}: {str(exc).splitlines()[-1][:160]}[/dim]"
        )
        return False

    console.print(f"[green]torch.compile active[/green] [dim](cc: {compiler})[/dim]")
    return True


def describe_agent(agent: Agent, cfg: Config, n_actions: int) -> str:
    """One line naming the algorithm, trunk and memory footprint."""
    params = sum(
        p.numel() for m in agent.modules.values() for p in m.parameters() if p.requires_grad
    )
    bits = [
        f"[bold]{agent.name}[/bold]",
        f"encoder {cfg.encoder.name}",
        f"{params / 1e6:.2f}M params",
        f"{n_actions} actions",
    ]
    buf = getattr(agent, "memory", None) or getattr(agent, "buffer", None)
    if buf is not None and hasattr(buf, "nbytes"):
        kind = "replay" if agent.on_policy is False else "rollout"
        bits.append(f"{kind} {buf.nbytes() / 1e9:.2f} GB")
    return " | ".join(bits)


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
    return ckpts[-1] if ckpts else None


def _moviepy_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("moviepy") is not None


def periodic_eval(cfg: Config, agent: Agent, writer: SummaryWriter, run_dir: Path) -> None:
    """Greedy rollouts on a separate env, logged as scalars plus a video.

    This is the "watch it get better" loop: every ``eval_every`` steps you get an
    honest, exploration-free measurement and a clip of the best episode, both
    visible in TensorBoard. Training-time return is a poor progress signal on its
    own because it is entangled with the epsilon schedule and with shaping.
    """
    from kungfu.emulator.integration import INTEGRATION_DIR, register
    from kungfu.envs.kungfu import YieArKungFuEnv
    from kungfu.envs.wrappers import ChannelStack
    from kungfu.evaluate import run_episode
    from kungfu.vision.atlas import DigitAtlas

    register(INTEGRATION_DIR)
    atlas = DigitAtlas.load(ATLAS_PATH)
    # Evaluation always starts from Level1, even when training uses a
    # start-state curriculum. Otherwise the score reports which savestate was
    # sampled -- an episode seeded at Stage20 inherits ~219,500 points -- and
    # the curve measures luck instead of skill.
    eval_env_cfg = cfg.env.model_copy(update={"start_states": None})
    env = ChannelStack(
        YieArKungFuEnv(eval_env_cfg, cfg.reward, atlas, render_mode="rgb_array"),
        cfg.env.frame_stack,
    )
    env.reset(seed=cfg.train.seed + 9973)

    was_training = next(iter(agent.modules.values())).training
    agent.train_mode(False)

    results, best_frames, best_score = [], [], -1
    try:
        for i in range(cfg.train.eval_episodes):
            capture = cfg.train.record_eval_video and i == 0
            result, frames = run_episode(env, agent, epsilon=0.01, capture=capture)
            results.append(result)
            if frames and result.score > best_score:
                best_score, best_frames = result.score, frames
    finally:
        env.close()
        if was_training:
            agent.train_mode(True)

    step = agent.steps
    writer.add_scalar("eval/score_mean", float(np.mean([r.score for r in results])), step)
    writer.add_scalar("eval/score_max", float(np.max([r.score for r in results])), step)
    writer.add_scalar("eval/return_mean", float(np.mean([r.ret for r in results])), step)
    writer.add_scalar("eval/stage_max", float(np.max([r.stage for r in results])), step)
    writer.add_scalar("eval/steps_mean", float(np.mean([r.steps for r in results])), step)

    if best_frames:
        # The MP4 on disk is the reliable artefact; the TensorBoard embed is a
        # convenience that needs moviepy, which is not a hard dependency.
        try:
            import imageio

            vid_dir = run_dir / "videos"
            vid_dir.mkdir(exist_ok=True)
            imageio.mimwrite(vid_dir / f"eval_{step:010d}.mp4", best_frames, fps=60, quality=8)
        except Exception as exc:  # pragma: no cover - video is a nicety
            console.print(f"[yellow]video write failed: {exc}[/yellow]")

        if _moviepy_available():
            # TensorBoard wants (N, T, C, H, W); subsample so the clip stays small.
            clip = np.stack(best_frames[::4])[None]
            writer.add_video(
                "eval/episode", torch.from_numpy(clip).permute(0, 1, 4, 2, 3), step, fps=15
            )
        elif not periodic_eval._warned_moviepy:
            # tensorboard's add_video prints "add_video needs package moviepy"
            # and returns, which looks alarming mid-run but breaks nothing.
            periodic_eval._warned_moviepy = True
            console.print(
                "[dim]inline TensorBoard video disabled (no moviepy). "
                f"Clips are still written to {run_dir / 'videos'}. "
                "`uv pip install moviepy` to embed them.[/dim]"
            )

    console.print(
        f"[cyan]eval[/cyan] step {step:,} | score mean "
        f"{np.mean([r.score for r in results]):,.0f} | max "
        f"{np.max([r.score for r in results]):,} | stage max "
        f"{np.max([r.stage for r in results])}"
    )


periodic_eval._warned_moviepy = False  # type: ignore[attr-defined]


def train(cfg: Config, args: Args) -> Path:
    seed = args.seed if args.seed is not None else cfg.train.seed
    torch.manual_seed(seed)

    run_name = args.run_name or cfg.train.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.train.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(run_dir / "config.yaml")

    device = resolve_device(cfg.train.device)
    console.print(f"[bold]run[/bold] {run_dir}  [bold]device[/bold] {device}")

    envs = build_vector_env(cfg, seed)
    obs_shape = envs.single_observation_space.shape
    n_actions = int(envs.single_action_space.n)

    agent = build_agent(
        cfg, obs_shape, n_actions, device, seed=seed, num_envs=cfg.train.num_envs
    )
    console.print(describe_agent(agent, cfg, n_actions))

    # Auto-resume: the old project defined load_models() and never called it.
    ckpt = args.resume or (latest_checkpoint(run_dir) if cfg.train.auto_resume else None)
    if ckpt and Path(ckpt).exists():
        agent.load(ckpt)
        console.print(f"[green]resumed[/green] from {ckpt} at step {agent.steps:,}")

    maybe_compile(agent, cfg, device, obs_shape)

    writer = SummaryWriter(run_dir / "tb")
    writer.add_text("config", f"```yaml\n{(run_dir / 'config.yaml').read_text()}\n```")
    writer.add_text("actions", ", ".join(f"{i}:{n}" for i, n in enumerate(ACTION_NAMES)))

    obs, _ = envs.reset(seed=seed)
    episode_starts = np.ones(cfg.train.num_envs, dtype=bool)
    returns = np.zeros(cfg.train.num_envs, dtype=np.float64)
    completed: list[float] = []
    best_score = 0
    end_counts: dict[int, int] = {}

    total = args.total_steps or cfg.train.total_steps
    t0 = time.time()
    last_report = agent.steps

    while agent.steps < total:
        actions, extras = agent.act(obs)
        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        dones = np.logical_or(terms, truncs)

        agent.observe(
            Transition(
                obs=obs,
                actions=actions,
                rewards=rewards,
                terminated=terms,
                truncated=truncs,
                next_obs=next_obs,
                episode_starts=episode_starts,
                extras=extras,
            )
        )
        # Only a true terminal breaks bootstrapping; a time-limit truncation
        # must still bootstrap or the agent learns the clock is a cliff.
        episode_starts = dones.copy()
        returns += rewards
        obs = next_obs

        # The agent owns its own update cadence: DQN trains every few steps,
        # PPO only when its rollout segment is full.
        stats = agent.maybe_update()
        if stats is not None and agent.steps % 1000 < cfg.train.num_envs:
            for key, value in stats.as_dict().items():
                writer.add_scalar(f"{agent.name}/{key}", value, agent.steps)
            writer.add_scalar("explore/value", agent.exploration(), agent.steps)

        for i, done in enumerate(dones):
            if not done:
                continue
            completed.append(float(returns[i]))
            returns[i] = 0.0
            reason = int(np.atleast_1d(infos.get("end_reason", 0))[i]) if "end_reason" in infos else 0
            end_counts[reason] = end_counts.get(reason, 0) + 1

        # Per-term reward accounting keeps shaping honest and debuggable.
        if "reward_terms" in infos and agent.steps % 2000 < cfg.train.num_envs:
            terms_list = [t for t in infos["reward_terms"] if isinstance(t, dict)]
            if terms_list:
                for key in terms_list[0]:
                    writer.add_scalar(
                        f"reward/{key}",
                        float(np.mean([t[key] for t in terms_list])),
                        agent.steps,
                    )
        if "best_score" in infos:
            scores = [s for s in np.atleast_1d(infos["best_score"]) if s is not None]
            if scores:
                best_score = max(best_score, int(max(scores)))

        if agent.steps - last_report >= 10_000:
            fps = (agent.steps - last_report) / max(time.time() - t0, 1e-6)
            window = completed[-100:]
            avg_return = float(np.mean(window)) if window else float("nan")
            writer.add_scalar("perf/fps", fps, agent.steps)
            writer.add_scalar("episode/return_mean100", avg_return, agent.steps)
            writer.add_scalar("episode/count", len(completed), agent.steps)
            writer.add_scalar("game/best_score", best_score, agent.steps)
            # Share of episodes ending each way. A run dominated by "stall" is
            # being cut off while alive, not beaten -- a config problem, not a
            # learning one.
            total_ends = sum(end_counts.values())
            if total_ends:
                for code, label in END_REASON_NAMES.items():
                    writer.add_scalar(
                        f"episode_end/{label}", end_counts.get(code, 0) / total_ends, agent.steps
                    )
            console.print(
                f"step {agent.steps:>10,} | {fps:6.0f} fps | "
                f"return(100) {avg_return:8.2f} | explore {agent.exploration():.3f} | "
                f"episodes {len(completed):,} | best score {best_score:,} | "
                f"ends {'/'.join(f'{END_REASON_NAMES[c][:4]}:{end_counts.get(c, 0)}' for c in END_REASON_NAMES)}"
            )
            last_report = agent.steps
            t0 = time.time()

        if agent.steps % cfg.train.checkpoint_every < cfg.train.num_envs:
            path = run_dir / f"checkpoint_{agent.steps:010d}.pt"
            agent.save(path)
            console.print(f"[dim]saved {path.name}[/dim]")

        if agent.steps > cfg.dqn.learn_start and agent.steps % cfg.train.eval_every < cfg.train.num_envs:
            periodic_eval(cfg, agent, writer, run_dir)

    final = run_dir / "final.pt"
    agent.save(final)
    envs.close()
    writer.close()
    console.print(f"[bold green]done[/bold green] -> {final}")
    return final


def main() -> None:
    args = tyro.cli(Args)
    cfg = Config.load(args.config)
    if args.total_steps is not None:
        cfg.train.total_steps = args.total_steps
    if args.num_envs is not None:
        cfg.train.num_envs = args.num_envs
    if args.seed is not None:
        cfg.train.seed = args.seed
    train(cfg, args)


if __name__ == "__main__":
    main()
