"""Live mosaic viewer: watch the agent play while training runs.

Training is headless on purpose -- rendering windows is most of what made the
2021 screen-capture loop slow, and re-introducing it would re-introduce the
coupling that broke the MDP. So this is a **separate playback process**. It
loads the newest checkpoint from a run directory, replays it across a grid of
envs, and streams the result as MJPEG to a browser. It never touches the
training loop, so it costs training nothing but a little CPU.

Because it re-reads the checkpoint on a timer, leaving it open shows the policy
actually changing over the course of a run.

    python -m tools.watch --run runs/my-run
    # then open http://localhost:8080

On Windows, WSL2 forwards localhost, so that URL works from a Windows browser
with no X server involved.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
from rich.console import Console

from kungfu.config import Config
from kungfu.rl.dqn import DQNAgent
from tools.mjpeg import FrameSlot, serve

console = Console()

# Native NES framebuffer.
NES_W, NES_H = 240, 224
LABEL_H = 26
HEADER_H = 64


@dataclass
class Args:
    run: Path
    """Run directory to watch (uses its newest checkpoint), or a checkpoint file."""
    config: Path | None = None
    """Config to build envs from; defaults to <run>/config.yaml, else built-in defaults."""
    envs: int = 6
    """How many games to play at once."""
    cols: int = 3
    scale: int = 2
    """Integer upscale of each 240x224 tile."""
    port: int = 8080
    host: str = "0.0.0.0"
    fps: int = 30
    """Playback frame rate (the emulator can go far faster; this is for your eyes)."""
    epsilon: float = 0.02
    """A little noise stops every tile playing the identical game."""
    reload_every: float = 30.0
    """Seconds between checkpoint reloads. 0 disables reloading."""
    title: str = "Yie Ar Kung-Fu - live agent"
    window: bool = False
    """Show a native OpenCV window instead of streaming (needs a display)."""
    max_steps: int | None = None
    """Override the episode length cap. NOTE: this cap lives in the env, so the
    config default (6000 agent steps) applies to the viewer and demo too, not
    just training -- it is why a run appears to jump back to stage 1."""
    stall_steps: int | None = None
    """Override the no-score-progress timeout (config default 600 steps)."""
    endless: bool = False
    """Remove both caps: play until the agent actually loses all its lives."""


def apply_limit_overrides(cfg, args) -> None:
    """Let a viewing/demo session outlive the training episode cap.

    Training wants short episodes: they keep the replay buffer diverse and stop
    one lucky run dominating a batch. Watching wants the opposite -- you want to
    see how far the agent really gets. Same env, different job, so the caps are
    overridable here rather than baked in.
    """
    if args.endless:
        cfg.env.max_episode_steps = 10_000_000
        cfg.env.stall_timeout = 10_000_000
    if args.max_steps is not None:
        cfg.env.max_episode_steps = args.max_steps
    if args.stall_steps is not None:
        cfg.env.stall_timeout = args.stall_steps


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def draw_tile(frame_rgb: np.ndarray, info: dict, scale: int) -> np.ndarray:
    """One upscaled game frame with a label strip underneath."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    tile = cv2.resize(bgr, (NES_W * scale, NES_H * scale), interpolation=cv2.INTER_NEAREST)

    strip = np.full((LABEL_H, NES_W * scale, 3), (34, 26, 20), dtype=np.uint8)
    score = info.get("score", -1)
    stage = info.get("stage", -1)
    lives = info.get("lives", -1)
    php = info.get("player_health", -1.0)

    left = f"score {score:,}" if score is not None and score >= 0 else "score --"
    mid = f"stage {stage}" if stage is not None and stage >= 0 else "stage --"
    right = f"lives {lives}" if lives is not None and lives >= 0 else "lives --"
    cv2.putText(strip, f"{left}   {mid}   {right}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (242, 246, 247), 1, cv2.LINE_AA)

    # Health pip: green when healthy, red when nearly dead.
    if php is not None and php >= 0:
        bar_w = int(70 * float(php))
        x0 = NES_W * scale - 80
        cv2.rectangle(strip, (x0, 8), (x0 + 70, 18), (60, 50, 44), -1)
        if bar_w > 0:
            colour = (103, 139, 46) if php > 0.3 else (60, 69, 196)
            cv2.rectangle(strip, (x0, 8), (x0 + bar_w, 18), colour, -1)

    return np.vstack([tile, strip])


def compose(tiles: list[np.ndarray], cols: int, header: str, sub: str) -> np.ndarray:
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)

    bar = np.full((HEADER_H, grid.shape[1], 3), (34, 26, 20), dtype=np.uint8)
    cv2.putText(bar, header, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (242, 246, 247), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (143, 128, 117), 1, cv2.LINE_AA)
    return np.vstack([bar, grid])


# --------------------------------------------------------------------------
def newest_checkpoint(run: Path) -> Path | None:
    if run.is_file():
        return run
    ckpts = sorted(run.glob("checkpoint_*.pt"))
    if ckpts:
        return ckpts[-1]
    final = run / "final.pt"
    return final if final.exists() else None


class ViewerEnvFactory:
    """Picklable, render-capable env constructor.

    stable-retro allows exactly one emulator instance per process, so a mosaic
    of N games means N worker processes -- the same reason training uses
    AsyncVectorEnv. Must be a module-level class so "spawn" can pickle it.
    """

    def __init__(self, cfg: Config, seed: int, idx: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.idx = idx

    def __call__(self):
        from kungfu.emulator.integration import INTEGRATION_DIR, register
        from kungfu.envs.kungfu import YieArKungFuEnv
        from kungfu.envs.wrappers import ChannelStack
        from kungfu.train import ATLAS_PATH
        from kungfu.vision.atlas import DigitAtlas

        register(INTEGRATION_DIR)
        atlas = DigitAtlas.load(ATLAS_PATH)
        env = ChannelStack(
            YieArKungFuEnv(self.cfg.env, self.cfg.reward, atlas, render_mode="rgb_array"),
            self.cfg.env.frame_stack,
        )
        env.reset(seed=self.seed + 1000 * self.idx)
        env.action_space.seed(self.seed + 1000 * self.idx)
        return env


def build_envs(cfg: Config, n: int, seed: int = 0):
    import gymnasium as gym

    fns = [ViewerEnvFactory(cfg, seed, i) for i in range(n)]
    return gym.vector.AsyncVectorEnv(fns, context="spawn")


def per_env(infos: dict, key: str, i: int, default=-1):
    """Pull one env's value out of Gymnasium's aggregated info arrays."""
    vals = infos.get(key)
    if vals is None:
        return default
    arr = np.atleast_1d(vals)
    if i >= len(arr):
        return default
    v = arr[i]
    return default if v is None else v


def run_viewer(args: Args) -> None:
    cfg_path = args.config
    if cfg_path is None:
        candidate = (args.run if args.run.is_dir() else args.run.parent) / "config.yaml"
        cfg_path = candidate if candidate.exists() else None
    cfg = Config.load(cfg_path)
    apply_limit_overrides(cfg, args)
    console.print(f"config: [dim]{cfg_path or 'built-in defaults'}[/dim]")
    console.print(
        f"episode cap: [dim]{cfg.env.max_episode_steps:,} steps, "
        f"stall {cfg.env.stall_timeout:,}[/dim]"
    )

    envs = build_envs(cfg, args.envs)
    obs, _ = envs.reset(seed=4242)
    obs_shape = envs.single_observation_space.shape
    n_actions = int(envs.single_action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        dqn_cfg=cfg.dqn,
        replay_cfg=cfg.replay,
        device=device,
        seed=1234,
        num_envs=1,
    )
    agent.online.eval()

    loaded_from: Path | None = None
    loaded_step = 0
    last_reload = 0.0
    rng = np.random.default_rng(0)

    def try_reload(force: bool = False) -> None:
        nonlocal loaded_from, loaded_step, last_reload
        now = time.time()
        if not force and (args.reload_every <= 0 or now - last_reload < args.reload_every):
            return
        last_reload = now
        ckpt = newest_checkpoint(args.run)
        if ckpt is None:
            return
        if ckpt == loaded_from and not force:
            # Same file may still have been rewritten (final.pt); cheap mtime check.
            return
        try:
            agent.load(ckpt)
            agent.online.eval()
            loaded_from, loaded_step = ckpt, agent.steps
            console.print(f"loaded [green]{ckpt.name}[/green] @ step {agent.steps:,}")
        except Exception as exc:
            # A checkpoint half-written by the trainer -- just retry next cycle.
            console.print(f"[dim]checkpoint not readable yet: {type(exc).__name__}[/dim]")

    try_reload(force=True)
    if loaded_from is None:
        console.print(
            f"[yellow]no checkpoint in {args.run} yet -- showing an untrained policy; "
            "it will pick one up as soon as training writes it[/yellow]"
        )

    slot = FrameSlot()
    server = None
    if not args.window:
        server = serve(
            slot, args.host, args.port, args.title,
            f"Reloads the newest checkpoint every {int(args.reload_every)}s "
            "&middot; separate process, training is unaffected",
        )
        console.print(
            f"\n  [bold green]{args.title}[/bold green]\n"
            f"  open [bold]http://localhost:{args.port}[/bold]  "
            f"[dim](Ctrl+C to stop)[/dim]\n"
        )

    frame_budget = 1.0 / max(args.fps, 1)
    shown = 0
    t_fps = time.time()
    fps_now = 0.0
    best = 0

    try:
        while True:
            t0 = time.time()
            try_reload()

            actions = agent.act(obs, greedy=True)
            explore = rng.random(args.envs) < args.epsilon
            if explore.any():
                rand = rng.integers(0, n_actions, size=args.envs)
                actions = np.where(explore, rand, actions)

            obs, _, _, _, infos = envs.step(actions)
            # Each worker holds its own emulator, so frames come back over the pipe.
            frames = envs.call("render")

            tiles = []
            for i in range(args.envs):
                frame = frames[i]
                if frame is None:
                    frame = np.zeros((NES_H, NES_W, 3), dtype=np.uint8)
                tiles.append(
                    draw_tile(
                        frame,
                        {
                            "score": per_env(infos, "score", i),
                            "stage": per_env(infos, "stage", i),
                            "lives": per_env(infos, "lives", i),
                            "player_health": per_env(infos, "player_health", i, -1.0),
                        },
                        args.scale,
                    )
                )
                best = max(best, int(per_env(infos, "best_score", i, 0) or 0))

            shown += 1
            if shown % 30 == 0:
                fps_now = 30.0 / max(time.time() - t_fps, 1e-6)
                t_fps = time.time()

            header = args.title
            sub = (
                f"checkpoint {loaded_from.name if loaded_from else 'none'}"
                f"   step {loaded_step:,}"
                f"   best score {best:,}"
                f"   {args.envs} envs @ {fps_now:0.0f} fps"
            )
            mosaic = compose(tiles, args.cols, header, sub)

            if args.window:
                cv2.imshow(args.title, mosaic)
                if cv2.waitKey(1) == 27:
                    break
            else:
                slot.publish_bgr(mosaic)

            spare = frame_budget - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        console.print("\nstopping")
    finally:
        slot.shutdown()
        if server is not None:
            server.shutdown()
        if args.window:
            cv2.destroyAllWindows()
        envs.close()


def main() -> None:
    # `make go` kills the viewer with SIGTERM when training exits. Turn that
    # into the same clean shutdown path as Ctrl+C, so the worker emulators are
    # closed properly instead of dying mid-pipe.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    run_viewer(tyro.cli(Args))


if __name__ == "__main__":
    main()
