"""Record a presentation-quality demo of the trained agent.

Produces an MP4 with the real game audio and a NES pad overlay that lights up
every button the agent presses, so you can see what it is actually doing rather
than just the result.

    python -m tools.demo --checkpoint runs/BIG-ML/final.pt --out out/demo.mp4

Unlike `tools.watch` (live, silent, many games at once), this renders one game
at presentation size, with sound, to a file you can share.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
from rich.console import Console

from kungfu.config import Config
from kungfu.envs.actions import ACTION_COMBOS
from kungfu.rl.dqn import DQNAgent
from tools.mjpeg import FrameSlot, serve

console = Console()

NES_W, NES_H = 240, 224
PANEL_W = 300
BG = (26, 20, 16)          # BGR
PANEL_BG = (38, 30, 24)
IDLE = (74, 62, 54)
LIT = (110, 196, 92)       # green
LIT_A = (90, 120, 240)     # A = warm
TEXT = (242, 246, 247)
DIM = (150, 136, 124)


@dataclass
class Args:
    checkpoint: Path
    """Trained checkpoint to demo."""
    out: Path = Path("out/demo.mp4")
    config: Path | None = None
    """Defaults to <checkpoint dir>/config.yaml."""
    seconds: float = 60.0
    """Seconds of gameplay to record. Ignored when --episodes is set."""
    episodes: int | None = None
    """Record this many COMPLETE episodes instead of a fixed duration -- the
    agent plays until it actually dies. Note this is separate from --endless:
    --endless lifts the env episode cap, --episodes lifts the recording cap.
    You normally want both."""
    max_seconds: float = 3600.0
    """Safety stop for --episodes, so a very long-lived agent cannot fill the
    disk. 1020x986 at 60fps is roughly 1 MB per second of footage."""
    scale: int = 3
    """Upscale of the 240x224 framebuffer."""
    epsilon: float = 0.01
    seed: int = 7
    fps: int = 60
    """Output video frame rate. The emulator runs 60 fps; one agent step is
    frame_skip frames, so each agent step becomes frame_skip video frames."""
    mute: bool = False
    no_vision: bool = False
    """Hide the second screen showing the frame stack the network receives."""
    max_steps: int | None = None
    """Override the episode length cap. NOTE: this cap lives in the env, so the
    config default (6000 agent steps) applies to the viewer and demo too, not
    just training -- it is why a run appears to jump back to stage 1."""
    stall_steps: int | None = None
    """Override the no-score-progress timeout (config default 600 steps)."""
    endless: bool = False
    """Remove both caps: play until the agent actually loses all its lives."""
    live: bool = False
    """Also stream what is being recorded to a browser, while it records."""
    port: int = 8081
    """Port for --live. Defaults clear of the training viewer on 8080."""
    host: str = "0.0.0.0"
    realtime: bool = False
    """Throttle to true game speed. Recording runs far faster than realtime by
    default; with --live you usually want to watch it at the speed it plays."""


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
# NES pad overlay
# --------------------------------------------------------------------------
def draw_pad(panel: np.ndarray, pressed: set[str], x0: int, y0: int) -> None:
    """A small NES controller; pressed buttons light up."""

    def on(name: str) -> bool:
        return name in pressed

    # D-pad: 3x3 cross of 26px cells.
    c = 26
    cx, cy = x0 + 46, y0 + 46
    cells = {
        "UP": (cx - c // 2, cy - c - c // 2),
        "DOWN": (cx - c // 2, cy + c // 2),
        "LEFT": (cx - c - c // 2, cy - c // 2),
        "RIGHT": (cx + c // 2, cy - c // 2),
    }
    cv2.rectangle(panel, (cx - c // 2, cy - c // 2), (cx + c // 2, cy + c // 2), IDLE, -1)
    for name, (bx, by) in cells.items():
        cv2.rectangle(panel, (bx, by), (bx + c, by + c), LIT if on(name) else IDLE, -1)
        cv2.rectangle(panel, (bx, by), (bx + c, by + c), BG, 1)

    # B and A, round, to the right.
    for name, dx, colour in (("B", 0, LIT), ("A", 58, LIT_A)):
        bx, by = x0 + 150 + dx, cy
        cv2.circle(panel, (bx, by), 19, colour if on(name) else IDLE, -1)
        cv2.circle(panel, (bx, by), 19, BG, 1)
        cv2.putText(panel, name, (bx - 7, by + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, BG if on(name) else DIM, 2, cv2.LINE_AA)

    cv2.putText(panel, "A = punch    B = kick", (x0, y0 + 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)


# Oldest -> newest. Max-blending these means static scenery (present in every
# frame) sums to near-white, while anything that moved leaves coloured ghosts.
# This is the 3-frames-as-RGB-channels idea generalised: it still works when the
# stack is not exactly 3 deep, and it says which way time runs.
TRAIL_BGR = (
    (210, 90, 70),     # t-3  blue
    (200, 170, 60),    # t-2  teal
    (90, 200, 225),    # t-1  amber
    (255, 255, 255),   # t    white = now
)


def motion_composite(stack: np.ndarray) -> np.ndarray:
    """(N, H, W) uint8 -> (H, W, 3) BGR colour-coded motion trail."""
    n, h, w = stack.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(n):
        colour = np.array(TRAIL_BGR[i % len(TRAIL_BGR)], dtype=np.float32)
        layer = (stack[i].astype(np.float32) / 255.0)[:, :, None] * colour
        np.maximum(out, layer, out=out)
    return out.clip(0, 255).astype(np.uint8)


def draw_vision_strip(
    stack: np.ndarray, width: int, zoom: int = 2, motion_zoom: int = 3
) -> np.ndarray:
    """Two screens in one strip.

    Left: the exact tensor the network receives -- `stack` is (N, 84, 84) uint8,
    greyscale, downscaled, cropped to the playfield. It looks far poorer than
    the game because that is genuinely all the agent has to go on.

    Right, larger: MOVEMENT. The same frames colour-coded oldest-to-newest and
    max-blended, so static scenery reads white and anything that moved leaves a
    coloured trail. A single frame cannot tell you whether a leg is extending or
    retracting; this is the information the stack exists to provide.
    """
    n, h, w = stack.shape
    tile_w, tile_h = w * zoom, h * zoom
    mot_w, mot_h = w * motion_zoom, h * motion_zoom
    pad, top, label_h = 12, 36, 26
    strip_h = top + max(tile_h, mot_h) + label_h

    strip = np.full((strip_h, width, 3), PANEL_BG, dtype=np.uint8)
    cv2.putText(strip, "WHAT THE NETWORK SEES", (18, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT, 1, cv2.LINE_AA)
    cv2.putText(strip, f"{n} x {h}x{w} greyscale", (250, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)

    # Bottom-align the filmstrip with the larger movement panel.
    base = top + max(tile_h, mot_h)
    x, y = 18, base - tile_h
    for i in range(n):
        tile = cv2.resize(stack[i], (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        strip[y:y + tile_h, x:x + tile_w] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(strip, (x, y), (x + tile_w, y + tile_h),
                      TRAIL_BGR[i % len(TRAIL_BGR)], 1)
        label = "t (now)" if i == n - 1 else f"t-{n - 1 - i}"
        cv2.putText(strip, label, (x, base + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    TRAIL_BGR[i % len(TRAIL_BGR)], 1, cv2.LINE_AA)
        x += tile_w + pad

    x += pad
    if x + mot_w <= width - 8:
        comp = cv2.resize(motion_composite(stack), (mot_w, mot_h),
                          interpolation=cv2.INTER_NEAREST)
        my = base - mot_h
        strip[my:my + mot_h, x:x + mot_w] = comp
        cv2.rectangle(strip, (x, my), (x + mot_w, my + mot_h), TEXT, 1)
        cv2.putText(strip, "MOVEMENT", (x, my - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, TEXT, 1, cv2.LINE_AA)
        cv2.putText(strip, "all 4 frames, oldest to newest", (x, base + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1, cv2.LINE_AA)
    return strip


def compose(frame_rgb: np.ndarray, pressed: set[str], action_name: str,
            info: dict, scale: int, elapsed: float,
            stack: np.ndarray | None = None) -> np.ndarray:
    game = cv2.resize(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        (NES_W * scale, NES_H * scale), interpolation=cv2.INTER_NEAREST,
    )
    h = game.shape[0]
    panel = np.full((h, PANEL_W, 3), PANEL_BG, dtype=np.uint8)

    def line(y: int, text: str, colour=TEXT, size=0.52, weight=1):
        cv2.putText(panel, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, size,
                    colour, weight, cv2.LINE_AA)

    line(38, "YIE AR KUNG-FU", TEXT, 0.62, 2)
    line(60, "trained agent, greedy policy", DIM, 0.4)

    score = info.get("score", -1)
    stage = info.get("stage", -1)
    lives = info.get("lives", -1)
    line(104, "SCORE", DIM, 0.4)
    line(134, f"{score:,}" if score is not None and score >= 0 else "--", TEXT, 0.8, 2)
    line(174, "STAGE", DIM, 0.4)
    line(202, f"{stage}" if stage is not None and stage >= 0 else "--", TEXT, 0.7, 2)
    line(174 + 68, "LIVES", DIM, 0.4)
    line(202 + 68, f"{lives}" if lives is not None and lives >= 0 else "--", TEXT, 0.7, 2)

    draw_pad(panel, pressed, 18, 330)

    line(470, "ACTION", DIM, 0.4)
    line(494, action_name, LIT, 0.5)
    cv2.putText(panel, f"{elapsed:5.1f}s", (18, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1, cv2.LINE_AA)

    top_row = np.hstack([game, panel])
    if stack is None:
        return top_row
    return np.vstack([top_row, draw_vision_strip(stack, top_row.shape[1])])


# --------------------------------------------------------------------------
def record(args: Args) -> Path:
    from kungfu.emulator.integration import INTEGRATION_DIR, register
    from kungfu.envs.kungfu import YieArKungFuEnv
    from kungfu.envs.wrappers import ChannelStack
    from kungfu.train import ATLAS_PATH
    from kungfu.vision.atlas import DigitAtlas

    cfg_path = args.config or (args.checkpoint.parent / "config.yaml")
    cfg = Config.load(cfg_path if Path(cfg_path).exists() else None)
    apply_limit_overrides(cfg, args)
    console.print(f"config: [dim]{cfg_path if Path(cfg_path).exists() else 'defaults'}[/dim]")
    console.print(
        f"episode cap: [dim]{cfg.env.max_episode_steps:,} steps, "
        f"stall {cfg.env.stall_timeout:,}[/dim]"
    )

    register(INTEGRATION_DIR)
    base = YieArKungFuEnv(
        cfg.env, cfg.reward, DigitAtlas.load(ATLAS_PATH),
        render_mode="rgb_array", capture_audio=not args.mute,
    )
    env = ChannelStack(base, cfg.env.frame_stack)
    obs, _ = env.reset(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        obs_shape=env.observation_space.shape,
        n_actions=int(env.action_space.n),
        dqn_cfg=cfg.dqn, replay_cfg=cfg.replay,
        device=device, seed=args.seed, num_envs=1,
    )
    agent.load(args.checkpoint)
    agent.online.eval()
    console.print(f"loaded [green]{args.checkpoint.name}[/green] @ step {agent.steps:,}")

    rng = np.random.default_rng(args.seed)
    skip = cfg.env.frame_skip
    # One agent step == frame_skip emulator frames == frame_skip output frames.
    # Two independent limits, and conflating them is an easy mistake:
    #   --endless  lifts the ENV episode cap (when the game ends)
    #   --episodes lifts the RECORDING cap   (when we stop filming)
    # A 60s recording of an uncapped episode still stops at 3600 frames.
    if args.episodes is not None:
        total_steps = int(args.max_seconds * args.fps / skip)
        stop_after_episodes = args.episodes
    else:
        total_steps = int(args.seconds * args.fps / skip)
        stop_after_episodes = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="kfdemo_"))
    video_path = tmp / "video.mp4"
    wav_path = tmp / "audio.wav"

    import imageio

    writer = imageio.get_writer(video_path, fps=args.fps, quality=8, macro_block_size=1)
    audio_chunks: list[np.ndarray] = []
    rate = base.audio_rate if not args.mute else 0

    slot = None
    server = None
    if args.live:
        slot = FrameSlot()
        server = serve(
            slot, args.host, args.port, "Yie Ar Kung-Fu - recording demo",
            f"live while writing {args.out.name}"
            + ("" if args.realtime else " &middot; faster than realtime, pass --realtime to slow it"),
        )
        console.print(
            f"  [bold green]live[/bold green] -> "
            f"[bold]http://localhost:{args.port}[/bold]"
        )

    info: dict = {}
    episodes = 0
    written = 0
    frame_budget = skip / args.fps  # wall-clock seconds one agent step represents
    try:
        for _ in range(total_steps):
            t0 = time.time()
            if rng.random() < args.epsilon:
                action = int(rng.integers(0, int(env.action_space.n)))
            else:
                action = int(agent.act(obs[None, ...], greedy=True)[0])

            obs, _, term, trunc, info = env.step(action)
            if not args.mute:
                chunk = base.pop_audio()
                if chunk.size:
                    audio_chunks.append(chunk)

            pressed = set(ACTION_COMBOS[action])
            name = "NOOP" if not pressed else " + ".join(ACTION_COMBOS[action])
            frame = env.render()
            if frame is None:
                continue
            canvas = compose(frame, pressed, name, info, args.scale,
                             written / args.fps,
                             stack=None if args.no_vision else obs)
            # Hold the composed frame for the whole skip so video matches audio.
            for _ in range(skip):
                writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                written += 1

            if slot is not None:
                slot.publish_bgr(canvas)
            if args.realtime:
                spare = frame_budget - (time.time() - t0)
                if spare > 0:
                    time.sleep(spare)

            if term or trunc:
                episodes += 1
                console.print(
                    f"[dim]episode {episodes} ended after {info.get('score', -1):,} "
                    f"points, stage {info.get('stage', -1)}[/dim]"
                )
                if stop_after_episodes is not None and episodes >= stop_after_episodes:
                    break
                obs, _ = env.reset()
    finally:
        writer.close()
        env.close()
        if slot is not None:
            slot.shutdown()
        if server is not None:
            server.shutdown()

    console.print(
        f"recorded {written:,} frames ({written / args.fps:.1f}s), {episodes} episode(s)"
    )
    if stop_after_episodes is None and episodes == 0:
        console.print(
            "[yellow]Recording hit the --seconds limit mid-episode. "
            "Use --episodes 1 --endless to record a full game.[/yellow]"
        )

    if args.mute or not audio_chunks:
        Path(video_path).replace(args.out)
        console.print(f"[green]wrote[/green] {args.out} [dim](no audio)[/dim]")
        return args.out

    audio = np.concatenate(audio_chunks, axis=0)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.astype(np.int16).tobytes())

    from imageio_ffmpeg import get_ffmpeg_exe

    cmd = [
        get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(video_path), "-i", str(wav_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(args.out),
    ]
    subprocess.run(cmd, check=True)
    dur = audio.shape[0] / max(rate, 1)
    console.print(
        f"[green]wrote[/green] {args.out}  "
        f"[dim]({dur:.1f}s audio @ {rate} Hz, {audio.shape[1]}ch)[/dim]"
    )
    return args.out


def main() -> None:
    record(tyro.cli(Args))


if __name__ == "__main__":
    main()
