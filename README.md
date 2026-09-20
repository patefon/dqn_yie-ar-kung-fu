# AI Kung-Fu

A deep RL agent that learns to play **Yie Ar Kung-Fu (NES)** from raw pixels.

The agent never reads emulator memory. It sees the framebuffer, and its reward is
extracted from the same pixels a human looks at — score digits, health bars, the
stage counter, the lives icons. Emulator RAM is touched in exactly one place: an
offline calibration tool that *verifies* the computer-vision pipeline is reading
the screen correctly. Nothing under `envs/` or `rl/` imports it.

<table>
<tr>
<td width="65%"><img src="resources/preview_2025.gif" alt="the 2025 agent, stage 21"></td>
<td width="35%"><img src="resources/preview.gif" alt="the 2021 agent, stage 5"></td>
</tr>
<tr>
<td><b>2025</b> — stage 21. Gameplay, the buttons the agent is pressing, the four
frames the network receives, and the movement composite.</td>
<td><b>2021</b> — stalled at ~stage 5.</td>
</tr>
</table>

*Recorded with `make demo`, which also writes an MP4 with the game audio.*

> An independent, non-commercial research project. Not affiliated with or
> endorsed by Konami or Nintendo, and no ROM is distributed — see
> [Disclaimer](#disclaimer). The 2021 implementation is preserved under
> [`legacy/`](legacy/).

---

## Timeline

**2021 — first attempt** Screen-scraped a Nestopia window, injected keystrokes,
vanilla DQN. Stalled at roughly **stage 5** and stayed there. Four silent bugs,
none of them a hyperparameter: a broken MDP, an action set with no punch, a
21 GB replay buffer, and a reward paid once per animation frame. Nothing could
measure whether the vision was even reading the screen correctly, so all four
looked identical — a flat reward curve*.

**2025 — rewrite.** Synchronous headless emulator, 17 actions, Double/Dueling
DQN with n-step and prioritised replay, and a calibration harness that proves
the CV pipeline works before a single training step. **Stage 21**, 10M steps,
~4.5 h on one GPU.

## Results

| | 2021 | 2025 |
|---|---|---|
| **Stage reached** | ~5 | **21** |
| **Score** | — | mean **235,900**, max **240,200** |
| Survival | — | ~13 min per game, dies at the stage-21 wall |

Greedy, uncapped, 3 episodes. `Yie Ar Kung-Fu (Japan) (Rev 1.4)`, sha1
`d07be428…`.

> Evaluate with `--endless`. The training default caps episodes at 6,000 steps,
> which measures the same agent at stage 13–14 — it was being cut off alive, not
> beaten. `make eval NAME=run1 ARGS=--endless`

### How it got there

| | 2021 | 2025 |
|---|---|---|
| Frame source | screen-scrape a Nestopia window (`mss`) | `stable-retro` framebuffer, synchronous |
| Action ↔ result | async queues that **dropped frames** | causally linked, deterministic, seeded |
| Throughput | ~100 fps | 2,209 emu fps/env; ~617 agent steps/s on 8 envs |
| Control | `pynput` key injection into a live window | emulator button vector |
| Action space | 7, **no punch**, held-B kick fires once | 17 combinations, press/release implicit |
| Algorithm | vanilla DQN, MSE loss | Double + Dueling + 3-step + PER, Huber |
| Replay, 200k transitions | **21.3 GB** (never fit) | **1.41 GB** |
| HUD read accuracy | unknown, unmeasurable | **100%** over 1,871 live frames |
| Config | `consts.py` globals | validated pydantic + YAML |
| Resume | never (`load_models` was never called) | automatic |
| Tests | 0 | **52** |
| Platform | Linux only (`wmctrl`, `os.killpg`) | WSL2 / Linux / macOS |

Still pixels-only. Still no RAM in the loop. That part was always the good idea.

---

## Requirements

- **Windows:** WSL2 + Ubuntu. `stable-retro` has no native Windows build, but CUDA
  passes through to WSL2 and the source lives on the Windows filesystem, so you
  edit from Windows and train in WSL.
- **Linux / macOS:** works directly.
- Python 3.11–3.13, NVIDIA GPU strongly recommended.
- A legally-obtained ROM. **None is distributed here**; `roms/*.nes` is gitignored.

Verified on Ubuntu 26.04 (WSL2), Python 3.12, torch 2.14+cu126, stable-retro
1.0.1, gymnasium 1.3.0.

For `torch.compile`, WSL also needs a C compiler (Triton builds kernels with it):
`sudo apt install build-essential`. Without one the trainer says so and runs
eager — it no longer dies mid-run.

---

## Setup

```bash
make install                      # uv venv + editable install with CUDA torch
cp "/path/to/Yie Ar Kung-Fu.nes" roms/
make rom                          # register a custom stable-retro integration + savestate
make calibrate                    # dump HUD regions, harvest digit glyphs
#   -> label out/glyphs/labels.json (ten glyphs, once)
make atlas
make check                        # must report ~100% HUD readability
```

Yie Ar Kung-Fu is **not** among stable-retro's 298 NES integrations, so `make rom`
builds one: `rom.sha`, an inert `scenario.json` (Python owns reward and
termination), and a savestate booted to the first fight.

`make check` is not ceremony. A misread HUD produces a reward signal that is
quietly noise, and that is indistinguishable from "the agent has not learned
yet" — which is exactly how the original hid four separate bugs for years.

---

## Training

```bash
make smoke                        # 40k steps, ~3 min, proves the pipeline end to end
make go NAME=run1                 # train AND watch live, one command; Ctrl+C stops both
make tb                           # TensorBoard at :6006 (separate terminal)
```

`make go` runs training in the foreground and the live mosaic on
**http://localhost:8080**. WSL2 forwards localhost, so that URL works from a
Windows browser with no X server. `NAME` is shared by `go`, `watch`, `demo` and
`eval`, so they always point at the same run. Knobs: `PORT=`, `WATCH_ENVS=`,
`CONFIG=`.

Separate processes instead:

```bash
make train NAME=run1              # terminal 1
make watch NAME=run1              # terminal 2, attaches to a run already going
```

Runs auto-resume from the newest checkpoint, so stopping and restarting loses
nothing.

### Watching it learn

- **Live mosaic** (`tools/watch.py`) — a separate playback process that reloads
  the newest checkpoint every 30s and streams a grid of games as MJPEG. Costs
  training nothing; leaving it open shows the policy changing over the run.
- **TensorBoard** — `eval/score_mean`, `eval/score_max`, `eval/stage_max` are the
  honest exploration-free numbers. `reward/*` breaks the shaped reward into its
  terms so you can see *which* part of the objective is moving. `loss/huber`,
  `q/mean`, `q/td_error`, `optim/grad_norm` cover the optimiser.
- **Console** every 10k steps: fps, mean return, epsilon, episodes, best score.

Watch `eval/score_mean`, not training return — training return is entangled with
the epsilon schedule and with reward shaping.

### Episode length

`max_episode_steps` (default 6,000) and `stall_timeout` (600) live **in the env**,
so they apply to the viewer and the demo too, not just training. That is why a
long run appears to jump back to stage 1. Short episodes are good for training —
they keep the replay buffer diverse — but for watching you usually want:

```bash
make watch NAME=run1 ARGS=--endless    # play until the agent actually loses
make eval  NAME=run1 ARGS=--endless
```

`make demo` already passes `--endless`. Both tools print the active cap on
startup, so it is never silently in play.

There are **two** independent limits and conflating them is easy: `--endless`
lifts the *env* cap (when the game ends), `--episodes N` lifts the *recording*
cap (when filming stops). A 60-second recording of an uncapped episode still
stops after 3,600 frames.

---

## Evaluating and demoing

```bash
make eval NAME=run1 ARGS=--endless   # greedy episodes, uncapped
make demo NAME=run1                  # one full game -> out/run1-demo.mp4, live at :8081
```

`make demo` records **one complete game**, start to death (~13 min at stage 21),
and **streams what it is recording** to http://localhost:8081 while it works, at
true game speed. The live preview is silent — MJPEG carries no audio — but the
MP4 has the game sound. Add `ARGS=--no-realtime` to render it in ~90 s instead
of real time, `--no-live` to skip the preview, `--episodes 3` for several games.

It produces a presentation-quality MP4 containing:

- **Real game audio**, 32 kHz stereo, muxed with ffmpeg.
- **A NES pad overlay** — D-pad and A/B light up on every press, with the action
  named, so you can see what the agent is doing rather than only the result.
- **What the network sees** — the exact tensor it receives: four 84×84 greyscale
  playfield crops, labelled `t-3 … t (now)`.
- **A MOVEMENT screen** — those same frames colour-coded oldest-to-newest and
  max-blended. Static scenery appears in all four so it reads white; anything
  that moved leaves a coloured trail. This is the 2021 "3 frames as RGB channels"
  idea generalised to a 4-deep stack, and it makes visible the one thing a single
  frame cannot tell you: whether a leg is extending or retracting.

Audio is deliberately only in `make demo`; the live viewer and the periodic
training eval clips stay silent.

---

## Layout

```
src/kungfu/
  config.py                validated config; every former magic number, with units
  emulator/integration.py  registers the ROM with stable-retro, builds a savestate
  vision/
    atlas.py               exact 8x8 tile OCR for HUD digits
    hud.py                 frame -> GameStats (score, stage, health, lives, game over)
    oracle.py              RAM ground truth -- CALIBRATION ONLY, never in training
  envs/
    kungfu.py              Gymnasium env: pixels in, CV reward out
    actions.py             17 button combinations
    reward.py              proportional, telescoping reward shaping
    wrappers.py            channel stacking, episode stats
  rl/
    networks.py            dueling CNN, optional NoisyNets
    replay.py              env-aware, frame-indexed, n-step, prioritised replay
    dqn.py                 Double/Dueling DQN agent
  train.py / evaluate.py
tools/
  calibrate_vision.py      dump / harvest / build-atlas / check
  discover_ram.py          locate RAM addresses empirically, for the oracle
  watch.py                 live mosaic of many games, MJPEG to a browser
  demo.py                  MP4 with audio, pad overlay, vision + movement screens
  mjpeg.py                 shared MJPEG-over-HTTP streaming used by both
legacy/                    the 2021 implementation, kept for comparison
```

---

## Status

- [x] Synchronous, deterministic, headless, parallel environment
- [x] Vision pipeline verified at 100% HUD readability
- [x] Modern DQN with auto-resume and honest evaluation
- [x] Live viewer, demo recorder, 52 tests
- [x] Beat the 2021 result (~5 levels → **stage 21**, 240,200 points)
- [ ] Get past the stage-21 wall
- [ ] 50+ levels
- [ ] Race PPO against the DQN

---

## References

- Mnih et al. 2015, *Human-level control through deep reinforcement learning*
- van Hasselt et al. 2016, *Deep RL with Double Q-learning*
- Wang et al. 2016, *Dueling Network Architectures*
- Schaul et al. 2016, *Prioritized Experience Replay*
- Hessel et al. 2018, *Rainbow*
- Machado et al. 2018, *Revisiting the ALE* (sticky actions)
- Maxim Lapan, *Deep Reinforcement Learning Hands-On* — the original reference
  for this project

## Disclaimer

This is an independent research and educational project, built to study computer
vision and reinforcement learning. It is **not affiliated with, authorised by,
sponsored by, or endorsed by Konami or Nintendo**, or any of their subsidiaries
or affiliates.

*Yie Ar Kung-Fu* is a trademark of Konami Digital Entertainment Co., Ltd.
*Nintendo Entertainment System*, *NES* and *Famicom* are trademarks of Nintendo
Co., Ltd. All game titles, characters, artwork, audio and other assets remain
the property of their respective owners. They are referenced here only to
identify the software this agent was evaluated against.

**No ROM or game code is distributed with this project.** `roms/*.nes`,
`integrations/*/rom.nes` and `*.state` are all gitignored, so nothing derived
from the cartridge is committed. To run anything here you must supply your own
copy of the game, obtained legally, and you are responsible for complying with
the laws of your jurisdiction.

Besides original source code, the repository contains only a SHA-1 checksum of
the ROM — used to verify your dump matches the one the HUD coordinates were
calibrated against — and two short gameplay clips
([`resources/preview.gif`](resources/preview.gif),
[`resources/preview_2025.gif`](resources/preview_2025.gif)) included to
illustrate the technical write-up. If you are a rights holder and would prefer
those clips removed, open an issue and they will be taken down.

Nothing here is offered for commercial use, and the project has no connection to
any commercial product or service.

## Licence

The code in this repository is MIT licensed (see [LICENSE](LICENSE)). The licence
covers **this project's own source code only** — it does not extend to the game,
its ROM, or any third-party assets referenced above.
