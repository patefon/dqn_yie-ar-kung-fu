# Postmortem: why the 2021 agent stalled at 5 levels

A review of the original `classes/` implementation (preserved under `legacy/`),
and what the 2025 rewrite does instead.

The short version: **the hyperparameters were never the problem.** The agent was
fitting a Bellman equation to data that did not satisfy the Markov property,
using a reward signal that no one could verify, with an action set that could not
punch, and a replay buffer that could not physically fit in RAM. Any one of those
caps performance on its own.

Findings are ordered by how much they cost you.

---

## 1. The transitions were not an MDP — the fatal one

`app.py` ran the grabber, detector, agent and GUI as four workers in a 16-process
pool, connected by `Manager().Queue(maxsize=2..3)`. Every producer wrote with
`put_nowait()` inside `try/except queue.Full: pass`.

```python
# legacy/classes/reader.py
try:
    out_queue.put_nowait(obs)
except queue.Full:
    ...
    continue          # frame silently discarded
```

So when the consumer lagged, frames were **dropped**. The agent would choose
action `a` from state `s`, and the next observation it received might predate the
keypress, or postdate it by an unknown number of frames, or belong to a different
moment entirely. Q-learning's entire premise is that `(s, a, r, s')` is causally
linked. Here the link was probabilistic and load-dependent.

This is not a subtle inefficiency. It means the target
`r + γ·max Q(s',·)` was regressing onto noise for an unknown fraction of samples,
and that fraction changed with CPU load. No amount of tuning fixes it.

**Now:** `env.step(a)` presses the buttons, advances the emulator exactly
`frame_skip` frames, and returns the frame that action produced. Nothing is
dropped, ever. Measured at **2,209 emulator fps on one env** versus the old
~100 fps ceiling, and it is reproducible from a seed.

---

## 2. The agent could not punch

```python
# legacy/classes/agent.py
self.action_space_named = ['stop','sit','kick-right','kick-left','left','right','up']
```

Seven actions, and no `A` button anywhere. In Yie Ar Kung-Fu **A is punch**. The
`__act` dispatch dict did define `sit-kick-right` / `sit-kick-left`, but they were
unreachable — never listed in `action_space_named`.

Worse, the kick was broken too:

```python
def kick(self, direction):
    self.kb.tap(key=k)
    self.kb.press(key=kick_k)      # pressed, never released
    self.current_move = kick_k
```

On NES a held attack button fires **once**. Holding B does not kick repeatedly.
So the agent had one working attack, and only when it happened to alternate away
and back.

**Now:** 17 explicit button combinations covering punch, kick, jump/crouch
variants and directional attacks. Because the emulator is stepped synchronously,
buttons are re-asserted from scratch each step, so press-and-release is implicit.

---

## 3. The replay buffer needed 21 GB

```python
Experience = namedtuple('Experience', ['state','action','reward','done','new_state'])
self.mem_size = 200_000
```

Each `state` was a `(3, 74, 240)` uint8 stack, and every transition stored **two**
of them:

```
  3 × 74 × 240          =  53,280 B per state
  × 2 (state, new_state)= 106,560 B per transition
  × 200,000             =   21.3 GB
```

It could never fill. The process would thrash or be OOM-killed first, so the
effective buffer was whatever fitted — with no warning that this had happened.

Sampling was independently broken:

```python
indices = np.random.choice(sum(1 for x in self.buffer if x is not None), ...)
```

That `sum(...)` walks all 200,000 slots **on every sample call**, i.e. on every
gradient step. And when the buffer was empty, `sample()` returned `None`, which
`learn()` unpacked into five variables — a `TypeError` swallowed by the bare
`except` in the training loop. Silent.

**Now:** frames are stored once and stacks rebuilt by index — the standard Atari
trick. Same 200k transitions at 84×84 with a 4-frame stack cost **1.41 GB**,
verified by a test. Sampling is O(log n) through a sum tree.

---

## 4. The reward paid out once per animation frame

```python
if state['highscore'] == previous_state['highscore'] and \
   state['player_health'] < previous_state['player_health']:
   reward -= 5
```

This compared against the immediately previous frame at ~100 fps. A health bar
takes many frames to animate downward, so **one hit produced −5 repeatedly** —
tens of times. Landing a hit was a flat `+1`, also repeated. The relative value of
attacking versus not being hit therefore depended on animation length and on the
current frame rate, not on the game.

Two further problems in the same function:

- It read `highscore`. On screen, `SCORE` is at x=16 and `HI` is at x=88, and they
  render identically. If the ROI was over `HI`, the value only changes when a
  record is beaten — a reward term that is almost always exactly zero.
- `read_text_stat` ended with `int(result or 0)`, so an unreadable score became
  **0**, which the very next frame read as a gigantic negative score delta.

**Now:** every term is proportional to the measured change, so it telescopes —
draining a full enemy bar is worth exactly `damage_dealt` regardless of how many
frames it spans or how often we sample. Two tests pin this:
`test_damage_is_not_double_counted_over_an_animation` and
`test_sampling_rate_does_not_change_the_payout`. An unreadable field returns
`None` and the last good value is held; implausible score jumps are counted as
misreads and logged.

---

## 5. Nothing could tell you the vision was wrong

This is the meta-failure that hid the other four. There was no way to distinguish
"the CV is misreading the screen" from "the agent has not learned yet". Both look
like a flat reward curve.

Game over, for instance, was:

```python
def __is_gameover(self, roi):
    return int(np.count_nonzero(roi) == 934)
```

An exact equality on a pixel count. One pixel of difference and episodes never
terminate.

**Now:** `tools/calibrate_vision.py --command check` reports hard numbers. On the
current ROM: **100% HUD readability across 1,871 live frames.** An optional RAM
oracle (`tools/discover_ram.py`) can cross-check the CV against emulator memory —
it is used only for calibration and is never imported by the training path, so
the agent stays strictly pixels-only.

Two bugs were caught by this machinery *during the rewrite*, both of which would
have been invisible otherwise:

- **Transition detection was suppressing 17% of live frames.** I used
  "playfield has fewer than 8 distinct colours" to detect inter-stage cards. But
  the NES palette is tiny: measured, live gameplay shows **7–10** distinct
  colours and stage cards show **1–2**. The threshold sat inside the live
  distribution, so real fighting frames were being classified as transitions and
  emitting zero reward. Now 4, with the measurement recorded in the config.
- **After the last life the ROM enters an attract "DEMO" loop** where the
  controller does nothing. The original had no way to detect this, so it would
  keep collecting thousands of transitions in which actions have no effect —
  actively teaching the agent that actions do not matter. Detected now from the
  lives icons (exactly 37 lit pixels each: 74 / 37 / 0 px = 2 / 1 / 0 spare
  lives) combined with both bars empty, debounced over 8 steps.

---

## 6. Learning algorithm and bookkeeping

| Issue | Original | Now |
|---|---|---|
| Target net init | `q_eval` and `q_target` independently random — the first 5,000 steps bootstrapped off noise | synced at construction |
| Overestimation | vanilla `max_a Q_target` | Double DQN |
| Loss | `nn.MSELoss` — one bad TD error spikes the gradient | Huber + grad-norm clipping |
| Value/advantage | single head | dueling head |
| Returns | 1-step | 3-step |
| Replay | uniform | prioritised (α=0.6, β annealed) |
| ε schedule | decayed inside `learn()`, so it silently depended on `learn_every` **and** on burn-in | pure function of env steps |
| Flatten size | `nn.Linear(8320, 512)` hard-coded the input resolution | inferred from a probe tensor |
| Normalisation | `T.tensor(input).to(device)` inside `forward()` | tensors arrive ready; module only scales |
| Exploration | ε-greedy only | ε-greedy or NoisyNets |

### The checkpointing bug worth singling out

```python
def load_models(self):     # defined...
    self.q_eval.load_checkpoint()
    self.q_target.load_checkpoint()
```

`load_models()` was **never called anywhere**. Every run started from scratch.
The stated goal — a self-improvement cycle — was not reachable by construction.

And saving was gated on a statistic that could only rise:

```python
if episode_score > 0 and episode_score > avg_score:
    scores.append(episode_score)       # only improving episodes are recorded
    avg_score = np.mean(scores)
    self.save_models()
```

`scores` only ever received episodes that *beat* the current average, so
`avg_score` was the mean of a monotonically-increasing subsequence. It drifted
upward until the agent could no longer exceed it, at which point **checkpointing
silently stopped**.

**Now:** checkpoints round-trip optimizer state, AMP scaler and step count; runs
auto-resume from the newest checkpoint; and a separate greedy evaluation measures
honest performance on a schedule.

---

## 7. Smaller things

- `traceback.format_exception(etype=...)` — the `etype` keyword was removed in
  Python 3.10. On any modern Python the **error handler itself** raises, in both
  `agent.py:212` and `environment.py:140`.
- `raise f'Path [{path}] does not exist'` (`utils.py:15`) — raising a `str` is a
  `TypeError`.
- FPS limiter mixed units: `t_per_frame` in milliseconds minus elapsed **seconds**,
  then `× 0.001`.
- Busy-wait in `app.py` — `while True:` polling `proc.ready()` with no sleep,
  burning a core.
- Linux-only: `os.setpgrp`, `os.killpg`, `wmctrl`, `/usr/games/nestopia`.
- `if self.current_move == k: pass` in `control.py` — a no-op guard that reads
  like it was meant to `return`.
- Dead `self.half` flag; `sit-kick-*` handlers wired but unreachable.

---

## What actually carried over

The parts worth keeping, and they were the interesting parts:

- **Reading game state from pixels rather than RAM.** Still the design, now with
  the calibration machinery to prove it works.
- **Glyph-matching OCR for the HUD.** The idea was right; at native 240×224 with
  no rescaling it becomes *exact* — NES text sits on an 8×8 tile grid, so a tile
  hashes to a digit in O(1) and an unknown tile is reported rather than guessed.
- **Max-pooling consecutive frames** to beat NES sprite flicker. Correct
  instinct, now applied to the right pair of frames.
- **Frame stacking** for motion. Correct, now with guaranteed spacing.

---

## Measured, on this ROM

`Yie Ar Kung-Fu (Japan) (Rev 1.4)`, sha1 `d07be428cf7d198453f4942f5288a05fd55720dc`

| | Original | Now |
|---|---|---|
| Throughput | ~100 fps, screen capture | 2,209 emu fps single env, 8 in parallel |
| Replay for 200k transitions | 21.3 GB (never fit) | 1.41 GB |
| HUD read accuracy | unknown, unmeasurable | 100% over 1,871 live frames |
| Reachable actions | 7, no punch | 17 |
| Resumable | no (`load_models` never called) | yes, auto |
| Reproducible | no | seeded |
| Tests | 0 | 46 |

A 40k-step smoke run (about 3 minutes) reaches **7,600 score and stage 2** — a
sanity check that the loop learns, not a result. Real runs are 10M+ steps.

---

*Independent research project, not affiliated with or endorsed by Konami or
Nintendo. Yie Ar Kung-Fu is a trademark of Konami; NES and Famicom are
trademarks of Nintendo. No ROM is distributed — see the
[README disclaimer](README.md#disclaimer).*
