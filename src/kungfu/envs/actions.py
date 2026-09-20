"""Discrete action set for Yie Ar Kung-Fu.

The original controller could not punch.
------------------------------------------
``action_space_named`` listed seven actions -- stop/sit/kick-right/kick-left/
left/right/up -- with no A button anywhere, so the agent had literally no access
to the punch. Worse, ``kick()`` did ``kb.press(B)`` and left B *held*: on NES a
held attack button fires once and then does nothing, so the agent could not even
kick repeatedly. Half the game's move set was unreachable, which alone caps how
far any amount of training could get.

Here every button combination the game actually uses is an explicit action, and
because the emulator is stepped synchronously the buttons are re-asserted fresh
on every step -- press-and-release is implicit.
"""

from __future__ import annotations

import numpy as np

# stable-retro's NES button ordering.
NES_BUTTONS: tuple[str, ...] = (
    "B", "NULL", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A",
)

# A = punch, B = kick. Direction + attack yields the distinct strikes.
ACTION_COMBOS: tuple[tuple[str, ...], ...] = (
    (),                      # 0  no-op
    ("LEFT",),               # 1  walk left
    ("RIGHT",),              # 2  walk right
    ("UP",),                 # 3  jump
    ("DOWN",),               # 4  crouch
    ("A",),                  # 5  punch
    ("B",),                  # 6  kick
    ("UP", "A"),             # 7  jump punch
    ("UP", "B"),             # 8  jump kick
    ("DOWN", "A"),           # 9  low punch
    ("DOWN", "B"),           # 10 sweep kick
    ("LEFT", "A"),           # 11 punch while retreating left
    ("LEFT", "B"),           # 12 kick while retreating left
    ("RIGHT", "A"),          # 13 punch while advancing right
    ("RIGHT", "B"),          # 14 kick while advancing right
    ("UP", "LEFT"),          # 15 jump left
    ("UP", "RIGHT"),         # 16 jump right
)

ACTION_NAMES: tuple[str, ...] = tuple(
    "NOOP" if not combo else "+".join(combo) for combo in ACTION_COMBOS
)

NUM_ACTIONS = len(ACTION_COMBOS)


def build_action_table() -> np.ndarray:
    """(NUM_ACTIONS, 9) uint8 button matrix, indexable by discrete action."""
    index = {name: i for i, name in enumerate(NES_BUTTONS)}
    table = np.zeros((NUM_ACTIONS, len(NES_BUTTONS)), dtype=np.uint8)
    for row, combo in enumerate(ACTION_COMBOS):
        for button in combo:
            table[row, index[button]] = 1
    return table
