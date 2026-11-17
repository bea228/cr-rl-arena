"""Play one local match between two policies (handoff B3)."""
from __future__ import annotations

import cr_sim

from bot.policy import DECIDE_EVERY

MAX_TICKS = 7000  # regulation 3,600 + overtime 2,400 is the real cap; this is a safety net


def to_step(action):
    """Policy action ([slot, x, y] or None) -> what Battle.step accepts (tuple or None)."""
    if action is None:
        return None
    return (int(action[0]), float(action[1]), float(action[2]))


def play(pol0, pol1, seed, deck0, deck1) -> dict:
    b = cr_sim.Battle()
    b.reset(seed, deck0, deck1)
    views = (b.view(0), b.view(1))
    pols = (pol0, pol1)
    pol0.reset(deck0)
    pol1.reset(deck1)

    actions: list = []
    ignored = [0, 0]
    by_reason: list[dict[str, int]] = [{}, {}]
    decisions = [0, 0]

    while not b.done():
        tick = b.tick
        if tick > MAX_TICKS:
            raise RuntimeError(f"game passed {MAX_TICKS} ticks (seed={seed})")
        step = [None, None]
        if tick % DECIDE_EVERY == 0:
            for seat in (0, 1):
                a = to_step(pols[seat].act(views[seat].observe()))
                if a is not None:
                    step[seat] = a
                    decisions[seat] += 1
                    actions.append([tick, seat, list(a)])
        b.step(step[0], step[1])
        for seat in (0, 1):
            if step[seat] is not None:
                r = b.last_action(seat)
                if r not in ("ok", "none"):
                    ignored[seat] += 1
                    by_reason[seat][r] = by_reason[seat].get(r, 0) + 1

    return {
        "result": views[0].result(),
        "crowns": list(b.crowns()),
        "ticks": b.tick,
        "actions": actions,
        "ignored": ignored,
        "ignored_by_reason": by_reason,
        "decisions": decisions,
        "seed": seed,
        "deck0": deck0,
        "deck1": deck1,
    }
