"""Save a match's actions and replay them deterministically (handoff B4).

Format: {"seed", "deck0", "deck1", "actions": [[tick, seat, [slot, x, y]], ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

import cr_sim

from evals.match import MAX_TICKS, to_step


def save(path, seed, deck0, deck1, actions) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"seed": seed, "deck0": deck0, "deck1": deck1, "actions": actions}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f)
        f.write("\n")


def save_match(path, match: dict) -> None:
    """Convenience: save the return value of evals.match.play()."""
    save(path, match["seed"], match["deck0"], match["deck1"], match["actions"])


def rerun(path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    by_tick: dict[int, list] = {}
    for tick, seat, a in data["actions"]:
        by_tick.setdefault(int(tick), [None, None])[int(seat)] = to_step(a)

    b = cr_sim.Battle()
    b.reset(data["seed"], data["deck0"], data["deck1"])
    while not b.done():
        if b.tick > MAX_TICKS:
            raise RuntimeError(f"replay passed {MAX_TICKS} ticks ({path})")
        a0, a1 = by_tick.get(b.tick, (None, None))
        b.step(a0, a1)
    return {"result": b.view(0).result(), "crowns": list(b.crowns()), "ticks": b.tick}
