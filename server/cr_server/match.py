"""Server-authoritative match runner.

The server owns the one true `cr_sim.Battle`. Agents never step their own sim
during a ranked match; they only receive observations and return actions. Because
the engine is deterministic, the full match is reproducible from
`(seed, actions)` — the returned log can be replayed or audited.

An "agent" here is any object with:
    .name : str
    .deck : list[str] | None            (8 card names, or None for default)
    async def act(obs: dict, timeout: float) -> action | None
where action is (slot, x, y) or None (no-op). A timeout / error / malformed
reply is treated as a no-op, so one misbehaving agent cannot stall or crash a match.
"""
from __future__ import annotations

import asyncio
import os
from numbers import Real

# [M7] Decision cadence (ticks between action requests). Default 4 => 5 Hz at the
# sim's 20 Hz tick. Deploys are instantaneous, so acting between decision points is
# unnecessary and keeps round-trips bounded (~900/match). Configurable so operators
# can trade granularity for round-trips WITHOUT touching the core tick rate (which is
# fixed at 20 Hz: every tuned stat, the determinism baseline, and the stat-reconciled
# level-11 numbers are calibrated to it). Set CR_DECISION_EVERY=1 to act every tick.
DECISION_EVERY = max(1, int(os.environ.get("CR_DECISION_EVERY", "4")))


def _validate(action):
    if action is None:
        return None
    try:
        slot, x, y = action
        if not isinstance(slot, int) or not isinstance(x, Real) or not isinstance(y, Real):
            return None
        return (int(slot), float(x), float(y))
    except (TypeError, ValueError):
        return None


async def run_match(sim, agent0, agent1, *, seed, decision_every=DECISION_EVERY,
                    move_timeout=1.0, max_ticks=None):
    b = sim.Battle()
    b.reset(seed, agent0.deck, agent1.deck)
    if max_ticks is None:
        max_ticks = sim.REGULATION_TICKS + sim.OVERTIME_TICKS + 2

    actions = []
    out = 0
    t = 0
    while out == 0 and t < max_ticks:
        if t % decision_every == 0:
            obs0, obs1 = dict(b.observe(0)), dict(b.observe(1))
            a0, a1 = await asyncio.gather(
                agent0.act(obs0, move_timeout),
                agent1.act(obs1, move_timeout),
            )
            a0, a1 = _validate(a0), _validate(a1)
            if a0 is not None:
                actions.append([t, 0, list(a0)])
            if a1 is not None:
                actions.append([t, 1, list(a1)])
            out = b.step(a0, a1)
        else:
            out = b.step(None, None)
        t += 1

    return {
        "outcome": out,                 # 0 running(shouldn't happen), 1 p0, 2 p1, 3 draw
        "crowns": list(b.crowns()),
        "ticks": t,
        "seed": seed,
        "actions": actions,             # [[tick, player, [slot,x,y]], ...] -> replayable
    }


def replay(sim, seed, actions, deck0=None, deck1=None):
    """Re-simulate a match from its (seed, actions) log and return the outcome.
    Used by the server to audit a result, and by anyone to verify a ladder game."""
    b = sim.Battle()
    b.reset(seed, deck0, deck1)
    by_tick = {}
    for t, p, a in actions:
        by_tick.setdefault(t, [None, None])[p] = tuple(a)
    out, t, maxt = 0, 0, sim.REGULATION_TICKS + sim.OVERTIME_TICKS + 2
    while out == 0 and t < maxt:
        a0, a1 = by_tick.get(t, [None, None])
        out = b.step(a0, a1)
        t += 1
    return {"outcome": out, "crowns": list(b.crowns()), "ticks": t}


def trace(sim, rec, every=2):
    """Re-simulate a saved replay and return a frame-by-frame trace for the viewer.

    `rec` is a replay record (seed, actions, deck0, deck1, plus the recorded
    result). Every `every`-th tick becomes a frame of unit positions. The trace
    also lists the deploys that actually happened, and says whether this re-run
    reached the same result as the recorded match: a replay saved by an older
    simulator build may not, and the viewer should say so rather than show a
    different game as if it were the real one.
    """
    b = sim.Battle()
    b.reset(rec["seed"], rec.get("deck0"), rec.get("deck1"))
    cards = b.deployable_cards() + ["princess_tower", "king_tower"]
    idx = {n: i for i, n in enumerate(cards)}
    by_tick = {}
    for t, p, a in rec["actions"]:
        by_tick.setdefault(t, [None, None])[p] = tuple(a)

    def frame(t):
        ents = []
        for (owner, card, x, y, hp, mhp, flying, is_tower, deploying, eid, hw, hh) in b.entities():
            flags = (1 if flying else 0) | (2 if is_tower else 0) | (4 if deploying else 0)
            ents.append([eid, owner, idx.get(card, 0), round(x, 2), round(y, 2),
                         round(hp / mhp, 3) if mhp else 0.0, flags, round(hw, 2), round(hh, 2)])
        return [t, list(b.crowns()), [round(b.elixir(0), 1), round(b.elixir(1), 1)], ents]

    frames, events = [frame(0)], []
    # stop where the recorded match stopped (a match can be cut short by the server)
    out, t = 0, 0
    maxt = rec.get("ticks") or (sim.REGULATION_TICKS + sim.OVERTIME_TICKS + 2)
    while out == 0 and t < maxt:
        acts = by_tick.get(t)
        if acts:
            before = (b.elixir(0), b.elixir(1))
            names = [None, None]
            for p in (0, 1):
                a = acts[p]
                if a is None:
                    continue
                if a[0] == sim.ABILITY_SLOT:
                    names[p] = "ability"
                elif 0 <= a[0] < 4:
                    names[p] = b.hand(p)[a[0]]
            out = b.step(acts[0], acts[1])
            for p in (0, 1):
                # elixir only ever falls on a tick when something was actually played
                if names[p] and b.elixir(p) < before[p]:
                    events.append([t, p, names[p], round(acts[p][1], 1), round(acts[p][2], 1)])
        else:
            out = b.step(None, None)
        t += 1
        if t % every == 0 or out != 0:
            frames.append(frame(t))

    crowns = list(b.crowns())
    want = rec.get("sim_outcome", rec.get("outcome"))
    faithful = (out == want and crowns == list(rec.get("crowns", crowns))
                and t == rec.get("ticks", t))
    return {
        "meta": {
            "w": sim.WIDTH, "h": sim.HEIGHT, "river_y": sim.RIVER_Y, "river_half": sim.RIVER_HALF,
            "bridges": list(sim.BRIDGES), "bridge_half": sim.BRIDGE_HALF,
            "tps": sim.TICKS_PER_SECOND, "every": every,
            "regulation": sim.REGULATION_TICKS, "cards": cards,
            "outcome": out, "crowns": crowns, "ticks": t, "faithful": faithful,
        },
        "events": events,      # [tick, player, card, x, y] for each deploy that happened
        "frames": frames,      # [tick, crowns, elixir, [[id, owner, card, x, y, hp, flags, hw, hh], ...]]
    }

