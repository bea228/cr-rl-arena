"""Read a saved replay as a timeline from one seat's point of view (handoff C12).

    ./.venv/Scripts/python scripts/show_replay.py <replay.json> --seat 0 [--every 40]
    ./.venv/Scripts/python scripts/show_replay.py --summary <folder-with-loss_*.json> [--seat S]

Read-only: reruns the replay in cr_sim and prints. "us" is --seat (owner 0 in its view, low y).
Their action tiles are shown in our frame (the other seat's frame flips y: y -> 32 - y).
"rule≈" is the bot.scripted_v0 rule that would fire on the obs we saw when we acted.
In --summary mode the seat comes from --seat, else from the gauntlet summary that saved the replay.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cr_sim  # noqa: E402

from bot import scripted_v0, state  # noqa: E402
from evals.match import MAX_TICKS, to_step  # noqa: E402

ARENA_H = 32.0
DROP_HP = 300
TOWERS = ("L", "R", "K")


def _load(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _by_tick(data: dict) -> dict[int, list]:
    by: dict[int, list] = {}
    for tick, seat, a in data["actions"]:
        by.setdefault(int(tick), [None, None])[int(seat)] = to_step(a)
    return by


def v0_rule(obs: dict) -> str:
    st = state.parse(obs)
    for name, fn in scripted_v0.RULES:
        if fn(obs, st) is not None:
            return name
    return "wait"


def _towers(st: state.State) -> str:
    me = " ".join(f"{st.my_tower_hp[k]:>4}" for k in TOWERS)
    them = " ".join(f"{st.enemy_tower_hp[k]:>4}" for k in TOWERS)
    return f"us[L R K] {me} | them {them}"


def _lane_units(units, lane) -> str:
    return ",".join(f"{u.name}:{u.hp}@{u.y:.0f}" for u in units if u.lane == lane and u.y < state.THREAT_Y) or "-"


def _status(st: state.State) -> str:
    lanes = "  ".join(
        f"{ln}: them[{_lane_units(st.enemy, ln)}] vs us[{_lane_units(st.mine, ln)}]" for ln in state.LANES
    )
    return f"{st.tick:>5}  ---- elix {st.elixir:4.1f}  {_towers(st)}\n         {lanes}"


def _all_hp(st: state.State) -> list[int]:
    return [st.my_tower_hp[k] for k in TOWERS] + [st.enemy_tower_hp[k] for k in TOWERS]


def timeline(path, seat: int, every: int = 40, out=print) -> dict:
    """Print the timeline; return {"result", "crowns", "ticks", "first_fall"} from seat's view."""
    data = _load(path)
    by = _by_tick(data)
    other = 1 - seat
    b = cr_sim.Battle()
    b.reset(data["seed"], data["deck0"], data["deck1"])
    vs, vo = b.view(seat), b.view(other)
    obs = vs.observe()
    st = state.parse(obs)
    last_hp = _all_hp(st)
    alive = {("us", k) for k in TOWERS} | {("them", k) for k in TOWERS}
    first_fall = None
    pending: list[str] = []
    while not b.done():
        if b.tick > MAX_TICKS:
            raise RuntimeError(f"replay passed {MAX_TICKS} ticks ({path})")
        tick = b.tick
        acts = by.get(tick, (None, None))
        lines = []
        if acts[seat] is not None:
            s, x, y = acts[seat]
            lines.append((seat, f"{tick:>5}  us    {obs['hand'][s]:<14} ({x:4.1f},{y:4.1f})", v0_rule(obs)))
        if acts[other] is not None:
            oo = vo.observe()
            s, x, y = acts[other]
            lines.append((other, f"{tick:>5}  them  {oo['hand'][s]:<14} ({x:4.1f},{ARENA_H - y:4.1f})", None))
        b.step(acts[0], acts[1])
        for who, text, rule in lines:
            res = b.last_action(who)
            out(f"{text} -> {res}" + (f"  rule≈{rule}" if rule else ""))
        obs = vs.observe()
        st = state.parse(obs)
        hp = _all_hp(st)
        for side, d in (("us", st.my_tower_hp), ("them", st.enemy_tower_hp)):
            for k in TOWERS:
                if (side, k) in alive and d[k] <= 0:
                    alive.discard((side, k))
                    out(f"{b.tick:>5}  **** {side} tower {k} destroyed")
                    if side == "us" and first_fall is None:
                        enemy_here = sorted({u.name for u in st.enemy if k == "K" or u.lane == k})
                        first_fall = {"tick": b.tick, "lane": k, "enemy": enemy_here}
        if b.tick % every == 0 or any(p - h > DROP_HP for p, h in zip(last_hp, hp)):
            out(_status(st))
            last_hp = hp
    crowns = list(b.crowns())
    result = vs.result()
    out(f"result {result}  crowns us {crowns[seat]} them {crowns[other]}  ticks {b.tick}")
    return {"result": result, "crowns": [crowns[seat], crowns[other]], "ticks": b.tick, "first_fall": first_fall}


def _seat_from_summaries(replay: Path) -> int | None:
    results = replay.parent.parent.parent  # results/replays/<stem>/loss_*.json
    summ = results / f"{replay.parent.name}.json"
    if not summ.exists():
        return None
    for e in _load(summ).get("worst_losses", []):
        if Path(e.get("path", "")).name == replay.name:
            return int(e["a_seat"])
    return None


def summary(folder, seat: int | None = None, out=print) -> int:
    files = sorted(Path(folder).glob("loss_*.json"), key=lambda p: int(re.search(r"loss_(\d+)", p.name).group(1)))
    for p in files:
        s = seat if seat is not None else _seat_from_summaries(p)
        if s is None:
            out(f"{p.name}: unknown seat (pass --seat)")
            continue
        g = re.search(r"_g(\d+)", p.name)
        r = timeline(p, s, every=10**9, out=lambda *_: None)
        ff = r["first_fall"]
        fall = f"first fall t{ff['tick']} lane {ff['lane']} vs {','.join(ff['enemy']) or '-'}" if ff else "no tower lost"
        out(f"g{g.group(1) if g else '?':>5} seat {s}  {r['result']:<4} {r['crowns'][0]}-{r['crowns'][1]}"
            f"  t{r['ticks']:<5} {fall}")
    return len(files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay", nargs="?", help="replay .json")
    ap.add_argument("--seat", type=int, choices=(0, 1), default=None)
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--summary", metavar="FOLDER", default=None)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # "≈" on a cp1252 Windows console
    if args.summary:
        summary(args.summary, args.seat)
        return 0
    if not args.replay:
        ap.error("give a replay file or --summary FOLDER")
    seat = args.seat if args.seat is not None else _seat_from_summaries(Path(args.replay))
    if seat is None:
        ap.error("--seat is required (no gauntlet summary found for this replay)")
    timeline(args.replay, seat, max(1, args.every))
    return 0


if __name__ == "__main__":
    sys.exit(main())
