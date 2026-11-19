"""Tournament client with reconnect (handoff step C5).

    ./.venv/Scripts/python -m bot.client --url ws://localhost:8765 --name dev-scripted \
        --policy scripted --deck bot/decks/default.json

Built from server/examples/random_agent.py. Every obs gets exactly one reply,
tagged with the obs's own tick; the policy is consulted only on decision ticks
(tick % DECIDE_EVERY == 0) and any error there becomes a null move. The
connection is wrapped in a loop that reconnects 5 s after it drops.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import cr_sim
import websockets

from bot.policy import DECIDE_EVERY, Policy

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DECK = str(REPO / "bot" / "decks" / "default.json")
DEFAULT_LOG = str(REPO / "runs" / "client.log")
RECONNECT_DELAY = 5.0
SLOW_MS = 200.0
NOT_IGNORED = ("ok", "none")  # every other last_action value means our move didn't happen

log = logging.getLogger("bot.client")


# --- setup helpers ---------------------------------------------------------

def setup_logging(path: str | None) -> None:
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(fmt)
    log.addHandler(out)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)


def load_deck(arg: str, seed: int = 0) -> list[str]:
    """Deck from a JSON file, or one random legal deck for the session. Exits if illegal."""
    if arg == "random":
        from evals.decks import random_deck

        deck = random_deck(random.Random(seed))
    else:
        try:
            with open(arg, encoding="utf-8") as f:
                deck = json.load(f)
        except (OSError, ValueError) as e:
            raise SystemExit(f"cannot read deck {arg!r}: {e}")
    if not isinstance(deck, list) or not all(isinstance(c, str) for c in deck):
        raise SystemExit(f"deck {arg!r} must be a JSON list of 8 card names")
    try:
        cr_sim.Battle().check_deck(list(deck))
    except ValueError as e:
        raise SystemExit(f"illegal deck {deck}: {e}")
    return list(deck)


def build_policy(kind: str, seed: int = 0, model: str | None = None) -> Policy:
    if kind == "scripted":
        from bot.scripted import ScriptedPolicy

        return ScriptedPolicy()
    if kind == "random":
        from bot.policy import RandomPolicy

        return RandomPolicy(seed)
    if kind == "learned":
        if not model:
            raise SystemExit("--policy learned needs --model <path>")
        try:
            import importlib

            learned = importlib.import_module("bot.learned")
        except ModuleNotFoundError as e:
            raise SystemExit(f"--policy learned is not available yet (bot/learned.py missing: {e})")
        return learned.LearnedPolicy(model)
    raise SystemExit(f"unknown policy {kind!r}")


def sanitise(action) -> list | None:
    """None, or [int slot, float x, float y]; anything else raises ValueError."""
    if action is None:
        return None
    slot, x, y = action  # raises on wrong length / non-iterable
    if isinstance(slot, bool) or isinstance(slot, str):
        raise ValueError(f"bad slot {slot!r}")
    out = [int(slot), float(x), float(y)]
    if not (math.isfinite(out[1]) and math.isfinite(out[2])):
        raise ValueError(f"non-finite position {out}")
    return out


# --- message handling ------------------------------------------------------

class Client:
    """Protocol state for one process. ``handle(msg)`` returns the reply dict or None."""

    def __init__(self, policy: Policy, deck: list[str], name: str = "bot"):
        self.policy = policy
        self.deck = list(deck)
        self.name = name
        self.matches_done = 0
        self.session = Counter()  # win / loss / draw / aborted
        self.rating = None
        self.last_summary: dict | None = None
        self._seen_errors: set[str] = set()
        self.slowest_ms = 0.0
        self.session_last_action = Counter()
        self.policy.reset(self.deck)  # in case a match starts before "start" is seen
        self._new_match(None)

    def _new_match(self, opponent):
        self.opponent = opponent
        self.last_action = Counter()
        self.ticks = 0
        self.decisions = 0
        self.moves = 0
        self.slow = 0
        self.errors = 0

    def join_message(self) -> dict:
        return {"t": "join", "name": self.name, "deck": self.deck}

    # obs -> act
    def _decide(self, obs: dict):
        tick = obs["tick"]
        if tick % DECIDE_EVERY != 0:
            return None
        self.decisions += 1
        t0 = time.perf_counter()
        try:
            raw = self.policy.act(obs)
        except Exception as e:
            self.errors += 1
            key = f"{type(e).__name__}: {e}"
            if key not in self._seen_errors:
                self._seen_errors.add(key)
                log.error("policy error at tick %s (logged once):\n%s", tick, traceback.format_exc())
            return None
        finally:
            ms = (time.perf_counter() - t0) * 1000
            self.slowest_ms = max(self.slowest_ms, ms)
            if ms > SLOW_MS:
                self.slow += 1
                log.warning("slow decision at tick %s: %.0f ms", tick, ms)
        try:
            action = sanitise(raw)
        except Exception as e:
            self.errors += 1
            log.error("bad action %r at tick %s (%s); sending null", raw, tick, e)
            return None
        if action is not None:
            self.moves += 1
        return action

    def on_obs(self, obs: dict) -> dict:
        tick = int(obs["tick"])
        self.ticks += 1
        la = obs.get("last_action")
        if la is not None:
            self.last_action[la] += 1
            self.session_last_action[la] += 1
        try:
            action = self._decide(obs)
        except Exception:  # never skip the reply
            log.exception("unexpected error deciding tick %s", tick)
            action = None
        return {"t": "act", "tick": tick, "action": action}

    def summary(self, msg: dict) -> dict:
        ignored = sum(n for k, n in self.last_action.items() if k not in NOT_IGNORED)
        return {
            "result": msg.get("result"),
            "crowns": msg.get("crowns"),
            "rating": msg.get("rating"),
            "delta": msg.get("delta"),
            "match": msg.get("match"),
            "replay": f"/replay?id={msg.get('match')}",
            "opponent": self.opponent,
            "ticks": self.ticks,
            "decisions": self.decisions,
            "moves": self.moves,
            "ignored": ignored,
            "slow": self.slow,
            "errors": self.errors,
            "last_action": dict(self.last_action),
        }

    async def handle(self, msg: dict) -> dict | None:
        t = msg.get("t")
        if t == "obs":
            return self.on_obs(msg["obs"])
        if t == "joined":
            self.rating = msg.get("rating")
            log.info("joined as %s (rating %s)", msg.get("name"), msg.get("rating"))
        elif t == "start":
            self._new_match(msg.get("opponent"))
            self.policy.reset(self.deck)
            log.info("match vs %s", self.opponent)
        elif t == "result":
            s = self.summary(msg)
            self.last_summary = s
            self.matches_done += 1
            self.session[s["result"]] += 1
            self.rating = s["rating"]
            d = s["delta"]
            log.info(
                "%s %s rating %s (%s) match %s replay %s vs %s | ticks %d decisions %d moves %d "
                "ignored %d slow %d errors %d | last_action %s",
                str(s["result"]).upper(), s["crowns"], s["rating"],
                f"{d:+}" if isinstance(d, (int, float)) else d, s["match"], s["replay"],
                s["opponent"], s["ticks"], s["decisions"], s["moves"], s["ignored"],
                s["slow"], s["errors"], s["last_action"],
            )
            if self.matches_done % 10 == 0:
                self.log_session()
            self._new_match(None)
        elif t == "aborted":
            log.warning("match void: %s", msg.get("reason"))
            self.session["aborted"] += 1
            self._new_match(None)
        elif t == "error":
            log.error("server error: %s", msg.get("msg"))
        else:
            log.warning("unknown message type %r", t)
        return None

    def log_session(self):
        log.info(
            "session: %d matches  W %d / L %d / D %d (aborted %d)  rating %s  slowest decision %.1f ms",
            self.matches_done, self.session["win"], self.session["loss"], self.session["draw"],
            self.session["aborted"], self.rating, self.slowest_ms,
        )


# --- network loop ----------------------------------------------------------

async def run(url: str, client: Client, max_matches: int = 0, reconnect_delay: float = RECONNECT_DELAY):
    """Play until max_matches results (0 = forever), reconnecting after any drop."""
    while True:
        try:
            async with websockets.connect(url, max_size=None) as ws:
                log.info("connected to %s", url)
                await ws.send(json.dumps(client.join_message()))
                async for raw in ws:
                    reply = None
                    try:
                        msg = json.loads(raw)
                        reply = await client.handle(msg)
                    except Exception:
                        log.exception("could not handle message %.200r", raw)
                        reply = _fallback_reply(raw)
                    if reply is not None:
                        await ws.send(json.dumps(reply))
                    if max_matches and client.matches_done >= max_matches:
                        log.info("reached --max-matches %d, closing", max_matches)
                        client.log_session()
                        return
                log.warning("connection closed by server (code %s %s)", ws.close_code, ws.close_reason)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # ConnectionClosed, OSError, handshake errors, ...
            log.warning("connection problem: %s: %s", type(e).__name__, e)
        log.info("reconnecting in %.0f s", reconnect_delay)
        await asyncio.sleep(reconnect_delay)


def _fallback_reply(raw) -> dict | None:
    """If a broken message was still an obs with an int tick, answer it with a null move."""
    try:
        msg = json.loads(raw)
        tick = msg["obs"]["tick"]
        if msg.get("t") == "obs" and isinstance(tick, int):
            return {"t": "act", "tick": tick, "action": None}
    except Exception:
        pass
    return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bot.client", description="cr-arena tournament client")
    ap.add_argument("--url", default="ws://localhost:8765")
    ap.add_argument("--name", required=True)
    ap.add_argument("--policy", choices=("scripted", "random", "learned"), default="scripted")
    ap.add_argument("--model", default=None, help="model path (learned only)")
    ap.add_argument("--deck", default=DEFAULT_DECK, help='deck JSON file, or "random"')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-matches", type=int, default=0, help="stop after N results (0 = forever)")
    ap.add_argument("--log", default=DEFAULT_LOG)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    setup_logging(a.log)
    deck = load_deck(a.deck, a.seed)
    log.info("deck: %s", deck)
    policy = build_policy(a.policy, a.seed, a.model)
    client = Client(policy, deck, a.name)
    log.info("policy %s, name %s, url %s", a.policy, a.name, a.url)
    try:
        asyncio.run(run(a.url, client, a.max_matches))
    except KeyboardInterrupt:
        log.info("interrupted, exiting")
        client.log_session()


if __name__ == "__main__":
    main()
