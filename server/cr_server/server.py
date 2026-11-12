"""Competition server: a continuous Elo ladder over the cr_sim engine.

Agents connect over WebSocket, join the queue, and are paired for
server-authoritative matches. The server owns the only real `cr_sim.Battle`;
agents just answer observations with actions. Results update a persistent Elo
leaderboard, and each match's (seed, action-log) is saved so it can be replayed.

Run:
    pip install cr-sim websockets        # cr-sim = the gym package
    python -m cr_server.server
Env:
    CR_ARENA_BIND        host:port         (default 0.0.0.0:8765)
    CR_ARENA_DATA        data dir          (default ./arena_data)
    CR_MOVE_TIMEOUT      seconds/decision  (default 1.0)
    CR_DECISION_EVERY    ticks/decision    (default 4 => 5 Hz; core tick stays 20 Hz)

Protocol (JSON text frames)
  C->S  {"t":"join","name":str,"deck":[8 card names]|null}
        {"t":"leaderboard"}                      (query then close)
  S->C  {"t":"joined","name","rating"}
        {"t":"start","player":0|1,"opponent","seed"}
        {"t":"obs","obs":{...}}                   (reply required)
  C->S  {"t":"act","action":[slot,x,y]|null}
  S->C  {"t":"result","you":0|1,"outcome","crowns","rating","delta"}
        {"t":"leaderboard","standings":[...]}
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time

import cr_sim
import websockets

from . import match
from .leaderboard import Leaderboard

BIND = os.environ.get("CR_ARENA_BIND", "0.0.0.0:8765")
DATA_DIR = os.environ.get("CR_ARENA_DATA", "./arena_data")
MOVE_TIMEOUT = float(os.environ.get("CR_MOVE_TIMEOUT", "1.0"))


class WsAgent:
    """Wraps a connected websocket as a match-runner agent."""

    def __init__(self, ws, name, deck):
        self.ws = ws
        self.name = name
        self.deck = deck
        self.alive = True

    async def send(self, obj):
        try:
            await self.ws.send(json.dumps(obj))
        except websockets.ConnectionClosed:
            self.alive = False

    async def act(self, obs, timeout):
        try:
            await self.ws.send(json.dumps({"t": "obs", "obs": obs}))
            raw = await asyncio.wait_for(self.ws.recv(), timeout)
            msg = json.loads(raw)
            if msg.get("t") == "act":
                return msg.get("action")
        except websockets.ConnectionClosed:
            self.alive = False
        except (asyncio.TimeoutError, json.JSONDecodeError, ValueError):
            pass  # slow/garbled reply -> treat this decision as a no-op
        return None


class Arena:
    def __init__(self):
        self.board = Leaderboard(os.path.join(DATA_DIR, "leaderboard.json"))
        self.idle: asyncio.Queue[WsAgent] = asyncio.Queue()
        self.replays_dir = os.path.join(DATA_DIR, "replays")
        os.makedirs(self.replays_dir, exist_ok=True)

    async def handler(self, ws):
        try:
            raw = await asyncio.wait_for(ws.recv(), 30)
            msg = json.loads(raw)
        except Exception:
            return
        if msg.get("t") == "leaderboard":
            await ws.send(json.dumps({"t": "leaderboard", "standings": self.board.standings()}))
            return
        if msg.get("t") != "join":
            await ws.send(json.dumps({"t": "error", "msg": "expected join or leaderboard"}))
            return

        name = str(msg.get("name") or f"anon-{random.randint(1000,9999)}")
        deck = msg.get("deck")
        if deck is not None and (not isinstance(deck, list) or len(deck) != 8):
            await ws.send(json.dumps({"t": "error", "msg": "deck must be 8 card names or null"}))
            return
        agent = WsAgent(ws, name, deck)
        await agent.send({"t": "joined", "name": name,
                          "rating": round(self.board.get(name).rating, 1)})
        self.idle.put_nowait(agent)
        try:
            await ws.wait_closed()
        finally:
            agent.alive = False

    async def matchmaker(self):
        while True:
            a = await self._next_live()
            b = await self._next_live(exclude=a)
            asyncio.create_task(self._play(a, b))

    async def _next_live(self, exclude=None):
        while True:
            ag = await self.idle.get()
            if ag.alive and ag is not exclude:
                return ag

    async def _play(self, a: WsAgent, b: WsAgent):
        try:
            seed = random.getrandbits(63)
            await a.send({"t": "start", "player": 0, "opponent": b.name, "seed": seed})
            await b.send({"t": "start", "player": 1, "opponent": a.name, "seed": seed})

            res = await match.run_match(
                cr_sim, a, b, seed=seed,
                move_timeout=MOVE_TIMEOUT, decision_every=match.DECISION_EVERY,
            )
            out = res["outcome"]
            # forfeit: a disconnected agent loses to a connected one
            if not a.alive and b.alive:
                out = 2
            elif not b.alive and a.alive:
                out = 1

            score_a = 1.0 if out == 1 else 0.0 if out == 2 else 0.5
            delta = self.board.report(a.name, b.name, score_a)
            self._save_replay(a, b, res, out)

            await a.send({"t": "result", "you": 0, "outcome": out, "crowns": res["crowns"],
                          "rating": round(self.board.get(a.name).rating, 1), "delta": round(delta, 1)})
            await b.send({"t": "result", "you": 1, "outcome": out, "crowns": res["crowns"],
                          "rating": round(self.board.get(b.name).rating, 1), "delta": round(-delta, 1)})
            print(f"[match] {a.name} vs {b.name} -> {['--','P0','P1','draw'][out]} "
                  f"{res['crowns']} in {res['ticks']} ticks")
        except Exception as e:  # never let one match kill the ladder
            print(f"[match error] {a.name} vs {b.name}: {e!r}")
        finally:
            for x in (a, b):
                if x.alive:
                    self.idle.put_nowait(x)

    def _save_replay(self, a, b, res, out):
        rid = f"{int(time.time()*1000)}_{a.name}_vs_{b.name}".replace("/", "_")[:120]
        path = os.path.join(self.replays_dir, rid + ".json")
        try:
            json.dump({"p0": a.name, "p1": b.name, "deck0": a.deck, "deck1": b.deck,
                       "outcome": out, **res}, open(path, "w"))
        except OSError:
            pass


async def main():
    host, port = BIND.rsplit(":", 1)
    arena = Arena()
    asyncio.create_task(arena.matchmaker())
    print(f"cr-arena listening on ws://{host}:{port}  (data: {DATA_DIR}, move timeout {MOVE_TIMEOUT}s)")
    async with websockets.serve(arena.handler, host, int(port), ping_interval=20):
        await asyncio.Future()  # run forever


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    run()
