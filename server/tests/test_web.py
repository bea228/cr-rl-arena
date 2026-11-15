"""The web side of the ladder: match log, replay traces and the HTTP routes."""
import asyncio
import json
import random
import urllib.error
import urllib.request

import cr_sim
import pytest
import websockets

from cr_server import match
from cr_server.ladder import MatchLog

DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]


class Bot:
    """A deterministic random agent, usable both in-process and over a socket."""

    def __init__(self, seed, name="bot"):
        self.r, self.deck, self.alive, self.name = random.Random(seed), DECK, True, name

    def choose(self, obs):
        legal = [i for i, ok in enumerate(obs["legal"]) if ok]
        if not legal or self.r.random() < 0.6:
            return None
        y = self.r.uniform(9, 14) if obs["player"] == 0 else self.r.uniform(18, 23)
        return [self.r.choice(legal), self.r.uniform(2, 16), y]

    async def act(self, obs, timeout):
        return self.choose(obs)


def _played(seed=7):
    res = asyncio.run(match.run_match(cr_sim, Bot(seed), Bot(seed + 1), seed=seed))
    return {**res, "deck0": DECK, "deck1": DECK, "sim_outcome": res["outcome"]}


def test_trace_reproduces_the_recorded_match():
    rec = _played()
    tr = match.trace(cr_sim, rec)
    m = tr["meta"]
    assert m["faithful"] and m["outcome"] == rec["outcome"]
    assert m["crowns"] == rec["crowns"] and m["ticks"] == rec["ticks"]
    assert tr["frames"][0][0] == 0 and tr["frames"][-1][0] == rec["ticks"]
    assert len(tr["frames"][0][3]) == 6                    # the six towers
    assert tr["events"], "a match with deploys must list them"
    for t, p, card, x, y in tr["events"]:
        assert p in (0, 1) and card in DECK and 0 <= x <= cr_sim.WIDTH


def test_trace_flags_a_replay_that_no_longer_matches():
    rec = _played()
    rec["crowns"] = [rec["crowns"][0] + 1, rec["crowns"][1]]   # as if recorded on another build
    assert match.trace(cr_sim, rec)["meta"]["faithful"] is False


def _entry(i, p0, p1, outcome, delta, r0, r1):
    return {"id": i, "at": 1000 + i, "p0": p0, "p1": p1, "outcome": outcome, "crowns": [3, 1],
            "ticks": 2000, "delta": delta, "r0": r0, "r1": r1, "upset": False, "forfeit": False,
            "deck0": DECK, "deck1": DECK, "file": f"{i:08d}.json"}


def test_match_log_summaries_and_reload(tmp_path):
    log = MatchLog(str(tmp_path))
    log.add(_entry(1, "a", "b", 1, 16.0, 1216.0, 1184.0))
    log.add(_entry(2, "b", "a", 1, 17.5, 1201.5, 1198.5))
    log.add(_entry(3, "a", "c", 3, -0.1, 1198.4, 1200.1))
    standings = [
        {"rank": 1, "name": "b", "rating": 1201.5, "wins": 1, "losses": 1, "draws": 0, "games": 2},
        {"rank": 2, "name": "c", "rating": 1200.1, "wins": 0, "losses": 0, "draws": 1, "games": 1},
        {"rank": 3, "name": "a", "rating": 1198.4, "wins": 1, "losses": 1, "draws": 1, "games": 3},
    ]
    snap = log.snapshot(standings, {"a"}, 1)
    a = next(r for r in snap["rows"] if r["name"] == "a")
    assert a["form"] == ["W", "L", "D"] and a["online"] is True
    assert a["history"] == [1200.0, 1216.0, 1198.5, 1198.4] and a["peak"] == 1216.0
    assert a["lastDelta"] == -0.1 and a["deck"] == DECK
    assert [m["id"] for m in a["recent"]] == [3, 2, 1] and a["recent"][1]["res"] == "L"
    assert [m["id"] for m in snap["matches"]] == [3, 2, 1]
    assert snap["stats"] == {"online": 1, "live": 1, "played": 3}
    json.dumps(snap)                                       # must be JSON-ready

    again = MatchLog(str(tmp_path))                        # survives a restart
    assert again.next_id() == 4
    assert again.snapshot(standings, set(), 0)["rows"] == [
        {**r, "online": False} for r in snap["rows"]]


def test_match_log_indexes_replays_from_older_versions(tmp_path):
    d = tmp_path / "replays"
    d.mkdir()
    (d / "1700000000000_x_vs_y.json").write_text(json.dumps(
        {"p0": "x", "p1": "y", "deck0": None, "deck1": None, "outcome": 2, "crowns": [0, 3],
         "ticks": 1500, "seed": 1, "actions": []}))
    (d / "junk.json").write_text("{not json")
    log = MatchLog(str(tmp_path))
    assert len(log.matches) == 1 and log.matches[0]["delta"] is None
    assert log.replay_path(1).endswith("1700000000000_x_vs_y.json")
    assert log.replay_path(2) is None
    assert (tmp_path / "matches.jsonl").exists()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def test_server_serves_page_data_and_replays(tmp_path, monkeypatch):
    """Two agents play real matches over the socket; the page data and a replay follow."""
    from cr_server import server, web

    monkeypatch.setattr(server, "DATA_DIR", str(tmp_path))
    orig = match.run_match

    async def short(sim, a0, a1, **kw):                    # keep matches brief
        return await orig(sim, a0, a1, max_ticks=400, **kw)

    monkeypatch.setattr(match, "run_match", short)

    async def agent(url, name, n):
        bot, done = Bot(hash(name) % 1000, name), 0
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"t": "join", "name": name, "deck": DECK}))
            async for raw in ws:
                m = json.loads(raw)
                if m["t"] == "obs":
                    await ws.send(json.dumps({"t": "act", "action": bot.choose(m["obs"])}))
                elif m["t"] == "result":
                    done += 1
                    if done >= n:
                        return

    async def main():
        arena = server.Arena()
        mm = asyncio.create_task(arena.matchmaker())
        async with websockets.serve(arena.handler, "127.0.0.1", 0,
                                    process_request=web.Web(arena).process_request) as srv:
            port = srv.sockets[0].getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            get = lambda path: asyncio.to_thread(_get, base + path)   # noqa: E731

            st, ct, body = await get("/api/ladder")
            assert st == 200 and json.loads(body)["rows"] == []      # empty ladder is fine

            await asyncio.wait_for(asyncio.gather(
                agent(f"ws://127.0.0.1:{port}", "alice", 3), agent(f"ws://127.0.0.1:{port}", "bob", 3)), 60)

            st, ct, body = await get("/api/ladder")
            snap = json.loads(body)
            assert st == 200 and ct.startswith("application/json")
            assert {r["name"] for r in snap["rows"]} == {"alice", "bob"}
            # a further match may already have started and ended as the agents left
            assert snap["stats"]["played"] >= 3 and len(snap["matches"]) >= 3
            assert all(m["replay"] and m["delta"] is not None for m in snap["matches"])
            assert len(snap["rows"][0]["history"]) >= 4 and len(snap["rows"][0]["recent"]) >= 3

            mid = snap["matches"][0]["id"]
            st, ct, body = await get(f"/api/replay/{mid}")
            tr = json.loads(body)
            assert st == 200 and tr["meta"]["faithful"] and tr["match"]["id"] == mid
            assert tr["meta"]["ticks"] == snap["matches"][0]["ticks"]

            for path, want in (("/", "<title>Arena Ladder"), ("/replay?id=1", "<title>Replay")):
                st, ct, body = await get(path)
                assert st == 200 and ct.startswith("text/html") and want in body.decode()
            assert (await get("/healthz"))[0] == 200
            for path in ("/api/replay/999", "/api/replay/abc", "/api/replay/1/../../x", "/nope", "/web/index.html"):
                assert (await get(path))[0] == 404, path
        mm.cancel()

    asyncio.run(main())
    assert (tmp_path / "matches.jsonl").exists()
    assert sorted(p.name for p in (tmp_path / "replays").iterdir())[:3] == [f"{i:08d}.json" for i in (1, 2, 3)]
