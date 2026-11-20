import asyncio
import importlib.util
import json
import os
import sys

import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cr_sim  # noqa: E402

from bot import client as C  # noqa: E402
from bot.policy import DECIDE_EVERY, Policy, RandomPolicy  # noqa: E402
from bot.scripted import ScriptedPolicy  # noqa: E402

DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]


def real_observations(n=60):
    """n consecutive seat-0 observations from a local Battle (opponent plays randomly)."""
    b = cr_sim.Battle()
    b.reset(3, DECK, DECK)
    opp = RandomPolicy(1)
    out = []
    for _ in range(n):
        out.append(b.view(0).observe())
        a1 = opp.act(b.view(1).observe()) if b.tick % DECIDE_EVERY == 0 else None
        b.step(None, tuple(a1) if a1 else None)
    return out


def send(client, msg):
    return asyncio.run(client.handle(msg))


class Fixed(Policy):
    def __init__(self, action):
        self.action = action

    def act(self, obs):
        return self.action


class Boom(Policy):
    def act(self, obs):
        raise RuntimeError("boom")


def test_every_obs_gets_one_tagged_reply():
    cl = C.Client(ScriptedPolicy(), DECK, "t")
    send(cl, {"t": "start", "opponent": "x"})
    obs_list = real_observations(60)
    for obs in obs_list:
        reply = send(cl, {"t": "obs", "obs": obs})
        assert reply is not None and reply["t"] == "act"
        assert reply["tick"] == obs["tick"] and type(reply["tick"]) is int
        if obs["tick"] % DECIDE_EVERY != 0:
            assert reply["action"] is None
        if reply["action"] is not None:
            slot, x, y = reply["action"]
            assert type(slot) is int and type(x) is float and type(y) is float
        json.dumps(reply)
    assert cl.ticks == 60 and cl.decisions == 6 and cl.errors == 0


def test_random_policy_moves_are_sanitised():
    cl = C.Client(RandomPolicy(0), DECK, "t")
    sent = [send(cl, {"t": "obs", "obs": o})["action"] for o in real_observations(200)]
    moves = [a for a in sent if a is not None]
    assert moves and all(type(a[0]) is int and type(a[1]) is float for a in moves)


def test_policy_error_still_replies_null():
    cl = C.Client(Boom(), DECK, "t")
    obs = real_observations(1)[0]
    assert obs["tick"] == 0
    reply = send(cl, {"t": "obs", "obs": obs})
    assert reply == {"t": "act", "tick": 0, "action": None}
    assert cl.errors == 1
    send(cl, {"t": "obs", "obs": {**obs, "tick": 10}})
    assert cl.errors == 2


@pytest.mark.parametrize("raw, sent", [
    ([1.0, 3, 9], [1, 3.0, 9.0]),
    ((2, 4.5, 10), [2, 4.5, 10.0]),
    (None, None),
    ("garbage", None),
    ([True, 3, 9], None),
    ([0, float("nan"), 9], None),
    ([0, 1], None),
])
def test_action_sanitising(raw, sent):
    cl = C.Client(Fixed(raw), DECK, "t")
    reply = send(cl, {"t": "obs", "obs": {**real_observations(1)[0], "tick": 20}})
    assert reply["action"] == sent
    if sent is not None:
        assert type(reply["action"][0]) is int
    json.dumps(reply)


def test_result_summary_has_last_action_counts():
    cl = C.Client(Fixed(None), DECK, "t")
    send(cl, {"t": "start", "opponent": "rando"})
    obs = real_observations(1)[0]
    for i, la in enumerate(["none", "ok", "ok", "not_enough_elixir", "none"]):
        send(cl, {"t": "obs", "obs": {**obs, "tick": i, "last_action": la}})
    out = send(cl, {"t": "result", "result": "win", "crowns": [3, 0], "rating": 1016.0,
                    "delta": 16.0, "match": "abc", "seed": 1})
    assert out is None
    s = cl.last_summary
    assert s["last_action"] == {"none": 2, "ok": 2, "not_enough_elixir": 1}
    assert s["ignored"] == 1 and s["opponent"] == "rando" and s["ticks"] == 5
    assert s["replay"] == "/replay?id=abc"
    assert cl.matches_done == 1 and cl.session["win"] == 1
    # counts reset for the next match
    assert cl.last_action == {}


def test_other_messages_do_not_reply_or_crash():
    cl = C.Client(Fixed(None), DECK, "t")
    for m in ({"t": "joined", "name": "t", "rating": 1000}, {"t": "aborted", "reason": "x"},
              {"t": "error", "msg": "bad"}, {"t": "weird"}):
        assert send(cl, m) is None
    assert cl.session["aborted"] == 1


def test_illegal_deck_exits_before_connecting(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["knight"] * 8), encoding="utf-8")
    with pytest.raises(SystemExit):
        C.load_deck(str(bad))
    short = tmp_path / "short.json"
    short.write_text(json.dumps(["knight", "archers"]), encoding="utf-8")
    with pytest.raises(SystemExit):
        C.load_deck(str(short))
    with pytest.raises(SystemExit):
        C.load_deck(str(tmp_path / "missing.json"))
    assert C.load_deck(C.DEFAULT_DECK) == DECK
    d = C.load_deck("random", 5)
    assert len(d) == 8 and d == C.load_deck("random", 5)


def test_learned_policy_missing_exits_cleanly():
    if importlib.util.find_spec("bot.learned") is not None:
        pytest.skip("bot/learned.py exists now")
    with pytest.raises(SystemExit):
        C.build_policy("learned", model="x.pt")


def _load_random_agent():
    spec = importlib.util.spec_from_file_location(
        "random_agent_example", os.path.join(ROOT, "server", "examples", "random_agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_end_to_end_with_local_server(tmp_path, monkeypatch):
    try:
        import websockets
        from cr_server import match, server, web
    except ImportError as e:
        pytest.skip(f"local server not installed (run: pip install ./server): {e}")

    monkeypatch.setattr(server, "DATA_DIR", str(tmp_path))
    orig = match.run_match

    async def short(sim, a0, a1, **kw):  # keep matches brief
        return await orig(sim, a0, a1, max_ticks=400, **kw)

    monkeypatch.setattr(match, "run_match", short)
    rnd = _load_random_agent()
    C.setup_logging(None)
    cl = C.Client(ScriptedPolicy(), DECK, "us")

    async def opponent(url):
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"t": "join", "name": "rando", "deck": rnd.DECK}))
            async for raw in ws:
                m = json.loads(raw)
                if m["t"] == "obs":
                    o = m["obs"]
                    await ws.send(json.dumps({"t": "act", "tick": o["tick"], "action": rnd.policy(o)}))

    async def main():
        arena = server.Arena()
        mm = asyncio.create_task(arena.matchmaker())
        async with websockets.serve(arena.handler, "127.0.0.1", 0,
                                    process_request=web.Web(arena).process_request) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}"
            opp = asyncio.create_task(opponent(url))
            try:
                await asyncio.wait_for(C.run(url, cl, max_matches=2, reconnect_delay=0.1), 90)
            finally:
                opp.cancel()
                mm.cancel()

    asyncio.run(main())
    assert cl.matches_done == 2
    assert cl.session_last_action.get("missed", 0) == 0
    assert cl.session_last_action.get("malformed", 0) == 0
    assert cl.session_last_action.get("ok", 0) > 0


# --- policy specs and --summary-json (C17) ---------------------------------

def test_policy_specs_with_colon_are_accepted():
    from bot.scripted_v0 import ScriptedPolicyV0

    assert isinstance(C.build_policy("class:bot.scripted_v0.ScriptedPolicyV0"), ScriptedPolicyV0)
    assert isinstance(C.build_policy("scripted"), ScriptedPolicy)
    assert isinstance(C.build_policy("random", seed=3), RandomPolicy)
    args = C.parse_args(["--name", "x", "--policy", "class:bot.scripted_v0.ScriptedPolicyV0"])
    assert args.policy == "class:bot.scripted_v0.ScriptedPolicyV0"
    assert C.parse_args(["--name", "x"]).policy == "scripted" and C.parse_args(["--name", "x"]).summary_json is None


@pytest.mark.parametrize("kind", ["nonsense", "class:bot.nope.Missing", "class:bot.scripted.NoSuchClass", "bogus:x"])
def test_unknown_policy_exits_with_code_2(kind, capsys):
    with pytest.raises(SystemExit) as e:
        C.build_policy(kind)
    assert e.value.code == 2
    assert kind in capsys.readouterr().err


def test_summary_json_written_after_each_result(tmp_path):
    path = tmp_path / "sub" / "summary.json"
    cl = C.Client(Fixed(None), DECK, "soak-x", summary_path=str(path), policy_label="scripted")
    assert not path.exists()
    obs = real_observations(1)[0]
    send(cl, {"t": "start", "opponent": "o"})
    for i, la in enumerate(["none", "ok", "missed", "malformed", "not_enough_elixir"]):
        send(cl, {"t": "obs", "obs": {**obs, "tick": i * DECIDE_EVERY, "last_action": la}})
    send(cl, {"t": "result", "result": "win", "crowns": [1, 0], "rating": 1016.0, "delta": 16.0, "match": "m1"})
    s = json.loads(path.read_text(encoding="utf-8"))
    assert s["matches"] == 1 and s["wins"] == 1 and s["losses"] == 0 and s["draws"] == 0
    assert s["aborted"] == 0 and s["rating"] == 1016.0 and s["policy"] == "scripted"
    assert s["last_action"] == {"malformed": 1, "missed": 1, "none": 1, "not_enough_elixir": 1, "ok": 1}
    assert s["ignored"] == 3 and s["decisions"] == 5
    assert s["slow_decisions"] == 0 and s["policy_errors"] == 0 and s["timestamp"]

    send(cl, {"t": "start", "opponent": "o"})
    send(cl, {"t": "result", "result": "loss", "crowns": [0, 1], "rating": 1000.0, "delta": -16.0, "match": "m2"})
    send(cl, {"t": "aborted", "reason": "x"})
    s = json.loads(path.read_text(encoding="utf-8"))
    assert (s["matches"], s["wins"], s["losses"], s["aborted"], s["rating"]) == (2, 1, 1, 1, 1000.0)
    assert s["last_action"]["ok"] == 1  # session totals, not reset per match
    assert [p.name for p in path.parent.iterdir()] == ["summary.json"]  # no temp files left behind


def test_summary_counts_policy_errors(tmp_path):
    path = tmp_path / "s.json"
    cl = C.Client(Boom(), DECK, "t", summary_path=str(path))
    send(cl, {"t": "obs", "obs": real_observations(1)[0]})
    send(cl, {"t": "result", "result": "draw", "crowns": [0, 0], "rating": 1000.0, "delta": 0.0, "match": "m"})
    s = json.loads(path.read_text(encoding="utf-8"))
    assert s["policy_errors"] == 1 and s["draws"] == 1


def test_summary_write_failure_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cl = C.Client(Fixed(None), DECK, "t", summary_path=str(blocker / "s.json"))  # parent is a file
    send(cl, {"t": "result", "result": "win", "crowns": [1, 0], "rating": 1.0, "delta": 1.0, "match": "m"})
    assert cl.matches_done == 1
