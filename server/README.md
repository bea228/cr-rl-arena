# cr-arena-server

The competition ladder for [`cr_sim`](../gym). Agents connect over WebSocket,
join a queue, and are continuously paired for **server-authoritative** matches.
The server owns the only real `cr_sim.Battle`; agents just answer observations
with actions. Results update a persistent **Elo** leaderboard, and every match's
`(seed, action-log)` is saved for replay/audit.

## Run

```bash
pip install ./gym ./server      # server depends on the gym package (cr-sim)
cr-arena                        # or: python -m cr_server.server
```

Environment variables:

| var | default | meaning |
|---|---|---|
| `CR_ARENA_BIND` | `0.0.0.0:8765` | host:port to listen on |
| `CR_ARENA_DATA` | `./arena_data` | leaderboard + replay storage |
| `CR_MOVE_TIMEOUT` | `1.0` | seconds an agent has to answer each decision (miss ⇒ no-op) |

## Protocol (JSON text frames over WebSocket)

```
C→S  {"t":"join","name":"myagent","deck":["knight",...8...]}
     {"t":"leaderboard"}                         # query standings, then close
S→C  {"t":"joined","name","rating"}
     {"t":"start","player":0|1,"opponent","seed"}
     {"t":"obs","obs":{...}}                      # sent each decision step; reply required
C→S  {"t":"act","action":[slot,x,y] | null}       # slot 0-3; x,y in tiles on your side
S→C  {"t":"result","you":0|1,"outcome":1|2|3,"crowns":[a,b],"rating","delta"}
```

- Decisions are requested at **5 Hz** (every 4 ticks). Between decisions the sim
  advances with no-ops — deploys are instantaneous, so that loses nothing.
- `outcome`: `1` player-0 win, `2` player-1 win, `3` draw.
- A disconnected agent **forfeits** to a connected opponent.
- A timed-out or malformed reply is treated as a no-op; one bad agent can't stall
  or crash a match or the ladder.

The `obs` dict mirrors `cr_sim.Battle.observe(player)`: `elixir`, `hand`,
`hand_cost`, `next_card`, `legal` (per-slot affordability mask), `crowns`,
`entities` (full board), `tick`, `phase`. See [`examples/random_agent.py`](examples/random_agent.py)
for a complete client.

## Reproducing / auditing a match

Each saved replay in `CR_ARENA_DATA/replays/` holds the `seed` and `actions`.
Re-simulate it deterministically:

```python
import cr_sim, json
from cr_server.match import replay
r = json.load(open("arena_data/replays/<id>.json"))
print(replay(cr_sim, r["seed"], r["actions"], r["deck0"], r["deck1"]))  # same outcome/crowns
```

## Design notes

- **Why Python, not Rust, for the server:** it imports the same `cr_sim` native
  extension the students use, so there is exactly one implementation of the game
  logic and no cross-language drift. Determinism holds because the engine itself
  is fixed-point.
- **Scaling:** matches are `asyncio` tasks and run concurrently; a full 3-minute
  match is ~900 decision round-trips and completes in well under a second of wall
  time against fast agents.
