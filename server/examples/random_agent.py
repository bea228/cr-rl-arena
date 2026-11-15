"""A minimal reference agent: connects, joins the ladder, and plays random legal
moves forever. Shows the full client protocol students build their RL agent on.

    python examples/random_agent.py [ws://host:port] [name]
"""
import asyncio
import json
import random
import sys

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765"
NAME = sys.argv[2] if len(sys.argv) > 2 else f"random-{random.randint(100, 999)}"


# A deck is required to join: 8 different playable cards, at most one champion.
DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]


def policy(obs):
    """obs: {player, elixir, hand, hand_cost, next_card, legal, crowns, entities, ...}
    Return [slot, x, y] to deploy hand[slot] at tile (x,y), or None to wait."""
    legal = obs.get("legal", [])
    playable = [i for i, ok in enumerate(legal) if ok]
    if not playable or random.random() < 0.6:
        return None
    slot = random.choice(playable)
    player = obs.get("player", 0)
    x = random.uniform(2.0, 16.0)
    y = random.uniform(9.0, 14.0) if player == 0 else random.uniform(18.0, 23.0)
    return [slot, x, y]


async def main():
    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"t": "join", "name": NAME, "deck": DECK}))
        async for raw in ws:
            msg = json.loads(raw)
            t = msg.get("t")
            if t == "obs":
                await ws.send(json.dumps({"t": "act", "action": policy(msg["obs"])}))
            elif t == "joined":
                print(f"joined as {msg['name']} (rating {msg['rating']})")
            elif t == "start":
                print(f"match vs {msg['opponent']} as player {msg['player']}")
            elif t == "result":
                r = {1: "WIN", 2: "LOSS", 3: "DRAW"}[msg["outcome"]] if msg["you"] == 0 \
                    else {1: "LOSS", 2: "WIN", 3: "DRAW"}[msg["outcome"]]
                print(f"  {r} {msg['crowns']}  rating {msg['rating']} ({msg['delta']:+})")
            elif t == "error":
                print("error:", msg.get("msg"))
                break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
