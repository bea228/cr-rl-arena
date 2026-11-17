"""Deck sampling and the fixed evaluation deck sets (handoff B2 + W5).

    python -m evals.decks --write   # regenerate the three JSON files (deterministic, seed 0)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cr_sim

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CARDS_JSON = REPO / "gym" / "data" / "cards.json"

GAUNTLET_FILE = "gauntlet_decks.json"
HELDOUT_CARDS_FILE = "heldout_cards.json"
HELDOUT_DECKS_FILE = "heldout_decks.json"

DEFAULT_DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]
N_GAUNTLET = 40
N_HELDOUT_CARDS = 12
N_HELDOUT_DECKS = 40

_BATTLE = cr_sim.Battle()  # one shared instance, used only for check_deck / deployable_cards


def deployable_cards() -> list[str]:
    """All deck-able card names, sorted so sampling never depends on simulator order."""
    return sorted(_BATTLE.deployable_cards())


def champions() -> set[str]:
    with open(CARDS_JSON, encoding="utf-8") as f:
        cards = json.load(f)["cards"]
    return {name for name, info in cards.items() if "champion" in info}


def is_legal(deck: list[str]) -> bool:
    try:
        _BATTLE.check_deck(list(deck))
    except ValueError:
        return False
    return True


def random_deck(rng: random.Random, exclude=()) -> list[str]:
    """8 cards sampled from the sorted deployable cards minus `exclude`, retried until legal."""
    ex = set(exclude)
    pool = [c for c in deployable_cards() if c not in ex]
    while True:
        deck = rng.sample(pool, 8)
        if is_legal(deck):
            return deck


def generate(seed: int = 0) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Return (heldout_cards, gauntlet_decks, heldout_decks), all from one rng."""
    rng = random.Random(seed)
    all_cards = deployable_cards()

    # held-out cards first: not champions, not in the default deck (training uses it 10% of games)
    banned = champions() | set(DEFAULT_DECK)
    candidates = [c for c in all_cards if c not in banned]
    heldout = sorted(rng.sample(candidates, N_HELDOUT_CARDS))

    gauntlet = [random_deck(rng, exclude=heldout) for _ in range(N_GAUNTLET)]

    others = [c for c in all_cards if c not in set(heldout)]
    heldout_decks = []
    while len(heldout_decks) < N_HELDOUT_DECKS:
        k = rng.randint(3, 5)
        deck = rng.sample(heldout, k) + rng.sample(others, 8 - k)
        if is_legal(deck):
            heldout_decks.append(deck)
    return heldout, gauntlet, heldout_decks


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def write_all(out_dir: Path = HERE, seed: int = 0) -> None:
    heldout, gauntlet, heldout_decks = generate(seed)
    out_dir = Path(out_dir)
    _dump(heldout, out_dir / HELDOUT_CARDS_FILE)
    _dump(gauntlet, out_dir / GAUNTLET_FILE)
    _dump(heldout_decks, out_dir / HELDOUT_DECKS_FILE)


def _load(name: str):
    with open(HERE / name, encoding="utf-8") as f:
        return json.load(f)


def load_gauntlet() -> list[list[str]]:
    return _load(GAUNTLET_FILE)


def load_heldout_cards() -> list[str]:
    return _load(HELDOUT_CARDS_FILE)


def load_heldout_decks() -> list[list[str]]:
    return _load(HELDOUT_DECKS_FILE)


if __name__ == "__main__":
    if "--write" in sys.argv[1:]:
        write_all()
        print(f"wrote {GAUNTLET_FILE}, {HELDOUT_CARDS_FILE}, {HELDOUT_DECKS_FILE} in {HERE}")
    else:
        print(__doc__)
