"""LearnedPolicy: a trained Net behind the Policy contract (handoff B10).

    LearnedPolicy("models/bc_v1.pt")              # argmax over masked logits
    LearnedPolicy(path, temperature=0.5, seed=1)  # sampled, for experiments

Loads on CPU, refuses a file whose FEAT_DIM / N_ACTIONS differ from the code (bot.model.load),
owns a Trackers, picks the highest masked logit and passes it through guardrails.apply.
Networks are cached per absolute path, so building one policy per game (the gauntlet) loads
the file once per process.
"""
from __future__ import annotations

import importlib
import math
import random
import time
from pathlib import Path

from bot import placements
from bot.encode import encode
from bot.policy import Policy
from bot.trackers import Trackers

_NETS: dict[str, object] = {}  # absolute path -> Net (eval mode)
LOAD_STATS = {"loads": 0, "hits": 0, "last_load_seconds": 0.0}
_THREADS_SET = False


def _load_net(path: str):
    global _THREADS_SET
    key = str(Path(path).resolve())
    net = _NETS.get(key)
    if net is not None:
        LOAD_STATS["hits"] += 1
        return net
    t0 = time.perf_counter()
    import torch

    from bot import model

    if not _THREADS_SET:
        torch.set_num_threads(1)
        _THREADS_SET = True
    net = model.load(key, map_location="cpu").eval()
    _NETS[key] = net
    LOAD_STATS["loads"] += 1
    LOAD_STATS["last_load_seconds"] = time.perf_counter() - t0
    return net


def _guardrails():
    """bot.guardrails (C10) if it exists, else None."""
    try:
        return importlib.import_module("bot.guardrails")
    except ModuleNotFoundError as e:
        if e.name == "bot.guardrails":
            return None
        raise


class LearnedPolicy(Policy):
    def __init__(self, path, temperature: float = 0.0, seed=None):
        self.path = str(path)
        self.net = _load_net(self.path)
        self.temperature = float(temperature)
        self.rng = random.Random(seed)
        self.trackers = Trackers()
        self.deck = None
        self.mask_fallbacks = 0
        self.decisions = 0
        # Guardrails (C10) don't exist yet: without bot/guardrails.py the net's choice is used as is.
        self._gr = _guardrails()
        self.guardrails_active = self._gr is not None and hasattr(self._gr, "apply")

    def reset(self, deck):
        self.deck = list(deck)
        self.trackers = Trackers()

    def _logits(self, obs):
        import torch

        from bot.model import to_torch

        with torch.no_grad():
            logits, _ = self.net(to_torch(encode(obs, self.trackers)))
        return logits[0]

    def _pick(self, logits) -> int:
        if self.temperature <= 0:
            return int(logits.argmax())
        vals = (logits / self.temperature).tolist()
        top = max(vals)
        weights = [math.exp(v - top) for v in vals]
        return self.rng.choices(range(len(weights)), weights=weights)[0]

    def act_index(self, obs):
        self.trackers.update(obs)
        self.decisions += 1
        idx = self._pick(self._logits(obs))
        if self.guardrails_active:
            idx = int(self._gr.apply(obs, idx))
        if not placements.mask(obs)[idx]:
            self.mask_fallbacks += 1
            return 0
        return idx
