"""League of opponents for PPO (handoff B13).

Each new episode's opponent: 50% a past snapshot (uniform over ALL saved ones), 30% scripted,
10% random, 10% bc_v1. Before any snapshot exists the snapshot share goes to bc_v1.

League is picklable: VecEnv workers each hold a copy and call it with their env's own
random.Random; the main process calls env.set_snapshots(league.snapshots) after add_snapshot.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHARES = (("snapshot", 0.5), ("scripted", 0.3), ("random", 0.1), ("bc", 0.1))
KINDS = tuple(k for k, _ in SHARES)


class League:
    def __init__(self, run_dir, bc_path="models/bc_v1.pt", snapshots=()):
        self.run_dir = str(run_dir)
        bc = Path(bc_path)
        self.bc_path = str(bc if bc.is_absolute() else (REPO / bc).resolve())
        self.snapshots: list[str] = [str(p) for p in snapshots]
        self.counts: Counter = Counter()

    def add_snapshot(self, path) -> None:
        p = str(Path(path).resolve())
        if p not in self.snapshots:
            self.snapshots.append(p)

    def set_snapshots(self, paths) -> None:
        self.snapshots = [str(p) for p in paths]

    def sample_kind(self, rng: random.Random | None = None) -> str:
        r = (rng or random).random()
        acc = 0.0
        for kind, share in SHARES:
            acc += share
            if r < acc:
                return kind
        return SHARES[-1][0]

    def sample(self, rng: random.Random | None = None) -> str:
        """A policy spec string for evals.policies.make_policy."""
        rng = rng or random
        kind = self.sample_kind(rng)
        if kind == "snapshot" and not self.snapshots:
            kind = "bc"
        self.counts[kind] += 1
        if kind == "snapshot":
            return f"learned:{rng.choice(self.snapshots)}"
        if kind == "bc":
            return f"learned:{self.bc_path}"
        return kind

    __call__ = sample

    def kind_of(self, spec: str) -> str:
        if spec in ("random", "scripted"):
            return spec
        path = spec[len("learned:"):] if spec.startswith("learned:") else spec
        if str(Path(path).resolve()) == self.bc_path:
            return "bc"
        return "snapshot" if Path(path).name.startswith("snap_") else "learned"

    def state(self) -> dict:
        return {"run_dir": self.run_dir, "bc_path": self.bc_path, "snapshots": list(self.snapshots),
                "counts": dict(self.counts)}

    @classmethod
    def from_state(cls, st: dict) -> "League":
        lg = cls(st["run_dir"], st["bc_path"], st.get("snapshots", ()))
        lg.counts.update(st.get("counts", {}))
        return lg
