"""Policy/value network (handoff B7).

One shared card layer (FEAT_DIM -> 32) for hand, units and enemy deck; a 2-layer net per
unit, masked mean + max pooled separately for our units and theirs; enemy deck pooled the
same way; concat with globals and the flattened hand; two hidden layers of 256; a policy
head of N_ACTIONS logits (masked entries = -1e9) and a value head of 1.

Input: the keys of ``bot.encode.encode`` / ``batch_encode`` as torch tensors (``to_torch``).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from bot import encode as enc

FEAT_DIM = enc.FEAT_DIM
N_ACTIONS = enc.N_ACTIONS
MASKED_LOGIT = -1e9

DEFAULT_CONFIG = {"card": 32, "unit": 64, "hidden": 256}


def to_torch(batch: dict, device="cpu") -> dict[str, torch.Tensor]:
    """encode()/batch_encode() output -> float32 tensors; adds a batch dim to single obs."""
    out = {}
    for k in enc.KEYS:
        t = torch.as_tensor(np.asarray(batch[k], dtype=np.float32), device=device)
        if t.dim() == len(enc.SHAPES[k]):
            t = t.unsqueeze(0)
        out[k] = t
    return out


def _masked_mean_max(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """x [B, N, D], m [B, N] (1 = real) -> [B, 2D]; an empty set pools to zeros."""
    m3 = m.unsqueeze(-1)
    count = m3.sum(1)
    mean = (x * m3).sum(1) / count.clamp(min=1.0)
    mx = x.masked_fill(m3 == 0, float("-inf")).amax(1)
    mx = torch.where(count > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


class Net(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = dict(DEFAULT_CONFIG, **(config or {}))
        self.FEAT_DIM = FEAT_DIM
        self.N_ACTIONS = N_ACTIONS
        c, u, h = self.config["card"], self.config["unit"], self.config["hidden"]

        self.card = nn.Sequential(nn.Linear(FEAT_DIM, c), nn.ReLU())
        self.unit = nn.Sequential(
            nn.Linear(c + enc.UNIT_EXTRA, u), nn.ReLU(),
            nn.Linear(u, u), nn.ReLU(),
        )
        trunk_in = (
            enc.N_GLOBALS
            + enc.N_HAND * (c + enc.HAND_EXTRA)
            + 2 * 2 * u  # ours + theirs, mean + max
            + 2 * (c + enc.ENEMY_EXTRA)  # enemy deck mean + max
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.policy = nn.Linear(h, N_ACTIONS)
        self.value = nn.Linear(h, 1)

    def _cards(self, rows: torch.Tensor) -> torch.Tensor:
        """[..., FEAT_DIM + k] -> [..., card + k]: shared card layer on the feature part."""
        return torch.cat([self.card(rows[..., :FEAT_DIM]), rows[..., FEAT_DIM:]], dim=-1)

    def forward(self, x: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        B = x["globals"].shape[0]
        hand = self._cards(x["hand"]).reshape(B, -1)

        units = self.unit(self._cards(x["units"]))
        um = x["units_mask"]
        owner = x["units"][..., FEAT_DIM]  # 0 ours, 1 theirs
        ours = _masked_mean_max(units, um * (owner < 0.5).float())
        theirs = _masked_mean_max(units, um * (owner >= 0.5).float())

        deck = _masked_mean_max(self._cards(x["enemy_deck"]), x["enemy_deck_mask"])

        z = self.trunk(torch.cat([x["globals"], hand, ours, theirs, deck], dim=-1))
        logits = self.policy(z).masked_fill(x["mask"] == 0, MASKED_LOGIT)
        return logits, self.value(z).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path) -> None:
        torch.save(
            {"state_dict": self.state_dict(), "FEAT_DIM": FEAT_DIM, "N_ACTIONS": N_ACTIONS, "config": self.config},
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu") -> "Net":
        ck = torch.load(path, map_location=map_location, weights_only=True)
        if ck.get("FEAT_DIM") != FEAT_DIM or ck.get("N_ACTIONS") != N_ACTIONS:
            raise ValueError(
                f"checkpoint {path} has FEAT_DIM={ck.get('FEAT_DIM')}, N_ACTIONS={ck.get('N_ACTIONS')}; "
                f"code has FEAT_DIM={FEAT_DIM}, N_ACTIONS={N_ACTIONS}"
            )
        net = cls(ck.get("config"))
        net.load_state_dict(ck["state_dict"])
        return net


def save(net: Net, path) -> None:
    net.save(path)


def load(path, map_location="cpu") -> Net:
    return Net.load(path, map_location)
