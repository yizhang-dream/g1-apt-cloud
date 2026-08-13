"""Phase-AR encoder: same phase-router but with the previous phase as input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .phase_router_encoder import Command, proprio_vector


class PhaseNetAR(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class PhaseAREncoder:
    """Frozen distilled encoder with autoregressive phase input."""

    def __init__(self, model_dir: Union[str, Path], device: str = "cuda:0"):
        self.device = device
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "phase_meta.json").read_text())
        norm = np.load(self.dir / "phase_norm.npz")
        self.pmean = norm["pmean"].ravel().astype(np.float32)
        self.pstd = norm["pstd"].ravel().astype(np.float32)
        self.modes_list = np.array(
            sorted({md["group"][0] for md in self.meta.values()}), dtype=np.int32
        )
        self.d_cmd = len(self.modes_list) + 9
        self.gmap: dict[tuple, int] = {}
        for gi, md in self.meta.items():
            self.gmap[tuple(md["group"])] = int(gi)
        self.nets = {}
        self.protos = {}
        for gi, md in self.meta.items():
            gi = int(gi)
            net = PhaseNetAR(int(md["d_in"])).to(device)
            net.load_state_dict(torch.load(self.dir / f"phase_g{gi}.pt", map_location=device))
            net.eval()
            self.nets[gi] = net
            self.protos[gi] = np.load(self.dir / f"proto_g{gi}.npy")
        self.reset()

    def reset(self):
        self._prev = None
        self._gi = None
        self._sc_prev = None

    def select_group(self, command: Command) -> int:
        key = command.key()
        if key in self.gmap:
            return self.gmap[key]
        mode, _speed, b = key
        cands = [(g, gi) for g, gi in self.gmap.items() if g[0] == mode and g[2] == b]
        if cands:
            return min(cands, key=lambda t: abs(t[0][1] - command.speed))[1]
        cands = [(g, gi) for g, gi in self.gmap.items() if g[0] == mode]
        if cands:
            return cands[0][1]
        raise KeyError(f"no router for command mode={command.mode} speed={command.speed} bin={b}")

    def _cmd_feature(self, command: Command) -> np.ndarray:
        oh = np.zeros(len(self.modes_list), dtype=np.float32)
        idx = int(np.where(self.modes_list == command.mode)[0][0])
        oh[idx] = 1.0
        return np.concatenate(
            [
                oh,
                command.mdir.astype(np.float32),
                command.fdir.astype(np.float32),
                np.array([command.speed, command.height, 1.0 if command.planner else 0.0], dtype=np.float32),
            ]
        ).astype(np.float32)

    def encode(self, command: Command, history: Union[dict, np.ndarray], ema: float = 0.3) -> np.ndarray:
        prop = proprio_vector(history) if isinstance(history, dict) else np.asarray(history, dtype=np.float32)
        gi = self.select_group(command)
        if self._gi != gi:
            self._gi = gi
            self._prev = None
            self._sc_prev = None
        prev = self._prev if self._prev is not None else np.zeros(2, dtype=np.float32)
        x = np.concatenate(
            [(prop - self.pmean) / self.pstd, self._cmd_feature(command), prev]
        ).astype(np.float32)
        with torch.no_grad():
            sc = self.nets[gi](torch.from_numpy(x[None]).to(self.device))[0].cpu().numpy().astype(np.float32)
        if ema > 0 and self._sc_prev is not None:
            sc = ema * self._sc_prev + (1.0 - ema) * sc
        self._sc_prev = sc
        n = float(np.linalg.norm(sc))
        if n < 1e-6:
            n = 1.0
        sc = sc / n
        self._prev = sc.astype(np.float32)
        phi = float(np.arctan2(sc[0], sc[1]))
        n_bins = len(self.protos[gi])
        b = int(np.floor((phi + np.pi) / (2 * np.pi) * n_bins) % n_bins)
        return self.protos[gi][b].copy()
