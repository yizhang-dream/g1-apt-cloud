"""Batched (vectorized) phase-router encoder for Isaac Lab envs.

Mirrors ``apt_g1.encoder.phase_router_encoder.PhaseRouterEncoder`` but
processes N environments in one forward pass (per-group masked MLP calls).
The per-env EMA phase smoothing is kept as explicit state so the caller can
reset it on episode resets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class PhaseNet(nn.Module):
    """Small MLP regressing the circular gait phase (sin(phi), cos(phi))."""

    def __init__(self, d_in: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class BatchedPhaseRouter:
    """(command, proprio history) -> 64-d SONIC token, vectorized over envs."""

    def __init__(self, model_dir: Union[str, Path], device: str = "cuda:0"):
        self.device = device
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "phase_meta.json").read_text(encoding="utf-8"))
        norm = np.load(self.dir / "phase_norm.npz")
        self.pmean = norm["pmean"].ravel().astype(np.float32)
        self.pstd = norm["pstd"].ravel().astype(np.float32)

        self.modes_list = np.array(
            sorted({md["group"][0] for md in self.meta.values()}), dtype=np.int32
        )
        self.d_cmd = len(self.modes_list) + 9  # one-hot + mdir(3) + fdir(3) + speed + height + planner

        self.gmap: dict[tuple, int] = {}
        self.group_rows: list[tuple] = []
        self.nets: dict[int, nn.Module] = {}
        self.protos: dict[int, np.ndarray] = {}
        self.n_bins: dict[int, int] = {}
        for gi_str, md in self.meta.items():
            gi = int(gi_str)
            g = tuple(md["group"])
            self.gmap[g] = gi
            self.group_rows.append(g)
            net = PhaseNet(930 + self.d_cmd).to(self.device)
            net.load_state_dict(
                torch.load(self.dir / f"phase_g{gi}.pt", map_location=self.device, weights_only=True)
            )
            net.eval()
            self.nets[gi] = net
            self.protos[gi] = np.load(self.dir / f"proto_g{gi}.npy")
            self.n_bins[gi] = len(self.protos[gi])
        self._groups_t = torch.tensor(
            [self.gmap[g] for g in self.group_rows], dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------ utils
    def select_group(self, command) -> int:
        """Mirror PhaseRouterEncoder.select_group for a single Command."""
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

    def select_groups(self, commands) -> np.ndarray:
        """Vectorized group selection over a list of Command objects."""
        return np.asarray([self.select_group(c) for c in commands], dtype=np.int64)

    def cmd_features(self, commands) -> np.ndarray:
        """(N, d_cmd) command features, same layout as PhaseRouterEncoder._cmd_feature."""
        rows = []
        for c in commands:
            oh = np.zeros(len(self.modes_list), dtype=np.float32)
            idx = int(np.where(self.modes_list == c.mode)[0][0])
            oh[idx] = 1.0
            rows.append(
                np.concatenate(
                    [
                        oh,
                        np.asarray(c.mdir, dtype=np.float32),
                        np.asarray(c.fdir, dtype=np.float32),
                        np.array(
                            [c.speed, c.height, 1.0 if c.planner else 0.0],
                            dtype=np.float32,
                        ),
                    ]
                ).astype(np.float32)
            )
        return np.stack(rows, axis=0)

    # ------------------------------------------------------------------ batch
    def phase_raw_batch(
        self,
        proprio_np: np.ndarray,
        commands,
        force_groups: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (normalized (sin,cos), group ids) for N envs (no EMA)."""
        N = proprio_np.shape[0]
        prop = (proprio_np.astype(np.float32) - self.pmean) / self.pstd
        cmd = self.cmd_features(commands)
        x = np.concatenate([prop, cmd], axis=1).astype(np.float32)
        groups = (
            self.select_groups(commands)
            if force_groups is None
            else np.asarray(force_groups, dtype=np.int64)
        )
        sc = np.zeros((N, 2), dtype=np.float32)
        for gi in self.nets:
            mask = groups == gi
            if not mask.any():
                continue
            with torch.no_grad():
                out = self.nets[gi](
                    torch.from_numpy(x[mask]).to(self.device)
                ).cpu().numpy()
            sc[mask] = out
        n = np.linalg.norm(sc, axis=1, keepdims=True)
        n[n < 1e-6] = 1.0
        sc = sc / n
        return sc.astype(np.float32), groups

    def encode_batch(
        self,
        proprio_np: np.ndarray,
        commands,
        state: dict | None = None,
        ema: float = 0.3,
        force_groups: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Return (N, 64) prototype tokens and updated router state dict.

        state: {"sc_prev": (N,2) float32 | None, "groups": (N,) int64 | None}
        """
        sc, groups = self.phase_raw_batch(proprio_np, commands, force_groups=force_groups)
        if state is None:
            state = {"sc_prev": None, "groups": None}
        sc_prev = state.get("sc_prev")
        if sc_prev is None or sc_prev.shape[0] != sc.shape[0]:
            state["sc_prev"] = sc.copy()
        else:
            prev_groups = state.get("groups")
            if prev_groups is not None and (prev_groups != groups).any():
                # reset EMA where the group changed
                changed = prev_groups != groups
                sc_prev[changed] = sc[changed]
            if ema > 0:
                sc = ema * sc_prev + (1.0 - ema) * sc
            state["sc_prev"] = sc.copy()
        state["groups"] = groups.copy()

        tokens = np.zeros((sc.shape[0], 64), dtype=np.float32)
        for gi in self.nets:
            mask = groups == gi
            if not mask.any():
                continue
            phi = np.arctan2(sc[mask, 0], sc[mask, 1])
            n_bins = self.n_bins[gi]
            bins = np.floor((phi + np.pi) / (2.0 * np.pi) * n_bins).astype(np.int64) % n_bins
            tokens[mask] = self.protos[gi][bins]
        return tokens, state

    def reset_state(self, n_envs: int) -> dict:
        return {"sc_prev": None, "groups": None}
