"""Single unified encoder distilled from the official closed loop.

``PhaseRouterEncoder`` replaces the official planner+encoder chain for the
covered command space: given a high-level command and a proprioception history,
it outputs a 64-d SONIC token (on the k/16 lattice) that the frozen SONIC
decoder turns into joint targets.

Internally it is a per-command-group phase router:

    command + proprio -> group (mode, speed, direction-bin)
                       -> PCA circular gait phase (MLP regresses sin/cos)
                       -> phase-bin prototype token (+ EMA phase smoothing)

Everything (per-group MLPs, PCA meta, prototypes, normalization) is loaded from
one model directory, so the class is a drop-in replacement for the planner +
encoder in the no-band MuJoCo loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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


def proprio_vector(history: dict) -> np.ndarray:
    """Concatenate the 10-frame history dict into the 930-d vector used by
    the official decoder obs (ang_vel, joint pos, joint vel, last actions,
    gravity), matching the C++ GatherHis* order."""
    parts = [
        history["base_angular_velocity"],
        history["body_joint_positions"],
        history["body_joint_velocities"],
        history["last_actions"],
        history["gravity_dir"],
    ]
    return np.concatenate([p.reshape(-1) for p in parts]).astype(np.float32)


@dataclass
class Command:
    """High-level command in the official command space."""

    mode: int
    speed: float
    mdir: np.ndarray  # (3,) unit movement direction
    fdir: np.ndarray  # (3,) unit facing direction
    height: float = -1.0
    planner: bool = True

    @classmethod
    def from_vxvy(cls, vx: float, vy: float, yaw: float = 0.0) -> "Command":
        """Map a linear-velocity command to the official (mode, speed, dir)
        space used by the distilled routers.

        - |v| < 0.05   -> IDLE (stand)
        - 0.05..0.5    -> SLOW_WALK (speed 0.2)
        - >= 0.5       -> WALK (mode default speed)
        - negative vx  -> backward (dir = pi)
        """
        speed_mag = float(np.hypot(vx, vy))
        if speed_mag < 0.05:
            mode, speed = 0, -1.0
            mdir = np.zeros(3, dtype=np.float32)
        elif speed_mag < 0.5:
            mode, speed = 1, 0.2
            ang = float(np.arctan2(vy, vx))
            mdir = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float32)
        else:
            mode, speed = 2, -1.0
            ang = float(np.arctan2(vy, vx))
            mdir = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float32)
        fdir = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        return cls(mode=mode, speed=speed, mdir=mdir, fdir=fdir)

    def angle(self) -> float:
        return float(np.arctan2(self.mdir[1], self.mdir[0]))

    def key(self, n_bins: int = 8) -> tuple:
        b = int(np.floor((self.angle() + np.pi) / (2 * np.pi) * n_bins)) % n_bins
        return (self.mode, round(float(self.speed), 2), b)


class PhaseRouterEncoder:
    """Single distilled encoder: (command, proprio history) -> 64-d token."""

    def __init__(self, model_dir: Union[str, Path], device: str = "cuda:0"):
        self.device = device
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "phase_meta.json").read_text())
        self.norm = np.load(self.dir / "phase_norm.npz")
        self.pmean = self.norm["pmean"].ravel().astype(np.float32)
        self.pstd = self.norm["pstd"].ravel().astype(np.float32)

        # command space: sorted unique modes across groups (== meta_modes)
        self.modes_list = np.array(
            sorted({md["group"][0] for md in self.meta.values()}), dtype=np.int32
        )
        self.d_cmd = len(self.modes_list) + 9  # one-hot + mdir(3) + fdir(3) + speed + height + planner

        self.gmap: dict[tuple, int] = {}
        self.group_rows: list[tuple] = []
        self.nets: dict[int, PhaseNet] = {}
        self.protos: dict[int, np.ndarray] = {}
        for gi, md in self.meta.items():
            gi = int(gi)
            g = tuple(md["group"])
            self.gmap[g] = gi
            self.group_rows.append(g)
            net = PhaseNet(930 + self.d_cmd).to(self.device)
            net.load_state_dict(torch.load(self.dir / f"phase_g{gi}.pt", map_location=self.device))
            net.eval()
            self.nets[gi] = net
            self.protos[gi] = np.load(self.dir / f"proto_g{gi}.npy")

        self.reset()

    def reset(self) -> None:
        """Clear inference state (EMA phase) — call when starting a new episode."""
        self._sc_prev: np.ndarray | None = None
        self._gi: int | None = None

    # ------------------------------------------------------------------ utils
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

    def phase_raw(self, command: Command, history: Union[dict, np.ndarray]):
        """Return (gi, normalized (sin,cos)) without EMA — used for supervised
        warm-start labels and diagnostics."""
        prop = proprio_vector(history) if isinstance(history, dict) else np.asarray(history, dtype=np.float32)
        gi = self.select_group(command)
        x = np.concatenate([(prop - self.pmean) / self.pstd, self._cmd_feature(command)]).astype(np.float32)
        with torch.no_grad():
            sc = self.nets[gi](torch.from_numpy(x[None]).to(self.device))[0].cpu().numpy().astype(np.float32)
        n = float(np.linalg.norm(sc))
        if n < 1e-6:
            n = 1.0
        return gi, (sc / n).astype(np.float32)

    # ------------------------------------------------------------------ core
    def encode(
        self,
        command: Command,
        history: Union[dict, np.ndarray],
        ema: float = 0.3,
    ) -> np.ndarray:
        """Return the 64-d token (k/16 lattice) for one control step.

        history: either the env's 10-frame history dict or a pre-built 930-d vector.
        """
        prop = proprio_vector(history) if isinstance(history, dict) else np.asarray(history, dtype=np.float32)
        gi = self.select_group(command)
        x = np.concatenate([(prop - self.pmean) / self.pstd, self._cmd_feature(command)]).astype(np.float32)
        with torch.no_grad():
            sc = self.nets[gi](torch.from_numpy(x[None]).to(self.device))[0].cpu().numpy().astype(np.float32)
        if self._gi != gi:
            self._gi = gi
            self._sc_prev = None
        if ema > 0 and self._sc_prev is not None:
            sc = ema * self._sc_prev + (1.0 - ema) * sc
        self._sc_prev = sc
        phi = float(np.arctan2(sc[0], sc[1]))
        n_bins = len(self.protos[gi])
        b = int(np.floor((phi + np.pi) / (2 * np.pi) * n_bins) % n_bins)
        return self.protos[gi][b].copy()

    def encode_from_env(self, env, command: Command, ema: float = 0.3) -> np.ndarray:
        """Convenience: build the history vector from a MujocoG1FlatEnv instance."""
        return self.encode(command, env._get_sonic_history(), ema=ema)

    def state_dict(self) -> dict:
        """Collect all torch state for later export (single artifact)."""
        return {str(gi): net.state_dict() for gi, net in self.nets.items()}
