"""Stability-gated command resolution for the phase routers (flat, no band).

Formalizes the v9 bin5 degradation discovered in the closed-loop battery:
commands whose exact router group is unstable (teacher-bound or noisy) are
deliberately mapped to the nearest stable anchor, so the no-band controller
never falls on flat ground even when it cannot execute the exact motion.

Stability table comes from ``outputs/flat_battery_v9.json``
(3 seeds x 20 s; stable = 3/3 completed without a fall; ``moved`` = the group
actually translates, |vx| >= 0.15 or displacement >= 3 m).

Resolution priority for a command (mode, speed, bin):
  1. exact group stable -> exact
  2. same mode, nearest trained speed, same bin
  3. direction-hemisphere anchor: forward-ish bins (3/4/5) -> walk fwd,
     backward-ish bins (0/1/7) -> walk back, lateral (2/6) -> walk fwd
  4. slow -> walk same bin (stable) -> anchors
  5. jump/stealth teacher-bound -> walk fwd degrade
  6. final anchor: idle (0,-1.0,4)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

BIN_N = 8


def circ_dist(a: int, b: int) -> int:
    d = abs(int(a) - int(b)) % BIN_N
    return min(d, BIN_N - d)


def forwardish(b: int) -> bool:
    return int(b) in (3, 4, 5)


def backwardish(b: int) -> bool:
    return int(b) in (0, 1, 7)


@dataclass
class ResolvedGroup:
    key: tuple
    degraded: bool
    reason: str


class StableResolver:
    """Deterministic (mode, speed, bin) -> group-key resolution."""

    def __init__(self, gmap: dict, battery: dict | None = None):
        self.gmap = gmap
        self.battery = battery or {}
        self.speeds_by_mode: dict[int, list[float]] = {}
        for (m, s, _b) in gmap:
            self.speeds_by_mode.setdefault(int(m), set()).add(round(float(s), 2))
        for m in self.speeds_by_mode:
            self.speeds_by_mode[m] = sorted(self.speeds_by_mode[m])

    def is_stable(self, key: tuple) -> bool:
        k = (int(key[0]), round(float(key[1]), 2), int(key[2]))
        if k not in self.gmap:
            return False
        if not self.battery:
            return True  # no battery -> assume stable (back-compat)
        row = self.battery.get(f"{k[0]}_{k[1]}_{k[2]}")
        return bool(row and row.get("completed", 0) == 3)

    def moved(self, key: tuple) -> bool:
        k = (int(key[0]), round(float(key[1]), 2), int(key[2]))
        row = self.battery.get(f"{k[0]}_{k[1]}_{k[2]}")
        if not row:
            return False
        return abs(row.get("vx_mean", 0.0)) >= 0.15 or row.get("disp_mean", 0.0) >= 3.0

    # ------------------------------------------------------------------ rules
    def _nearest_speed(self, mode: int, speed: float) -> float:
        speeds = self.speeds_by_mode.get(mode, [])
        if not speeds:
            return speed
        return min(speeds, key=lambda s: abs(s - speed))

    def _stable_key(self, mode: int, speed: float, bin_: int) -> tuple | None:
        k = (int(mode), round(float(speed), 2), int(bin_))
        return k if k in self.gmap and self.is_stable(k) else None

    def _walk_anchor(self, bin_: int) -> tuple | None:
        """Data-driven nearest stable walk anchor (ties -> larger |vx|)."""
        stable_bins = [
            b
            for b in range(BIN_N)
            if self.is_stable((2, -1.0, b))
        ]
        if not stable_bins:
            return None
        best = min(
            stable_bins,
            key=lambda b: (
                circ_dist(bin_, b),
                -abs(
                    self.battery.get(f"2_-1.0_{b}", {}).get("vx_mean", 0.0)
                ),
            ),
        )
        k = (2, -1.0, best)
        return k if k in self.gmap else None

    def resolve(self, mode: int, speed: float, bin_: int) -> ResolvedGroup:
        mode = int(mode)
        bin_ = int(bin_)
        exact = (mode, round(float(speed), 2), bin_)
        if exact in self.gmap and self.is_stable(exact):
            return ResolvedGroup(exact, False, "exact")

        # same mode, trained speeds sorted by distance, same bin
        speeds = sorted(
            self.speeds_by_mode.get(mode, []),
            key=lambda s: (abs(s - round(float(speed), 2)), s),
        )
        for ns in speeds:
            if ns == round(float(speed), 2):
                continue
            k = self._stable_key(mode, ns, bin_)
            if k:
                return ResolvedGroup(k, True, f"speed {speed}->{ns} same bin")

        if mode == 2:
            # walk: skip slow (standing or unstable), go straight to the anchor
            anchor = self._walk_anchor(bin_)
            if anchor:
                return ResolvedGroup(anchor, True, "direction anchor")
        if mode == 1:
            # slow -> walk same bin (stable) -> anchor
            k = self._stable_key(2, -1.0, bin_)
            if k:
                return ResolvedGroup(k, True, f"slow->walk same bin")
            anchor = self._walk_anchor(bin_)
            if anchor:
                return ResolvedGroup(anchor, True, "direction anchor")

        if mode in (17, 18):
            anchor = self._walk_anchor(bin_ if mode == 18 else 4)
            if anchor:
                return ResolvedGroup(anchor, True, "teacher-bound degrade")

        # final anchor: idle
        for k in [(0, -1.0, 4), (0, 0.0, 4)]:
            if k in self.gmap and self.is_stable(k):
                return ResolvedGroup(k, True, "idle fallback")
        return ResolvedGroup(exact, True, "unresolved (exact)")


def load_resolver(router_dir: str, battery_path: str) -> StableResolver:
    meta = json.load(open(os.path.join(router_dir, "phase_meta.json")))
    gmap = {tuple(md["group"]): int(gi) for gi, md in meta.items() if not gi.startswith("_")}
    battery = json.load(open(battery_path)) if os.path.exists(battery_path) else {}
    return StableResolver(gmap, battery)
