"""Flat-ground full-command-space battery WITH the StableResolver fallback.

Commands enumerated: modes {0,1,2,17,18}, all 8 direction bins for slow/walk,
the training speeds per mode, plus continuous speeds (0.15/0.35/0.5/0.9 m/s) at
bin4.  Each command is resolved through ``StableResolver`` and the RESOLVED
group's phase net + prototypes are driven (anchor feature, in-distribution).

Criterion (priority-2 gate):
  * every command: 3/3 x 20 s without a fall (degraded direction is allowed)
  * slow fwd: moved (|vx| >= 0.15 or disp >= 3 m) -- not standing
  * walk back: negative displacement

Outputs: outputs/flat_battery_fallback_v9.json
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import torch

LOCAL = r"C:\Users\zyz\Documents\gr00t\apt_g1"
REPO = r"D:\GR00T-WholeBodyControl"
sys.path.insert(0, os.path.dirname(LOCAL))
sys.path.insert(0, LOCAL)
sys.path.insert(0, REPO)

from rough_sweep import load_router, feat_for, run_one, make_env
from router_fallback import StableResolver, load_resolver

STEPS = 1200
SEEDS = [0, 1, 2]


def bin_angle(b):
    return b * np.pi / 4.0 - np.pi


def command_space():
    cmds = []
    for b in range(8):
        cmds.append((2, -1.0, b))  # walk all directions
        cmds.append((1, 0.2, b))  # slow all directions
    cmds += [(0, -1.0, 4), (0, 0.0, 4), (17, -1.0, 4), (18, -1.0, 3)]
    for v in (0.15, 0.35, 0.5, 0.9):
        cmds.append((2, v, 4))
    return cmds


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    name = "distill_v9"
    pm, ps, nets, protos, gmap = load_router(name)
    resolver = load_resolver(
        os.path.join(LOCAL, "outputs", name),
        os.path.join(LOCAL, "outputs", "flat_battery_v9.json"),
    )
    env = make_env(0.0)
    results = {}
    for (m, s, b) in command_space():
        res = resolver.resolve(m, s, b)
        key = res.key
        gi = gmap[key]
        B = len(protos[gi])
        km, ks, kb = key
        feat = feat_for(
            dict(
                mode=km,
                speed=ks,
                mdir=[float(np.cos(bin_angle(kb))), float(np.sin(bin_angle(kb))), 0.0],
                fdir=[float(np.cos(bin_angle(kb))), float(np.sin(bin_angle(kb))), 0.0],
            )
        )
        sc_prev = {}

        def token_fn(e, t, seed, _pm=pm, _ps=ps, _nets=nets, _protos=protos, _gi=gi, _B=B):
            from apt_g1.eval_distill import hist_to_proprio

            prop = hist_to_proprio(e._get_sonic_history())
            x = np.concatenate([(prop - _pm) / _ps, feat]).astype(np.float32)
            with torch.no_grad():
                sc = _nets[_gi](torch.from_numpy(x[None]))[0].numpy().astype(np.float32)
            prev = sc_prev.get(seed)
            if prev is not None:
                sc = 0.3 * prev + 0.7 * sc
            sc_prev[seed] = sc
            phi = float(np.arctan2(sc[0], sc[1]))
            b_ = int(np.floor((phi + np.pi) / (2 * np.pi) * _B) % _B)
            return _protos[_gi][b_]

        per_seed = {}
        for seed in SEEDS:
            per_seed[f"s{seed}"] = run_one(env, 0.0, token_fn, seed, STEPS)
        ok = sum(
            1 for r in per_seed.values() if r["fall"] is None or r["fall"] >= 999
        )
        vx = round(float(np.mean([r["vx_est"] for r in per_seed.values()])), 3)
        disp = round(float(np.mean([r["disp"] for r in per_seed.values()])), 2)
        results[f"{m}_{s}_{b}"] = {
            "cmd": [m, s, b],
            "resolved": list(key),
            "degraded": res.degraded,
            "reason": res.reason,
            "completed": ok,
            "seeds": per_seed,
            "vx_mean": vx,
            "disp_mean": disp,
        }
        print(
            f"cmd({m},{s},{b}) -> {key} deg={int(res.degraded)} ok={ok}/3 "
            f"vx={vx} disp={disp}",
            flush=True,
        )
    json.dump(
        results,
        open(os.path.join(LOCAL, "outputs", "flat_battery_fallback_v9.json"), "w"),
        indent=1,
    )
    # summary
    n_fail = sum(1 for r in results.values() if r["completed"] < 3)
    slow = results.get("1_0.2_4")
    wback = results.get("2_-1.0_0")
    print("commands:", len(results), "fail(<3/3):", n_fail, flush=True)
    print(
        "slow_fwd moved:",
        bool(slow and (abs(slow["vx_mean"]) >= 0.15 or slow["disp_mean"] >= 3.0)),
        slow,
        flush=True,
    )
    print(
        "walk_back negative:",
        bool(wback and wback["disp_mean"] < 0),
        wback,
        flush=True,
    )
    print("saved flat_battery_fallback_v9.json")


if __name__ == "__main__":
    main()
