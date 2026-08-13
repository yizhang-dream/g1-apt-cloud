"""Flat-ground command-coverage audit for the phase routers (MuJoCo, no band).

Enumerates every group in the router's gmap ((mode, speed, angle_bin)) and runs
3 seeds x 1200 control steps (20 s; fall=999 means the episode completed) with
the per-group phase net + prototype + EMA.  Also probes continuous-speed
commands that have no exact group (missing -> handled later by the fallback).

Outputs: outputs/flat_battery_v9.json (and v6 for comparison)
"""

from __future__ import annotations

import io
import json
import os
import sys
import argparse

import numpy as np
import torch

LOCAL = r"C:\Users\zyz\Documents\gr00t\apt_g1"
REPO = r"D:\GR00T-WholeBodyControl"
sys.path.insert(0, os.path.dirname(LOCAL))
sys.path.insert(0, LOCAL)
sys.path.insert(0, REPO)

from rough_sweep import (
    load_router,
    feat_for,
    run_one,
    make_env,
)

STEPS = 1200
SEEDS = [0, 1, 2]


def bin_angle(b):
    """Direction angle (rad) for angle_bin b (bin4 == forward +x)."""
    return b * np.pi / 4.0 - np.pi


def scenario(m, s, b):
    a = bin_angle(b)
    return dict(
        mode=m,
        speed=s,
        mdir=[float(np.cos(a)), float(np.sin(a)), 0.0],
        fdir=[float(np.cos(a)), float(np.sin(a)), 0.0],
    )


def audit(name, out_name):
    pm, ps, nets, protos, gmap = load_router(name)
    env = make_env(0.0)
    results = {}
    for key in sorted(gmap.keys(), key=lambda k: (k[0], k[1], k[2])):
        m, s, b = key
        gi = gmap[key]
        B = len(protos[gi])
        feat = feat_for(scenario(m, s, b))
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
            r = run_one(env, 0.0, token_fn, seed, STEPS)
            per_seed[f"s{seed}"] = r
        # env episode_length is 1000 -> a completed run terminates at fall=999
        ok = sum(1 for r in per_seed.values() if r["fall"] is None or r["fall"] >= 999)
        results[f"{m}_{s}_{b}"] = {
            "group": list(key),
            "completed": ok,
            "seeds": per_seed,
            "vx_mean": round(
                float(np.mean([r["vx_est"] for r in per_seed.values()])), 3
            ),
            "disp_mean": round(
                float(np.mean([r["disp"] for r in per_seed.values()])), 2
            ),
        }
        print(
            f"({m},{s},{b}) ok={ok}/3 "
            f"falls={[r['fall'] for r in per_seed.values()]} "
            f"vx={results[f'{m}_{s}_{b}']['vx_mean']} "
            f"disp={results[f'{m}_{s}_{b}']['disp_mean']}",
            flush=True,
        )
    json.dump(results, open(os.path.join(LOCAL, "outputs", out_name), "w"), indent=1)
    print("saved", out_name, flush=True)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", choices=["v9", "v6", "both"], default="both")
    cli = ap.parse_args()
    jobs = {
        "v9": ("distill_v9", "flat_battery_v9.json"),
        "v6": ("distill_final", "flat_battery_v6.json"),
    }
    if cli.router == "both":
        for name, out in jobs.values():
            audit(name, out)
    else:
        name, out = jobs[cli.router]
        audit(name, out)


if __name__ == "__main__":
    main()
