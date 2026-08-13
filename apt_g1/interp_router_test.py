"""Continuous-latent test: phase-interpolated prototype readout (v9 router).

Replaces the discrete 40-bin prototype lookup with a linear interpolation of
the two nearest prototypes weighted by the fractional phase.  The token stays
on the convex hull of the training prototypes (never leaves the manifold),
and the phase->token map becomes continuous -- the property the RL phase-line
(E3) was missing.  Tests closed-loop stability on flat (walk/slow/idle) and
measures phase->token smoothness.
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

import mujoco

from rough_sweep import load_router, feat_for, make_env
from apt_g1.eval_distill import hist_to_proprio

SEEDS = [0, 1, 2]
STEPS = 1200


def bin_angle(b):
    return b * np.pi / 4.0 - np.pi


def interp_token(protos, phi, B):
    x = (phi + np.pi) / (2 * np.pi) * B
    b0 = int(np.floor(x)) % B
    frac = x - np.floor(x)
    b1 = (b0 + 1) % B
    return (1.0 - frac) * protos[b0] + frac * protos[b1]


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    pm, ps, nets, protos, gmap = load_router("distill_v9")
    env = make_env(0.0)
    results = {}
    for tag, m, s, b in [
        ("walk_fwd", 2, -1.0, 4),
        ("slow06_fwd", 1, 0.6, 4),
        ("idle", 0, -1.0, 4),
    ]:
        gi = gmap[(m, s, b)]
        B = len(protos[gi])
        feat = feat_for(
            dict(
                mode=m,
                speed=s,
                mdir=[float(np.cos(bin_angle(b))), float(np.sin(bin_angle(b))), 0.0],
                fdir=[float(np.cos(bin_angle(b))), float(np.sin(bin_angle(b))), 0.0],
            )
        )
        per = {}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            env.reset()
            env.data.qpos[2] = 0.76
            env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(
                np.float32
            )
            env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
            mujoco.mj_forward(env.model, env.data)
            env._reset_history()
            env._fill_history_from_state()
            sc_prev = None
            heights, xs, ys = [], [], []
            fall = None
            for t in range(STEPS):
                xs.append(float(env.data.qpos[0]))
                ys.append(float(env.data.qpos[1]))
                prop = hist_to_proprio(env._get_sonic_history())
                x = np.concatenate([(prop - pm) / ps, feat]).astype(np.float32)
                with torch.no_grad():
                    sc = nets[gi](torch.from_numpy(x[None]))[0].numpy().astype(np.float32)
                if sc_prev is not None:
                    sc = 0.3 * sc_prev + 0.7 * sc
                sc_prev = sc
                phi = float(np.arctan2(sc[0], sc[1]))
                token = interp_token(protos[gi], phi, B).astype(np.float32)
                obs, reward, terminated, info = env.step(
                    {"token": token, "aux": np.zeros(12, dtype=np.float32)}
                )
                heights.append(float(env.data.qpos[2]))
                if terminated:
                    fall = t
                    break
            n_steps = STEPS if fall is None else fall
            dx = float(env.data.qpos[0] - xs[0]) if fall is None else float(xs[fall] - xs[0])
            dy = float(env.data.qpos[1] - ys[0]) if fall is None else float(ys[fall] - ys[0])
            per[f"s{seed}"] = {
                "fall": fall,
                "h_min": round(float(min(heights)), 3),
                "disp": round(float(np.hypot(dx, dy)), 2),
                "vx_est": round(float(np.hypot(dx, dy) / max(1e-6, n_steps * 0.02)), 3),
            }
            print(tag, "seed", seed, per[f"s{seed}"], flush=True)
        results[tag] = per
    # smoothness: token distance for a phase step of one 40th of a cycle
    gi = gmap[(2, -1.0, 4)]
    B = len(protos[gi])
    dphi = 2 * np.pi / B
    for phi in np.linspace(-np.pi, np.pi, 9):
        t0 = interp_token(protos[gi], float(phi), B)
        t1 = interp_token(protos[gi], float(phi + dphi), B)
        print(
            "smoothness phi=%.2f d_token=%.4f (discrete bin-jump would be ~%.4f)"
            % (phi, float(np.linalg.norm(t1 - t0)), float(np.linalg.norm(protos[gi][1] - protos[gi][0]))),
            flush=True,
        )
    json.dump(
        results,
        open(os.path.join(LOCAL, "outputs", "interp_router_flat.json"), "w"),
        indent=1,
    )
    print("saved interp_router_flat.json")


if __name__ == "__main__":
    main()
