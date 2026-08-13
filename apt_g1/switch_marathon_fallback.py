"""60s+ command-switch marathon through the StableResolver (flat, no band).

Schedule S1 (no jump): idle 5s -> slow_fwd 10s -> walk_fwd 10s -> walk_back
10s -> slow 0.6 fwd 10s -> idle 5s -> walk_fwd 10s -> walk bin5 (degraded) 5s.
Schedule S2: S1 + jump 5s at the end.

Criterion (priority-2): S1 3/3 x 65 s; S2 recorded (jump is teacher-marginal).
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
from router_fallback import load_resolver


def bin_angle(b):
    return b * np.pi / 4.0 - np.pi


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    name = "distill_v9"
    pm, ps, nets, protos, gmap = load_router(name)
    resolver = load_resolver(
        os.path.join(LOCAL, "outputs", name),
        os.path.join(LOCAL, "outputs", "flat_battery_v9.json"),
    )
    env = make_env(0.0)
    env.episode_length = 100000
    schedules = {
        "S1_nojump": [
            (0, -1.0, 4, 250), (1, 0.2, 4, 500), (2, -1.0, 4, 500),
            (2, -1.0, 0, 500), (1, 0.6, 4, 500), (0, -1.0, 4, 250),
            (2, -1.0, 4, 500), (2, -1.0, 5, 250),
        ],
        "S2_jump": [
            (0, -1.0, 4, 250), (1, 0.2, 4, 500), (2, -1.0, 4, 500),
            (2, -1.0, 0, 500), (1, 0.6, 4, 500), (17, -1.0, 4, 250),
        ],
    }
    results = {}
    for sname, sched in schedules.items():
        per_seed = {}
        for seed in [0, 1, 2]:
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
            sc_prev = {}
            fall = None
            cmd_idx = 0
            steps_in_cmd = 0
            heights = []
            total = sum(x[3] for x in sched)
            for t in range(total):
                m, s, b, dur = sched[cmd_idx]
                if steps_in_cmd >= dur:
                    cmd_idx += 1
                    steps_in_cmd = 0
                    if cmd_idx >= len(sched):
                        break
                    m, s, b, dur = sched[cmd_idx]
                res = resolver.resolve(m, s, b)
                km, ks, kb = res.key
                gi = gmap[res.key]
                B = len(protos[gi])
                feat = feat_for(
                    dict(
                        mode=km,
                        speed=ks,
                        mdir=[float(np.cos(bin_angle(kb))), float(np.sin(bin_angle(kb))), 0.0],
                        fdir=[float(np.cos(bin_angle(kb))), float(np.sin(bin_angle(kb))), 0.0],
                    )
                )
                from apt_g1.eval_distill import hist_to_proprio

                prop = hist_to_proprio(env._get_sonic_history())
                x = np.concatenate([(prop - pm) / ps, feat]).astype(np.float32)
                with torch.no_grad():
                    sc = nets[gi](torch.from_numpy(x[None]))[0].numpy().astype(np.float32)
                prev = sc_prev.get((cmd_idx, seed))
                if prev is not None:
                    sc = 0.3 * prev + 0.7 * sc
                sc_prev[(cmd_idx, seed)] = sc
                phi = float(np.arctan2(sc[0], sc[1]))
                b_ = int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)
                obs, reward, terminated, info = env.step(
                    {"token": protos[gi][b_], "aux": np.zeros(12, dtype=np.float32)}
                )
                heights.append(float(env.data.qpos[2]))
                steps_in_cmd += 1
                if terminated:
                    fall = t
                    break
            per_seed[f"s{seed}"] = {
                "fall": fall,
                "h_min": round(float(min(heights)), 3),
                "steps": t + 1,
                "completed": fall is None,
            }
            print(sname, "seed", seed, per_seed[f"s{seed}"], flush=True)
        results[sname] = per_seed
    json.dump(
        results,
        open(os.path.join(LOCAL, "outputs", "switch_marathon_fallback.json"), "w"),
        indent=1,
    )
    print("saved switch_marathon_fallback.json")


if __name__ == "__main__":
    main()
