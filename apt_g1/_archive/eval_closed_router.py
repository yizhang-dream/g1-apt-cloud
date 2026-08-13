"""Re-evaluate a phase-router directory on A/B/C/D (no aux) with closed cycles."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

from apt_g1.encoder import Command, PhaseRouterEncoder
from apt_g1.encoder.phase_ar_encoder import PhaseAREncoder
from stress_test import make_env, run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-router-dir", default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_closed")
    ap.add_argument("--out", default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_closed/eval_closed.json")
    ap.add_argument("--ar", action="store_true")
    args = ap.parse_args()

    router = (
        PhaseAREncoder(args.phase_router_dir)
        if args.ar
        else PhaseRouterEncoder(args.phase_router_dir)
    )
    out = {}
    cmd_walk = Command.from_vxvy(0.8, 0.0)
    cmd_jump = Command(
        mode=17, speed=-1.0,
        mdir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        fdir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    # A. 60s walk
    env = make_env(70.0)
    out["A_walk60"] = {}
    for seed in [0, 1, 2]:
        r = run_episode(router, env, [(cmd_walk, 60)], 3000, seed)
        out["A_walk60"][f"seed{seed}"] = r
        print(f"A walk60 seed{seed} done={r['completed']} h_min={r['h_min']} vx={r['vx']} disp={r['displacement']}", flush=True)

    # B. disturbance grid (500 N)
    env = make_env(50.0)
    dirs = {"fwd": [500.0, 0, 0], "back": [-500.0, 0, 0], "left": [0, 500.0, 0], "right": [0, -500.0, 0]}
    out["B_disturbance"] = {}
    for dname, dvec in dirs.items():
        for seed in [0, 1, 2]:
            imp = [(500, dvec), (1250, dvec)]
            r = run_episode(router, env, [(cmd_walk, 45)], 2250, seed, impulses=imp)
            out["B_disturbance"][f"{dname}_seed{seed}"] = r
            print(f"B {dname} seed{seed} done={r['completed']} h_min={r['h_min']}", flush=True)

    # C. command-switch marathon (vx/vy)
    env = make_env(75.0)
    sched = [
        (Command.from_vxvy(0.0, 0.0), 5), (Command.from_vxvy(0.8, 0.0), 8), (Command.from_vxvy(0.0, 0.0), 3),
        (Command.from_vxvy(-0.8, 0.0), 6), (Command.from_vxvy(0.0, 0.0), 3), (Command.from_vxvy(0.25, 0.0), 6),
        (Command.from_vxvy(0.0, 0.0), 3), (Command.from_vxvy(0.25, -0.43), 6), (Command.from_vxvy(0.0, 0.0), 3),
        (Command.from_vxvy(0.25, 0.43), 6), (Command.from_vxvy(0.0, 0.0), 3), (Command.from_vxvy(0.8, 0.0), 8),
    ]
    out["C_switch"] = {}
    for seed in [0, 1, 2]:
        r = run_episode(router, env, sched, 3400, seed)
        out["C_switch"][f"seed{seed}"] = r
        print(f"C switch seed{seed} done={r['completed']} fall={r['fall_step']} h_min={r['h_min']}", flush=True)

    # D. jump 20s
    env = make_env(30.0)
    out["D_jump"] = {}
    for seed in [0, 1, 2]:
        r = run_episode(router, env, [(cmd_jump, 20)], 1000, seed)
        out["D_jump"][f"seed{seed}"] = r
        print(f"D jump seed{seed} done={r['completed']} h_min={r['h_min']} vx={r['vx']}", flush=True)

    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
