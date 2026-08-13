"""Guarded eval: only runs the requested test sections (A/B/C/D)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tests", default="A,B,C,D")
    ap.add_argument("--env", choices=["apt", "vanilla"], default="apt")
    ap.add_argument(
        "--terrain",
        choices=["plane", "rough", "stairs", "stairs_hi", "stones", "discrete"],
        default="plane",
    )
    ap.add_argument("--terrain-noise", type=float, default=0.04)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--keys", default="aux,noaux")
    ap.add_argument("--use-elevation", type=int, default=0)
    ap.add_argument("--gate-sel", type=int, default=0)
    ap.add_argument("--phase-mode", action="store_true")
    ap.add_argument("--use-2hz-gate", type=int, default=1)
    ap.add_argument(
        "--router-model-dir",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final",
    )
    ap.add_argument(
        "--decoder-path",
        default=(
            "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
            "gear_sonic_deploy/policy/release/model_decoder.onnx"
        ),
    )
    cli = ap.parse_args()
    tests = set(cli.tests.split(","))
    keys = [k for k in cli.keys.split(",") if k]

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = 1
    launcher_args.headless = cli.headless
    launcher_args.env_spacing = 4.0
    launcher_args.output_dir = str(Path(cli.out).parent)
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    from apt_g1.encoder import Command
    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.apt_flat_env_vanilla import (
        AptFlatG1VanillaEnv,
        AptFlatG1VanillaEnvCfg,
    )
    from apt_g1.isaac.eval_apt_isaac import rollout
    from apt_g1.isaac.ppo_core import AptPPOPolicy
    from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg

    if cli.env == "vanilla":
        cfg = AptFlatG1VanillaEnvCfg()
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space, aux_dim=29, use_phase=False
        ).to("cuda:0")
    else:
        cfg = AptFlatG1EnvCfg()
        cfg.use_elevation = bool(cli.use_elevation)
        if cfg.use_elevation:
            cfg.observation_space += cfg.elev_grid * cfg.elev_grid
        cfg.use_gate_sel = bool(cli.gate_sel)
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space,
            aux_dim=12,
            gate_k=3 if cfg.use_gate_sel else 0,
            use_phase=not cfg.use_gate_sel,
        ).to("cuda:0")
    cfg.scene.num_envs = 1
    cfg.terrain = make_terrain_importer_cfg(
        cli.terrain, cli.terrain_noise, seed=cli.terrain_seed
    )
    cfg.sonic_decoder_path = cli.decoder_path
    cfg.router_model_dir = cli.router_model_dir
    cfg.use_2hz_gate = bool(cli.use_2hz_gate)
    cfg.phase_mode = cli.phase_mode
    cfg.disturbance_prob = 0.0
    cfg.episode_length_s = 120.0
    # HfRandomUniformTerrainCfg samples heights from the GLOBAL numpy RNG (it
    # ignores TerrainGeneratorCfg.seed), so pin the global state to the terrain
    # seed right before env creation to make terrain reproducible.
    np.random.seed(cli.terrain_seed)
    if cli.env == "vanilla":
        env = AptFlatG1VanillaEnv(cfg)
    else:
        env = AptFlatG1Env(cfg)

    try:
        policy.load_state_dict(torch.load(cli.checkpoint, map_location="cuda:0"))
        policy.eval()
        print("[eval] loaded checkpoint", cli.checkpoint)
    except FileNotFoundError:
        print("[eval] WARNING: checkpoint not found, aux=0 only")

    env._vanilla = cli.env == "vanilla"
    env._gate_policy = bool(cli.gate_sel)
    obs_dict, _ = env.reset()
    env._last_obs = obs_dict["policy"]

    pp = cli.phase_mode
    out = {}

    if "A" in tests:
        out["A_walk60"] = {}
        for key in keys:
            out["A_walk60"][key] = {}
            use_aux = key != "noaux"
            for seed in [0, 1, 2]:
                r = rollout(env, policy, [(0.8, 0.0, 60)], seed, use_aux, phase_policy=pp)
                out["A_walk60"][key][f"seed{seed}"] = r
                print(f"A walk60 {key} seed{seed} done={r['completed']} h_min={r['h_min']} vx={r['vx']} disp={r['disp']}", flush=True)

    if "B" in tests:
        out["B_disturbance"] = {}
        dirs = {"fwd": [500.0, 0, 0], "back": [-500.0, 0, 0], "left": [0, 500.0, 0], "right": [0, -500.0, 0]}
        for key in keys:
            out["B_disturbance"][key] = {}
            use_aux = key != "noaux"
            for dname, dvec in dirs.items():
                for seed in [0, 1, 2]:
                    imp = [(500, dvec), (1250, dvec)]
                    r = rollout(env, policy, [(0.8, 0.0, 45)], seed, use_aux, impulses=imp, phase_policy=pp)
                    out["B_disturbance"][key][f"{dname}_seed{seed}"] = r
                    print(f"B {dname} {key} seed{seed} done={r['completed']} h_min={r['h_min']}", flush=True)

    if "C" in tests:
        out["C_switch"] = {}
        sched = [
            (0.0, 0.0, 5), (0.8, 0.0, 8), (0.0, 0.0, 3), (-0.8, 0.0, 6),
            (0.0, 0.0, 3), (0.25, 0.0, 6), (0.0, 0.0, 3), (0.25, -0.43, 6),
            (0.0, 0.0, 3), (0.25, 0.43, 6), (0.0, 0.0, 3), (0.8, 0.0, 8),
        ]
        for key in keys:
            out["C_switch"][key] = {}
            use_aux = key != "noaux"
            for seed in [0, 1, 2]:
                r = rollout(env, policy, [(vx, vy, s) for vx, vy, s in sched], seed, use_aux, phase_policy=pp)
                out["C_switch"][key][f"seed{seed}"] = r
                print(f"C switch {key} seed{seed} done={r['completed']} fall={r['fall_step']} h_min={r['h_min']}", flush=True)

    if "D" in tests:
        out["D_jump"] = {}
        jump_cmd = Command(
            mode=17, speed=-1.0,
            mdir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            fdir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        for key in keys:
            out["D_jump"][key] = {}
            use_aux = key != "noaux"
            for seed in [0, 1, 2]:
                r = rollout(env, policy, [(jump_cmd, 20)], seed, use_aux, phase_policy=pp)
                out["D_jump"][key][f"seed{seed}"] = r
                print(f"D jump {key} seed{seed} done={r['completed']} h_min={r['h_min']} vx={r['vx']}", flush=True)

    with open(cli.out, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", cli.out)
    os._exit(0)


if __name__ == "__main__":
    import os

    main()
