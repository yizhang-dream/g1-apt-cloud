"""Smoke test: create the APT Isaac env and step it a few times."""

from __future__ import annotations

import argparse
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--router-model-dir",
                    default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final")
    ap.add_argument(
        "--decoder-path",
        default=(
            "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
            "gear_sonic_deploy/policy/release/model_decoder.onnx"
        ),
    )
    cli = ap.parse_args()

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = cli.num_envs
    launcher_args.headless = cli.headless
    launcher_args.env_spacing = 4.0
    launcher_args.output_dir = "/tmp/isaac_smoke"
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg

    cfg = AptFlatG1EnvCfg()
    cfg.scene.num_envs = cli.num_envs
    cfg.sonic_decoder_path = cli.decoder_path
    cfg.router_model_dir = cli.router_model_dir
    cfg.vx_max = 0.8
    cfg.disturbance_prob = 0.0
    cfg.episode_length_s = 20.0
    env = AptFlatG1Env(cfg)
    print("[smoke] env created, num_envs=", env.num_envs)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    print("[smoke] obs shape", tuple(obs.shape), "device", obs.device)
    assert obs.shape[1] == cfg.observation_space

    t0 = time.time()
    fall = 0
    for i in range(cli.steps):
        action = torch.zeros(cli.num_envs, 14, device=env.device)
        obs_dict, rew, term, trunc, _ = env.step(action)
        fall += int(term.sum().item())
    dt = time.time() - t0
    h = float(env.robot.data.root_pos_w[:, 2].mean().item())
    print(f"[smoke] stepped {cli.steps} control steps in {dt:.2f}s "
          f"({cli.steps / dt:.1f} steps/s), falls={fall}, mean_h={h:.3f}")
    print("[smoke] OK")
    os._exit(0)


if __name__ == "__main__":
    main()
