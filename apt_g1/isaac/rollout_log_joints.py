"""Log a latent-policy rollout's joint trajectory for offline skeleton animation.

Camera-free (enable_cameras=False) so it avoids the Isaac viewport/hydra render
segfault on this server. Runs the E27/E29-style latent policy for N steps and
saves per-step base pose + 29 joint positions (SONIC G1_ISAACLab_ORDER) to npz.
Feed the npz to animate_skeleton.py for a matplotlib stick-figure video.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latent-mode", action="store_true")
    ap.add_argument(
        "--latent-vae-path",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/token_vae_e27/vae.pt",
    )
    ap.add_argument("--latent-speed-bins", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--command-vx", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--decoder-path",
        default=(
            "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
            "gear_sonic_deploy/policy/release/model_decoder.onnx"
        ),
    )
    ap.add_argument(
        "--router-model-dir",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final",
    )
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    from isaaclab.app import AppLauncher
    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = 1
    launcher_args.headless = True
    launcher_args.enable_cameras = False  # key: no render -> no hydra segfault
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.ppo_core import AptPPOPolicy
    from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg

    cfg = AptFlatG1EnvCfg()
    cfg.scene.num_envs = 1
    cfg.terrain = make_terrain_importer_cfg("plane", 0.0, seed=cli.seed)
    cfg.sonic_decoder_path = cli.decoder_path
    cfg.router_model_dir = cli.router_model_dir
    cfg.use_2hz_gate = True
    cfg.episode_length_s = 120.0
    if cli.latent_mode:
        cfg.latent_mode = True
        cfg.latent_vae_path = cli.latent_vae_path
        cfg.latent_speed_bins = cli.latent_speed_bins
        cfg.action_space = 16
        cfg.observation_space += 14

    env = AptFlatG1Env(cfg)
    policy = AptPPOPolicy(
        obs_dim=cfg.observation_space,
        aux_dim=12,
        hidden_dim=256,
        use_phase=not cfg.latent_mode,
        latent_dim=16 if cfg.latent_mode else 0,
    ).to("cuda:0")
    policy.load_state_dict(torch.load(cli.checkpoint, map_location="cuda:0"))
    policy.eval()

    obs_dict, _ = env.reset()
    env._last_obs = obs_dict["policy"]
    env._commands[0] = torch.tensor(
        [cli.command_vx, 0.0, 0.0], dtype=torch.float32, device=env.device
    )

    sonic_idx = env._body_idx  # maps SONIC order -> articulation order
    N = cli.steps
    base_xyz = np.zeros((N, 3), dtype=np.float32)
    base_quat = np.zeros((N, 4), dtype=np.float32)
    joint_pos = np.zeros((N, 29), dtype=np.float32)
    fell_at = -1

    with torch.no_grad():
        for i in range(N):
            act, _, _, _, _ = policy.act(env._last_obs, deterministic=True)
            if cli.latent_mode:
                action = act["phase"]
            else:
                action = torch.zeros(1, cfg.action_space, dtype=torch.float32, device=env.device)
            obs_dict, rew, term, trunc, _ = env.step(action)
            env._last_obs = obs_dict["policy"]
            base_xyz[i] = env.robot.data.root_pos_w[0, :3].detach().cpu().numpy()
            base_quat[i] = env.robot.data.root_quat_w[0, :4].detach().cpu().numpy()
            jp = env.robot.data.joint_pos[0]
            joint_pos[i] = jp[sonic_idx].detach().cpu().numpy()
            if bool(term[0]) and fell_at < 0:
                fell_at = i
            # keep the commanded vx pinned (some envs resample)
            env._commands[0, 0] = cli.command_vx

    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        base_xyz=base_xyz,
        base_quat=base_quat,
        joint_pos=joint_pos,
        command_vx=np.float32(cli.command_vx),
        fell_at=np.int32(fell_at),
        body_names=np.array(env._body_names),
    )
    disp = float(np.linalg.norm(base_xyz[-1, :2] - base_xyz[0, :2]))
    mean_vx = float((base_xyz[-1, 0] - base_xyz[0, 0]) / (N / 50.0))
    print(f"[rollout] steps={N} fell_at={fell_at} disp={disp:.2f}m mean_vx={mean_vx:.3f}")
    print(f"[rollout] saved {out}")
    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
