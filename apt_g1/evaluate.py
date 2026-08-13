"""Deterministic MuJoCo rollout for a saved APT policy."""

from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--config", default="apt_g1/configs/flat_g1.yaml")
    parser.add_argument("--no-band", action="store_true", help="Disable elastic band")
    args = parser.parse_args()

    import yaml

    from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
    from apt_g1.policies.apt_policy import APTPolicy
    from apt_g1.sonic.sonic_wrapper import SonicOnnxDecoder

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    decoder = SonicOnnxDecoder(args.onnx_path)
    motion_cfg = cfg.get("motion", {})
    reference_tokens = None
    base_token = None
    if motion_cfg.get("reference_tokens_path"):
        reference_tokens = np.load(motion_cfg["reference_tokens_path"])
    if motion_cfg.get("base_token_path"):
        base_token = np.load(motion_cfg["base_token_path"])
    token_vae = None
    if motion_cfg.get("token_vae_path"):
        from apt_g1.sonic.token_vae import TokenVAE

        token_vae = TokenVAE(token_dim=64, latent_dim=cfg["apt"]["latent_dim"])
        token_vae.load_state_dict(torch.load(motion_cfg["token_vae_path"], map_location="cpu"))
        token_vae.eval()
    skill_tokens = None
    if motion_cfg.get("skill_tokens_path"):
        skill_tokens = np.load(motion_cfg["skill_tokens_path"])
    token_seq_vae = None
    if motion_cfg.get("token_seq_vae_path"):
        from apt_g1.sonic.token_seq_vae import TokenSeqVAE

        token_seq_vae = TokenSeqVAE(
            token_dim=64,
            latent_dim=cfg["apt"]["latent_dim"],
            window=motion_cfg.get("vae_window", 10),
        )
        token_seq_vae.load_state_dict(
            torch.load(motion_cfg["token_seq_vae_path"], map_location="cpu")
        )
        token_seq_vae.eval()
    joint_seq_vae = None
    if motion_cfg.get("joint_seq_vae_path"):
        from apt_g1.sonic.token_seq_vae import TokenSeqVAE

        joint_seq_vae = TokenSeqVAE(
            token_dim=29,
            latent_dim=cfg["apt"]["latent_dim"],
            window=motion_cfg.get("vae_window", 10),
        )
        joint_seq_vae.load_state_dict(
            torch.load(motion_cfg["joint_seq_vae_path"], map_location="cpu")
        )
        joint_seq_vae.eval()
    use_elastic_band = False if args.no_band else cfg["mujoco"].get("use_elastic_band", False)
    band_scale = 0.0 if args.no_band else cfg["mujoco"].get("band_scale", 1.0)
    env = MujocoG1FlatEnv(
        sonic_decoder=decoder,
        repo_root=args.repo_root,
        robot_scene=cfg["mujoco"]["robot_scene"],
        wbc_config_path=cfg["mujoco"]["wbc_config_path"],
        sim_dt=cfg["mujoco"]["sim_dt"],
        control_decimation=cfg["mujoco"]["control_decimation"],
        episode_length_s=cfg["mujoco"]["episode_length_s"],
        aux_scale=cfg["apt"]["aux_scale"],
        stand_only=cfg["commands"].get("stand_only", False),
        use_elastic_band=use_elastic_band,
        band_scale=band_scale,
        command_vx_min=cfg["commands"].get("vx_min", 0.0),
        command_vx_max=cfg["commands"].get("vx_max", 1.0),
        command_vy_min=cfg["commands"].get("vy_min", -0.5),
        command_vy_max=cfg["commands"].get("vy_max", 0.5),
        command_yaw_min=cfg["commands"].get("yaw_min", -0.5),
        command_yaw_max=cfg["commands"].get("yaw_max", 0.5),
        reference_token_sequence=reference_tokens,
        residual_scale=motion_cfg.get("residual_scale", 0.1),
        base_token=base_token,
        token_vae=token_vae,
        skill_tokens=skill_tokens,
        token_seq_vae=token_seq_vae,
        add_phase_obs=motion_cfg.get("add_phase_obs", True),
        joint_seq_vae=joint_seq_vae,
        joint_reset_path=motion_cfg.get("joint_reset_path"),
    )

    policy = APTPolicy(
        obs_dim=(
            100
            if motion_cfg.get("reference_tokens_path")
            and motion_cfg.get("add_phase_obs", True)
            else 99
        ),
        token_dim=cfg["apt"]["latent_dim"],
        aux_dim=cfg["apt"]["aux_dim"],
        num_skills=cfg["apt"]["num_skills"],
    )
    policy.load_state_dict(torch.load(args.policy, map_location="cpu"))
    policy.eval()

    obs = env.reset()
    total_reward = 0.0
    for step in range(args.steps):
        action, _, _ = policy.act(
            torch.as_tensor(obs, dtype=torch.float32),
            deterministic=True,
        )
        obs, reward, done, _ = env.step(
            {
                "token": action["token"].detach().numpy(),
                "aux": action["aux"].detach().numpy(),
                "skill": action["skill"].detach().numpy(),
            }
        )
        total_reward += float(reward)
        if done:
            print(f"done at control step {step} (~{step * 0.02:.2f}s)")
            break
    else:
        print(f"survived all {args.steps} control steps (~{args.steps * 0.02:.2f}s)")
    print(f"total reward: {total_reward:.3f}")
    print(f"final root height: {env.data.qpos[2]:.3f} m")


if __name__ == "__main__":
    main()
