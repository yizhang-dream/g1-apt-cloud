"""Render a short walk video from the APT Isaac env (best effort).

Uses a TiledCamera behind the robot; saves rgb frames to
<out_dir>/frames/*.png and an mp4 if ffmpeg is available.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default="/home/cvgluser/ros2_data/apt_g1/outputs/isaac_walk_render")
    ap.add_argument("--terrain", choices=["plane", "rough"], default="plane")
    ap.add_argument("--terrain-noise", type=float, default=0.04)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--use-elevation", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--use-aux", type=int, default=0)
    ap.add_argument("--label", default="")
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
    ap.add_argument("--latent-mode", action="store_true")
    ap.add_argument(
        "--latent-vae-path",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/token_vae_e27/vae.pt",
    )
    cli = ap.parse_args()

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = 1
    launcher_args.headless = cli.headless
    launcher_args.enable_cameras = True
    launcher_args.env_spacing = 4.0
    launcher_args.output_dir = cli.out
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.ppo_core import AptPPOPolicy
    from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg

    cfg = AptFlatG1EnvCfg()
    cfg.scene.num_envs = 1
    cfg.terrain = make_terrain_importer_cfg(
        cli.terrain, cli.terrain_noise, seed=cli.terrain_seed
    )
    cfg.sonic_decoder_path = cli.decoder_path
    cfg.router_model_dir = cli.router_model_dir
    cfg.use_2hz_gate = True
    cfg.use_elevation = bool(cli.use_elevation)
    if cfg.use_elevation:
        cfg.observation_space += cfg.elev_grid * cfg.elev_grid
    cfg.episode_length_s = 120.0
    if cli.latent_mode:
        cfg.latent_mode = True
        cfg.latent_vae_path = cli.latent_vae_path
        cfg.action_space = 16  # latent z only (no aux / gate)
        cfg.observation_space += 14  # _last_phase 2 -> 16 in the observation
    np.random.seed(cli.terrain_seed)
    env = AptFlatG1Env(cfg)

    policy = None
    if cli.checkpoint:
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space,
            aux_dim=12,
            hidden_dim=256,
            use_phase=not cfg.use_gate_sel and not cfg.latent_mode,
            latent_dim=16 if cfg.latent_mode else 0,
        ).to("cuda:0")
        policy.load_state_dict(torch.load(cli.checkpoint, map_location="cuda:0"))
        policy.eval()

    # attach a follow camera behind the robot
    cam_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        height=360,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.5, 0.0, 1.4),
            rot=(0.996, 0.0, 0.089, 0.0),
            convention="world",
        ),
    )
    from isaaclab.sensors import TiledCamera

    env.scene._sensors["walkcam"] = TiledCamera(cam_cfg)
    env.scene._sensors["walkcam"]._initialize_impl()
    env.scene._sensors["walkcam"].reset()

    obs_dict, _ = env.reset()
    env._last_obs = obs_dict["policy"]
    out_dir = Path(cli.out)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # fixed command: walk 0.8 m/s
    if not cli.latent_mode:
        env.router_commands[0] = None
    env._commands[0] = torch.tensor([0.8, 0.0, 0.0], dtype=torch.float32, device=env.device)

    save_every = 3  # 50 Hz -> 16.7 fps
    count = 0
    for i in range(cli.steps):
        if policy is not None:
            with torch.no_grad():
                act, _, _, _, _ = policy.act(env._last_obs, deterministic=True)
            if cli.latent_mode:
                action = act["phase"]  # (1, 16) latent z -> VAE -> token -> SONIC
            elif cli.use_aux:
                action = torch.zeros(1, 14, dtype=torch.float32, device=env.device)
                action[:, 2:] = act["aux"]
            else:
                action = torch.zeros(1, 14, dtype=torch.float32, device=env.device)
        else:
            action = torch.zeros(1, cfg.action_space, dtype=torch.float32, device=env.device)
        obs_dict, rew, term, trunc, _ = env.step(action)
        env._last_obs = obs_dict["policy"]
        if term.any():
            print(f"[render] fell at step {i}", flush=True)
            break
        if i % save_every == 0:
            env.sim.render()
            import omni.replicator.core as rep

            rep.orchestrator.step()
            env.scene._sensors["walkcam"].update(env.cfg.sim.dt)
            rgb = env.scene._sensors["walkcam"].data.output["rgb"][0].cpu().numpy()
            from PIL import Image

            Image.fromarray(rgb).save(frame_dir / f"frame_{count:04d}.png")
            count += 1

    # quick brightness sanity + ffmpeg assembly
    import glob

    files = sorted(glob.glob(str(frame_dir / "*.png")))
    print("[render] frames:", len(files), flush=True)
    if files:
        from PIL import Image

        a = np.asarray(Image.open(files[len(files) // 2])).astype(np.float32)
        print("[render] mid-frame mean brightness:", round(float(a.mean()), 2), flush=True)
    os.system(
        f"ffmpeg -y -framerate 16 -i {frame_dir}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p {out_dir / 'walk.mp4'} >/dev/null 2>&1"
    )
    print("[render] mp4:", (out_dir / "walk.mp4").exists(), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
