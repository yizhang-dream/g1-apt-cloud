"""Native same-task eval of the mjlab M-FROM0 policy under OUR task condition:
60 s straight command (vx 0.8, yaw 0) in mjlab's own G1-Flat velocity env.

This is the "non-SONIC" anchor evaluated in its native sim with our test
condition, avoiding cross-stack wrapper confounds. Reports survival (falls?),
net displacement and mean forward speed per seed.

Usage (server, .venv_mjlab, from unitree_rl_mjlab):
    python eval_mjlab_fwd.py [seed] [steps]
"""
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

sys.path.insert(0, "/home/cvgluser/ros2_data/unitree_rl_mjlab")
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from src.tasks.velocity.rl import VelocityOnPolicyRunner

CKPT = ("/home/cvgluser/ros2_data/unitree_rl_mjlab/logs/rsl_rl/"
        "g1_velocity/2026-08-14_00-52-58/model_6499.pt")
DEVICE = "cuda:0"


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    configure_torch_backends()
    env_cfg = unitree_g1_flat_env_cfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 60.0  # no truncation before 60 s
    # disable command resampling BEFORE the term is built (the term caches the
    # resample interval at construction)
    env_cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
    if noise > 0.0:
        # flat task env (no terrain scan -> blind policy, obs matches ckpt),
        # with the ground replaced by a fixed-noise rough heightfield
        from mjlab.terrains import HfRandomUniformTerrainCfg
        from mjlab.terrains.config import TerrainGeneratorCfg
        tg = TerrainGeneratorCfg(
            seed=0, curriculum=False, size=(8.0, 8.0), border_width=20.0,
            num_rows=10, num_cols=20,
            sub_terrains={"random_rough": HfRandomUniformTerrainCfg(
                proportion=1.0, size=(8.0, 8.0), noise_range=(noise, noise),
                noise_step=0.01, horizontal_scale=0.1, vertical_scale=0.005,
            )},
        )
        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.terrain_generator = tg
        print(f"[mjlab-eval] terrain -> rough noise {noise}", flush=True)
    raw = ManagerBasedRlEnv(
        env_cfg, device=DEVICE,
        render_mode="rgb_array" if (len(sys.argv) > 4 and sys.argv[4] == "video") else None,
    )
    want_video = len(sys.argv) > 4 and sys.argv[4] == "video"
    if want_video:
        from mjlab.utils.wrappers import VideoRecorder
        vdir = f"/home/cvgluser/ros2_data/apt_g1/outputs/mjlab_terrain_videos_r{noise:.2f}"
        raw = VideoRecorder(
            raw,
            video_folder=str(Path(vdir)),
            step_trigger=lambda step: step == 0,
            video_length=400,
            disable_logger=True,
        )
        print(f"[mjlab-eval] video recording on -> {vdir}", flush=True)
    env = RslRlVecEnvWrapper(raw, clip_actions=None)
    runner = VelocityOnPolicyRunner(
        env, asdict(unitree_g1_ppo_runner_cfg()), str(Path("logs/eval_tmp")), DEVICE
    )
    runner.load(CKPT)
    policy = runner.get_inference_policy()

    # pin a fixed forward command AFTER reset (reset resamples the command)
    raw.seed(seed)
    obs, _ = env.reset()
    cm = raw.command_manager
    twist = cm.get_term("twist")
    twist.vel_command_b[:] = torch.tensor([0.8, 0.0, 0.0], device=DEVICE)
    robot = raw.scene["robot"]
    # pin heading target to the robot's current yaw -> pure straight command
    quat_names = ["root_link_quat_w", "root_quat_w", "quat_w"]
    q_attr = next((n for n in quat_names if hasattr(robot.data, n)), None)
    if q_attr is not None and hasattr(twist, "heading_target"):
        q = getattr(robot.data, q_attr)[0].detach().cpu().numpy()
        w, x, y, z = q
        yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        twist.heading_target[:] = torch.tensor([yaw], device=DEVICE)
        print(f"[mjlab-eval] pinned heading target to {yaw:.3f} rad", flush=True)
    else:
        print(f"[mjlab-eval] warn: heading pin skipped (q_attr={q_attr})", flush=True)

    pos_names = ["root_pos_w", "root_pos", "root_link_pos_w", "xpos"]
    vel_names = ["root_link_lin_vel_b", "root_lin_vel_b",
                 "root_link_lin_vel_w", "root_lin_vel_w", "lin_vel_w"]
    pos_attr = next((n for n in pos_names if hasattr(robot.data, n)), None)
    vel_attr = next((n for n in vel_names if hasattr(robot.data, n)), None)
    print(f"[mjlab-eval] seed={seed} steps={N} pos_attr={pos_attr} vel_attr={vel_attr}",
          flush=True)

    def get_pos():
        if pos_attr is not None:
            return getattr(robot.data, pos_attr)
        return raw.sim.data.qpos[0:3]

    def get_vel():
        if vel_attr is not None:
            return getattr(robot.data, vel_attr)
        return raw.sim.data.qvel[0:3]

    fell = None
    p0 = None
    p_last = None
    vxs = []
    with torch.no_grad():
        for t in range(N):
            # read position BEFORE stepping: on the truncation step the env
            # resets inside step(), so a post-step read would be the reset pose
            p = get_pos()
            if p0 is None:
                p0 = p.detach().cpu().numpy().copy()
            p_last = p.detach().cpu().numpy().copy()
            act = policy(obs)
            obs, rew, term, extras = env.step(act)
            v = get_vel()
            vxs.append(float(v[0, 0].item()))
            if bool(term.any()):
                fell = t
                break
    p1 = p_last if p_last is not None else get_pos().detach().cpu().numpy()
    disp = float(np.linalg.norm(p1 - p0))
    mean_vx = float(np.mean(vxs)) if vxs else 0.0
    print(f"[mjlab-eval] RESULT seed={seed} fell={fell} steps_done={len(vxs)} "
          f"disp={disp:.2f}m mean_vx={mean_vx:.3f}", flush=True)


if __name__ == "__main__":
    main()
