"""Replay an Isaac rollout (npz from rollout_log_joints.py) in MuJoCo and render
a REAL 3D G1 model video, offscreen (EGL, no display server needed).

The npz holds per-step base pose + 29 joint angles (SONIC/IsaacLab order) logged
from the Isaac RL rollout; here we set the same state in the MuJoCo G1 model and
render camera frames with mujoco.Renderer, then encode an mp4 with the bundled
imageio-ffmpeg.

Run on the server with .venv_mjlab and MUJOCO_GL=egl:
    MUJOCO_GL=egl .venv_mjlab/bin/python replay_render_mujoco.py \
        --npz outputs/e29_rollout.npz --out outputs/e29_mujoco.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# G1 ordering constants (same as envs/mujoco_g1_flat_env.py).
G1_ISAACLAB_TO_MUJOCO_DOF = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16,
    20, 22, 24, 26, 28,
]

MODEL_PATH = (
    "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
    "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="outputs/replay.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=640)  # GLContext framebuffer cap
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--view", choices=["side", "threequarter"], default="threequarter")
    ap.add_argument("--cam-dist", type=float, default=2.6)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    import mujoco

    d = np.load(args.npz)
    base = d["base_xyz"]
    quat = d["base_quat"]
    jp = d["joint_pos"]  # (N, 29) SONIC order
    N = len(base)
    fell_at = int(d["fell_at"])
    print(f"[replay] {N} steps, fell_at={fell_at}, "
          f"mean_vx={(base[-1,0]-base[0,0])/(N/50.0):.3f} m/s")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # joint qpos addresses for the 29 body joints (skipping hands), in MuJoCo order
    body_qpos_adr = []
    for act_id in range(model.nu):
        jid = model.actuator_trnid[act_id, 0]
        if "hand" in model.joint(jid).name:
            continue
        body_qpos_adr.append(model.jnt_qposadr[jid])
    body_qpos_adr = np.asarray(body_qpos_adr, dtype=int)
    isaac_to_mujoco = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF, dtype=int)

    # camera setup (follow the robot)
    cam = mujoco.MjvCamera()
    cam.distance = args.cam_dist
    cam.elevation = -12.0
    cam.azimuth = 90.0 if args.view == "side" else 135.0
    cam.lookat = base[0].copy()

    renderer = mujoco.Renderer(model, args.height, args.width)
    frames = []
    for i in range(N):
        # free joint: base pose (MuJoCo free qpos = xyz + wxyz quat)
        data.qpos[0:3] = base[i]
        data.qpos[3:7] = quat[i]
        # body joints: SONIC order -> MuJoCo order -> qpos addresses
        data.qpos[body_qpos_adr] = jp[i][isaac_to_mujoco]
        mujoco.mj_forward(model, data)
        cam.lookat = base[i]
        renderer.update_scene(data, cam)
        frames.append(renderer.render().copy())
    renderer.close()
    print(f"[replay] rendered {len(frames)} frames {frames[0].shape}")

    # encode mp4 with bundled ffmpeg
    import imageio.v2 as iio

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(
        str(out), fps=args.fps, codec="libx264", quality=8,
        macro_block_size=2, format="FFMPEG",
    ) as w:
        for f in frames:
            w.append_data(f)
    print(f"[replay] wrote {out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
