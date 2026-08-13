"""Oracle ceiling check for the new walk direction data (exp3 sweeps)."""
import sys

import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder

D = "/home/cvgluser/ros2_data/apt_g1/data/exp_all3"
token = np.load(D + "/token.npy")
mode = np.load(D + "/mode.npy")
speed = np.load(D + "/speed.npy")
ab = np.load(D + "/angle_bin.npy")

env = MujocoG1FlatEnv(
    NoQuantDecoder(
        "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
    ),
    "/home/cvgluser/ros2_data/GR00T-WholeBodyControl",
    use_elastic_band=False,
    stand_only=True,
)
env.command = np.zeros(3, dtype=np.float32)


def replay(rows, steps=600):
    import mujoco

    env.reset()
    vxs, vys, hs = [], [], []
    for t in range(min(steps, len(rows))):
        tok = token[rows[t]]
        obs, reward, terminated, info = env.step(
            {"token": tok, "aux": np.zeros(12, dtype=np.float32)}
        )
        v = env._get_base_linear_velocity()
        vxs.append(float(v[0]))
        vys.append(float(v[1]))
        hs.append(float(env.data.qpos[2]))
        if terminated:
            return dict(
                fall=t,
                vx=round(float(np.mean(vxs)), 3),
                vy=round(float(np.mean(vys)), 3),
                h_min=round(float(min(hs)), 3),
            )
    return dict(
        fall=None,
        vx=round(float(np.mean(vxs)), 3),
        vy=round(float(np.mean(vys)), 3),
        h_min=round(float(min(hs)), 3),
    )


for b in [1, 2, 3, 5, 6, 7]:
    rows = np.where((mode == 2) & (np.abs(speed + 1) < 1e-6) & (ab == b))[0]
    if len(rows) < 200:
        print(f"walk_bin{b}: only {len(rows)} rows, skip")
        continue
    r = replay(rows)
    print(f"walk_bin{b}: n={len(rows)} {r}")
