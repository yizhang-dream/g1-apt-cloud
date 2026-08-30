"""Official SONIC full stack (planner -> encoder -> decoder) replicated in Python/MuJoCo.

Replicates the complete 3-model GEAR-SONIC pipeline:
  1. kinematic planner (10 Hz ONNX) : 4 recent qpos frames + locomotion command
       (mode / target_vel / movement_direction / facing_direction / height)
       -> future whole-body qpos trajectory (27 LocomotionMode styles)
  2. encoder (model_encoder.onnx)   : motion (joint_pos + joint_vel + 14 body quats
       via MuJoCo FK) -> 64-d token
  3. decoder (model_decoder.onnx)   : token + 10-frame live proprio -> 29 joint targets

Verified modes (fall=None): idle(0) walk(2) run(3) stealth(18) kneelTwoLeg(5) squat(4).
crawl(8) falls at ~27 steps -- the stand->hands-knees transition needs the official
ADAPTING state machine, not yet replicated.

Usage (server, mjlab venv):
    python planner_sonic.py --mode 2          # walk
    python planner_sonic.py --mode 8          # crawl (currently falls)
"""

from __future__ import annotations

import argparse

import numpy as np

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
PLANNER_ONNX = f"{REPO}/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
ENC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_encoder.onnx"
DEC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"

BODY_IDX = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]


def _qn(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _heading(q):
    q = _qn(q)
    return np.array([q[0], 0.0, 0.0, q[3]])


def _heading_inv(q):
    q = _qn(q)
    return np.array([q[0], 0.0, 0.0, -q[3]])


def _rotmat(q):
    w, x, y, z = _qn(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def track(mode: int, vel: float = -1.0, height: float = -1.0, mdir=(1.0, 0, 0)):
    import sys
    sys.path.insert(0, "/home/cvgluser/ros2_data")
    sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
    sys.path.insert(0, REPO)
    import mujoco
    import onnxruntime as ort
    from apt_g1.envs.mujoco_g1_flat_env import (MujocoG1FlatEnv, SONIC_DEFAULT_ANGLES_MUJOCO,
                                                G1_MUJOCO_TO_ISAACLAB_DOF)
    from eval_distill import NoQuantDecoder

    planner = ort.InferenceSession(PLANNER_ONNX, providers=["CPUExecutionProvider"])
    enc = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])
    iname = enc.get_inputs()[0].name
    env = MujocoG1FlatEnv(NoQuantDecoder(DEC_ONNX), REPO, use_elastic_band=False, stand_only=True)
    m2i = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)

    def standing():
        q = np.zeros(36, dtype=np.float32)
        q[0:3] = [0, 0, 0.76]; q[3:7] = [1, 0, 0, 0]; q[7:36] = SONIC_DEFAULT_ANGLES_MUJOCO
        return q

    def fk(q36):
        q = np.zeros(env.model.nq)
        q[:3] = q36[:3]; q[3:7] = q36[3:7]; q[env.body_qpos_adr] = q36[7:36]
        env.data.qpos[:] = q
        mujoco.mj_forward(env.model, env.data)
        return env.data.xquat[BODY_IDX].copy()

    inp = {"context_mujoco_qpos": np.tile(standing()[None, None], (1, 4, 1)).astype(np.float32),
           "target_vel": np.array([vel], dtype=np.float32), "mode": np.array([mode], dtype=np.int64),
           "movement_direction": np.array([mdir], dtype=np.float32),
           "facing_direction": np.array([[1.0, 0, 0]], dtype=np.float32),
           "random_seed": np.array([0], dtype=np.int64), "has_specific_target": np.array([[0]], dtype=np.int64),
           "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
           "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
           "allowed_pred_num_tokens": np.ones((1, 11), dtype=np.int64),
           "height": np.array([height], dtype=np.float32)}
    qpos_out, nframes_out = planner.run(None, inp)
    traj = qpos_out[0][:int(nframes_out[0])]
    jp = traj[:, 7:36][:, m2i]
    # finite difference gives rad/step; the encoder expects rad/s at 50 Hz control
    jv = np.vstack([np.zeros((1, 29)), np.diff(jp, axis=0) * 50.0])
    bq = np.array([fk(t) for t in traj])
    apply_delta = _qn(_qmul(_heading(np.array([1.0, 0, 0, 0])), _heading_inv(bq[0, 0])))

    env.reset()
    hs, fall = [], None
    for t in range(len(traj)):
        live = env.data.qpos[3:7].astype(np.float64)
        obs = np.zeros(1762, dtype=np.float32); obs[0] = 0.0; p = 4
        for f in range(10):
            idx = min(t + f * 5, len(jp) - 1); obs[p:p + 29] = jp[idx]; p += 29
        for f in range(10):
            idx = min(t + f * 5, len(jv) - 1); obs[p:p + 29] = jv[idx]; p += 29
        p += 17
        for f in range(10):
            idx = min(t + f * 5, len(bq) - 1)
            nr = _qn(_qmul(apply_delta, bq[idx, 0]))
            btr = _qn(_qmul(_qconj(live), nr))
            rot = _rotmat(btr); obs[p:p + 6] = rot[:, :2].flatten(); p += 6
        tok = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
        _, _, term, _ = env.step({"token": tok, "aux": np.zeros(12, dtype=np.float32)})
        hs.append(float(env.data.qpos[2]))
        if term:
            fall = t; break
    h = np.asarray(hs)
    return dict(fall=fall, n=len(h), x=round(float(env.data.qpos[0]), 2),
                h_min=round(float(h.min()), 2), h_end=round(float(h[-1]), 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, required=True)
    ap.add_argument("--vel", type=float, default=-1.0)
    ap.add_argument("--height", type=float, default=-1.0)
    args = ap.parse_args()
    print(track(args.mode, args.vel, args.height))
