"""Official SONIC reference-tracking closed loop, replicated in Python/MuJoCo.

Tracks a pre-recorded reference motion (squat / kick / lunge / dance / ...) by
replicating the official C++ deploy stack's loop:

  1. advance the reference motion one frame per 50 Hz step
  2. build the 1762-d encoder observation (mode 0 = "g1"):
       encoder_mode_4(4) + motion_joint_positions_10frame_step5(290)
       + motion_joint_velocities_10frame_step5(290)
       + [root_z_10frame(10) + root_z(1) + anchor_single(6)  -- ZEROED]
       + motion_anchor_orientation_10frame_step5(60)
  3. encode -> 64-d token (model_encoder.onnx)
  4. decode token + 10-frame live proprio -> joint targets (frozen decoder)
  5. step MuJoCo

The two critical details (both were bugs in the earlier naive replay):
  * anchor orientation sits at offset 601, NOT 584 -- the three zeroed
    observations in between still occupy 17 dims.
  * the anchor orientation's "left quaternion" is the robot's LIVE base
    quaternion (mode 0 = full base), re-read every step -- the token must be
    encoded online, not precomputed offline with a fixed identity base.

Usage (on the server, mjlab venv which has mujoco + onnxruntime):
    python replay_reference_track.py --motion squat_001__A359 [--video out.mp4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = "/home/cvgluser/ros2_data"
ENC_ONNX = f"{REPO}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
DEC_ONNX = f"{REPO}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
REF_ROOT = Path(f"{REPO}/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example")

# quaternion helpers (mirror export_reference_tokens.py)
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


def load_reference(motion_dir: Path):
    jp = np.genfromtxt(motion_dir / "joint_pos.csv", delimiter=",", skip_header=1)
    jv = np.genfromtxt(motion_dir / "joint_vel.csv", delimiter=",", skip_header=1)
    bq = np.genfromtxt(motion_dir / "body_quat.csv", delimiter=",", skip_header=1).reshape(-1, 14, 4)
    return jp.astype(np.float32), jv.astype(np.float32), bq.astype(np.float64)


def build_encoder_obs(frame, jp, jv, bq, apply_delta, live_base, num_frames=10, step=5):
    obs = np.zeros(1762, dtype=np.float32)
    obs[0] = 0.0  # encoder_mode_4 = 0 (mode "g1")
    p = 4
    for f in range(num_frames):
        idx = min(frame + f * step, len(jp) - 1)
        obs[p:p + 29] = jp[idx]; p += 29
    for f in range(num_frames):
        idx = min(frame + f * step, len(jp) - 1)
        obs[p:p + 29] = jv[idx]; p += 29
    p += 17  # root_z_10frame(10) + root_z(1) + anchor_single(6), zeroed for "g1"
    for f in range(num_frames):
        idx = min(frame + f * step, len(jp) - 1)
        new_ref = _qn(_qmul(apply_delta, bq[idx, 0]))
        base_to_ref = _qn(_qmul(_qconj(live_base), new_ref))
        rot = _rotmat(base_to_ref)
        obs[p:p + 6] = rot[:, :2].flatten(); p += 6
    return obs


def track(motion_name: str, video: str | None = None):
    import sys
    sys.path.insert(0, REPO)
    sys.path.insert(0, f"{REPO}/apt_g1")
    sys.path.insert(0, f"{REPO}/GR00T-WholeBodyControl")
    import onnxruntime as ort
    from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
    from eval_distill import NoQuantDecoder

    jp, jv, bq = load_reference(REF_ROOT / motion_name)
    enc = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])
    iname = enc.get_inputs()[0].name
    env = MujocoG1FlatEnv(NoQuantDecoder(DEC_ONNX), f"{REPO}/GR00T-WholeBodyControl",
                          use_elastic_band=False, stand_only=True)

    apply_delta = _qn(_qmul(_heading(np.array([1.0, 0, 0, 0])), _heading_inv(bq[0, 0])))
    env.reset()
    heights, fall = [], None
    for t in range(len(jp)):
        live_base = env.data.qpos[3:7].astype(np.float64)
        obs = build_encoder_obs(t, jp, jv, bq, apply_delta, live_base)
        tok = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
        _, _, term, _ = env.step({"token": tok, "aux": np.zeros(12, dtype=np.float32)})
        heights.append(float(env.data.qpos[2]))
        if term:
            fall = t
            break
    h = np.asarray(heights)
    return dict(fall=fall, n=len(h), h_min=round(float(h.min()), 3),
                h_max=round(float(h.max()), 3), h_end=round(float(h[-1]), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    r = track(args.motion, args.video)
    print(f"{args.motion}: {r}")
