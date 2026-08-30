"""Export SONIC tokens from an official reference motion using encoder mode 0.

The official C++ deploy stack builds the encoder observation with mode 0:

    encoder_mode_4                       4
    motion_joint_positions_10frame_step5 290
    motion_joint_velocities_10frame_step5 290
    motion_anchor_orientation_10frame_step5 60

Every other encoder input is explicitly zeroed. This script reproduces that
layout so downstream RL uses the same token space as the official deployment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_norm(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def heading_quat(q):
    q = quat_norm(q)
    return np.array([q[0], 0.0, 0.0, q[3]])


def heading_quat_inv(q):
    q = quat_norm(q)
    return np.array([q[0], 0.0, 0.0, -q[3]])


def quat_to_rotmat(q):
    w, x, y, z = quat_norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_reference(motion_dir: Path):
    jp = np.genfromtxt(motion_dir / "joint_pos.csv", delimiter=",", skip_header=1)
    jv = np.genfromtxt(motion_dir / "joint_vel.csv", delimiter=",", skip_header=1)
    bq = np.genfromtxt(
        motion_dir / "body_quat.csv", delimiter=",", skip_header=1
    ).reshape(-1, 14, 4)
    if jp.ndim != 2 or jp.shape[0] != jv.shape[0] or jp.shape[0] != bq.shape[0]:
        raise ValueError("reference motion files have inconsistent frame counts")
    return jp.astype(np.float32), jv.astype(np.float32), bq.astype(np.float64)


def build_mode0_obs(
    frame: int,
    num_frames: int,
    step: int,
    jp: np.ndarray,
    jv: np.ndarray,
    bq: np.ndarray,
    base_quat: np.ndarray,
    apply_delta: np.ndarray,
    input_dim: int,
) -> np.ndarray:
    total_frames = jp.shape[0]
    obs = np.zeros(input_dim, dtype=np.float32)
    obs[0] = 0.0  # encoder mode 0; remaining mode fields stay zero

    p = 4
    for f in range(num_frames):
        idx = min(frame + f * step, total_frames - 1)
        obs[p : p + 29] = jp[idx]
        p += 29
    for f in range(num_frames):
        idx = min(frame + f * step, total_frames - 1)
        obs[p : p + 29] = jv[idx]
        p += 29

    # The "g1" encoder mode requires only 4 observations; the three between
    # joint_velocities and anchor_orientation_10frame are zero-filled but still
    # occupy their offsets in the 1762-d layout:
    #   motion_root_z_position_10frame_step5 (10) + motion_root_z_position (1)
    #   + motion_anchor_orientation (6) = 17
    p += 17

    for f in range(num_frames):
        idx = min(frame + f * step, total_frames - 1)
        new_ref = quat_norm(quat_mul(apply_delta, bq[idx, 0]))
        base_to_ref = quat_norm(quat_mul(quat_conj(base_quat), new_ref))
        rot = quat_to_rotmat(base_to_ref)
        obs[p : p + 6] = rot[:, :2].flatten()
        p += 6

    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--encoder-onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-quat", default="1,0,0,0")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--no-heading-align", action="store_true")
    args = parser.parse_args()

    jp, jv, bq = load_reference(args.motion_dir)
    base_quat = quat_norm(np.fromstring(args.base_quat, sep=",", dtype=np.float64))
    if base_quat.shape != (4,):
        raise ValueError("--base-quat must contain four numbers")

    if args.no_heading_align:
        apply_delta = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        apply_delta = quat_norm(
            quat_mul(heading_quat(base_quat), heading_quat_inv(bq[0, 0]))
        )

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(args.encoder_onnx), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    input_dim = session.get_inputs()[0].shape[1]
    if input_dim != 1762:
        raise ValueError(f"expected 1762-d encoder input, got {input_dim}")

    tokens = []
    for frame in range(jp.shape[0]):
        obs = build_mode0_obs(
            frame,
            args.num_frames,
            args.step,
            jp,
            jv,
            bq,
            base_quat,
            apply_delta,
            input_dim,
        )
        tokens.append(
            session.run(None, {input_name: obs[None]})[0][0].astype(np.float32)
        )

    tokens = np.asarray(tokens, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, tokens)
    print(f"saved {tokens.shape} tokens to {args.output}")


if __name__ == "__main__":
    main()
