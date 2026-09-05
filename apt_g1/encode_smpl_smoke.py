"""B2-s smoke (D040, DS_OFFICIAL_DATA_PLAN §3.1): offline-encode the SMPL-format
sample with encoder mode 2 ("smpl") and validate against the g1-mode encoding of
the SAME motion (robot_filtered pkl; D038-validated ref-rel tokens).

Layout ground truth (deploy g1_deploy_onnx_ref.cpp obs registry +
observation_config.yaml; g1-mode offsets cross-checked against
encode_bones_smoke.py). Total encoder input 1762, packed in the config's
encoder_observations list order:
    [0:4)      encoder_mode_4          obs[0]=mode_id, obs[1:4)=0
    [4:294)    g1 joint pos 10x step5  [294:584) g1 joint vel 10x step5
    [584:601)  root_z x10 + root_z + anchor_single (zeroed in g1 & smpl modes)
    [601:661)  g1 anchor 10x step5
    [661:781)  lowerbody pos (120)     [781:901) lowerbody vel (120)
    [901:910)  vr_3point_local_target (9)   [910:922) vr_3point_local_orn (12)
    [922:1642) smpl_joints_10frame_step1      (24*3*10)
    [1642:1702) smpl_anchor_orientation_10frame_step1 (6*10)
    [1702:1762) motion_joint_positions_wrists_10frame_step1 (6*10;
                Isaac dof indices {23..28}, policy_parameters.hpp)

smpl mode (mode_id=2) needs ONLY smpl fields -> encodable offline with no robot
dof. Anchor semantics (offline, no live robot; both from the C++/Python naming):
  ref-rel    btr = conj(q_t) * q_idx            (D036 root-cause fix; g1-validated)
  refheading btr = conj(heading(q_0)) * q_idx  (C++ "no robot state" variant)
Candidate matrix: {transform: identity | y-up->z-up perm (robot=(z,x,y) of
smpl, quat composed with the 120-deg (0.5,0.5,0.5,0.5) rotation)} x
{anchor: ref-rel | refheading}. Wrists are ZERO-filled everywhere (the 131k
mirror carries no robot dof; the paired robot pkl supplies the G1 reference
for comparison and the decoder-oracle history). Metrics per candidate:
lattice violation, paired token mean-L2 + frame-wise L2 vs g1 tokens, and the
decisive one -- decoder roundtrip MAE vs the robot reference (g1-mode
reference value for this motion: 0.109 rad, D038).

Usage (server, mjlab venv):
    cd ~/ros2_data/apt_g1 && ~/ros2_data/.venv_mjlab/bin/python \
        encode_smpl_smoke.py [--out data/ds_bones/b2_smpl/b2s_result.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
REPO = f"{HOME}/ros2_data/GR00T-WholeBodyControl"
SD = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b1/sample_data"
ENC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_encoder.onnx"
DEC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"
VENDOR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b2/_vendor"
FPS_ENC = 50.0
LATTICE_TOL = 0.05
WRIST_ISAAC_IDX = np.array([23, 24, 25, 26, 27, 28])  # policy_parameters.hpp

sys.path.insert(0, HOME + "/ros2_data")
sys.path.insert(0, HOME + "/ros2_data/apt_g1")
sys.path.insert(0, REPO)
if os.path.isdir(VENDOR):
    sys.path.insert(0, VENDOR)

import encode_bones_smoke as eb  # verbatim helpers, no drift
from apt_g1.envs.mujoco_g1_flat_env import (
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MUJOCO_TO_ISAACLAB_DOF,
    SONIC_ACTION_SCALE_MUJOCO,
    SONIC_DEFAULT_ANGLES_MUJOCO,
)
from apt_g1.sonic.sonic_wrapper import SonicOnnxDecoder

M2I = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)
I2M = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF)
DEFAULT_MJ = SONIC_DEFAULT_ANGLES_MUJOCO.astype(np.float64)
SCALE_ISAAC = SONIC_ACTION_SCALE_MUJOCO[M2I]
DEFAULT_ISAAC = SONIC_DEFAULT_ANGLES_MUJOCO.astype(np.float64)[M2I]

# y-up(SMPL) -> z-up(robot): robot = (z_s, x_s, y_s); active 120-deg rot about (1,1,1)
PERM_Q = np.array([0.5, 0.5, 0.5, 0.5])


def perm_vec(v):
    return v[:, [2, 0, 1]]


def perm_quat(q):
    return eb._qn(eb._qmul(PERM_Q, q))


def aa_to_quat(aa):
    """(N,3) axis-angle -> (N,4) wxyz."""
    th = np.linalg.norm(aa, axis=1)
    axis = np.divide(aa, np.where(th[:, None] < 1e-12, 1.0, th)[:, None])
    out = np.zeros((len(aa), 4))
    small = th[:, None] < 1e-12
    out[:, 0] = np.where(small[:, 0], 1.0, np.cos(th / 2))
    out[:, 1:] = np.where(small, 0.0, axis * np.sin(th / 2)[:, None])
    return out


def build_obs_smpl(t, joints72, root_quat, wrists60, anchor="ref-rel"):
    """1762-d encoder obs, mode 2 (smpl). joints72 stride-1 10 frames."""
    obs = np.zeros(1762, dtype=np.float32)
    obs[0] = 2.0
    p = 922
    for f in range(10):
        idx = min(t + f, len(joints72) - 1)
        obs[p:p + 72] = joints72[idx]
        p += 72
    if anchor == "refheading":
        left = eb._heading(root_quat[0])
    for f in range(10):
        idx = min(t + f, len(root_quat) - 1)
        if anchor == "ref-rel":
            btr = eb._qn(eb._qmul(eb._qconj(root_quat[t]), root_quat[idx]))
        else:  # refheading: no live robot, no delta (offline form)
            btr = eb._qn(eb._qmul(eb._qconj(left), root_quat[idx]))
        rot = eb._rotmat(btr)
        obs[p:p + 6] = rot[:, :2].flatten()
        p += 6
    for f in range(10):
        idx = min(t + f, len(wrists60) - 1)
        obs[p:p + 6] = wrists60[idx]
        p += 6
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl-pkl",
                    default=f"{SD}/smpl_filtered/walk_forward_amateur_001__A001.pkl")
    ap.add_argument("--robot-pkl",
                    default=f"{SD}/robot_filtered/210531/walk_forward_amateur_001__A001.pkl")
    ap.add_argument("--out", default=f"{HOME}/ros2_data/apt_g1/data/ds_bones/b2_smpl/b2s_result.json")
    args = ap.parse_args()

    import onnxruntime as ort

    # ---- 1. g1-mode reference tokens from the paired robot pkl
    obj, _ = eb.load_pkl(args.robot_pkl)
    obj = eb.unwrap(obj)
    dof = obj["dof"].astype(np.float64)
    root_rot = obj["root_rot"].astype(np.float64)
    trans = obj["root_trans_offset"].astype(np.float64)
    fps_src = float(obj["fps"])
    q0 = eb._qn(root_rot[0])
    layout = "wxyz" if abs(q0[0]) >= abs(q0[3]) else "xyzw"
    quat = root_rot.copy() if layout == "wxyz" else root_rot[:, [3, 0, 1, 2]]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    # joint order discriminator (corr of per-column mean vs default stance;
    # D036 decided MuJoCo for this exact pkl -- re-check cheaply, no MuJoCo env)
    means = dof.mean(axis=0)
    c_mj = float(np.corrcoef(means, DEFAULT_MJ)[0, 1])
    c_isa = float(np.corrcoef(means[I2M], DEFAULT_MJ)[0, 1])
    order = "mujoco" if c_mj >= c_isa else "isaac"
    print(f"[robot] {os.path.basename(args.robot_pkl)} fps={fps_src} quat={layout} "
          f"order={order} (corr mj={c_mj:.3f} isaac={c_isa:.3f})")
    dof_mj = dof if order == "mujoco" else dof[:, I2M]
    dof_mj_rs = eb.resample(dof_mj, fps_src, FPS_ENC)
    quat_rs = eb.resample_quat(quat, fps_src, FPS_ENC)
    jp_isaac = dof_mj_rs[:, M2I]
    jv_isaac = (np.vstack([np.zeros((1, 29)), np.diff(dof_mj_rs, axis=0) * FPS_ENC]))[:, M2I]
    bq = quat_rs.reshape(-1, 1, 4)
    apply_delta = eb._qn(eb._qmul(eb._heading(np.array([1.0, 0, 0, 0])), eb._heading_inv(bq[0, 0])))
    n_g1 = len(dof_mj_rs)

    enc = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])
    iname = enc.get_inputs()[0].name
    tokens_g1 = np.zeros((n_g1, 64), dtype=np.float32)
    for t in range(n_g1):
        obs = eb.build_obs(t, jp_isaac, jv_isaac, bq, apply_delta, anchor="ref-rel")
        tokens_g1[t] = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
    print(f"[g1-ref] tokens {tokens_g1.shape} (anchor=ref-rel, D038-validated form)")
    m_g1, s_g1 = tokens_g1.astype(np.float64).mean(axis=0), tokens_g1.astype(np.float64).std(axis=0)

    # ---- 2. smpl pkl
    objs, _ = eb.load_pkl(args.smpl_pkl)
    vs = eb.unwrap(objs)
    pa = vs["pose_aa"].astype(np.float64).reshape(len(vs["pose_aa"]), 24, 3)
    joints = vs["smpl_joints"].astype(np.float64).reshape(len(pa), 72)
    trans_s = vs["transl"].astype(np.float64)
    fps_s = float(vs["fps"])
    print(f"[smpl] {os.path.basename(args.smpl_pkl)} fps={fps_s} frames={len(pa)}")
    root_q = aa_to_quat(pa[:, 0, :])
    n_s = len(pa)

    # wrists: zero-filled (mirror has no robot dof); paired robot wrists kept
    # for an optional true-wrist contrast on the winning candidate
    wrists_zero = np.zeros((n_s, 6), dtype=np.float64)
    dof_isaac_rs = None
    if len(dof_mj_rs) >= n_s:
        dof_isaac_rs = dof_mj_rs[:, M2I][:n_s]
    wrists_true = dof_isaac_rs[:, WRIST_ISAAC_IDX] if dof_isaac_rs is not None else None

    # ---- 3. candidate matrix
    cands = []
    for tf in ("identity", "y2z"):
        for an in ("ref-rel", "refheading"):
            cands.append((tf, an))

    def prep(tf):
        if tf == "identity":
            return joints, root_q, trans_s
        return perm_vec(joints), perm_quat(root_q), perm_vec(trans_s)

    ref_path = float(np.linalg.norm(np.diff(trans_s, axis=0), axis=1).sum())

    # decoder-oracle history from the robot reference (same as encode_bones_smoke check3)
    omega_body = np.zeros((n_g1, 3))
    for t in range(n_g1):
        a, b = quat_rs[min(t + 1, n_g1 - 1)], quat_rs[max(t - 1, 0)]
        step = (min(t + 1, n_g1 - 1) - max(t - 1, 0)) / FPS_ENC
        dq = eb._qmul(a, eb._qconj(b))
        if dq[0] < 0:
            dq = -dq
        w_world = 2.0 * dq[1:] / max(dq[0], 1e-6) / max(step, 1e-6)
        omega_body[t] = eb._quat_rotate_inverse(quat_rs[t], w_world)
    grav = np.array([eb._quat_rotate_inverse(qq, np.array([0.0, 0.0, -1.0])) for qq in quat_rs])
    dec = SonicOnnxDecoder(DEC_ONNX)

    def sonic_history(t):
        idx = np.clip(np.arange(t - 9, t + 1), 0, n_g1 - 1)
        return {
            "base_angular_velocity": omega_body[idx].astype(np.float32),
            "body_joint_positions": ((dof_mj_rs[idx] - DEFAULT_MJ)[:, M2I]).astype(np.float32),
            "body_joint_velocities": jv_isaac[idx].astype(np.float32),
            "last_actions": (((dof_mj_rs[idx] - DEFAULT_MJ) / SONIC_ACTION_SCALE_MUJOCO)[:, M2I]).astype(np.float32),
            "gravity_dir": grav[idx].astype(np.float32),
        }

    results = {}
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    for tf, an in cands:
        j72, rq, _ = prep(tf)
        name = f"{tf}__{an}"
        tokens = np.zeros((n_s, 64), dtype=np.float32)
        for t in range(n_s):
            obs = build_obs_smpl(t, j72, rq, wrists_zero, anchor=an)
            tokens[t] = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
        lat = tokens.astype(np.float64) * 16.0
        viol = float((np.abs(lat - np.round(lat)) > LATTICE_TOL).mean())
        m_u, s_u = tokens.astype(np.float64).mean(axis=0), tokens.astype(np.float64).std(axis=0)
        mean_l2 = float(np.linalg.norm(m_u - m_g1))
        std_ratio = float(np.median(s_u / np.maximum(s_g1, 1e-6)))
        nmin = min(n_s, n_g1)
        frame_l2 = float(np.linalg.norm(tokens[:nmin] - tokens_g1[:nmin], axis=1).mean())
        err = []
        for t in range(nmin):
            obs = dec.build_decoder_obs(tokens[t], sonic_history(t))
            act = dec.session.run([dec.output_name], {dec.input_name: obs})[0][0]
            q_des = DEFAULT_ISAAC + act.astype(np.float64) * SCALE_ISAAC
            err.append(np.abs(q_des - jp_isaac[t]))
        rt_mae = float(np.mean(err))
        tok_path = os.path.join(out_dir, f"tokens_smpl_{name}.npy")
        np.save(tok_path, tokens)
        results[name] = {
            "transform": tf, "anchor": an, "wrists": "zero",
            "lattice_violation_rate": viol,
            "paired_mean_l2_vs_g1": mean_l2,
            "paired_frame_l2_vs_g1": frame_l2,
            "std_ratio_median_vs_g1": std_ratio,
            "roundtrip_mae_rad": rt_mae,
            "tokens_npy": tok_path,
        }
        print(f"[cand] {name:<20} lattice={viol:.2e} meanL2={mean_l2:.3f} "
              f"frameL2={frame_l2:.3f} stdR={std_ratio:.2f} roundtripMAE={rt_mae:.4f} rad", flush=True)

    best = min(results, key=lambda k: results[k]["roundtrip_mae_rad"])
    print(f"[best] {best} (roundtrip MAE {results[best]['roundtrip_mae_rad']:.4f} rad; "
          f"g1-mode reference for this motion = 0.109 rad, D038)")

    # ---- 4. true-wrist contrast on the winning candidate (paired robot dof)
    tf, an = results[best]["transform"], results[best]["anchor"]
    if wrists_true is not None:
        j72, rq, _ = prep(tf)
        tokens = np.zeros((n_s, 64), dtype=np.float32)
        for t in range(n_s):
            obs = build_obs_smpl(t, j72, rq, wrists_true, anchor=an)
            tokens[t] = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
        lat = tokens.astype(np.float64) * 16.0
        viol = float((np.abs(lat - np.round(lat)) > LATTICE_TOL).mean())
        m_u = tokens.astype(np.float64).mean(axis=0)
        err = []
        for t in range(min(n_s, n_g1)):
            obs = dec.build_decoder_obs(tokens[t], sonic_history(t))
            act = dec.session.run([dec.output_name], {dec.input_name: obs})[0][0]
            q_des = DEFAULT_ISAAC + act.astype(np.float64) * SCALE_ISAAC
            err.append(np.abs(q_des - jp_isaac[t]))
        name = f"{tf}__{an}__wristtrue"
        results[name] = {
            "transform": tf, "anchor": an, "wrists": "true(robot pkl)",
            "lattice_violation_rate": viol,
            "paired_mean_l2_vs_g1": float(np.linalg.norm(m_u - m_g1)),
            "roundtrip_mae_rad": float(np.mean(err)),
            "tokens_npy": os.path.join(out_dir, f"tokens_smpl_{name}.npy"),
        }
        np.save(results[name]["tokens_npy"], tokens)
        print(f"[cand] {name:<20} lattice={viol:.2e} "
              f"meanL2={results[name]['paired_mean_l2_vs_g1']:.3f} "
              f"roundtripMAE={results[name]['roundtrip_mae_rad']:.4f} rad (zero-wrist "
              f"contrast: {results[best]['roundtrip_mae_rad']:.4f})", flush=True)

    res = {
        "exp": "D040",
        "line": "DS plan B2-s smoke (encoder mode 2 = smpl)",
        "robot_pkl": args.robot_pkl,
        "smpl_pkl": args.smpl_pkl,
        "robot_joint_order": order,
        "layout": {
            "smpl_joints": [922, 1642], "smpl_anchor": [1642, 1702],
            "wrists": [1702, 1762], "mode_header": [0, 4],
            "source": "g1_deploy_onnx_ref.cpp obs registry + observation_config.yaml",
        },
        "g1_reference_tokens": {"n": n_g1, "anchor": "ref-rel"},
        "candidates": results,
        "best": best,
        "note": "wrists zero-filled = deployable form for the 131k mirror (no robot dof); "
                "true-wrist row quantifies the wrist-information gap",
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[out] JSON -> {args.out}")


if __name__ == "__main__":
    main()
