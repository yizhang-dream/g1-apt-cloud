"""M2 stage-2 IK pilot (D041): retarget SMPL mirror joints -> G1 dof by
per-frame full-pose IK, validated against the paired OFFICIAL retargeting
(robot_filtered pkl) and the g1-mode encode/decode loopback (D038 = 0.109 rad).

Frame facts (stage-1 calibration, m2_calibration.json):
  - session->robot: official_root(t) = s * R * perm(transl(t)) + b, Umeyama
    residual 3 mm, s = 0.773 (human->G1).
  - smpl_joints are SESSION-frame root-relative offsets (hip axis rotates with
    the body over the take: max 178 deg on the loop walk) -> world targets
    = root_pos_robot(t) + s * R * rel(t, j).
  - robot frame: z-up, this take walks along +y; pelvis height 0.79 m.
  - root orientation is SOLVED by the IK (no quat-convention assumption);
    official quat comparison is reported as a diagnostic only.

Usage (server, mjlab venv):
    python retarget_ik_pilot.py [--max-frames 0] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
REPO = f"{HOME}/ros2_data/GR00T-WholeBodyControl"
SD = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b1/sample_data"
DEC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"
ENC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_encoder.onnx"
VENDOR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b2/_vendor"
OUT_DIR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/m2"

sys.path.insert(0, f"{HOME}/ros2_data")
sys.path.insert(0, f"{HOME}/ros2_data/apt_g1")
sys.path.insert(0, REPO)
if os.path.isdir(VENDOR):
    sys.path.insert(0, VENDOR)

import encode_bones_smoke as eb
from retarget_smpl_g1 import (aa_to_quat, geodesic_deg, mat_to_quat, perm_vec,
                              qconj_batch, qmul_batch, umeyama, unwrap1)

from apt_g1.envs.mujoco_g1_flat_env import (
    G1_MUJOCO_TO_ISAACLAB_DOF,
    SONIC_ACTION_SCALE_MUJOCO,
    SONIC_DEFAULT_ANGLES_MUJOCO,
)
from apt_g1.sonic.sonic_wrapper import SonicOnnxDecoder

M2I = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)
DEFAULT_MJ = SONIC_DEFAULT_ANGLES_MUJOCO.astype(np.float64)
SCALE_ISAAC = SONIC_ACTION_SCALE_MUJOCO[M2I]
DEFAULT_ISAAC = DEFAULT_MJ[M2I]

# soma name -> G1 link (position targets); Hips = root (fixed), rest skipped
TARGET_MAP = {
    "LeftLeg": "left_hip_pitch_link",
    "RightLeg": "right_hip_pitch_link",
    "LeftShin": "left_knee_link",
    "RightShin": "right_knee_link",
    "LeftFoot": "left_ankle_pitch_link",
    "RightFoot": "right_ankle_pitch_link",
    "LeftToeBase": "left_ankle_roll_link",
    "RightToeBase": "right_ankle_roll_link",
    "Spine2": "torso_link",
    "Chest": "torso_link",
    "LeftArm": "left_shoulder_pitch_link",
    "RightArm": "right_shoulder_pitch_link",
    "LeftForeArm": "left_elbow_link",
    "RightForeArm": "right_elbow_link",
    "LeftHand": "left_wrist_roll_link",
    "RightHand": "right_wrist_roll_link",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0, help="debug cap (0 = all)")
    ap.add_argument("--out", default=f"{OUT_DIR}/m2_ik_pilot.json")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    rob = unwrap1(eb.load_pkl(f"{SD}/robot_filtered/210531/walk_forward_amateur_001__A001.pkl")[0])
    smp = unwrap1(eb.load_pkl(f"{SD}/smpl_filtered/walk_forward_amateur_001__A001.pkl")[0])
    som = unwrap1(eb.load_pkl(f"{SD}/soma_filtered/210531/walk_forward_amateur_001__A001.pkl")[0])

    dof30 = rob["dof"].astype(np.float64)
    quat30 = rob["root_rot"].astype(np.float64)
    if abs(eb._qn(quat30[0])[0]) < abs(eb._qn(quat30[0])[3]):
        quat30 = quat30[:, [3, 0, 1, 2]]
    quat30 /= np.linalg.norm(quat30, axis=1, keepdims=True)
    trans30 = rob["root_trans_offset"].astype(np.float64)

    joints50 = smp["smpl_joints"].astype(np.float64).reshape(-1, 24, 3)
    transl50 = smp["transl"].astype(np.float64)
    fps_s = float(smp["fps"])
    som_names = list(som["joint_names"])

    # ---- calibration constants (stage 1, redone here in-process)
    X30 = eb.resample(perm_vec(transl50), fps_s, 30.0)
    n30 = min(len(dof30), len(X30))
    sR, R, b, resid = umeyama(X30[:n30], trans30[:n30])
    print(f"[calib] Umeyama s={sR:.4f} resid={resid.mean():.4f} m")

    # ---- world root positions + joint targets at the IK rate (30 Hz)
    root_pos30 = sR * (R @ perm_vec(transl50).T).T + b          # (2002,3)
    J30 = eb.resample(joints50.reshape(-1, 72), fps_s, 30.0).reshape(-1, 24, 3)
    n = min(len(J30), n30)
    rel_ses = J30[:n] - J30[:n][:, 0:1, :]
    targets = root_pos30[:n, None, :] + sR * (R @ rel_ses.reshape(-1, 3).T).T.reshape(-1, 24, 3)
    if args.max_frames:
        n = min(n, args.max_frames)
    print(f"[ik] frames={n} (30 Hz), targets={targets.shape}")

    # ---- smpl joint index -> g1 body id
    som_j = som["soma_joints"].astype(np.float64)
    som_rel = som_j[:n] - som_j[:n][:, 0:1, :]
    smpl_mean = (J30[:n] - J30[:n][:, 0:1, :]).mean(axis=0)
    som_mean = som_rel.mean(axis=0)
    idx_by_name = {}
    for j in range(24):
        k = int(np.argmin(np.linalg.norm(som_mean - smpl_mean[j], axis=1)))
        idx_by_name[som_names[k]] = j
    from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
    env = MujocoG1FlatEnv(SonicOnnxDecoder(DEC_ONNX), REPO,
                          use_elastic_band=False, stand_only=True)
    m, data = env.model, env.data
    qadr = env.body_qpos_adr
    targets_mask = []  # (smpl_idx, body_id)
    for nm_soma, nm_g1 in TARGET_MAP.items():
        if nm_soma in idx_by_name:
            try:
                bid = m.body(nm_g1).id
            except KeyError:
                continue
            targets_mask.append((idx_by_name[nm_soma], bid))
    print(f"[ik] target pairs: {[(som_names and nm) for nm, _ in [(k, v) for k, v in TARGET_MAP.items() if k in idx_by_name]]}")
    tgt_idx = np.array([a for a, _ in targets_mask])
    tgt_bid = np.array([bb for _, bb in targets_mask])
    print(f"[ik] {len(targets_mask)} target bodies: {[m.body(bb).name for bb in tgt_bid]}")

    # ---- IK loop
    import mujoco as mjc
    from scipy.optimize import least_squares

    def set_pose(root_quat, dof, root_pos):
        q = np.zeros(m.nq)
        q[:3] = root_pos
        q[3:7] = root_quat / np.linalg.norm(root_quat)
        q[qadr] = dof
        data.qpos[:] = q
        mjc.mj_forward(m, data)

    def residual(x, root_pos, tgt):
        set_pose(x[:4], x[4:], root_pos)
        pos = data.xpos[tgt_bid]           # (K,3) world
        return (pos - tgt[tgt_idx]).ravel()

    # init root quat from skeleton frame (session->robot), sign picked at t=0
    l_hip, r_hip = J30[0, idx_by_name["LeftLeg"]], J30[0, idx_by_name["RightLeg"]]
    spine = J30[0, idx_by_name["Spine2"]]
    lat = r_hip - l_hip
    up0 = spine - 0.5 * (l_hip + r_hip)
    lat /= np.linalg.norm(lat)
    up0 /= np.linalg.norm(up0)
    f = np.cross(lat, up0)
    f /= np.linalg.norm(f)
    u = np.cross(f, lat)
    R_pel = np.stack([f, lat, u], axis=1)
    q0_a = mat_to_quat(R @ R_pel)
    R_pel2 = np.stack([f, -lat, np.cross(f, -lat)], axis=1)
    q0_b = mat_to_quat(R @ R_pel2)
    set_pose(q0_a, DEFAULT_MJ, root_pos30[0])
    res_a = np.linalg.norm(residual(np.concatenate([q0_a, DEFAULT_MJ]), root_pos30[0], targets[0]))
    set_pose(q0_b, DEFAULT_MJ, root_pos30[0])
    res_b = np.linalg.norm(residual(np.concatenate([q0_b, DEFAULT_MJ]), root_pos30[0], targets[0]))
    quat_cur = q0_a if res_a <= res_b else q0_b
    print(f"[ik] init sign residuals: {res_a:.3f} / {res_b:.3f} -> "
          f"{'a' if res_a <= res_b else 'b'}")

    dof_ik = np.zeros((n, 29))
    quat_ik = np.zeros((n, 4))
    res_hist = []
    x = np.concatenate([quat_cur, DEFAULT_MJ])
    for t in range(n):
        sol = least_squares(residual, x, args=(root_pos30[t], targets[t]),
                            method="lm", max_nfev=64, xtol=1e-8, ftol=1e-8)
        x = sol.x.copy()
        x[4:] = np.clip(x[4:], -3.14, 3.14)
        dof_ik[t] = x[4:]
        quat_ik[t] = x[:4] / np.linalg.norm(x[:4])
        res_hist.append(float(np.sqrt(np.mean(sol.fun ** 2))))
        if (t + 1) % 200 == 0:
            print(f"[ik] {t + 1}/{n} frames, rms {res_hist[-1] * 100:.1f} cm, "
                  f"{(time.time() - t_start):.0f}s", flush=True)
    print(f"[ik] done in {time.time() - t_start:.0f}s; residual rms mean "
          f"{np.mean(res_hist) * 100:.2f} cm, p95 {np.percentile(res_hist, 95) * 100:.2f} cm")

    # ---- validation vs official
    res = {"n_frames": n, "ik_residual_rms_cm": round(float(np.mean(res_hist)) * 100, 2),
           "ik_residual_p95_cm": round(float(np.percentile(res_hist, 95)) * 100, 2)}
    mae = np.abs(dof_ik[:n30] - dof30[:n]).mean(axis=1)
    res["dof_mae_vs_official_rad"] = round(float(mae.mean()), 4)
    pj = np.abs(dof_ik[:n30] - dof30[:n]).mean(axis=0)
    worst = np.argsort(-pj)[:5]
    res["dof_mae_worst5_mujoco_dim"] = [{"dim": int(j), "mae_rad": round(float(pj[j]), 4)} for j in worst]
    qerr = geodesic_deg(quat_ik[:n30], quat30[:n])
    res["root_quat_geodesic_deg_mean"] = round(float(qerr.mean()), 2)
    res["root_quat_geodesic_deg_p95"] = round(float(np.percentile(qerr, 95)), 2)
    print(f"[val] dof MAE vs official = {mae.mean():.4f} rad "
          f"(worst mujoco dims {[(int(j), round(float(pj[j]), 3)) for j in worst]})")
    print(f"[val] root quat geodesic vs official: mean {qerr.mean():.2f} p95 {np.percentile(qerr, 95):.2f} deg")

    # ---- loopback: g1-mode encode OUR dof -> decoder roundtrip vs official
    dof50 = eb.resample(dof_ik, 30.0, 50.0)
    quat50 = eb.resample_quat(quat_ik, 30.0, 50.0)
    n50 = len(dof50)
    jp_mj = dof50
    jp_isaac = dof50[:, M2I]
    jv_isaac = (np.vstack([np.zeros((1, 29)), np.diff(dof50, axis=0) * 50.0]))[:, M2I]
    bq = quat50.reshape(-1, 1, 4)
    apply_delta = eb._qn(eb._qmul(eb._heading(np.array([1.0, 0, 0, 0])), eb._heading_inv(bq[0, 0])))
    import onnxruntime as ort
    enc = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])
    iname = enc.get_inputs()[0].name
    tokens = np.zeros((n50, 64), dtype=np.float32)
    for t in range(n50):
        obs = eb.build_obs(t, jp_isaac, jv_isaac, bq, apply_delta, anchor="ref-rel")
        tokens[t] = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
    lat_tokens = tokens.astype(np.float64) * 16.0
    viol = float((np.abs(lat_tokens - np.round(lat_tokens)) > 0.05).mean())

    dof50_official = eb.resample(dof30[:n30], 30.0, 50.0)
    jp_isaac_ref = dof50_official[:, M2I]
    dec = SonicOnnxDecoder(DEC_ONNX)
    omega_body = np.zeros((n50, 3))
    for t in range(n50):
        a, bb = quat50[min(t + 1, n50 - 1)], quat50[max(t - 1, 0)]
        step = (min(t + 1, n50 - 1) - max(t - 1, 0)) / 50.0
        dq = eb._qmul(a, eb._qconj(bb))
        if dq[0] < 0:
            dq = -dq
        w_world = 2.0 * dq[1:] / max(dq[0], 1e-6) / max(step, 1e-6)
        omega_body[t] = eb._quat_rotate_inverse(quat50[t], w_world)
    grav = np.array([eb._quat_rotate_inverse(qq, np.array([0.0, 0.0, -1.0])) for qq in quat50])

    def sonic_history(t):
        idx = np.clip(np.arange(t - 9, t + 1), 0, n50 - 1)
        return {
            "base_angular_velocity": omega_body[idx].astype(np.float32),
            "body_joint_positions": ((dof50[idx] - DEFAULT_MJ)[:, M2I]).astype(np.float32),
            "body_joint_velocities": jv_isaac[idx].astype(np.float32),
            "last_actions": (((dof50[idx] - DEFAULT_MJ) / SONIC_ACTION_SCALE_MUJOCO)[:, M2I]).astype(np.float32),
            "gravity_dir": grav[idx].astype(np.float32),
        }

    err = []
    for t in range(n50):
        obs = dec.build_decoder_obs(tokens[t], sonic_history(t))
        act = dec.session.run([dec.output_name], {dec.input_name: obs})[0][0]
        q_des = DEFAULT_ISAAC + act.astype(np.float64) * SCALE_ISAAC
        err.append(np.abs(q_des - jp_isaac_ref[t]))
    rt_mae = float(np.mean(err))
    print(f"[loopback] lattice viol {viol:.2e}; roundtrip MAE vs official = {rt_mae:.4f} rad "
          f"(D038 official-dof reference: 0.109)")

    res["loopback"] = {"lattice_violation_rate": viol, "roundtrip_mae_rad": round(rt_mae, 4),
                       "reference_d038_rad": 0.109}
    np.savez(os.path.join(OUT_DIR, "m2_retargeted.npz"),
             dof30=dof_ik, quat30=quat_ik, dof50=dof50, quat50=quat50,
             tokens=tokens, root_pos30=root_pos30[:n])
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[out] {args.out} + m2_retargeted.npz; total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
