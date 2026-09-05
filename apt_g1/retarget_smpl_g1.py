"""M2 stage-1 calibration (D041, DS_OFFICIAL_DATA_PLAN M2): pin down every
convention of the SMPL(mirror)->G1 mapping using the paired official samples
(smpl_filtered + robot_filtered = SAME motion, official Bones-Studio
retargeting) plus soma_filtered (named 26-joint skeleton, disambiguates the
SMPL joint order).

Findings so far (see JSON): the mocap session frame differs from the robot
world frame by a per-take rotation (Umeyama-fitted, 3 mm residual, scale
0.773 = human->G1); the official robot frame is y-forward / z-up.

Root-orientation candidates: exhaustive quat-field component orders +
compositions, and a convention-free skeleton-derived pelvis frame (hip axis +
spine axis) validated against the official root_rot.

Run (server, mjlab venv):
    python retarget_smpl_g1.py            # mode = calibration
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
DEC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"
VENDOR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b2/_vendor"
OUT_DIR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/m2"

sys.path.insert(0, f"{HOME}/ros2_data")
sys.path.insert(0, f"{HOME}/ros2_data/apt_g1")
sys.path.insert(0, REPO)
if os.path.isdir(VENDOR):
    sys.path.insert(0, VENDOR)

import encode_bones_smoke as eb  # verbatim quat helpers / resample / pkl loader

PERM_Q = np.array([0.5, 0.5, 0.5, 0.5])  # y-up->z-up candidate (subsumed by R)


def perm_vec(v):
    return v[..., [2, 0, 1]]


def aa_to_quat(aa):
    th = np.linalg.norm(aa, axis=1)
    axis = np.divide(aa, np.where(th < 1e-12, 1.0, th)[:, None])
    out = np.zeros((len(aa), 4))
    small = th < 1e-12
    out[:, 0] = np.where(small, 1.0, np.cos(th / 2))
    out[:, 1:] = np.where(small[:, None], 0.0, axis * np.sin(th / 2)[:, None])
    return out


def geodesic_deg(qa, qb):
    d = np.abs(np.sum(qa * qb, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(d, 0.0, 1.0)))


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def umeyama(X, Y):
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = float(np.trace(np.diag(S) @ D) / max((Xc ** 2).sum(), 1e-12))
    b = my - s * R @ mx
    resid = np.linalg.norm(s * (R @ X.T).T + b - Y, axis=1)
    return s, R, b, resid


def unwrap1(obj):
    if isinstance(obj, dict) and len(obj) == 1:
        v = next(iter(obj.values()))
        if isinstance(v, dict):
            return v
    return obj


def qmul_batch(a, b):
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def qconj_batch(q):
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calibration"], default="calibration")
    ap.add_argument("--out", default=f"{OUT_DIR}/m2_calibration.json")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    rob = unwrap1(eb.load_pkl(f"{SD}/robot_filtered/210531/walk_forward_amateur_001__A001.pkl")[0])
    smp = unwrap1(eb.load_pkl(f"{SD}/smpl_filtered/walk_forward_amateur_001__A001.pkl")[0])
    som = unwrap1(eb.load_pkl(f"{SD}/soma_filtered/210531/walk_forward_amateur_001__A001.pkl")[0])

    dof30 = rob["dof"].astype(np.float64)
    quat30 = rob["root_rot"].astype(np.float64)
    if abs(eb._qn(quat30[0])[0]) < abs(eb._qn(quat30[0])[3]):
        quat30 = quat30[:, [3, 0, 1, 2]]
        print("[robot] root_rot was xyzw -> converted wxyz")
    quat30 /= np.linalg.norm(quat30, axis=1, keepdims=True)
    trans30 = rob["root_trans_offset"].astype(np.float64)
    n30 = len(dof30)
    print(f"[robot] dof {dof30.shape} fps={rob['fps']}")
    print(f"[robot] trans30 first={np.round(trans30[0], 3)} last={np.round(trans30[-1], 3)} "
          f"std={np.round(trans30.std(axis=0), 3)}")

    orig_aa = smp["original_pose_aa"].astype(np.float64)
    joints50 = smp["smpl_joints"].astype(np.float64).reshape(-1, 24, 3)
    transl50 = smp["transl"].astype(np.float64)
    fps_s = float(smp["fps"])
    som_j = som["soma_joints"].astype(np.float64)
    som_names = list(som["joint_names"])
    res = {}

    # ---- A. root position: official = s * R * perm(transl) + b (Umeyama)
    X30 = eb.resample(perm_vec(transl50), fps_s, 30.0)
    n = int(min(n30, len(X30), len(som_j)))
    sR, R, b, resid = umeyama(X30[:n], trans30[:n])
    print(f"[A root-pos] Umeyama s={sR:.4f} resid MAE={resid.mean():.4f} m "
          f"p95={np.percentile(resid, 95):.4f} m")
    print(f"[A] R=\n{np.round(R, 3)}\n    b={np.round(b, 3).tolist()}")
    res["root_position"] = {"scale": sR, "R": np.round(R, 4).tolist(), "offset": b.tolist(),
                            "resid_mae_m": round(float(resid.mean()), 4)}
    q_R = mat_to_quat(R)

    # ---- B. smpl joint names via soma (session frame, 30Hz)
    J30 = eb.resample(joints50.reshape(-1, 72), fps_s, 30.0).reshape(-1, 24, 3)
    smpl_rel_ses = J30[:n] - J30[:n][:, 0:1, :]
    som_rel = som_j[:n] - som_j[:n][:, 0:1, :]
    smpl_mean_ses = smpl_rel_ses.mean(axis=0)
    som_mean = som_rel.mean(axis=0)
    name_of = []
    for j in range(24):
        d = np.linalg.norm(som_mean - smpl_mean_ses[j], axis=1)
        k = int(np.argmin(d))
        name_of.append({"smpl_idx": j, "soma_name": som_names[k],
                        "dist_m": round(float(d[k]), 4)})
    res["smpl_joint_names"] = name_of
    print("[B smpl<-soma names] (session frame)")
    for e in name_of:
        print(f"   smpl[{e['smpl_idx']:>2}] ~ {e['soma_name']:<18} d={e['dist_m']:.3f}")
    nm_of = {e["smpl_idx"]: e["soma_name"] for e in name_of}

    # ---- C. root orientation
    root_ori = {}
    som_q = som["soma_root_quat"].astype(np.float64)[:n]
    if np.abs(som_q[:, 0]).mean() < np.abs(som_q[:, 3]).mean():
        som_q = som_q[:, [3, 0, 1, 2]]
        print("[soma] root_quat xyzw -> wxyz")
    som_q /= np.linalg.norm(som_q, axis=1, keepdims=True)
    q_root = aa_to_quat(orig_aa[:n, :3])
    q_R_tiled = np.tile(q_R, (n, 1))
    cands = {}
    for src_name, q_src in (("soma", som_q), ("aa_root", q_root)):
        for order in ("wxyz", "xyzw"):
            qq = q_src if order == "wxyz" else q_src[:, [3, 0, 1, 2]]
            for conj in (False, True):
                base = qconj_batch(qq) if conj else qq
                base = base / np.linalg.norm(base, axis=1, keepdims=True)
                for comp_name, comp in (("raw", None), ("qR", q_R_tiled)):
                    q_c = base if comp is None else qmul_batch(comp, base)
                    q_c = q_c / np.linalg.norm(q_c, axis=1, keepdims=True)
                    err = geodesic_deg(q_c, quat30[:n])
                    key = f"{src_name}/{order}{'/conj' if conj else ''}/{comp_name}"
                    cands[key] = (err, q_c)
    # skeleton-derived pelvis frame (convention-free): hip axis + spine axis
    idx = {nm_of[j]: j for j in nm_of}
    if all(k in idx for k in ("LeftLeg", "RightLeg", "Spine1")):
        l_hip = J30[:n, idx["LeftLeg"]]
        r_hip = J30[:n, idx["RightLeg"]]
        spine = J30[:n, idx["Spine1"]]
        hips_mid = 0.5 * (l_hip + r_hip)
        lat = r_hip - l_hip
        up0 = spine - hips_mid
        lat /= np.linalg.norm(lat, axis=1, keepdims=True)
        up0 /= np.linalg.norm(up0, axis=1, keepdims=True)
        for sign, sg in ((1.0, "p"), (-1.0, "m")):
            f = np.cross(sign * lat, up0)
            f /= np.linalg.norm(f, axis=1, keepdims=True)
            u = np.cross(f, sign * lat)
            R_pel = np.stack([f, sign * lat, u], axis=2)  # session->pelvis cols
            q_sk = np.zeros((n, 4))
            for t in range(n):
                q_sk[t] = mat_to_quat(R @ R_pel[t])
            q_sk /= np.linalg.norm(q_sk, axis=1, keepdims=True)
            cands[f"skel/sign{sg}"] = (None, q_sk)

    best = None
    for key, (err, q_c) in cands.items():
        if err is None:
            err = geodesic_deg(q_c, quat30[:n])
        qn = q_c / np.linalg.norm(q_c, axis=1, keepdims=True)
        q_corr = qmul_batch(quat30[:n], qconj_batch(qn))
        q_corr /= np.linalg.norm(q_corr, axis=1, keepdims=True)
        sign_fix = np.where(q_corr[:, 0] < 0, -1.0, 1.0)[:, None]
        q_corr_m = q_corr * sign_fix
        spread = float(np.linalg.norm(q_corr_m - q_corr_m.mean(axis=0), axis=1).mean())
        root_ori[key] = {"mean_deg": round(float(err.mean()), 2),
                         "p95_deg": round(float(np.percentile(err, 95)), 2),
                         "corr_spread": round(spread, 3)}
        if best is None or err.mean() < best[1]:
            best = (key, float(err.mean()), qn)
    for k in sorted(root_ori, key=lambda k: root_ori[k]["mean_deg"])[:8]:
        print(f"[C root-ori] {k:<28} mean {root_ori[k]['mean_deg']:>7.2f} "
              f"p95 {root_ori[k]['p95_deg']:>7.2f} corrSpread {root_ori[k]['corr_spread']:.3f}")
    print(f"[C] BEST = {best[0]} ({best[1]:.2f} deg)")
    res["root_orientation"] = root_ori
    res["root_orientation_best"] = best[0]

    # ---- D. correspondence in robot frame: p_robot = s * R * p_session
    smpl_rel_rob = sR * (R @ smpl_rel_ses.reshape(-1, 3).T).T.reshape(-1, 24, 3)
    smpl_mean_rob = smpl_rel_rob.mean(axis=0)
    from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
    from apt_g1.sonic.sonic_wrapper import SonicOnnxDecoder
    env = MujocoG1FlatEnv(SonicOnnxDecoder(DEC_ONNX), REPO,
                          use_elastic_band=False, stand_only=True)
    m, data = env.model, env.data
    qadr = env.body_qpos_adr
    pelvis_id = m.body("pelvis").id
    import mujoco as mjc
    g1_rel = {}
    for t in range(0, n, 5):
        q = np.zeros(m.nq)
        q[3:7] = quat30[t]
        q[:3] = trans30[t]
        q[qadr] = dof30[t]
        data.qpos[:] = q
        mjc.mj_forward(m, data)
        base = data.xpos[pelvis_id].copy()
        for i in range(m.nbody):
            nm = m.body(i).name
            if nm == "pelvis" or "link" not in nm or "hand" in nm:
                continue
            g1_rel.setdefault(nm, []).append(data.xpos[i] - base)
    g1_mean = {k: np.mean(v, axis=0) for k, v in g1_rel.items()}
    g1_names = sorted(g1_mean)
    g1_mat = np.stack([g1_mean[k] for k in g1_names])
    corr = []
    for j in range(24):
        d = np.linalg.norm(g1_mat - smpl_mean_rob[j], axis=1)
        k = int(np.argmin(d))
        corr.append({"smpl_idx": j, "soma_name": nm_of[j], "g1_body": g1_names[k],
                     "dist_m": round(float(d[k]), 3)})
    res["smpl_to_g1_proposal"] = corr
    print("[D smpl -> g1 nearest links] (robot frame)")
    for e in corr:
        print(f"   smpl[{e['smpl_idx']:>2}] {e['soma_name']:<18} -> {e['g1_body']:<26} d={e['dist_m']:.3f}")
    ratio_table = {}
    for j, e in enumerate(corr):
        if e["g1_body"] in g1_mean:
            r = np.linalg.norm(smpl_mean_rob[j]) / max(np.linalg.norm(g1_mean[e["g1_body"]]), 1e-6)
            ratio_table[f"smpl{j}:{e['soma_name']}->{e['g1_body']}"] = round(float(r), 3)
    res["scale_ratio_by_joint"] = ratio_table

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
