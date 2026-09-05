"""M2 stage-1 calibration (D041, DS_OFFICIAL_DATA_PLAN M2): pin down every
convention of the SMPL(mirror)->G1 mapping using the paired official samples
(smpl_filtered + robot_filtered = SAME motion, official Bones-Studio
retargeting) plus soma_filtered (named 26-joint skeleton, disambiguates the
SMPL joint order).

Outputs (JSON + console):
  1. root orientation convention: SMPL root axis-angle -> quat, candidate
     compositions (identity / y-up->z-up 120-deg perm) vs official robot
     root_rot (wxyz, D036) -- geodesic error per frame.
  2. root position convention: official trans vs s * perm(transl) + b
     (scalar scale + per-axis offset, least squares) -- residual MAE.
  3. SMPL joint names: soma joint_names (26, named) matched to the 24 SMPL
     joints by mean root-relative position.
  4. G1 correspondence: mean root-relative G1 link positions (official dof FK)
     vs mean root-relative SMPL joints -> proposed smpl_idx -> g1_body table
     + per-bone scale ratios.

Run (server, mjlab venv):
    python retarget_smpl_g1.py --mode calibration
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

PERM_Q = np.array([0.5, 0.5, 0.5, 0.5])  # y-up->z-up: robot=(z,x,y) of smpl


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


def perm_quat(q):
    w1, x1, y1, z1 = PERM_Q
    w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def geodesic_deg(qa, qb):
    d = np.abs(np.sum(qa * qb, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(d, 0.0, 1.0)))


def mat_to_quat(R):
    """Rotation matrix (3,3) -> quat wxyz."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def umeyama(X, Y):
    """Y ~= s * R @ X + b  (row-wise points). Returns s, R, b, residual."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calibration"], default="calibration")
    ap.add_argument("--out", default=f"{OUT_DIR}/m2_calibration.json")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- paired pkls (30 Hz native for robot + original smpl pose)
    rob = eb.unwrap(eb.load_pkl(f"{SD}/robot_filtered/210531/walk_forward_amateur_001__A001.pkl")[0])
    smp = eb.unwrap(eb.load_pkl(f"{SD}/smpl_filtered/walk_forward_amateur_001__A001.pkl")[0])
    som = eb.unwrap(eb.load_pkl(f"{SD}/soma_filtered/210531/walk_forward_amateur_001__A001.pkl")[0])

    dof30 = rob["dof"].astype(np.float64)            # (1202,29) MuJoCo order
    quat30 = rob["root_rot"].astype(np.float64)      # wxyz (D036)
    if abs(eb._qn(quat30[0])[0]) < abs(eb._qn(quat30[0])[3]):
        quat30 = quat30[:, [3, 0, 1, 2]]
        print("[robot] root_rot was xyzw -> converted wxyz")
    quat30 /= np.linalg.norm(quat30, axis=1, keepdims=True)
    trans30 = rob["root_trans_offset"].astype(np.float64)
    n30 = len(dof30)
    print(f"[robot] dof {dof30.shape} fps={rob['fps']}")

    orig_aa = smp["original_pose_aa"].astype(np.float64)  # (1202,72) @30Hz
    joints50 = smp["smpl_joints"].astype(np.float64).reshape(-1, 24, 3)
    transl50 = smp["transl"].astype(np.float64)
    fps_s = float(smp["fps"])
    print(f"[smpl] original_pose_aa {orig_aa.shape} @30, joints50 {joints50.shape} @{fps_s:g}")

    res = {}

    # ---- 1. root position: official = s * R * perm(transl) + b  (Umeyama)
    # the mocap session world frame differs from the robot frame by an
    # arbitrary yaw (per-take capture orientation), so fit the FULL rotation.
    X = perm_vec(transl50)                              # (2002,3) @50
    X30 = eb.resample(X, fps_s, 30.0)                   # (1201,3)
    n = int(min(n30, len(X30)))
    sR, R, b, resid = umeyama(X30[:n], trans30[:n])
    print(f"[root-pos] Umeyama scale s={sR:.4f}, residual MAE={resid.mean():.4f} m, "
          f"p95={np.percentile(resid, 95):.4f} m")
    print(f"[root-pos] R (smpl_session -> robot world, incl. y2z perm + yaw):\n"
          f"{np.round(R, 3)}\n offset b={np.round(b, 3).tolist()}")
    res["root_position"] = {"scale": sR, "R": np.round(R, 4).tolist(),
                            "offset": b.tolist(),
                            "residual_mae_m": round(float(resid.mean()), 4),
                            "residual_p95_m": round(float(np.percentile(resid, 95)), 4)}

    # ---- 2. root orientation: q_official ~ quat(R) (x) q_smpl_root
    q_root = aa_to_quat(orig_aa[:, :3])
    q_R = mat_to_quat(R)

    def qmul_batch(a, b):
        w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], axis=1)

    root_ori = {}
    cand = {
        "qR*x_smpl": qmul_batch(np.tile(q_R, (n, 1)), q_root[:n]),
        "qR*x_smpl_Rt": qmul_batch(np.tile(mat_to_quat(R.T), (n, 1)), q_root[:n]),
        "identity": q_root[:n],
    }
    for name, q in cand.items():
        qn = q / np.linalg.norm(q, axis=1, keepdims=True)
        err = geodesic_deg(qn, quat30[:n])
        root_ori[name] = {"mean_deg": round(float(err.mean()), 2),
                          "p95_deg": round(float(np.percentile(err, 95)), 2)}
        print(f"[root-ori] {name:<14} geodesic mean {err.mean():.2f} deg, p95 {np.percentile(err, 95):.2f}")
    res["root_orientation"] = root_ori
    res["R_quat_wxyz"] = q_R.tolist()

    # ---- 3. SMPL joint names via soma (named 26-joint skeleton)
    som_j = som["soma_joints"].astype(np.float64)       # (1202,26,3)
    names = list(som["joint_names"])
    print(f"[soma] {len(names)} joints: {names}")
    som_rel = som_j - som_j[:, 0:1, :]                  # hips as root
    # smpl joints @30Hz, root-relative, robot frame (n = aligned length)
    J30 = eb.resample(joints50.reshape(-1, 72), fps_s, 30.0).reshape(-1, 24, 3)
    smpl_rel = perm_vec(J30 - J30[:, 0:1, :])
    som_mean = som_rel.mean(axis=0)
    smpl_mean = smpl_rel[:n].mean(axis=0)
    name_of = []
    for j in range(24):
        d = np.linalg.norm(som_mean - smpl_mean[j], axis=1)
        k = int(np.argmin(d))
        name_of.append({"smpl_idx": j, "soma_name": names[k],
                        "dist_m": round(float(d[k]), 4),
                        "mean_pos": [round(float(x), 3) for x in smpl_mean[j]]})
    res["smpl_joint_names"] = name_of
    print("[smpl<-soma names]")
    for e in name_of:
        print(f"   smpl[{e['smpl_idx']:>2}] ~ {e['soma_name']:<18} d={e['dist_m']:.3f} pos={e['mean_pos']}")

    # ---- 4. G1 link positions (root-relative FK from official dof)
    from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
    from apt_g1.sonic.sonic_wrapper import SonicOnnxDecoder
    env = MujocoG1FlatEnv(SonicOnnxDecoder(DEC_ONNX), REPO,
                          use_elastic_band=False, stand_only=True)
    m, data = env.model, env.data
    qadr = env.body_qpos_adr
    pelvis_id = m.body("pelvis").id
    g1_rel = {}
    for t in range(0, n, 5):  # subsample 6 Hz is plenty for means
        q = np.zeros(m.nq)
        q[3:7] = quat30[t]
        q[:3] = trans30[t]
        q[qadr] = dof30[t]
        data.qpos[:] = q
        import mujoco as mjc
        mjc.mj_forward(m, data)
        base = data.xpos[pelvis_id].copy()
        for i in range(m.nbody):
            nm = m.body(i).name
            if nm in ("pelvis",) or "link" not in nm:
                continue
            g1_rel.setdefault(nm, []).append(data.xpos[i] - base)
    g1_mean = {k: np.mean(v, axis=0) for k, v in g1_rel.items()}

    # proposed correspondence: nearest g1 link for each smpl joint
    g1_names = sorted(g1_mean)
    g1_mat = np.stack([g1_mean[k] for k in g1_names])
    corr = []
    for j in range(24):
        d = np.linalg.norm(g1_mat - smpl_mean[j], axis=1)
        k = int(np.argmin(d))
        corr.append({"smpl_idx": j, "soma_name": name_of[j]["soma_name"],
                     "g1_body": g1_names[k], "dist_m": round(float(d[k]), 3)})
    res["smpl_to_g1_proposal"] = corr
    print("[smpl -> g1 nearest links]")
    for e in corr:
        print(f"   smpl[{e['smpl_idx']:>2}] {e['soma_name']:<18} -> {e['g1_body']:<26} d={e['dist_m']:.3f}")

    # bone scale ratios for confident leg/arm pairs (by soma name)
    def g1_pos(nm):
        return g1_mean.get(nm)

    pairs = [("Left_Knee" if "Left_Knee" in names else None,)]
    ratio_table = {}
    for j, e in enumerate(corr):
        nm = e["soma_name"]
        gb = e["g1_body"]
        if gb in g1_mean:
            r = np.linalg.norm(smpl_mean[j]) / max(np.linalg.norm(g1_mean[gb]), 1e-6)
            ratio_table[f"smpl{j}:{nm}->{gb}"] = round(float(r), 3)
    res["scale_ratio_by_joint"] = ratio_table
    print("[scale ratios |smpl_rel| / |g1_rel|]")
    for k, v in ratio_table.items():
        print(f"   {k:<44} {v}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
