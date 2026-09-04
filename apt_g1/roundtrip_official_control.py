"""D036 control arm: the OFFICIAL closed-loop WALK tokens (ds_smoke
policy_input.csv, events.json walk_fwd_60s_baseline window) pushed through the
SAME offline decoder roundtrip harness as encode_bones_smoke.py check3, with
oracle proprio taken from target_motion.csv (official planner reference, 50 Hz
native -- no resample).

Discriminating experiment for B2's 0.564 rad roundtrip MAE:
  - official tokens also ~0.5  -> 0.564 is the harness metric's own magnitude
    (frame-aligned decoder-vs-smooth-reference on a closed loop is inherently
    strict); no B-line alarm.
  - official tokens ~0.15      -> our offline encode path has a real problem
    (report per-joint difference pattern; do NOT touch the encode path here).

target_motion.csv layout (probed, 22645 rows x 37 fields @ 50 Hz):
  col 0-2   root xyz (deploy frame; col 0 = root x)
  col 3-6   root quat, wxyz (row-norm 1.0; near-identity inside the walk window)
  col 7-35  29 joints (joint order decided below, same two discriminators as
            encode_bones_smoke.joint_order_evidence: default-pose corr + FK
            foot clearance)
  col 36    empty trailing field (trailing comma, parsed as 0) -- dropped

Roundtrip machine = encode_bones_smoke.py check3 verbatim (quat utils,
sonic_history semantics, sonic_scale/default conversion all IMPORTED from that
read-only module to prevent drift).

Usage (server, mjlab venv):
    cd ~/ros2_data/apt_g1 && ~/ros2_data/.venv_mjlab/bin/python \
        roundtrip_official_control.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
HOME = os.path.expanduser("~")
DEC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"
OFFICIAL_CSV = f"{HOME}/ros2_data/apt_g1/data/ds_smoke/policy_input.csv"
OFFICIAL_EVENTS = f"{HOME}/ros2_data/apt_g1/data/ds_smoke/events.json"
OFFICIAL_MOTION = f"{HOME}/ros2_data/apt_g1/data/ds_smoke/target_motion.csv"
DEFAULT_OUT = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b2/control_official.json"
FPS = 50.0
# B2 results this experiment discriminates against (smoke_result.json)
B2_MAE = 0.5638091343626204
B2_BASELINE = 0.2228111863314196

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, REPO)

from encode_bones_smoke import (  # noqa: E402  (read-only module, reused not copied)
    LATTICE_TOL,
    _qn,
    _qmul,
    _qconj,
    _quat_rotate_inverse,
    joint_order_evidence,
    official_walk_window,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--fk-stride", type=int, default=3,
                    help="subsample stride for the FK foot-clearance check")
    args = ap.parse_args()

    from apt_g1.envs.mujoco_g1_flat_env import (
        G1_ISAACLAB_TO_MUJOCO_DOF,
        MujocoG1FlatEnv,
        SONIC_DEFAULT_ANGLES_MUJOCO,
        G1_MUJOCO_TO_ISAACLAB_DOF,
    )
    from eval_distill import NoQuantDecoder

    m2i = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)
    i2m = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF)
    default_mj = SONIC_DEFAULT_ANGLES_MUJOCO.astype(np.float64)

    res = {
        "exp": "D036",
        "line": "DS plan-B B2 control: official WALK tokens through the B2 roundtrip harness",
        "purpose": (f"discriminate B2 roundtrip MAE {B2_MAE:.4f} rad (baseline {B2_BASELINE:.4f}): "
                    "harness metric magnitude vs our encode path defect"),
        "official_csv": OFFICIAL_CSV,
        "motion_csv": OFFICIAL_MOTION,
    }

    # ---- official tokens (usecols: only the 64 token columns matter here)
    tokens_all = np.genfromtxt(OFFICIAL_CSV, delimiter=",", dtype=np.float32,
                               filling_values=0.0, usecols=list(range(64)))
    print(f"[csv] policy_input token cols: {tokens_all.shape}")

    # ---- target_motion structure probe
    tm = np.genfromtxt(OFFICIAL_MOTION, delimiter=",", dtype=np.float64, filling_values=0.0)
    print(f"[csv] target_motion: {tm.shape}")
    layout_note = f"raw {tm.shape[1]} fields"
    if tm.shape[1] == 37 and np.all(tm[:, 36] == 0.0):
        tm = tm[:, :36]
        layout_note += "; dropped trailing all-zero field (trailing comma artifact)"
    assert tm.shape[1] == 36, f"target_motion has {tm.shape[1]} usable cols, expected 36 (3+4+29)"
    trans_all = tm[:, 0:3]
    quat_raw = tm[:, 3:7]
    dof_all = tm[:, 7:36]
    qn = quat_raw / np.linalg.norm(quat_raw, axis=1, keepdims=True)
    norm_dev = float(np.abs(np.linalg.norm(quat_raw, axis=1) - 1.0).max())
    q0 = _qn(quat_raw[1896])  # first row of the walk window
    qlayout = "wxyz" if abs(q0[0]) >= abs(q0[3]) else "xyzw"
    assert qlayout == "wxyz", f"unexpected quat layout {qlayout} (q0={q0})"
    res["target_motion"] = {
        "n_rows": int(tm.shape[0]),
        "n_cols_used": 36,
        "layout": "col0-2 root xyz (deploy frame, col0=root x); col3-6 root quat wxyz; col7-35 29 joints",
        "layout_note": layout_note,
        "quat_wxyz_confidence": f"window row1896 quat={np.round(q0, 4).tolist()}; "
                                f"max |row-norm-1| = {norm_dev:.2e}",
        "joint_col_range": [7, 36],
    }
    print(f"[tm] {layout_note}; quat wxyz (norm dev {norm_dev:.2e})")

    # ---- window (same source as B2 check2)
    r0, r1, src = official_walk_window(OFFICIAL_EVENTS, OFFICIAL_MOTION)
    n_win = r1 - r0
    ref_disp = float(trans_all[r1 - 1, 0] - trans_all[r0, 0])
    ref_vx = ref_disp / (n_win / FPS)
    ref_path = float(np.linalg.norm(np.diff(trans_all[r0:r1, :2], axis=0), axis=1).sum())
    print(f"[win] rows [{r0},{r1}) {n_win} rows ({src}); ref x disp {ref_disp:.1f} m "
          f"-> vx {ref_vx:.3f} m/s; horiz path len {ref_path:.1f} m")

    # ---- joint order decision (two discriminators, subsampled FK)
    env = MujocoG1FlatEnv(NoQuantDecoder(DEC_ONNX), REPO,
                          use_elastic_band=False, stand_only=True)
    sub = slice(r0, r1, max(1, args.fk_stride))
    ev = joint_order_evidence(env, dof_all[sub], qn[sub], trans_all[sub], m2i, i2m, default_mj)
    fk = {h: ev[h].get("foot_check") if isinstance(ev[h].get("foot_check"), dict) else None
          for h in ev}
    order = (min(fk, key=lambda h: fk[h]["mean_abs_foot_z"])
             if all(fk.get(h) for h in fk)
             else max(ev, key=lambda h: ev[h]["corr_with_default"]))
    print(f"[order] decision: target_motion joints are {order.upper()} order")
    for h in ev:
        print(f"[order]   hyp {h}: {ev[h]}")
    dof_mj_all = dof_all if order == "mujoco" else dof_all[:, i2m]
    res["joint_order"] = {"decision": order, "evidence": ev,
                          "fk_stride": args.fk_stride}

    # ---- windowed reference in MuJoCo + Isaac orders
    jp_mj = dof_mj_all[r0:r1]
    quat = qn[r0:r1]
    n = len(jp_mj)
    jp_isaac = jp_mj[:, m2i]
    jv_mj = np.vstack([np.zeros((1, 29)), np.diff(jp_mj, axis=0) * FPS])
    jv_isaac = jv_mj[:, m2i]

    # body-frame angular velocity from root quat finite diff + gravity dir
    # (check3 formulas verbatim)
    omega_body = np.zeros((n, 3))
    for t in range(n):
        a, b = quat[min(t + 1, n - 1)], quat[max(t - 1, 0)]
        step = (min(t + 1, n - 1) - max(t - 1, 0)) / FPS
        dq = _qmul(a, _qconj(b))
        if dq[0] < 0:
            dq = -dq
        w_world = 2.0 * dq[1:] / max(dq[0], 1e-6) / max(step, 1e-6)
        omega_body[t] = _quat_rotate_inverse(quat[t], w_world)
    grav = np.array([_quat_rotate_inverse(qq, np.array([0.0, 0.0, -1.0])) for qq in quat])

    # ---- roundtrip (check3 machine; oracle proprio = target_motion itself)
    dec = env.sonic_decoder
    din = dec.session.get_inputs()[0]
    ddim = din.shape[1] if isinstance(din.shape[1], int) else None
    if ddim != 994:
        raise SystemExit(f"[rt] FAIL: decoder input dim {ddim} != 994")
    lat = tokens_all[r0:r1].astype(np.float64) * 16.0
    viol_off = float((np.abs(lat - np.round(lat)) > LATTICE_TOL).mean())

    def sonic_history(t):
        idx = np.clip(np.arange(t - 9, t + 1), 0, n - 1)
        return {
            "base_angular_velocity": omega_body[idx].astype(np.float32),
            "body_joint_positions": ((jp_mj[idx] - default_mj)[:, m2i]).astype(np.float32),
            "body_joint_velocities": jv_isaac[idx].astype(np.float32),
            "last_actions": (((jp_mj[idx] - default_mj) / env.sonic_scale_mujoco)[:, m2i]).astype(np.float32),
            "gravity_dir": grav[idx].astype(np.float32),
        }

    err, err0 = [], []
    q_des_all = np.zeros((n, 29), dtype=np.float32)
    for t in range(n):
        tok = tokens_all[r0 + t]
        obs = dec.build_decoder_obs(tok, sonic_history(t))
        act_isaac = dec.session.run([dec.output_name], {dec.input_name: obs})[0][0]
        q_des_isaac = env.sonic_default_isaac + act_isaac.astype(np.float64) * env.sonic_scale_isaac
        q_des_all[t] = q_des_isaac.astype(np.float32)
        err.append(np.abs(q_des_isaac - jp_isaac[t]))
        err0.append(np.abs(default_mj[m2i] - jp_isaac[t]))
    err = np.asarray(err)
    err0 = np.asarray(err0)
    pj = err.mean(axis=0)
    pj0 = err0.mean(axis=0)
    worst = np.argsort(-pj)[:5]
    mae = float(err.mean())
    mae0 = float(err0.mean())
    if mae <= 0.30:
        verdict = "ENCODE_PATH_SUSPECT (official << our 0.564)"
    elif mae >= 0.40:
        verdict = "HARNESS_MAGNITUDE (official ~ our 0.564 -> frame-aligned metric is inherently strict)"
    else:
        verdict = "AMBIGUOUS (official between 0.30 and 0.40)"
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    q_des_path = os.path.join(out_dir, "q_des_official_roundtrip.npy")
    np.save(q_des_path, q_des_all)
    res.update({
        "window": {"rows": [r0, r1], "n_rows": int(n), "source": src,
                   "ref_vx_mps": round(ref_vx, 4),
                   "ref_horiz_path_len_m": round(ref_path, 2),
                   "ref_x_disp_m": round(ref_disp, 2)},
        "official_lattice_violation_rate": viol_off,
        "roundtrip": {
            "condition": ("oracle: live proprio from target_motion.csv itself "
                          "(official planner reference, 50 Hz native); last_actions=reference normalized action"),
            "decoder_input": {"name": din.name, "shape": list(din.shape)},
            "mae_rad": mae,
            "mae_default_stance_baseline_rad": mae0,
            "per_joint_mae_isaac_order": [round(float(x), 4) for x in pj],
            "per_joint_mae_default_baseline": [round(float(x), 4) for x in pj0],
            "per_joint_mae_top5": [{"isaac_dim": int(j), "mae_rad": round(float(pj[j]), 4),
                                    "baseline_rad": round(float(pj0[j]), 4)} for j in worst],
            "q_des_npy": q_des_path,
        },
        "comparison": {
            "b2_ours_mae_rad": B2_MAE, "b2_ours_baseline_rad": B2_BASELINE,
            "official_mae_rad": mae, "official_baseline_rad": mae0,
            "ratio_official_vs_ours": mae / B2_MAE,
            "verdict": verdict,
        },
    })
    print(f"[rt] OFFICIAL token roundtrip MAE = {mae:.4f} rad "
          f"(default-stance baseline {mae0:.4f} rad); ours was {B2_MAE:.4f}/{B2_BASELINE:.4f}")
    print(f"[rt] worst Isaac dims: {[(int(j), round(float(pj[j]), 3)) for j in worst]}")
    print(f"[verdict] {verdict}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[out] JSON -> {args.out}")


if __name__ == "__main__":
    main()
