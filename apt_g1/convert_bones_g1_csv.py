"""D044 / DS official-data line: BONES-SEED G1 native CSV -> 1762-d g1-mode
encoder obs -> frozen release encoder tokens (+ optional decoder roundtrip).

Data (D043): 142,220 CSVs on lab-ts under
  ~/ros2_data/apt_g1/data/ds_bones/g1/csv/{date}/*.csv
36 columns = Frame + root_translate XYZ (cm) + root_rotate XYZ (Euler, deg)
+ 29 semantic-named dof columns (deg); 120 fps native, frame step constant 1.
The 29 dof columns are already G1 retargeted by NVIDIA (no M2 needed) and the
semantic order matches the MuJoCo XML joint order (name-asserted here, never
assumed).

Converter five points (D043 -> this script):
  1. 120 -> 50 Hz linear resample (D036 precedent, same helpers)
  2. deg -> rad
  3. cm -> m
  4. root Euler -> quaternion wxyz -- the axis ORDER is DECIDED EMPIRICALLY,
     not assumed: `--calibrate` sweeps the 6 intrinsic Euler conventions
     against the paired official sample_data pkl (same take, both formats
     exist for walk_forward_amateur_001__A001, 210531) and picks the minimum
     mean rotation error. D036 ref-rel anchor semantics downstream make the
     anchor yaw-invariant, but roll/pitch feed gravity_dir + omega of the
     decoder history, so the convention must still be right.
  5. 29 dof column names -> deploy joint order: name-asserted mapping to
     MuJoCo order, then G1_MUJOCO_TO_ISAACLAB_DOF for the encoder obs
     (build_obs consumes Isaac-order jp/jv).

Obs construction is imported from encode_bones_smoke.py VERBATIM (the
D036/D038-verified v2 ref-rel path, roundtrip 0.109 rad): build_obs,
resample, resample_quat, quat utils. The decoder roundtrip mirrors that
script's check-3 sonic_history exactly (default-relative positions,
normalized last_actions, finite-diff omega in body frame, gravity dir).

Outputs (server): one npz per segment under --out-dir with
  tokens (n,64) f32 @50Hz | jp_isaac/jp_mj (n,29) | quat_wxyz (n,4) |
  trans_m (n,3) | jv_isaac (n,29) | meta json string
and a manifest.json (consumed by isaac/b3p_gate_isaac.py for the B3' gate).

D045 (B4-lite protocol §5, refine-logs/DS_B4LITE_ACTION_SPLIT_PLAN.md):
every successful meta additionally carries a nested "b4lite" block --
source / frames / actor_group / semantics / kinematics / quality / split /
exclusion / flat_replay (the four-question manifest). Defaults are
null/pass; --labels-json merges semantics hits; --tar-version stamps the
source archive. flat_replay is backfilled from the B3' gate result by
backfill_b4lite_manifest.py and frozen there.

Usage (server, mjlab venv, cwd=~/ros2_data/apt_g1):
  # 1) calibrate Euler axis order against the paired pkl (once):
  python convert_bones_g1_csv.py --calibrate \
      --pkl data/ds_bones/b1/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
      --csv-root data/ds_bones/g1/csv --out data/ds_bones/g1_b3p/calibration.json
  # 2) sample >=10 segments per core class (walk/run/jump/dance), distinct
  #    actors, no _M mirrors, then convert with roundtrip MAE:
  python convert_bones_g1_csv.py --sample 12 \
      --csv-root data/ds_bones/g1/csv --calibration-json <calib.json> \
      --out-dir data/ds_bones/g1_b3p/npz --roundtrip
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
HOME = os.path.expanduser("~")
DEFAULT_CSV_ROOT = f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1/csv"
DEFAULT_OUT_DIR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/g1_b3p"
FPS_SRC = 120.0  # BONES-SEED G1 CSV native rate (D043)
FPS_ENC = 50.0  # encoder stride-5 assumption (planner_sonic.py)
DEG2RAD = np.pi / 180.0

# D043-verified CSV header (36 cols). Asserted verbatim on every load.
CSV_COLUMNS = [
    "Frame",
    "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
    "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
    "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
    "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof", "right_wrist_yaw_joint_dof",
]

# Canonical MuJoCo G1 29-dof joint order (gear_sonic deploy semantics; the
# CSV dof columns are semantically named, so the mapping is by NAME).
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Core classes for the B3' gate (DS_OFFICIAL_DATA_PLAN D2 default list;
# "run" is realised by the jog_/run_ prefixes per M1 jog-family evidence).
CLASS_PATTERNS = {
    "walk": re.compile(r"^walk_"),
    "run": re.compile(r"^(jog|run)_"),
    "jump": re.compile(r"^jump_"),
    "dance": re.compile(r"^(dance|dancing)_"),
}

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, REPO)

# D036-verified pieces, zero-drift import (see module docstring).
from encode_bones_smoke import (  # noqa: E402
    FPS_ENC as _FPS_ENC_SMOKE,
    LATTICE_TOL,
    _heading,
    _heading_inv,
    _qconj,
    _qn,
    _qmul,
    _quat_rotate_inverse,
    _rotmat,
    build_obs,
    load_pkl,
    resample,
    resample_quat,
    unwrap,
)

assert _FPS_ENC_SMOKE == FPS_ENC, "encoder rate assumption diverged from encode_bones_smoke"


# ------------------------------------------------------------ Euler handling
def _mat_rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _mat_ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _mat_rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_AXES = {"x": _mat_rx, "y": _mat_ry, "z": _mat_rz}

# The 6 intrinsic Euler conventions ("i:xyz" = rotate about body x, then new
# y, then new z == matrix product Rx@Ry@Rz). Extrinsic orders are the same 6
# matrices with reversed angle assignment, so sweeping these 6 covers all.
EULER_CONVENTIONS = [
    "i:xyz", "i:xzy", "i:yxz", "i:yzx", "i:zxy", "i:zyx",
]


def euler_to_mat(euler_deg, conv):
    """Euler angles (N,3) in degrees -> rotation matrices (N,3,3) for the
    given intrinsic convention."""
    ang = np.asarray(euler_deg, dtype=np.float64) * DEG2RAD
    mats = np.zeros((len(ang), 3, 3))
    for i, (ax, ay, az) in enumerate(ang):
        m = {"x": ax, "y": ay, "z": az}
        order = conv.split(":")[1]
        R = np.eye(3)
        for axis in order:  # intrinsic: first rotation goes leftmost
            R = R @ _AXES[axis](m[axis])
        mats[i] = R
    return mats


def mat_to_quat_wxyz(R):
    """Rotation matrices (N,3,3) -> quats (N,4) wxyz (Shepperd)."""
    R = np.asarray(R, dtype=np.float64)
    n = len(R)
    q = np.zeros((n, 4))
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(n):
        tr = t[i]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            q[i] = [0.25 * s, (R[i, 2, 1] - R[i, 1, 2]) / s,
                    (R[i, 0, 2] - R[i, 2, 0]) / s, (R[i, 1, 0] - R[i, 0, 1]) / s]
        else:
            d = np.argmax(np.diag(R[i]))
            if d == 0:
                s = np.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2
                q[i] = [(R[i, 2, 1] - R[i, 1, 2]) / s, 0.25 * s,
                        (R[i, 0, 1] + R[i, 1, 0]) / s, (R[i, 0, 2] + R[i, 2, 0]) / s]
            elif d == 1:
                s = np.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2]) * 2
                q[i] = [(R[i, 0, 2] - R[i, 2, 0]) / s, (R[i, 0, 1] + R[i, 1, 0]) / s,
                        0.25 * s, (R[i, 1, 2] + R[i, 2, 1]) / s]
            else:
                s = np.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1]) * 2
                q[i] = [(R[i, 1, 0] - R[i, 0, 1]) / s, (R[i, 0, 2] + R[i, 2, 0]) / s,
                        (R[i, 1, 2] + R[i, 2, 1]) / s, 0.25 * s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def euler_to_quat_wxyz(euler_deg, conv):
    return mat_to_quat_wxyz(euler_to_mat(euler_deg, conv))


def quat_angle_error(qa, qb):
    """Per-frame rotation angle (rad) between two wxyz quat sets."""
    out = np.zeros(len(qa))
    for i in range(len(qa)):
        d = _qmul(_qn(qa[i]), _qconj(_qn(qb[i])))
        if d[0] < 0:
            d = -d
        out[i] = 2.0 * np.arctan2(np.linalg.norm(d[1:]), d[0])
    return out


# ------------------------------------------------------------------ CSV load
def class_of(stem):
    for cls, pat in CLASS_PATTERNS.items():
        if pat.match(stem):
            return cls
    return stem.split("_")[0]


def parse_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    actor = None
    m = re.search(r"__(A\d+)", stem)
    if m:
        actor = m.group(1)
    return stem, actor, stem.endswith("_M"), class_of(stem)


def load_csv(path):
    """Returns (dof_deg_mj, euler_deg, trans_m, n_frames, frame_start,
    frame_end, warnings).

    Order is BY NAME (never positional); units deg+cm per D043.
    frame_start/frame_end are the raw Frame-column first/last values
    (D045 B4-lite manifest "frames" question).
    """
    with open(path) as f:
        header = f.readline().strip().split(",")
    if header != CSV_COLUMNS:
        raise ValueError(f"header mismatch in {path}: {len(header)} cols, "
                         f"first diff at {[i for i, (a, b) in enumerate(zip(header, CSV_COLUMNS)) if a != b][:3]}")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if raw.shape[1] != 36:
        raise ValueError(f"expected 36 numeric cols, got {raw.shape[1]} in {path}")
    frame = raw[:, 0]
    warn = []
    if not np.all(np.diff(frame) == 1):
        warn.append(f"frame step not constant: {np.unique(np.diff(frame))[:5]}")
    dof_cols = header[7:]
    dof_names = [c[: -len("_dof")] for c in dof_cols]
    if sorted(dof_names) != sorted(MUJOCO_JOINT_NAMES):
        raise ValueError(f"dof column names do not match MuJoCo 29 set in {path}")
    csv2mj = np.asarray([dof_names.index(nm) for nm in MUJOCO_JOINT_NAMES])
    dof_deg_mj = raw[:, 7:][:, csv2mj] * DEG2RAD  # -> MuJoCo order, rad
    euler_deg = raw[:, 4:7]
    trans_m = raw[:, 1:4] / 100.0
    return (dof_deg_mj, euler_deg, trans_m, len(raw),
            float(frame[0]), float(frame[-1]), warn)


# --------------------------------------------- B4-lite manifest block (D045)
B4LITE_DEFAULT_TAR_VERSION = "g1.tar.gz@D043-scp-bitwise-verified"


def kinematics_stats(trans_m):
    """v_med / v_p90 (m/s @50 Hz) + total heading change (rad), from the npz
    trans_m via transl differencing (D040 M1 method).

    Zero-displacement segments are legal (in-place actions -> v_med = 0);
    heading accumulation skips sub-mm steps so atan2 noise on stationary
    frames cannot inflate the total.
    """
    d = np.diff(np.asarray(trans_m)[:, :2], axis=0)
    step = np.linalg.norm(d, axis=1)
    v = step * FPS_ENC
    moving = step > 1e-3
    if moving.sum() >= 2:
        h = np.arctan2(d[moving, 1], d[moving, 0])
        dh = np.abs((np.diff(h) + np.pi) % (2.0 * np.pi) - np.pi)
        heading_total = float(dh.sum())
    else:
        heading_total = 0.0
    return float(np.median(v)), float(np.percentile(v, 90)), heading_total


def build_b4lite_block(stem, mirror, csv_path, csv_root, tar_version,
                       frame_start, frame_end, trans_m, labels=None):
    """D045: B4-lite protocol §5 four-question manifest fields, nested as a
    single meta["b4lite"] sub-dict (D044 top-level fields untouched -> old
    readers, e.g. b3p_gate_isaac.py, stay source-compatible). Everything
    defaults to null/pass; --labels-json hits (JSON keyed by stem) merge
    into semantics. quality is S3's, split is S4's, flat_replay is
    backfilled from gate_result.json by backfill_b4lite_manifest.py --
    the npz-embedded meta stays the fact source (D044 pitfall #1)."""
    v_med, v_p90, heading_total = kinematics_stats(trans_m)
    semantics = {
        "family": None,
        "labels_5dim": {"locomotion": None, "upper_body": None,
                        "posture": None, "contact": None, "temporal": None},
        "label_confidence": None,
        "label_sources": {"filename": None, "trajectory": None, "manual": None},
    }
    if labels:
        for k, v in labels.items():  # shallow merge; sub-dicts update key-wise
            if isinstance(semantics.get(k), dict) and isinstance(v, dict):
                semantics[k].update(v)
            else:
                semantics[k] = v
    try:
        csv_relpath = os.path.relpath(csv_path, csv_root)
    except ValueError:  # e.g. different drive -- keep the absolute path
        csv_relpath = csv_path
    return {
        "source": {"csv_relpath": csv_relpath, "tar_version": tar_version},
        "frames": {"frame_start": frame_start, "frame_end": frame_end},
        "actor_group": {"parent_take": stem[:-2] if mirror else stem},
        "semantics": semantics,
        "kinematics": {"v_med": v_med, "v_p90": v_p90,
                       "heading_change_total": heading_total},
        "quality": {"layer1": "pass", "layer2": "pass", "layer3": "none",
                    "boundary_set": False, "quality_notes": []},
        "split": {"set_role": None, "t_role": None,
                  "repr_participation": {"vae": None, "norm_stats": None,
                                         "sonic_pretrain": "not_excludable"}},
        "exclusion": {"excluded": False, "reason": None},
        # 占位：gate_result.json 回填后冻结（backfill_b4lite_manifest.py，
        # 协议 §5「平地回放结果」问）；回填前保持 null，不伪造
        "flat_replay": None,
    }


# ------------------------------------------------------------- encoder chain
def encode_segment(dof_deg_mj, euler_deg, trans_m, conv, enc, iname,
                   m2i, default_mj, anchor="ref-rel", dec_bundle=None):
    """CSV arrays (rad, deg, m; native 120 fps, MuJoCo order) -> tokens +
    reference arrays + optional decoder roundtrip. Mirrors encode_bones_smoke
    main flow (30Hz pkl there, 120fps CSV here)."""
    dof_mj = dof_deg_mj.astype(np.float64)
    quat = euler_to_quat_wxyz(euler_deg, conv)
    dof_mj_rs = resample(dof_mj, FPS_SRC, FPS_ENC)
    quat_rs = resample_quat(quat, FPS_SRC, FPS_ENC)
    trans_rs = resample(trans_m, FPS_SRC, FPS_ENC)
    n_rs = len(dof_mj_rs)

    jp_mj = dof_mj_rs
    jp_isaac = jp_mj[:, m2i]
    # planner_sonic.py L109: finite diff gives rad/step; encoder wants rad/s
    jv_mj = np.vstack([np.zeros((1, 29)), np.diff(jp_mj, axis=0) * FPS_ENC])
    jv_isaac = jv_mj[:, m2i]
    bq = np.asarray(quat_rs, dtype=np.float64).reshape(-1, 1, 4)
    apply_delta = _qn(_qmul(_heading(np.array([1.0, 0, 0, 0])), _heading_inv(bq[0, 0])))

    tokens = np.zeros((n_rs, 64), dtype=np.float32)
    for t in range(n_rs):
        obs = build_obs(t, jp_isaac, jv_isaac, bq, apply_delta, anchor=anchor)
        tokens[t] = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)

    # ref-rel sanity at t=0: f=0 anchor is exactly identity
    obs0 = build_obs(0, jp_isaac, jv_isaac, bq, apply_delta, anchor=anchor)
    sanity = float(np.abs(obs0[601:607] - np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)).max())
    if sanity > 1e-5:
        raise RuntimeError(f"f=0 anchor sanity broken: {sanity}")

    lat = tokens.astype(np.float64) * 16.0
    lattice_rate = float((np.abs(lat - np.round(lat)) > LATTICE_TOL).mean())

    out = {
        "tokens": tokens, "jp_mj": jp_mj, "jp_isaac": jp_isaac,
        "jv_isaac": jv_isaac, "quat_wxyz": quat_rs, "trans_m": trans_rs,
        "lattice_rate": lattice_rate, "anchor_sanity": sanity, "n_rows": n_rs,
        "roundtrip_mae": None, "roundtrip_mae_default_baseline": None,
    }

    if dec_bundle is not None:
        env = dec_bundle["env"]
        dec = env.sonic_decoder
        m2i_b, default_mj_b = dec_bundle["m2i"], dec_bundle["default_mj"]
        # body-frame angular velocity from root quat finite diff (smoke verbatim)
        omega_body = np.zeros((n_rs, 3))
        for t in range(n_rs):
            a, b = quat_rs[min(t + 1, n_rs - 1)], quat_rs[max(t - 1, 0)]
            step = (min(t + 1, n_rs - 1) - max(t - 1, 0)) / FPS_ENC
            dq = _qmul(a, _qconj(b))
            if dq[0] < 0:
                dq = -dq
            w_world = 2.0 * dq[1:] / max(dq[0], 1e-6) / max(step, 1e-6)
            omega_body[t] = _quat_rotate_inverse(quat_rs[t], w_world)
        grav = np.array([_quat_rotate_inverse(qq, np.array([0.0, 0.0, -1.0])) for qq in quat_rs])

        err, err0 = [], []
        for t in range(n_rs):
            idx = np.clip(np.arange(t - 9, t + 1), 0, n_rs - 1)
            hist = {
                "base_angular_velocity": omega_body[idx].astype(np.float32),
                "body_joint_positions": ((jp_mj[idx] - default_mj_b)[:, m2i_b]).astype(np.float32),
                "body_joint_velocities": jv_isaac[idx].astype(np.float32),
                "last_actions": (((jp_mj[idx] - default_mj_b) / env.sonic_scale_mujoco)[:, m2i_b]).astype(np.float32),
                "gravity_dir": grav[idx].astype(np.float32),
            }
            obs = dec.build_decoder_obs(tokens[t], hist)
            act_isaac = dec.session.run([dec.output_name], {dec.input_name: obs})[0][0]
            q_des_isaac = env.sonic_default_isaac + act_isaac.astype(np.float64) * env.sonic_scale_isaac
            err.append(np.abs(q_des_isaac - jp_isaac[t]))
            err0.append(np.abs(default_mj_b[m2i_b] - jp_isaac[t]))
        out["roundtrip_mae"] = float(np.mean(err))
        out["roundtrip_mae_default_baseline"] = float(np.mean(err0))
    return out


# ------------------------------------------------------------------ sampling
def build_sample_list(csv_root, per_class, seed, min_rows_src=240):
    """>=per_class segments per core class: distinct actors, no _M mirror,
    deterministic seed. Durations unknown pre-load; min_rows_src (2s @120fps)
    is enforced post-load by the converter via manifest filtering."""
    by_class = {c: [] for c in CLASS_PATTERNS}
    for path in sorted(glob.glob(os.path.join(csv_root, "*", "*.csv"))):
        stem, actor, mirror, cls = parse_stem(path)
        if cls in by_class and not mirror and actor:
            by_class[cls].append((path, actor))
    rng = random.Random(seed)
    picks = {}
    for cls, cands in by_class.items():
        rng.shuffle(cands)
        chosen, used = [], set()
        for path, actor in cands:
            if actor in used:
                continue
            chosen.append(path)
            used.add(actor)
            if len(chosen) >= per_class:
                break
        picks[cls] = chosen
        print(f"[sample] {cls}: {len(chosen)}/{per_class} (pool {len(cands)} unique-actor candidates)")
    return picks


# ----------------------------------------------------------------- calibrate
def calibrate(args):
    """Decide the Euler axis order empirically: paired official pkl (30 Hz,
    root_rot wxyz) vs same-take CSV (120 fps, root_rotate XYZ deg)."""
    obj = unwrap(load_pkl(args.pkl)[0])
    fps_pkl = float(obj["fps"])
    dof_pkl = obj["dof"].astype(np.float64)
    quat_pkl = obj["root_rot"].astype(np.float64)
    q0 = _qn(quat_pkl[0])
    layout = "wxyz" if abs(q0[0]) >= abs(q0[3]) else "xyzw"
    if layout != "wxyz":
        quat_pkl = quat_pkl[:, [3, 0, 1, 2]]
    quat_pkl = quat_pkl / np.linalg.norm(quat_pkl, axis=1, keepdims=True)
    trans_pkl = obj["root_trans_offset"].astype(np.float64)

    stem = os.path.splitext(os.path.basename(args.pkl))[0]
    date_dir = os.path.basename(os.path.dirname(args.pkl))
    csv_path = os.path.join(args.csv_root, date_dir, stem + ".csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"paired CSV not found: {csv_path}")
    dof_csv_mj, euler_csv, trans_csv_m, n_rows_csv, _fs, _fe, warn = load_csv(csv_path)

    # CSV -> pkl rate for frame-aligned comparison
    dof_csv_rs = resample(dof_csv_mj, FPS_SRC, fps_pkl)
    trans_csv_rs = resample(trans_csv_m, FPS_SRC, fps_pkl)
    n = min(len(dof_pkl), len(dof_csv_rs))

    # dof order+units+frame-lag check (semantic-name order vs pkl as-loaded)
    lags = {}
    for lag in range(-10, 11):
        a = dof_csv_rs[max(0, lag): n + min(0, lag)]
        b = dof_pkl[max(0, -lag): n + min(0, -lag)]
        lags[lag] = float(np.abs(a[: len(b)] - b[: len(a)]).mean())
    best_lag = min(lags, key=lags.get)
    dof_mae = lags[best_lag]

    # trans unit/origin check at best lag: per-axis median diff (m) + Z ratio
    o = best_lag
    a = trans_csv_rs[max(0, o): n + min(0, o)]
    b = trans_pkl[max(0, -o): n + min(0, -o)]
    m = min(len(a), len(b))
    with np.errstate(all="ignore"):
        med_diff = np.median(a[:m] - b[:m], axis=0)
        z_ratio = float(np.nanmedian(a[:m, 2] / b[:m, 2])) if m > 10 else float("nan")

    convs = {}
    for conv in EULER_CONVENTIONS:
        quat_csv = resample_quat(euler_to_quat_wxyz(euler_csv, conv), FPS_SRC, fps_pkl)
        err = quat_angle_error(quat_csv[:n], quat_pkl[:n])
        convs[conv] = {
            "mean_deg": round(float(np.degrees(err.mean())), 3),
            "p50_deg": round(float(np.degrees(np.median(err))), 3),
            "p95_deg": round(float(np.degrees(np.percentile(err, 95))), 3),
            "max_deg": round(float(np.degrees(err.max())), 3),
        }
    best_conv = min(convs, key=lambda c: convs[c]["mean_deg"])

    res = {
        "exp": "D044 calibration",
        "pkl": args.pkl, "csv": csv_path,
        "fps_pkl": fps_pkl, "fps_csv": FPS_SRC, "n_pkl": len(dof_pkl),
        "n_csv_src": n_rows_csv, "csv_warnings": warn,
        "pkl_quat_layout": layout,
        "euler_conventions": convs,
        "best_euler_convention": best_conv,
        "dof_mae_rad_at_best_lag": round(dof_mae, 5),
        "dof_lag_scan_pkl_frames": {str(k): round(v, 5) for k, v in lags.items()},
        "best_lag_pkl_frames": best_lag,
        "trans_median_diff_m_per_axis": [round(float(x), 4) for x in med_diff],
        "trans_z_median_ratio_csv_over_pkl": round(z_ratio, 5),
        "note": ("CSV dof deg->rad, name-mapped to MuJoCo order; rotation error "
                 "deg between CSV-Euler quats (resampled to pkl rate) and pkl root_rot. "
                 "dof_mae ~0 => order+units+lag all confirmed."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in
                      ("best_euler_convention", "dof_mae_rad_at_best_lag", "best_lag_pkl_frames",
                       "trans_median_diff_m_per_axis", "trans_z_median_ratio_csv_over_pkl")}, indent=1))
    for c, v in convs.items():
        print(f"[calib] {c}: mean {v['mean_deg']:6.2f} deg  p95 {v['p95_deg']:7.2f}  max {v['max_deg']:7.2f}")
    print(f"[calib] -> {args.out}")
    return res


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--csv-root", default=DEFAULT_CSV_ROOT)
    ap.add_argument("--out", default=f"{DEFAULT_OUT_DIR}/calibration.json")
    ap.add_argument("--calibration-json", default=None,
                    help="read best_euler_convention from a calibration result")
    ap.add_argument("--euler-conv", default=None, choices=EULER_CONVENTIONS)
    ap.add_argument("--sample", type=int, default=0,
                    help="build a sampled conversion list: N segments per core class")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--list", default=None, help="file with one CSV path per line")
    ap.add_argument("--csv", default=None, help="convert a single CSV")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--roundtrip", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tar-version", default=B4LITE_DEFAULT_TAR_VERSION,
                    help="source archive version string for the B4-lite "
                         "manifest (D045, protocol §5 source/version)")
    ap.add_argument("--labels-json", default=None,
                    help="JSON {stem: {family/labels_5dim/label_confidence/"
                         "label_sources}} merged into the b4lite semantics "
                         "block when the stem hits (D045)")
    args = ap.parse_args()

    if args.calibrate:
        if not args.pkl:
            ap.error("--calibrate requires --pkl")
        calibrate(args)
        return

    # ---- resolve euler convention
    conv = args.euler_conv
    if conv is None and args.calibration_json:
        conv = json.load(open(args.calibration_json))["best_euler_convention"]
        print(f"[conv] euler convention from calibration: {conv}")
    if conv is None:
        ap.error("euler convention required: --euler-conv or --calibration-json")

    import onnxruntime as ort
    from apt_g1.envs.mujoco_g1_flat_env import (
        G1_MUJOCO_TO_ISAACLAB_DOF,
        SONIC_DEFAULT_ANGLES_MUJOCO,
        MujocoG1FlatEnv,
    )
    from eval_torque_srb import NoQuantDecoder

    m2i = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)
    default_mj = SONIC_DEFAULT_ANGLES_MUJOCO.astype(np.float64)
    enc = ort.InferenceSession(
        f"{REPO}/gear_sonic_deploy/policy/release/model_encoder.onnx",
        providers=["CPUExecutionProvider"])
    ins = enc.get_inputs()
    assert len(ins) == 1 and ins[0].shape[-1] == 1762, f"encoder input {ins[0].shape}"

    npz_dir = os.path.join(args.out_dir, "npz")
    os.makedirs(npz_dir, exist_ok=True)
    labels = {}
    if args.labels_json:
        with open(args.labels_json) as f:
            labels = json.load(f)
        print(f"[b4lite] labels json: {len(labels)} stems")
    dec_bundle = None
    if args.roundtrip:
        env = MujocoG1FlatEnv(NoQuantDecoder(
            f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"), REPO,
            use_elastic_band=False, stand_only=True)
        dec_bundle = {"env": env, "m2i": m2i, "default_mj": default_mj}

    # ---- resolve input file list (+ sampling mode)
    if args.sample:
        picks = build_sample_list(args.csv_root, args.sample, args.sample_seed)
        files = [p for c in sorted(picks) for p in picks[c]]
    elif args.list:
        files = [l.strip() for l in open(args.list) if l.strip()]
    elif args.csv:
        files = [args.csv]
    else:
        ap.error("one of --sample N / --list FILE / --csv FILE required")

    manifest, skipped = [], 0
    for i, path in enumerate(files):
        stem, actor, mirror, cls = parse_stem(path)
        npz_path = os.path.join(npz_dir, stem + ".npz")
        if os.path.isfile(npz_path) and not args.force:
            skipped += 1
            continue
        try:
            dof_mj, euler, trans_m, n_src, f_start, f_end, warn = load_csv(path)
            seg = encode_segment(dof_mj, euler, trans_m, conv, enc, ins[0].name,
                                 m2i, default_mj, dec_bundle=dec_bundle)
        except Exception as e:
            print(f"[convert] FAIL {stem}: {type(e).__name__}: {e}", flush=True)
            manifest.append({"stem": stem, "csv": path, "class": cls, "actor": actor,
                             "error": f"{type(e).__name__}: {e}"})
            continue
        meta = {
            "stem": stem, "actor": actor, "mirror": mirror, "class": cls,
            "fps_src": FPS_SRC, "fps_enc": FPS_ENC, "euler_conv": conv,
            "n_rows_src": n_src, "n_rows_enc": seg["n_rows"],
            "lattice_rate": seg["lattice_rate"],
            "roundtrip_mae": seg["roundtrip_mae"],
            "roundtrip_mae_default_baseline": seg["roundtrip_mae_default_baseline"],
            "path_len_m": float(np.linalg.norm(np.diff(seg["trans_m"][:, :2], axis=0), axis=1).sum()),
            "npz_path": os.path.abspath(npz_path),
            "warnings": warn,
            # D045: B4-lite §5 four-question fields, nested (backwards
            # compatible); npz meta + manifest are written from this same
            # dict so the two stay isomorphic (D044 pitfall #1).
            "b4lite": build_b4lite_block(
                stem, mirror, path, args.csv_root, args.tar_version,
                f_start, f_end, seg["trans_m"], labels=labels.get(stem)),
        }
        np.savez(npz_path,
                 tokens=seg["tokens"], jp_isaac=seg["jp_isaac"], jp_mj=seg["jp_mj"],
                 jv_isaac=seg["jv_isaac"], quat_wxyz=seg["quat_wxyz"],
                 trans_m=seg["trans_m"], meta=np.array(json.dumps(meta)))
        manifest.append(meta)
        rt = "" if seg["roundtrip_mae"] is None else f" rt_mae={seg['roundtrip_mae']:.4f}"
        print(f"[convert] ({i + 1}/{len(files)}) {stem}: {meta['n_rows_src']}@120 -> "
              f"{meta['n_rows_enc']}@50 class={cls} actor={actor} "
              f"lat={seg['lattice_rate']:.1e}{rt}", flush=True)

    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[manifest] {len(manifest)} entries -> {mpath}")
    ok = [m for m in manifest if "error" not in m]
    rts = [m["roundtrip_mae"] for m in ok if m.get("roundtrip_mae") is not None]
    print(f"SUMMARY: converted {len(ok)}/{len(manifest)} (skipped existing {skipped})"
          + (f" | roundtrip MAE mean {np.mean(rts):.4f} rad, max {np.max(rts):.4f}" if rts else ""))


if __name__ == "__main__":
    main()
