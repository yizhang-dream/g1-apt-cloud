"""D036 / DS plan-B line B2 smoke: offline-encode the official GEAR-SONIC
sample_data G1 walk motion (BONES-SEED format pkl) into SONIC tokens with the
frozen release encoder, then run three checks:

  1. lattice legality  : token*16 distance to nearest integer (same 0.05
                         tolerance as isaac/oracle_token_replay_isaac.py).
  2. distribution cmp  : per-dim mean/std vs the official closed-loop WALK
                         baseline token window (ds_smoke recording), mean-L2,
                         top-10 |mean diff| dims.
  3. decoder roundtrip : best-effort. Oracle condition (live proprio taken
                         from the reference trajectory itself), decode each
                         frame with the frozen release decoder, MAE(rad) of
                         29 joint targets vs reference jp.

Encoder obs layout = planner_sonic.py g1 mode, 1762-d:
  [0]    encoder_mode=0, [1:4) zero
  [4:294)   10 future frames joint pos  (IsaacLab DOF order, step 5 @ 50 Hz)
  [294:584) 10 future frames joint vel (rad/s @ 50 Hz)
  [584:601) 17 zeros (root_z + anchor_single, skipped exactly like planner_sonic)
  [601:661) 10 anchor 6D orientations: rot(btr)[:, :2]
            live = reference root quat at frame t (offline oracle)
  rest zero.

Usage (server, mjlab venv):
    cd ~/ros2_data/apt_g1 && ~/ros2_data/.venv_mjlab/bin/python \
        encode_bones_smoke.py --pkl <robot_filtered pkl> --out <json>

Anchor semantics (--anchor, default ref-rel since the D036 root-cause fix):
ref-rel = btr = conj(q_t)*q_idx (relative rotation along the reference; the
only self-consistent closed-loop form, yaw-invariant). heading-norm = the
original planner_sonic apply_delta block (kept for A/B; injects the reference
frame-0 heading -- invisible in planner_sonic where frame 0 is standing, but
a constant ~90 deg yaw for this pkl which starts at yaw ~ -87 deg). ref-rel
tokens are saved as tokens_<stem>_anchorrefrel.npy; heading-norm keeps the
original tokens_<stem>.npy name.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zlib

import numpy as np

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
HOME = os.path.expanduser("~")
ENC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_encoder.onnx"
DEC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"
DEFAULT_PKL = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b1/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl"
DEFAULT_OUT = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b2/smoke_result.json"
OFFICIAL_CSV = f"{HOME}/ros2_data/apt_g1/data/ds_smoke/policy_input.csv"
OFFICIAL_EVENTS = f"{HOME}/ros2_data/apt_g1/data/ds_smoke/events.json"
OFFICIAL_MOTION = f"{HOME}/ros2_data/apt_g1/data/ds_smoke/target_motion.csv"
# joblib is required to unpickle joblib.dump'ed pkl but is absent from both
# server venvs; a vendored pure-python copy is pip-installed --target here.
VENDOR = f"{HOME}/ros2_data/apt_g1/data/ds_bones/b2/_vendor"
FPS_ENC = 50.0  # encoder stride-5 assumption (planner_sonic.py)
LATTICE_TOL = 0.05  # same tolerance as oracle_token_replay_isaac.py L119-120

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, REPO)
if os.path.isdir(VENDOR):
    sys.path.insert(0, VENDOR)


# ---------------------------------------------------------------- quat utils
# copied verbatim from planner_sonic.py (do not drift)
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


def _quat_rotate_inverse(q, v):
    # copied from apt_g1/envs/mujoco_g1_flat_env.py quat_rotate_inverse
    q = q / np.linalg.norm(q)
    qv = q[1:]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q[0] * v)


# ------------------------------------------------------------------ pkl load
def load_pkl(path):
    """GEAR-SONIC BONES-SEED pkls are joblib.dump'ed; robot_* ones are
    zlib-compressed on top (magic 78 5e)."""
    raw = open(path, "rb").read()
    data = raw
    if raw[:1] == b"\x78":  # zlib header
        data = zlib.decompress(raw)
    import io

    import joblib

    return joblib.load(io.BytesIO(data)), data


def unwrap(obj):
    """Top level of these pkls is {<motion stem>: {field: array}}."""
    if isinstance(obj, dict) and len(obj) == 1:
        v = next(iter(obj.values()))
        if isinstance(v, dict):
            return v
    return obj


def pkl_summary(obj):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                s = {"shape": list(v.shape), "dtype": str(v.dtype)}
                if v.size and np.issubdtype(v.dtype, np.number):
                    with np.errstate(all="ignore"):
                        s["min"] = float(np.nanmin(v))
                        s["max"] = float(np.nanmax(v))
                out[k] = s
            else:
                out[k] = repr(v)[:60]
    return out


def sample_data_dir(pkl_path):
    sd = os.path.abspath(pkl_path)
    while os.path.basename(sd) != "sample_data" and os.path.dirname(sd) != sd:
        sd = os.path.dirname(sd)
    if os.path.basename(sd) != "sample_data":
        raise ValueError(f"cannot locate sample_data dir above {pkl_path}")
    return sd


def inventory(sample_dir):
    """Light structural inventory of every pkl under sample_data/."""
    inv = {}
    for path in sorted(glob.glob(os.path.join(sample_dir, "**", "*.pkl"), recursive=True)):
        try:
            obj, _ = load_pkl(path)
            inv[os.path.relpath(path, sample_dir)] = pkl_summary(unwrap(obj))
        except Exception as e:  # record, never block
            inv[os.path.relpath(path, sample_dir)] = {"error": f"{type(e).__name__}: {e}"}
    return inv


# ------------------------------------------------------ joint order evidence
def joint_order_evidence(env, dof_mj_order, quat_wxyz, trans, m2i, i2m, default_mj):
    """Two independent discriminators for whether the pkl `dof` array is in
    MuJoCo XML order (hyp A, needs [:, m2i] for Isaac) or already IsaacLab
    order (hyp B, needs [:, i2m] to reach MuJoCo for FK):
      (a) correlation of per-column mean with SONIC_DEFAULT_ANGLES_MUJOCO
          (mocap walking oscillates around the default stance);
      (b) MuJoCo FK foot clearance: under the right hypothesis feet hover
          near the ground and are anti-phase (walking)."""
    means = dof_mj_order.mean(axis=0)  # as-loaded columns
    ev = {}
    for hyp, col_mujoco in (
        ("mujoco", means),  # as-loaded IS MuJoCo order
        ("isaac", means[i2m]),  # as-loaded is Isaac -> map to MuJoCo
    ):
        d = default_mj - col_mujoco
        r = float(np.corrcoef(col_mujoco, default_mj)[0, 1])
        ev[hyp] = {"corr_with_default": round(r, 4), "max_abs_mean_minus_default": round(float(np.max(np.abs(d))), 4)}

    import mujoco as mjc

    def foot_bodies():
        lo = ri = None
        for i in range(env.model.nbody):
            nm = env.model.body(i).name
            if "ankle_roll" in nm and "left" in nm:
                lo = i
            if "ankle_roll" in nm and "right" in nm:
                ri = i
        return lo, ri

    lo, ri = foot_bodies()
    if lo is None or ri is None:
        for hyp in ev:
            ev[hyp]["foot_check"] = "skipped: no ankle_roll body found"
        return ev

    q = np.zeros(env.model.nq)
    zl, zr = {}, {}
    for hyp, cols in (("mujoco", None), ("isaac", i2m)):
        dof_mj = dof_mj_order if cols is None else dof_mj_order[:, cols]
        a, b = [], []
        for t in range(len(dof_mj)):
            q[:3] = trans[t]
            q[3:7] = quat_wxyz[t]
            q[env.body_qpos_adr] = dof_mj[t]
            env.data.qpos[:] = q
            mjc.mj_forward(env.model, env.data)
            a.append(env.data.xpos[lo][2])
            b.append(env.data.xpos[ri][2])
        zl[hyp], zr[hyp] = np.asarray(a), np.asarray(b)
    for hyp in ev:
        zmean = float((np.abs(zl[hyp]) + np.abs(zr[hyp])).mean())
        anti = float(np.corrcoef(zl[hyp] - zl[hyp].mean(), zr[hyp] - zr[hyp].mean())[0, 1])
        ev[hyp]["foot_check"] = {
            "mean_abs_foot_z": round(zmean, 4),
            "left_right_z_corr": round(anti, 4),
            "left_z_min_max": [round(float(zl[hyp].min()), 4), round(float(zl[hyp].max()), 4)],
        }
    return ev


# ----------------------------------------------------------------- resample
def resample(arr, fps_in, fps_out):
    """Per-column linear interpolation (joints)."""
    n_in = len(arr)
    n_out = int(round(n_in * fps_out / fps_in))
    t_in = np.arange(n_in) / fps_in
    t_out = np.arange(n_out) / fps_out
    out = np.zeros((n_out, arr.shape[1]), dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(t_out, t_in, arr[:, j])
    return out


def resample_quat(quats, fps_in, fps_out):
    """Hemisphere-aligned per-component lerp + renormalize (quats)."""
    q = quats.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    lin = resample(q, fps_in, fps_out)
    return lin / np.linalg.norm(lin, axis=1, keepdims=True)


# ------------------------------------------------------------- encoder obs
def build_obs(t, jp, jv, bq, apply_delta, anchor="ref-rel"):
    """planner_sonic.py L117-127 g1 layout, live = bq[t] (offline oracle).

    anchor semantics (D036 root-cause fix, 2026-09-04):
      - "ref-rel" (default, CORRECT): relative rotation along the reference,
            btr = conj(q_t) * q_idx
        Rationale: (a) a perfectly-tracking deploy loop (ref(t) ~= live q_t)
        degenerates exactly to this form; (b) if BOTH sides applied a constant
        delta, conj(delta*q_t)*delta*q_idx cancels it identically, so this is
        the only self-consistent choice; (c) yaw invariance -- the encoder
        must not see world-frame yaw. Sanity: f=0 anchor is exactly identity
        (obs[601:607] == [1,0,0,1,0,0]).
      - "heading-norm": original planner_sonic behaviour,
            btr = conj(q_t) * (delta * q_idx)
        Kept only for A/B. The bug: planner_sonic's reference frame 0 is
        standing (quat ~ identity) so delta ~ identity and the flaw is
        invisible there, but the BONES sample pkl starts at yaw ~ -87 deg
        (walking along -y), injecting a constant ~90 deg yaw into every
        anchor vector (D036: yaw-class roundtrip errors 1.3-1.9 rad).
    """
    obs = np.zeros(1762, dtype=np.float32)
    obs[0] = 0.0
    p = 4
    for f in range(10):
        idx = min(t + f * 5, len(jp) - 1)
        obs[p:p + 29] = jp[idx]
        p += 29
    for f in range(10):
        idx = min(t + f * 5, len(jv) - 1)
        obs[p:p + 29] = jv[idx]
        p += 29
    p += 17
    for f in range(10):
        idx = min(t + f * 5, len(bq) - 1)
        if anchor == "ref-rel":
            btr = _qn(_qmul(_qconj(bq[t, 0]), bq[idx, 0]))
        else:
            nr = _qn(_qmul(apply_delta, bq[idx, 0]))
            btr = _qn(_qmul(_qconj(bq[t, 0]), nr))
        rot = _rotmat(btr)
        obs[p:p + 6] = rot[:, :2].flatten()
        p += 6
    return obs


# --------------------------------------------------------- official window
def official_walk_window(events_path, motion_csv):
    """Row range of the WALK baseline in policy_input.csv. Primary source:
    events.json timings relative to start_control @ 50 Hz; cross-checked with
    target_motion.csv col 0 (planner reference root x) mean vx. Fallback:
    vx-onset detection (oracle_token_replay style)."""
    fps = 50.0
    win = None
    try:
        events = json.load(open(events_path))
        labels = [e[1] for e in events]
        t0 = next(e[0] for e in events if e[1] == "start_control")
        i = labels.index("walk_fwd_60s_baseline")
        t1, t2 = events[i][0], events[i + 1][0]
        r0, r1 = int(round((t1 - t0) * fps)), int(round((t2 - t0) * fps))
        src = "events.json walk_fwd_60s_baseline (rel. start_control @50Hz)"
    except Exception as e:
        r0, r1, src = None, None, f"events.json failed: {e}"
    if r0 is not None:
        x = np.genfromtxt(motion_csv, delimiter=",", dtype=np.float32, filling_values=0.0, usecols=(0,))
        r1 = min(r1, len(x))
        r0 = max(r0, 0)
        vx_in = (x[min(r1 - 1, len(x) - 1)] - x[r0]) / max((r1 - r0) / fps, 1e-6)
        pre0 = max(r0 - 600, 0)
        vx_pre = (x[r0] - x[pre0]) / max((r0 - pre0) / fps, 1e-6)
        if vx_in > 0.2 and vx_pre < 0.15:
            win = (r0 + 50, min(r1 - 50, r1), src + f"; vx cross-check in={vx_in:.2f} pre={vx_pre:.2f} m/s")
    if win is None:  # fallback: vx onset
        x = np.genfromtxt(motion_csv, delimiter=",", dtype=np.float32, filling_values=0.0, usecols=(0,))
        vx1 = x[50:] - x[:-50]
        onset = int(np.argmax(vx1 > 0.5)) if (vx1 > 0.5).any() else 0
        win = (onset + 50, min(onset + 3000, len(x) - 50), f"vx-onset fallback (events cross-check failed: {src})")
    return int(win[0]), int(win[1]), win[2]


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=DEFAULT_PKL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--encoder-onnx", default=ENC_ONNX)
    ap.add_argument("--decoder-onnx", default=DEC_ONNX)
    ap.add_argument("--order", choices=["auto", "mujoco", "isaac"], default="auto",
                    help="joint order of pkl `dof`: auto = decide by default-pose corr + FK foot check")
    ap.add_argument("--anchor", choices=["ref-rel", "heading-norm"], default="ref-rel",
                    help="encoder anchor semantics: ref-rel = relative rotation along the "
                         "reference (correct; D036 root-cause fix), heading-norm = original "
                         "planner_sonic apply_delta behaviour (kept for A/B; injects the "
                         "reference-frame-0 heading as a constant world yaw)")
    ap.add_argument("--skip-roundtrip", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort
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

    # ---- load pkl
    obj, _ = load_pkl(args.pkl)
    obj = unwrap(obj)
    fps_src = float(obj["fps"])
    dof = obj["dof"].astype(np.float64)
    root_rot = obj["root_rot"].astype(np.float64)
    trans = obj["root_trans_offset"].astype(np.float64)
    assert dof.ndim == 2 and dof.shape[1] == 29, f"dof shape {dof.shape}, expected (N,29)"
    n_in = len(dof)
    print(f"[pkl] {os.path.basename(args.pkl)}: {sorted(obj.keys())} fps={fps_src} "
          f"dof={dof.shape} root_rot={root_rot.shape} trans={trans.shape}")

    # root quat convention: wxyz (MuJoCo/Isaac) vs xyzw (SMPL-ish)
    q0 = _qn(root_rot[0])
    layout = "wxyz" if abs(q0[0]) >= abs(q0[3]) else "xyzw"
    quat = root_rot.copy() if layout == "wxyz" else root_rot[:, [3, 0, 1, 2]]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    disp = float(np.linalg.norm(trans[-1, :2] - trans[0, :2]))
    path_len = float(np.linalg.norm(np.diff(trans, axis=0), axis=1).sum())
    dur = n_in / fps_src
    print(f"[pkl] root quat layout={layout} (frame0={np.round(q0, 3)}); "
          f"first-to-last horiz displacement {disp:.2f} m, path length {path_len:.2f} m "
          f"over {dur:.1f}s -> mean speed {path_len / dur:.2f} m/s")

    # ---- joint order decision (needs MuJoCo model)
    env = MujocoG1FlatEnv(NoQuantDecoder(args.decoder_onnx), REPO,
                          use_elastic_band=False, stand_only=True)
    ev = joint_order_evidence(env, dof, quat, trans, m2i, i2m, default_mj)
    if args.order != "auto":
        order = args.order
    else:
        fk = {h: ev[h].get("foot_check") if isinstance(ev[h].get("foot_check"), dict) else None for h in ev}
        if fk["mujoco"] and fk["isaac"]:
            order = min(fk, key=lambda h: fk[h]["mean_abs_foot_z"])
        else:
            order = max(ev, key=lambda h: ev[h]["corr_with_default"])
    print(f"[order] decision: pkl dof is {order.upper()} order")
    for h in ev:
        print(f"[order]   hyp {h}: {ev[h]}")

    # to MuJoCo order (for decoder history + FK bookkeeping) and Isaac order
    dof_mj = dof if order == "mujoco" else dof[:, i2m]

    # ---- resample 30 -> 50 Hz (encoder stride-5 assumes 50 Hz)
    if fps_src != FPS_ENC:
        dof_mj_rs = resample(dof_mj, fps_src, FPS_ENC)
        quat_rs = resample_quat(quat, fps_src, FPS_ENC)
    else:
        dof_mj_rs, quat_rs = dof_mj, quat
    n_rs = len(dof_mj_rs)
    jp_mj = dof_mj_rs
    jp_isaac = jp_mj[:, m2i]
    # planner_sonic.py L109: finite diff gives rad/step; encoder wants rad/s @50 Hz
    jv_mj = np.vstack([np.zeros((1, 29)), np.diff(jp_mj, axis=0) * FPS_ENC])
    jv_isaac = jv_mj[:, m2i]
    bq = np.asarray(quat_rs, dtype=np.float64).reshape(-1, 1, 4)
    apply_delta = _qn(_qmul(_heading(np.array([1.0, 0, 0, 0])), _heading_inv(bq[0, 0])))
    print(f"[resample] {n_in}@{fps_src:g}Hz -> {n_rs}@{FPS_ENC:g}Hz "
          f"(linear per-joint; quats hemisphere-aligned lerp+renorm)")

    # ---- encoder
    enc = ort.InferenceSession(args.encoder_onnx, providers=["CPUExecutionProvider"])
    ins = enc.get_inputs()
    info = [{"name": i.name, "shape": i.shape, "type": i.type} for i in ins]
    print(f"[enc] inputs: {info}")
    if len(ins) != 1:
        print(f"[enc] FAIL: expected single input, got {len(ins)} -> stopping (no guessing)")
        sys.exit(2)
    shp = ins[0].shape
    if isinstance(shp[-1], int) and shp[-1] != 1762:
        print(f"[enc] FAIL: encoder input dim {shp[-1]} != 1762 -> stopping")
        sys.exit(2)
    iname = ins[0].name
    tokens = np.zeros((n_rs, 64), dtype=np.float32)
    for t in range(n_rs):
        obs = build_obs(t, jp_isaac, jv_isaac, bq, apply_delta, anchor=args.anchor)
        tokens[t] = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
    # ref-rel sanity: at t=0, f=0 the anchor is exactly identity -> 6D block
    # [r00,r10,r01,r11,r02,r12] = [1,0,0,1,0,0]  (heading-norm has no such fix)
    obs0 = build_obs(0, jp_isaac, jv_isaac, bq, apply_delta, anchor=args.anchor)
    sanity = float(np.abs(obs0[601:607] - np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)).max())
    print(f"[enc] anchor={args.anchor} f=0 anchor-6D sanity |dev| = {sanity:.2e} "
          f"(ref-rel must be ~0; heading-norm deviates by the frame-0 yaw)")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tok_suffix = "" if args.anchor == "heading-norm" else "_anchorrefrel"
    tok_path = os.path.join(out_dir, f"tokens_{os.path.splitext(os.path.basename(args.pkl))[0]}{tok_suffix}.npy")
    np.save(tok_path, tokens)
    print(f"[enc] tokens {tokens.shape} -> {tok_path}")

    res = {
        "exp": "D036",
        "line": "DS plan-B B2 smoke (BONES-SEED pre-check)",
        "pkl_path": args.pkl,
        "pkl_summary": pkl_summary(obj),
        "root_quat_layout": layout,
        "fps_source": fps_src,
        "fps_encoder": FPS_ENC,
        "resample": f"linear per-joint np.interp {n_in}@{fps_src:g}Hz -> {n_rs}@{FPS_ENC:g}Hz; quats hemisphere-aligned lerp+renorm",
        "n_frames_in": n_in,
        "n_tokens": n_rs,
        "tokens_npy": tok_path,
        "joint_order": {"decision": order, "evidence": ev,
                        "mappings": {"G1_MUJOCO_TO_ISAACLAB_DOF": m2i.tolist()},
                        "dof_column_stats": {
                            "mean_as_loaded": [round(float(x), 4) for x in dof.mean(axis=0)],
                            "std_as_loaded": [round(float(x), 4) for x in dof.std(axis=0)],
                            "default_mujoco": [round(float(x), 4) for x in default_mj],
                        },
                        "root_path_length_m": path_len,
                        "root_first_to_last_horiz_disp_m": disp},
        "sample_data_inventory": inventory(sample_data_dir(args.pkl)),
        "encoder": {"inputs": info, "output_shape": list(tokens.shape),
                    "anchor": {"mode": args.anchor, "f0_anchor6d_sanity_dev": sanity}},
    }

    # ---- check 1: lattice legality
    lat = tokens.astype(np.float64) * 16.0
    viol = np.abs(lat - np.round(lat)) > LATTICE_TOL
    res["lattice"] = {"tol": LATTICE_TOL, "n_elem": int(viol.size),
                      "violation_rate_ours": float(viol.mean()),
                      "max_dist_to_lattice": float(np.abs(lat - np.round(lat)).max())}
    print(f"[check1] lattice violation rate (ours) = {viol.mean():.6f} "
          f"(tol {LATTICE_TOL} on token*16, max dev {res['lattice']['max_dist_to_lattice']:.4f})")

    # ---- check 2: distribution vs official WALK baseline
    dist = {}
    try:
        r0, r1, src = official_walk_window(OFFICIAL_EVENTS, OFFICIAL_MOTION)
        official = np.genfromtxt(OFFICIAL_CSV, delimiter=",", dtype=np.float32,
                                 filling_values=0.0)[:, :64]
        r1 = min(r1, len(official))
        off_win = official[r0:r1].astype(np.float64)
        m_o, s_o = off_win.mean(axis=0), off_win.std(axis=0)
        m_u, s_u = tokens.astype(np.float64).mean(axis=0), tokens.astype(np.float64).std(axis=0)
        l2 = float(np.linalg.norm(m_u - m_o))
        top = np.argsort(-np.abs(m_u - m_o))[:10]
        lat_off = off_win * 16.0
        viol_off = float((np.abs(lat_off - np.round(lat_off)) > LATTICE_TOL).mean())
        dist = {
            "official_window": {"rows": [r0, r1], "n_rows": int(r1 - r0), "source": src},
            "official_lattice_violation_rate": viol_off,
            "mean_l2": l2,
            "per_dim_mean_ours": [round(float(x), 4) for x in m_u],
            "per_dim_std_ours": [round(float(x), 4) for x in s_u],
            "per_dim_mean_official": [round(float(x), 4) for x in m_o],
            "per_dim_std_official": [round(float(x), 4) for x in s_o],
            "top10_abs_mean_diff": [{"dim": int(d), "ours": round(float(m_u[d]), 4),
                                     "official": round(float(m_o[d]), 4)} for d in top],
        }
        print(f"[check2] official WALK window rows [{r0},{r1}) ({r1 - r0} rows, {src})")
        print(f"[check2] official lattice rate = {viol_off:.6f}; mean-L2(ours, official) = {l2:.4f}")
        print(f"[check2] top-10 |mean diff| dims: {[(int(d), round(float(m_u[d] - m_o[d]), 3)) for d in top]}")
    except Exception as e:
        dist = {"error": f"{type(e).__name__}: {e}"}
        print(f"[check2] FAILED: {dist['error']}")
    res["distribution"] = dist

    # ---- check 3: decoder roundtrip (oracle condition, best effort)
    rt = {"status": "skipped"}
    if not args.skip_roundtrip:
        try:
            dec = env.sonic_decoder
            din = dec.session.get_inputs()[0]
            rt = {"status": "ok", "decoder_input": {"name": din.name, "shape": din.shape}}
            ddim = din.shape[1] if isinstance(din.shape[1], int) else None
            if ddim != 994:
                raise ValueError(f"decoder input dim {ddim} != 994, format unknown -> skip")

            def sonic_history(t):
                """Mirror env._get_sonic_history semantics offline: 10 rows,
                oldest first, most recent last; positions are default-relative,
                last_actions are normalized effective actions, all Isaac order."""
                idx = np.clip(np.arange(t - 9, t + 1), 0, n_rs - 1)
                return {
                    "base_angular_velocity": omega_body[idx].astype(np.float32),
                    "body_joint_positions": ((jp_mj[idx] - default_mj)[:, m2i]).astype(np.float32),
                    "body_joint_velocities": jv_isaac[idx].astype(np.float32),
                    "last_actions": (((jp_mj[idx] - default_mj) / env.sonic_scale_mujoco)[:, m2i]).astype(np.float32),
                    "gravity_dir": grav[idx].astype(np.float32),
                }

            # body-frame angular velocity from root quat finite diff (small-angle)
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
            q_des_all = np.zeros((n_rs, 29), dtype=np.float32)
            for t in range(n_rs):
                tok = tokens[t]
                obs = dec.build_decoder_obs(tok, sonic_history(t))
                act_isaac = dec.session.run([dec.output_name], {dec.input_name: obs})[0][0]
                q_des_isaac = env.sonic_default_isaac + act_isaac.astype(np.float64) * env.sonic_scale_isaac
                q_des_all[t] = q_des_isaac.astype(np.float32)
                err.append(np.abs(q_des_isaac - jp_isaac[t]))
                err0.append(np.abs(default_mj[m2i] - jp_isaac[t]))
            err = np.asarray(err)
            err0 = np.asarray(err0)
            pj = err.mean(axis=0)
            worst = np.argsort(-pj)[:5]
            rt.update({
                "condition": "oracle: live proprio from reference trajectory itself; last_actions=reference normalized action",
                "mae_rad": float(err.mean()),
                "mae_default_stance_baseline_rad": float(err0.mean()),
                "per_joint_mae_top5": [{"isaac_dim": int(j), "mae_rad": round(float(pj[j]), 4)} for j in worst],
                "q_des_npy": os.path.join(out_dir, "q_des_roundtrip.npy"),
            })
            np.save(rt["q_des_npy"], q_des_all)
            print(f"[check3] decoder roundtrip MAE = {rt['mae_rad']:.4f} rad "
                  f"(default-stance baseline {rt['mae_default_stance_baseline_rad']:.4f} rad); "
                  f"worst Isaac dims: {[(int(j), round(float(pj[j]), 3)) for j in worst]}")
        except Exception as e:
            rt = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
            print(f"[check3] roundtrip failed (non-blocking): {rt['reason']}")
    res["roundtrip"] = rt

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[out] JSON -> {args.out}")
    print("=" * 70)
    print(f"SUMMARY: frames {n_in}@{fps_src:g}Hz -> {n_rs} tokens@50Hz | joint order: {order} "
          f"(pkl) | lattice viol rate {res['lattice']['violation_rate_ours']:.2e} "
          f"(official walk {dist.get('official_lattice_violation_rate', float('nan')):.2e}) | "
          f"mean-L2 vs official walk {dist.get('mean_l2', float('nan')):.3f} | "
          f"roundtrip MAE {rt.get('mae_rad', float('nan')) if rt.get('status') == 'ok' else rt['status']} rad")


if __name__ == "__main__":
    main()
