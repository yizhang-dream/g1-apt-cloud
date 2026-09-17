"""D060 source-1 generator: planner-commanded token scripts (command-aligned corpus).

One npz per script ("剧本" = one physical rollout of one command point):

    tokens      (n, 64) float32   SONIC FSQ token stream @ 50 Hz (one token per ctrl step)
    root_pos    (n, 3)  float32   world root position, PRE-step state at token index t
    root_quat   (n, 4)  float32   world root orientation (w, x, y, z), same indexing
    jp_mujoco   (n, 29) float32   joint positions, MuJoCo order (env.data.qpos[body_qpos_adr])
    jp_isaaclab (n, 29) float32   joint positions, IsaacLab order (== decoder/encoder order)
    meta        ()      <U..      JSON string (nominal command / seed / realized / planner OL)

Command -> script mechanism (all three official models frozen, no learning):
    1. planner_sonic.onnx -- 10 Hz kinematic planner. The latent autoregression lives
       INSIDE the ONNX graph (masked-token prediction gated by allowed_pred_num_tokens
       (1, 11)); the python side is a single session.run -> mujoco_qpos (1, 64, 36) +
       num_pred_frames (1,). Frames are consumed at the 50 Hz control rate (jv = diff*50).
    2. model_encoder.onnx -- obs_dict (1, 1762) -> encoded_tokens (1, 64), one token per
       control step (obs layout identical to the proven harness).
    3. model_decoder.onnx -- frozen, executed in MuJoCo via envs.mujoco_g1_flat_env +
       eval_distill.NoQuantDecoder, so the recorded token stream is physically realized.
    Closed loop = planner_closed_loop.py lineage: replan every 5 control steps (10 Hz) with
    the LIVE 4-frame qpos context (the proven form; a frozen/canonical context is a known
    failure mode, see that file's comment), obs assembled step-for-step as in that harness.
    Harness env cap: envs/mujoco_g1_flat_env.MujocoG1FlatEnv hard-caps an episode at
    episode_length_s = 20 s (= 1000 steps @ 50 Hz), and its `terminated` flag covers BOTH a
    real fall (root z < 0.2) and that time limit. `--dur-s > 20` is therefore clamped, and
    meta["realized"]["terminated_cause"] separates "horizon_cap" from "fall" (a 20 s script
    ends exactly ON the cap -- that is not a fall). Step-count keys: meta carries
    n_steps_requested_full (pre-clamp, from --dur-s), n_steps_effective_planned (post-clamp)
    and env_episode_length, but the AUTHORITATIVE executed count is meta["realized"]["n_steps"]
    -- a real fall ends the script early. Per-script wall time is deliberately kept OUT of the
    npz meta (manifest.json only) so the per-file bytes/md5 stay reproducible (D059 pitfall).

Command convention (§5t, D033 in force -- target_vel is non-monotonic in this ONNX):
    - meta["nominal"] is what was ASKED of the planner (grid value, mode, direction, height);
      meta["realized"] is the MEASURED net speed of the physical rollout. The D033 decay gap
      (planner bare output 2.12 m/s vs deploy loop ~1.0 m/s) is therefore measured per command
      point instead of assumed; meta["planner_open_loop"] records the same point's raw ONNX
      trajectory speed (the D033 numerator, standing-context protocol == sim/planner_sonic.py).
    - target_vel = -1 is the mode-default sentinel: RUN measures 2.226 m/s of planner bare
      output in this repo (D033 logged 2.12 on the same standing-context protocol; D032 logged
      -1 ~= explicit 2.5, bit-identical trajectory). Source 1 uses the explicit grid
      {0, 0.2, ..., 1.2}; -1 stays available by passing --target-vel -1 (kept for the D033
      reference point).
    - realized triage labels (meta["flags"], descriptive only, NOT gates): low_net =
      net_speed_xy < 0.2, low_posture = h_min < 0.55, fall = early real-fall termination. They
      are reported side by side with the per-point counts n_low_net / n_low_posture so a slow
      script is never silently lumped with a collapsed one ("塌陷" is not a category here).
    - movement_direction / facing_direction are hpp MovementState [x, y, z] unit vectors in the
      body frame (front=+x, left=+y, right=-y). ASSUMPTION flagged: the direction axis is set by
      convention here, not verified; the realized net displacement vector is recorded in meta so
      the full-grid run can check commanded-vs-realized direction (this smoke only covers front).

Windowing handoff (build/d060_windows.py -- a separate agent's module; NOT imported here):
    §5t windows are 4 s @ 50 Hz (D057 builder convention, stride 2 s). Arrays are causally
    aligned at the same 50 Hz index: root_pos[t] / jp_*[t] is the state BEFORE executing
    tokens[t], so a window anchored at step t = state history ending at t + token stream
    starting at t (and command/terrain_desc come from meta["nominal"] plus the terrain
    sources). Nothing else is needed from this file.

Usage (server, .venv_mjlab -- the only venv with both mujoco and onnxruntime):
    python gen_planner_scripts_d060.py --target-vel 0.6 --mode SLOW_WALK --direction front \
        --n-scripts 10 --dur-s 20 --out-dir data/d060/planner_scripts_smoke
    python gen_planner_scripts_d060.py --dry-run --target-vel grid    # enumerate only
    python gen_planner_scripts_d060.py --selftest                     # pure numpy, no mujoco/ORT
    # explicit points (negative vel needs the '=' form):
    python gen_planner_scripts_d060.py --point=-1:RUN:front --point 1.0:RUN:front --n-scripts 3

Server deployment note: the execution root is FLAT (~/ros2_data/apt_g1/, no build/ subdir --
the 2026-09-15 domain split is not deployed there), so the file runs as
`../.venv_mjlab/bin/python gen_planner_scripts_d060.py ...` from ~/ros2_data/apt_g1.

D060 / §5t; tracks refine-logs/ds/DS_TERRAIN_AUTHOR_PLAN.md §6 source 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# constants (mirror sim/planner_sonic.py + planner_closed_loop.py)
# ---------------------------------------------------------------------------
ROOT = os.path.expanduser("~")
REPO_DEFAULT = f"{ROOT}/ros2_data/GR00T-WholeBodyControl"
APT_DIR_DEFAULT = f"{ROOT}/ros2_data/apt_g1"
ROS2_DATA_DEFAULT = f"{ROOT}/ros2_data"
PLANNER_REL = "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
ENC_REL = "gear_sonic_deploy/policy/release/model_encoder.onnx"
DEC_REL = "gear_sonic_deploy/policy/release/model_decoder.onnx"
SCENE_REL = "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"

BODY_IDX = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]
CTRL_HZ = 50.0
REPLAN_EVERY = 5  # 10 Hz replanning at 50 Hz control
OBS_DIM = 1762
OBS_HDR = 17  # skipped block between motion history and body quats

MODE_IDS = {"SLOW_WALK": 1, "WALK": 2, "RUN": 3}
DIR_VECS = {"front": (1.0, 0.0, 0.0), "left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0)}
FACING_DEFAULT = (1.0, 0.0, 0.0)
TARGET_VEL_GRID = [round(0.2 * i, 1) for i in range(7)]  # §5t: {0, 0.2, ..., 1.2}

# descriptive triage thresholds only (see triage_flags; NOT gates)
LOW_NET_THRESH = 0.2
LOW_POSTURE_H_THRESH = 0.55

SCHEMA = "d060_src1_v1"


# ---------------------------------------------------------------------------
# quaternion helpers (verbatim from the planner harness lineage)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# pure helpers (selftest covers all of these; no mujoco / onnxruntime needed)
# ---------------------------------------------------------------------------
def parse_target_vels(spec: str) -> list[float]:
    """'grid' -> §5t grid; '-1' -> sentinel; '0,0.6,1.2' -> explicit list."""
    spec = str(spec).strip()
    if spec == "grid":
        return list(TARGET_VEL_GRID)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError(f"empty --target-vel spec: {spec!r}")
    return out


def parse_modes(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in str(spec).split(","):
        name = part.strip().upper()
        if not name:
            continue
        if name.isdigit():
            name = {v: k for k, v in MODE_IDS.items()}.get(int(name), name)
        if name not in MODE_IDS:
            raise ValueError(f"unknown mode {part!r}; known: {sorted(MODE_IDS)} or ids {sorted(MODE_IDS.values())}")
        out.append((name, MODE_IDS[name]))
    if not out:
        raise ValueError(f"empty --mode spec: {spec!r}")
    return out


def parse_directions(spec: str) -> list[str]:
    out = []
    for part in str(spec).split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in DIR_VECS:
            raise ValueError(f"unknown direction {part!r}; known: {sorted(DIR_VECS)}")
        out.append(name)
    if not out:
        raise ValueError(f"empty --direction spec: {spec!r}")
    return out


def seed_for(script_idx: int, seed_base: int) -> int:
    """Planner random_seed for script k. Same k -> same seed across command points, so the
    command axis is compared on paired scripts (diversity inside a point == planner seed)."""
    return int(seed_base) + int(script_idx)


def build_points(vels, modes, directions) -> list[dict]:
    """Full §5t grid enumeration (deterministic order: vel x mode x direction)."""
    pts = []
    for v in vels:
        for name, mid in modes:
            for d in directions:
                pts.append({"target_vel": float(v), "mode_name": name, "mode_id": int(mid),
                            "direction": d, "movement_direction": list(DIR_VECS[d]),
                            "facing_direction": list(FACING_DEFAULT)})
    return pts


def parse_point_spec(spec: str) -> dict:
    """'vel:MODE:direction' -> one explicit command point (bypasses the cartesian product)."""
    parts = [p.strip() for p in str(spec).split(":")]
    if len(parts) != 3:
        raise ValueError(f"bad --point {spec!r}; expected 'vel:MODE:direction' e.g. '0.6:SLOW_WALK:front'")
    vel = parse_target_vels(parts[0])[0]
    mode_name, mode_id = parse_modes(parts[1])[0]
    d = parse_directions(parts[2])[0]
    return {"target_vel": float(vel), "mode_name": mode_name, "mode_id": int(mode_id),
            "direction": d, "movement_direction": list(DIR_VECS[d]),
            "facing_direction": list(FACING_DEFAULT)}


def point_tag(point_idx: int, p: dict) -> str:
    return f"p{point_idx:02d}_vel{p['target_vel']:g}_{p['mode_name']}_{p['direction']}"


def net_speed_stats(root_pos: np.ndarray, ctrl_hz: float = CTRL_HZ) -> dict:
    """Net (XY) and per-axis speeds from the PRE-step root trace; empty trace is safe.

    Conventions: `dur_s` = record length (n / ctrl_hz, the window-relevant duration) while
    speeds divide by the endpoint interval ((n-1) / ctrl_hz) -- the physically elapsed time
    between the first and last pre-step sample. At n >= 1000 the two differ by < 0.1%, so the
    numbers stay comparable with the D032/D033 protocol (x displacement / window duration).
    """
    x = np.asarray(root_pos, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return {"n_steps": 0, "dur_s": 0.0, "elapsed_s": 0.0, "disp_x": None, "disp_y": None,
                "disp_xy": None, "net_speed_xy": None, "net_vx": None, "net_vy": None,
                "path_len_xy": None, "straightness": None}
    n = x.shape[0]
    dur = n / float(ctrl_hz)
    elapsed = max(n - 1, 1) / float(ctrl_hz)
    d = x[-1, :2] - x[0, :2]
    seg = np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1) if n > 1 else np.zeros(0)
    path = float(seg.sum())
    return {"n_steps": int(n), "dur_s": round(dur, 4), "elapsed_s": round(elapsed, 4),
            "disp_x": round(float(d[0]), 5), "disp_y": round(float(d[1]), 5),
            "disp_xy": round(float(np.linalg.norm(d)), 5),
            "net_speed_xy": round(float(np.linalg.norm(d)) / elapsed, 5),
            "net_vx": round(float(d[0]) / elapsed, 5), "net_vy": round(float(d[1]) / elapsed, 5),
            "path_len_xy": round(path, 5),
            "straightness": round(float(np.linalg.norm(d) / path), 4) if path > 1e-9 else None}


def token_stats(tokens: np.ndarray) -> dict:
    t = np.asarray(tokens, dtype=np.float64)
    if t.size == 0:
        return {"n_tokens": 0}
    per_dim_std = t.std(axis=0) if t.ndim == 2 else np.array([t.std()])
    return {"n_tokens": int(t.shape[0]), "dim": int(t.shape[1]) if t.ndim == 2 else int(t.size),
            "mean": round(float(t.mean()), 5), "std": round(float(t.std()), 5),
            "abs_mean": round(float(np.abs(t).mean()), 5), "min": round(float(t.min()), 5),
            "max": round(float(t.max()), 5), "frac_neg": round(float((t < 0).mean()), 5),
            "per_dim_std_mean": round(float(per_dim_std.mean()), 5)}


def decay_ratio(realized: float | None, nominal: float) -> float | None:
    """realized / nominal; None when nominal <= 0 -- the target_vel=0 stop point and the
    -1 mode-default sentinel have no meaningful ratio (use planner_open_loop_over_realized
    for the sentinel instead)."""
    if realized is None or nominal is None or float(nominal) <= 0:
        return None
    return round(float(realized) / float(nominal), 4)


def classify_termination(term: bool, step_count: int, episode_length: int) -> str | None:
    """The harness env returns `terminated=True` for BOTH a real fall (qpos z < 0.2) and the
    episode time limit (env.step_count >= episode_length, default 20 s @ 50 Hz = 1000 steps).
    Keep them apart: a 20 s script ends exactly ON the cap, so 'fall' must not swallow it
    (the env's own reward does the same test, see mujoco_g1_flat_env.py:518/549).
    `step_count` must be the POST-increment value (pre-step counter + 1): env.step() calls
    self.reset() before returning, which zeroes its own counter, so it cannot be read after."""
    if not term:
        return None
    return "horizon_cap" if int(step_count) >= int(episode_length) else "fall"


def triage_flags(realized: dict) -> dict:
    """Descriptive per-script labels for triage -- NOT gates, no pass/fail semantics.
    Kept separate on purpose (don't lump them into one "collapse" word):
      low_net      net_speed_xy < LOW_NET_THRESH  (harness under-tracking; can happen upright)
      low_posture  h_min < LOW_POSTURE_H_THRESH   (root sank toward the fall threshold)
      fall         the env terminated early on a real fall
    A script can be low_net while upright, or low_posture at a decent net speed."""
    net = realized.get("net_speed_xy")
    hmin = realized.get("h_min")
    return {"low_net": (net is not None and float(net) < LOW_NET_THRESH),
            "low_posture": (hmin is not None and float(hmin) < LOW_POSTURE_H_THRESH),
            "fall": bool(realized.get("fall", False)),
            "thresholds": {"low_net_net_speed_xy": LOW_NET_THRESH,
                           "low_posture_h_min": LOW_POSTURE_H_THRESH}}


def build_script_meta(*, point: dict, point_tag: str, point_idx: int, script_idx: int,
                      seed: int, seed_base: int, dur_s: float, height: float,
                      spawn_x: float, spawn_z: float, ol: dict, rollout: dict,
                      realized: dict, tstat: dict) -> dict:
    """Pure builder for the npz meta block (selftest asserts its key contract).

    Invariants: the AUTHORITATIVE step count is realized["n_steps"] (rollout["n_steps_requested"]
    is the pre-clamp --dur-s request, rollout["n_steps_effective"] the post-clamp plan), and
    `wall_s` is deliberately ABSENT -- wall time goes to the directory-level manifest only, so
    the per-file bytes/md5 stay reproducible (D059 created_at pitfall)."""
    rpos = rollout["root_pos"]
    h_min = round(float(np.min(rpos[:, 2])), 4) if len(rpos) else None
    h_end = round(float(rpos[-1, 2]), 4) if len(rpos) else None
    return {
        "schema": SCHEMA, "point_tag": point_tag, "point_idx": point_idx, "script_idx": script_idx,
        "nominal": {"target_vel": point["target_vel"], "target_vel_sent": point["target_vel"],
                    "mode_id": point["mode_id"], "mode_name": point["mode_name"],
                    "movement_direction": point["movement_direction"],
                    "facing_direction": point["facing_direction"],
                    "direction_name": point["direction"], "height": height,
                    "ctrl_hz": CTRL_HZ, "replan_hz": CTRL_HZ / REPLAN_EVERY},
        "seed": seed, "seed_base": seed_base, "dur_s": dur_s,
        "n_steps_requested_full": rollout["n_steps_requested"],
        "n_steps_effective_planned": rollout["n_steps_effective"],
        "env_episode_length": rollout["episode_length"],
        "realized": {**realized, "terminated_cause": rollout["terminated_cause"],
                     "env_step_count_at_term": rollout["env_step_count_at_term"],
                     "fall": rollout["fall"], "fall_step": rollout["fall_step"],
                     "h_min": h_min, "h_end": h_end,
                     "ratio_to_nominal": decay_ratio(realized["net_speed_xy"], point["target_vel"])},
        "planner_open_loop": ol,
        "planner_open_loop_first_plan": rollout["first_plan_open_loop"],
        "decay_note": ("D033: planner bare output 2.12 m/s (RUN, -1 sentinel) vs official deploy "
                       "loop ~1.0 m/s; planner_open_loop here is THIS point's raw ONNX speed "
                       "(standing-context protocol) and realized is the measured MuJoCo "
                       "realization -- measured, not assumed"),
        "token_stats": tstat,
        "flags": triage_flags({**realized, "h_min": h_min, "fall": rollout["fall"]}),
        "spawn": {"x": spawn_x, "z": spawn_z, "settle_steps": 40},
    }


def save_script(path, tokens, root_pos, root_quat, jp_mujoco, jp_isaaclab, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        tokens=np.asarray(tokens, dtype=np.float32),
        root_pos=np.asarray(root_pos, dtype=np.float32),
        root_quat=np.asarray(root_quat, dtype=np.float32),
        jp_mujoco=np.asarray(jp_mujoco, dtype=np.float32),
        jp_isaaclab=np.asarray(jp_isaaclab, dtype=np.float32),
        meta=np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
    )


def load_script(path) -> tuple[dict, dict]:
    with np.load(str(path), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    return arrays, meta


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# heavy harness (mujoco + onnxruntime; instantiated only for real runs)
# ---------------------------------------------------------------------------
class SonicHarness:
    """Official SONIC stack in MuJoCo: planner -> encoder -> frozen decoder (env)."""

    def __init__(self, repo: str, apt_dir: str, ros2_data: str, planner_onnx: str | None = None):
        for p in (ros2_data, apt_dir, repo):
            if p not in sys.path:
                sys.path.insert(0, p)
        import mujoco
        import onnxruntime as ort
        from envs.mujoco_g1_flat_env import (MujocoG1FlatEnv, SONIC_DEFAULT_ANGLES_MUJOCO,
                                             G1_MUJOCO_TO_ISAACLAB_DOF)
        from eval_distill import NoQuantDecoder

        self.mujoco = mujoco
        self.repo = repo
        self.apt_dir = apt_dir
        self.ros2_data = ros2_data
        self.planner_onnx = planner_onnx or f"{repo}/{PLANNER_REL}"
        self.enc_onnx = f"{repo}/{ENC_REL}"
        self.dec_onnx = f"{repo}/{DEC_REL}"
        self.scene = f"{repo}/{SCENE_REL}"
        self.planner = ort.InferenceSession(self.planner_onnx, providers=["CPUExecutionProvider"])
        self.enc = ort.InferenceSession(self.enc_onnx, providers=["CPUExecutionProvider"])
        self.iname = self.enc.get_inputs()[0].name
        self.defaults = SONIC_DEFAULT_ANGLES_MUJOCO
        self.m2i = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)
        self.env = MujocoG1FlatEnv(NoQuantDecoder(self.dec_onnx), repo, robot_scene=self.scene,
                                   use_elastic_band=False, stand_only=True)
        self.fk_data = mujoco.MjData(self.env.model)
        # harness env hard-caps an episode at episode_length_s = 20 s (mujoco_g1_flat_env.py:113)
        self.episode_length = int(getattr(self.env, "episode_length", 0) or 0)

    # --- state plumbing ------------------------------------------------
    def standing(self):
        q = np.zeros(36, dtype=np.float32)
        q[0:3] = [0, 0, 0.76]
        q[3:7] = [1, 0, 0, 0]
        q[7:36] = self.defaults
        return q

    def live_q36(self):
        q = np.zeros(36, dtype=np.float32)
        q[0:3] = self.env.data.qpos[0:3]
        q[3:7] = self.env.data.qpos[3:7]
        q[7:36] = self.env.data.qpos[self.env.body_qpos_adr]
        return q

    def fk(self, q36):
        q = np.zeros(self.env.model.nq)
        q[:3] = q36[:3]
        q[3:7] = q36[3:7]
        q[self.env.body_qpos_adr] = q36[7:36]
        self.fk_data.qpos[:] = q
        self.mujoco.mj_forward(self.env.model, self.fk_data)
        return self.fk_data.xquat[BODY_IDX].copy()

    # --- planner / encoder ---------------------------------------------
    def plan(self, mode_id: int, vel: float, mdir, fdir, seed: int, height: float, ctx36):
        inp = {"context_mujoco_qpos": np.tile(np.asarray(ctx36, dtype=np.float32)[None, None], (1, 4, 1)),
               "target_vel": np.array([vel], dtype=np.float32),
               "mode": np.array([mode_id], dtype=np.int64),
               "movement_direction": np.array([mdir], dtype=np.float32),
               "facing_direction": np.array([fdir], dtype=np.float32),
               "random_seed": np.array([seed], dtype=np.int64),
               "has_specific_target": np.array([[0]], dtype=np.int64),
               "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
               "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
               "allowed_pred_num_tokens": np.ones((1, 11), dtype=np.int64),
               "height": np.array([height], dtype=np.float32)}
        qpos_out, nframes_out = self.planner.run(None, inp)
        traj = qpos_out[0][:int(nframes_out[0])]
        jp = traj[:, 7:36][:, self.m2i]
        jv = np.vstack([np.zeros((1, 29)), np.diff(jp, axis=0) * CTRL_HZ])
        bq = np.array([self.fk(t) for t in traj])
        ad = _qn(_qmul(_heading(np.array([1.0, 0, 0, 0])), _heading_inv(bq[0, 0])))
        return jp, jv, bq, ad, np.asarray(traj)

    def planner_open_loop(self, mode_id: int, vel: float, mdir, fdir, seed: int, height: float) -> dict:
        """Raw ONNX trajectory from the standing context -- the D033 numerator protocol
        (sim/planner_sonic.py first plan), no physics involved."""
        _, _, _, _, traj = self.plan(mode_id, vel, mdir, fdir, seed, height, self.standing())
        x = np.asarray(traj[:, 0], dtype=np.float64)
        n = len(x)
        dur = n / CTRL_HZ
        return {"n_frames": int(n), "horizon_s": round(dur, 4),
                "disp_x": round(float(x[-1] - x[0]), 4),
                "vx": round(float(x[-1] - x[0]) / dur, 4) if dur > 0 else None,
                "vy": round(float(traj[-1, 1] - traj[0, 1]) / dur, 4) if dur > 0 else None,
                "z_min": round(float(traj[:, 2].min()), 4), "z_max": round(float(traj[:, 2].max()), 4)}

    # --- rollout --------------------------------------------------------
    def settle(self, spawn_x: float, spawn_z: float, n_settle: int = 40):
        self.env.reset()
        self.env.data.qpos[0] = spawn_x
        self.env.data.qpos[1] = 0.0
        self.env.data.qpos[2] = spawn_z
        for _ in range(n_settle):
            self.env._step_physics(self.defaults.copy())

    def rollout(self, point: dict, seed: int, n_steps: int, spawn_x: float, spawn_z: float,
                height: float = -1.0) -> dict:
        mode_id = point["mode_id"]
        vel = point["target_vel"]
        mdir, fdir = point["movement_direction"], point["facing_direction"]
        n_steps_req = int(n_steps)
        n_steps = min(n_steps_req, self.episode_length) if self.episode_length else n_steps_req
        if n_steps < n_steps_req:
            print(f"[d060-src1] WARN clamped n_steps {n_steps_req} -> {n_steps} "
                  f"(env episode_length={self.episode_length}); per-script duration capped at "
                  f"{self.episode_length / CTRL_HZ:g}s", flush=True)
        self.settle(spawn_x, spawn_z)
        jp = jv = bq = ad = None
        cur_frame = 0
        first_plan_ol = None
        toks, rpos, rquat, jpm, jpi = [], [], [], [], []
        fall_step = None
        cause = None
        for step in range(n_steps):
            if step % REPLAN_EVERY == 0:
                jp, jv, bq, ad, traj = self.plan(mode_id, vel, mdir, fdir, seed, height, self.live_q36())
                cur_frame = 0
                if step == 0:
                    x = np.asarray(traj[:, 0], dtype=np.float64)
                    dur = len(x) / CTRL_HZ
                    first_plan_ol = {"n_frames": int(len(x)),
                                     "vx": round(float(x[-1] - x[0]) / dur, 4) if dur > 0 else None}
            live = self.env.data.qpos[3:7].astype(np.float64)
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            obs[0] = 0.0
            p = 4
            for f in range(10):
                idx = min(cur_frame + f * 5, len(jp) - 1)
                obs[p:p + 29] = jp[idx]
                p += 29
            for f in range(10):
                idx = min(cur_frame + f * 5, len(jv) - 1)
                obs[p:p + 29] = jv[idx]
                p += 29
            p += OBS_HDR
            for f in range(10):
                idx = min(cur_frame + f * 5, len(bq) - 1)
                nr = _qn(_qmul(ad, bq[idx, 0]))
                btr = _qn(_qmul(_qconj(live), nr))
                rot = _rotmat(btr)
                obs[p:p + 6] = rot[:, :2].flatten()
                p += 6
            jp_mu = np.asarray(self.env.data.qpos[self.env.body_qpos_adr], dtype=np.float32)
            rpos.append(np.asarray(self.env.data.qpos[0:3], dtype=np.float32).copy())
            rquat.append(np.asarray(self.env.data.qpos[3:7], dtype=np.float32).copy())
            jpm.append(jp_mu.copy())
            jpi.append(jp_mu[self.m2i].copy())
            tok = self.enc.run(None, {self.iname: obs[None]})[0][0].astype(np.float32)
            toks.append(tok)
            sc_before = int(self.env.step_count)  # env.step() resets this on termination
            _, _, term, _ = self.env.step({"token": tok, "aux": np.zeros(12, dtype=np.float32)})
            cur_frame += 1
            if term:
                cause = classify_termination(term, sc_before + 1, self.episode_length)
                if cause == "fall":
                    fall_step = step
                break
        return {"tokens": np.asarray(toks, dtype=np.float32),
                "root_pos": np.asarray(rpos, dtype=np.float32),
                "root_quat": np.asarray(rquat, dtype=np.float32),
                "jp_mujoco": np.asarray(jpm, dtype=np.float32),
                "jp_isaaclab": np.asarray(jpi, dtype=np.float32),
                "fall": cause == "fall", "fall_step": fall_step, "terminated_cause": cause,
                "env_step_count_at_term": (int(sc_before) + 1) if cause else None,
                "n_steps_requested": n_steps_req, "n_steps_effective": n_steps,
                "episode_length": self.episode_length,
                "first_plan_open_loop": first_plan_ol}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_all(args) -> int:
    vels = parse_target_vels(args.target_vel)
    modes = parse_modes(args.mode)
    dirs = parse_directions(args.direction)
    points = ([parse_point_spec(s) for s in args.point] if args.point
              else build_points(vels, modes, dirs))
    if args.point:
        vels = sorted({p["target_vel"] for p in points})
        modes = [(m, MODE_IDS[m]) for m in dict.fromkeys(p["mode_name"] for p in points)]
        dirs = list(dict.fromkeys(p["direction"] for p in points))
    n_steps = int(round(float(args.dur_s) * CTRL_HZ))

    print(f"[d060-src1] points={len(points)} scripts/point={args.n_scripts} "
          f"steps={n_steps} ({args.dur_s:g}s @ {CTRL_HZ:g}Hz) out={args.out_dir}", flush=True)
    if args.dry_run:
        for i, pt in enumerate(points):
            seeds_preview = (np.arange(args.n_scripts) + args.seed_base)
            print(f"[dry-run] {point_tag(i, pt)}  seeds={seeds_preview[:3].tolist()}..{int(seeds_preview[-1])}")
        print(f"[dry-run] total scripts={len(points) * args.n_scripts} "
              f"total sim time={len(points) * args.n_scripts * args.dur_s / 3600:.2f} h")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = SonicHarness(args.repo, args.apt_dir, args.ros2_data, args.planner_onnx)
    print(f"[d060-src1] sessions ready in {time.time() - t0:.1f}s "
          f"(planner={h.planner_onnx} enc={h.enc_onnx} dec={h.dec_onnx})", flush=True)

    manifest = {"schema": SCHEMA, "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "host": os.uname().nodename if hasattr(os, "uname") else "windows",
                "python": sys.version.split()[0], "venv_hint": args.venv_hint,
                "script_path": os.path.abspath(__file__),
                "script_md5": md5_file(__file__) if os.path.exists(__file__) else None,
                "repo": args.repo, "assets": {"planner_onnx": h.planner_onnx, "encoder_onnx": h.enc_onnx,
                                              "decoder_onnx": h.dec_onnx, "scene": h.scene},
                "ctrl_hz": CTRL_HZ, "replan_every": REPLAN_EVERY, "replan_hz": CTRL_HZ / REPLAN_EVERY,
                "env_episode_length_steps": h.episode_length,
                "env_episode_length_s": h.episode_length / CTRL_HZ,
                "dur_s": args.dur_s, "n_scripts": args.n_scripts, "seed_base": args.seed_base,
                "args": {k: str(v) for k, v in sorted(vars(args).items()) if k != "planner_onnx"},
                "points": [], "files": []}
    point_summaries = []
    total_scripts = 0
    for pidx, pt in enumerate(points):
        pdir_name = point_tag(pidx, pt)
        pdir = out_dir / pdir_name
        pdir.mkdir(parents=True, exist_ok=True)
        ol = h.planner_open_loop(pt["mode_id"], pt["target_vel"], pt["movement_direction"],
                                 pt["facing_direction"], seed_for(0, args.seed_base), args.height)
        recs, all_tok, t_p = [], [], time.time()
        for k in range(args.n_scripts):
            seed = seed_for(k, args.seed_base)
            ts = time.time()
            r = h.rollout(pt, seed, n_steps, args.spawn_x, args.spawn_z, args.height)
            wall = time.time() - ts
            realized = net_speed_stats(r["root_pos"])
            tstat = token_stats(r["tokens"])
            meta = build_script_meta(point=pt, point_tag=pdir_name, point_idx=pidx, script_idx=k,
                                     seed=seed, seed_base=args.seed_base, dur_s=args.dur_s,
                                     height=args.height, spawn_x=args.spawn_x, spawn_z=args.spawn_z,
                                     ol=(ol if args.open_loop_point_probe else r["first_plan_open_loop"]),
                                     rollout=r, realized=realized, tstat=tstat)
            fname = f"script_{k:03d}.npz"
            save_script(pdir / fname, r["tokens"], r["root_pos"], r["root_quat"],
                        r["jp_mujoco"], r["jp_isaaclab"], meta)
            # wall_s lives ONLY here (directory-level manifest), never inside the npz: it would
            # make the per-file bytes/md5 non-reproducible (D059 created_at pitfall).
            manifest["files"].append({"point_tag": pdir_name, "script_idx": k, "path": f"{pdir_name}/{fname}",
                                      "n_tokens": int(r["tokens"].shape[0]), "seed": seed,
                                      "wall_s": round(wall, 2), "md5": md5_file(pdir / fname)})
            recs.append(meta)
            all_tok.append(r["tokens"])
            total_scripts += 1
            print(f"[d060-src1] {pdir_name} script={k:03d} seed={seed} n={realized['n_steps']} "
                  f"cause={r['terminated_cause']} net_xy={realized['net_speed_xy']} "
                  f"tok_mean={tstat.get('mean')} wall={wall:.1f}s", flush=True)
        tokn = np.concatenate(all_tok, axis=0) if all_tok else np.zeros((0, 64), dtype=np.float32)
        nets = [m["realized"]["net_speed_xy"] for m in recs if m["realized"]["net_speed_xy"] is not None]
        ratios = [m["realized"]["ratio_to_nominal"] for m in recs if m["realized"]["ratio_to_nominal"] is not None]
        psum = {"point_tag": pdir_name, "point_idx": pidx, "nominal": recs[0]["nominal"],
                "n_scripts": len(recs),
                "n_fall": int(sum(1 for m in recs if m["realized"]["fall"])),
                "n_horizon_cap": int(sum(1 for m in recs if m["realized"]["terminated_cause"] == "horizon_cap")),
                "n_completed_no_term": int(sum(1 for m in recs if m["realized"]["terminated_cause"] is None)),
                "n_low_net": int(sum(1 for m in recs if m["flags"]["low_net"])),
                "n_low_posture": int(sum(1 for m in recs if m["flags"]["low_posture"])),
                "flags_note": ("low_net = net_speed_xy < 0.2 ; low_posture = h_min < 0.55 ; both are "
                               "descriptive triage labels, not gates (a script can be low_net upright)"),
                "realized_net_speed_xy": {"mean": round(float(np.mean(nets)), 5) if nets else None,
                                          "median": round(float(np.median(nets)), 5) if nets else None,
                                          "std": round(float(np.std(nets)), 5) if nets else None,
                                          "min": round(float(np.min(nets)), 5) if nets else None,
                                          "max": round(float(np.max(nets)), 5) if nets else None},
                "realized_net_vx_mean": round(float(np.mean([m["realized"]["net_vx"] for m in recs])), 5),
                "realized_net_vy_mean": round(float(np.mean([m["realized"]["net_vy"] for m in recs])), 5),
                "realized_straightness_median": round(float(np.median(
                    [m["realized"]["straightness"] for m in recs
                     if m["realized"]["straightness"] is not None])), 4),
                "ratio_to_nominal_mean": round(float(np.mean(ratios)), 4) if ratios else None,
                "planner_open_loop": ol,
                "planner_open_loop_over_realized": round(float(ol["vx"]) / float(np.mean(nets)), 3)
                if ol.get("vx") and nets and np.mean(nets) > 1e-9 else None,
                "token_stats": token_stats(tokn),
                "h_min_over_scripts": round(float(min(m["realized"]["h_min"] for m in recs
                                                      if m["realized"]["h_min"] is not None)), 4),
                "wall_s": round(time.time() - t_p, 2)}
        point_summaries.append(psum)
        manifest["points"].append(psum)

    summary = {"schema": SCHEMA, "out_dir": str(out_dir), "n_points": len(points),
               "n_scripts_total": total_scripts, "dur_s": args.dur_s,
               "wall_s_total": round(time.time() - t0, 1),
               "grid": {"target_vel": vels, "modes": [m[0] for m in modes], "directions": dirs},
               "sp5t_check": {"scripts_per_point_ge_10": bool(args.n_scripts >= 10),
                              "note": "§5t 抽查门=每命令点 >=10 剧本; 全网格={0,0.2,..,1.2}x{SLOW_WALK,RUN}x{front,left,right}"},
               "points": point_summaries}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest["summary"] = {p["point_tag"]: {"n_scripts": p["n_scripts"],
                                            "realized_net_speed_xy_mean": p["realized_net_speed_xy"]["mean"],
                                            "ratio_to_nominal_mean": p["ratio_to_nominal_mean"],
                                            "planner_open_loop_vx": p["planner_open_loop"]["vx"],
                                            "n_fall": p["n_fall"], "n_horizon_cap": p["n_horizon_cap"]}
                          for p in point_summaries}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n=== D060 SRC1 SUMMARY ===")
    hdr = (f"{'point':26s} {'n':>3s} {'fall':>4s} {'cap':>3s} {'nom':>5s} {'real_xy':>8s} {'ratio':>6s} "
           f"{'ol_vx':>6s} {'ol/real':>7s} {'tok_mean':>9s} {'tok_std':>8s} {'wall_s':>7s}")
    print(hdr)
    for p in point_summaries:
        rn = p["realized_net_speed_xy"]
        print(f"{p['point_tag']:26s} {p['n_scripts']:3d} {p['n_fall']:4d} {p['n_horizon_cap']:3d} "
              f"{p['nominal']['target_vel']:5.2f} "
              f"{(rn['mean'] if rn['mean'] is not None else float('nan')):8.3f} "
              f"{(p['ratio_to_nominal_mean'] if p['ratio_to_nominal_mean'] is not None else float('nan')):6.3f} "
              f"{(p['planner_open_loop']['vx'] or float('nan')):6.3f} "
              f"{(p['planner_open_loop_over_realized'] or float('nan')):7.3f} "
              f"{p['token_stats'].get('mean', float('nan')):9.5f} {p['token_stats'].get('std', float('nan')):8.5f} "
              f"{p['wall_s']:7.1f}")
    print(f"total: {total_scripts} scripts, wall {time.time() - t0:.1f}s -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# selftest (pure numpy; no mujoco / onnxruntime / files beyond tmp)
# ---------------------------------------------------------------------------
def selftest() -> int:
    import tempfile
    checks = []

    def ck(name, cond, extra=""):
        checks.append((name, bool(cond), extra))

    vels = parse_target_vels("grid")
    ck("grid_spec_len7", len(vels) == 7 and vels[0] == 0.0 and vels[-1] == 1.2, str(vels))
    ck("grid_step_0.2", all(abs((vels[i + 1] - vels[i]) - 0.2) < 1e-9 for i in range(6)))
    ck("list_spec", parse_target_vels("0,0.6,1.0") == [0.0, 0.6, 1.0])
    ck("sentinel_spec", parse_target_vels("-1") == [-1.0])
    try:
        parse_target_vels("")
        ck("empty_spec_raises", False)
    except ValueError:
        ck("empty_spec_raises", True)

    modes = parse_modes("SLOW_WALK,RUN")
    ck("mode_names", modes == [("SLOW_WALK", 1), ("RUN", 3)], str(modes))
    ck("mode_ids", parse_modes("1,3") == [("SLOW_WALK", 1), ("RUN", 3)])
    try:
        parse_modes("CRAWL")
        ck("bad_mode_raises", False)
    except ValueError:
        ck("bad_mode_raises", True)

    dirs = parse_directions("front,left,right")
    ck("dir_order", dirs == ["front", "left", "right"], str(dirs))
    ck("dir_vecs", DIR_VECS["front"] == (1.0, 0.0, 0.0) and DIR_VECS["left"] == (0.0, 1.0, 0.0)
       and DIR_VECS["right"] == (0.0, -1.0, 0.0))

    pts = build_points(vels, modes, dirs)
    ck("full_grid_42", len(pts) == 42, str(len(pts)))
    ck("grid_unique", len({(p["target_vel"], p["mode_id"], p["direction"]) for p in pts}) == 42)
    ck("grid_mode_split", sum(1 for p in pts if p["mode_name"] == "SLOW_WALK") == 21
       and sum(1 for p in pts if p["mode_name"] == "RUN") == 21)
    ck("grid_dir_split", sum(1 for p in pts if p["direction"] == "front") == 14)
    ck("point_tag_format", point_tag(3, {"target_vel": 0.6, "mode_name": "RUN", "direction": "left"})
       == "p03_vel0.6_RUN_left", point_tag(3, {"target_vel": 0.6, "mode_name": "RUN", "direction": "left"}))
    pp = parse_point_spec("0.6:SLOW_WALK:front")
    ck("point_spec_fields", pp["target_vel"] == 0.6 and pp["mode_id"] == 1 and pp["direction"] == "front"
       and pp["movement_direction"] == [1.0, 0.0, 0.0] and pp["facing_direction"] == [1.0, 0.0, 0.0], str(pp))
    ck("point_spec_ids", parse_point_spec("1:RUN:left")["mode_id"] == 3
       and parse_point_spec("1:3:left")["mode_id"] == 3)
    ck("point_spec_sentinel", parse_point_spec("-1:RUN:front")["target_vel"] == -1.0)
    try:
        parse_point_spec("0.6:SLOW_WALK")
        ck("point_spec_bad_raises", False)
    except ValueError:
        ck("point_spec_bad_raises", True)

    s = [seed_for(k, 1000) for k in range(10)]
    ck("seed_deterministic", s == [1000 + k for k in range(10)], str(s))
    ck("seed_distinct", len(set(s)) == 10)

    rp = np.zeros((101, 3), dtype=np.float32)
    rp[:, 0] = np.arange(101) / 50.0  # 1 m/s straight, 2.0 s endpoint interval
    st = net_speed_stats(rp)
    ck("net_vx_1ms", abs(st["net_vx"] - 1.0) < 1e-6 and abs(st["net_speed_xy"] - 1.0) < 1e-6, str(st))
    ck("net_n_101", st["n_steps"] == 101 and abs(st["dur_s"] - 2.02) < 1e-9 and abs(st["elapsed_s"] - 2.0) < 1e-9)
    ck("net_straightness_1", st["straightness"] is not None and abs(st["straightness"] - 1.0) < 1e-3)
    rp2 = np.zeros((101, 3), dtype=np.float32)
    rp2[:, 0] = np.concatenate([np.arange(51) / 50.0, 1.0 - np.arange(1, 51) / 50.0])
    st2 = net_speed_stats(rp2)
    ck("net_back_forth_net0", st2["net_speed_xy"] is not None and st2["net_speed_xy"] < 0.02, str(st2["net_speed_xy"]))
    ck("net_back_forth_pathlen", st2["path_len_xy"] is not None and abs(st2["path_len_xy"] - 2.0) < 0.02,
       str(st2["path_len_xy"]))
    ck("net_safe_empty", net_speed_stats(np.zeros((0, 3), dtype=np.float32))["net_speed_xy"] is None)

    t = np.zeros((10, 64), dtype=np.float32)
    t[:5] = 0.5
    t[5:] = -0.5
    ts = token_stats(t)
    ck("tok_stats_mean0", abs(ts["mean"]) < 1e-9 and abs(ts["std"] - 0.5) < 1e-6, str(ts))
    ck("tok_stats_shape", ts["n_tokens"] == 10 and ts["dim"] == 64)
    ck("tok_stats_frac_neg", abs(ts["frac_neg"] - 0.5) < 1e-9)
    ck("tok_stats_safe", token_stats(np.zeros((0, 64), dtype=np.float32)) == {"n_tokens": 0})

    ck("decay_ratio", decay_ratio(1.0, 2.12) == round(1.0 / 2.12, 4))
    ck("decay_ratio_zero_nominal", decay_ratio(0.5, 0.0) is None)
    ck("decay_ratio_sentinel_nominal", decay_ratio(0.5, -1.0) is None)
    ck("decay_ratio_none", decay_ratio(None, 1.0) is None)

    ck("term_none_when_not_terminated", classify_termination(False, 500, 1000) is None)
    ck("term_horizon_cap", classify_termination(True, 1000, 1000) == "horizon_cap")
    ck("term_horizon_cap_over", classify_termination(True, 1001, 1000) == "horizon_cap")
    ck("term_real_fall", classify_termination(True, 321, 1000) == "fall")
    ck("term_real_fall_last_minus1", classify_termination(True, 999, 1000) == "fall")

    tf = triage_flags({"net_speed_xy": 0.31, "h_min": 0.67, "fall": False})
    ck("flags_clean", tf["low_net"] is False and tf["low_posture"] is False and tf["fall"] is False, str(tf))
    ck("flags_low_net_upright", triage_flags({"net_speed_xy": 0.19, "h_min": 0.69, "fall": False})["low_net"] is True
       and triage_flags({"net_speed_xy": 0.19, "h_min": 0.69, "fall": False})["low_posture"] is False)
    ck("flags_low_posture_fast", triage_flags({"net_speed_xy": 0.49, "h_min": 0.53, "fall": False})["low_posture"] is True
       and triage_flags({"net_speed_xy": 0.49, "h_min": 0.53, "fall": False})["low_net"] is False)
    ck("flags_thresholds_recorded", tf["thresholds"] == {"low_net_net_speed_xy": LOW_NET_THRESH,
                                                         "low_posture_h_min": LOW_POSTURE_H_THRESH})
    ck("flags_empty_safe", triage_flags({"net_speed_xy": None, "h_min": None})["low_net"] is False)
    ck("flags_fall_propagates", triage_flags({"net_speed_xy": 1.0, "h_min": 0.21, "fall": True})["fall"] is True)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sub" / "script_000.npz"
        meta = {"schema": SCHEMA, "nominal": {"target_vel": 0.6, "mode_name": "SLOW_WALK"},
                "seed": 7, "realized": {"net_speed_xy": 0.51}, "中文键": "值"}
        save_script(p, np.ones((3, 64), dtype=np.float32), np.ones((3, 3), dtype=np.float32),
                    np.tile([1, 0, 0, 0], (3, 1)), np.zeros((3, 29)), np.zeros((3, 29)), meta)
        arr, m2 = load_script(p)
        ck("npz_roundtrip_keys",
           set(arr) == {"tokens", "root_pos", "root_quat", "jp_mujoco", "jp_isaaclab"}, str(sorted(arr)))
        ck("npz_dtypes", all(v.dtype == np.float32 for v in arr.values()))
        ck("npz_shapes", arr["tokens"].shape == (3, 64) and arr["root_quat"].shape == (3, 4)
           and arr["jp_isaaclab"].shape == (3, 29))
        ck("npz_meta_roundtrip", m2 == meta, str(m2.get("中文键")))
        ck("npz_meta_is_str", isinstance(m2["realized"]["net_speed_xy"], float))
        ck("md5_stable", md5_file(p) == md5_file(p) and len(md5_file(p)) == 32)

    # --- npz meta key contract (build_script_meta is pure -> testable without mujoco) ---
    pt = {"target_vel": 0.6, "mode_id": 1, "mode_name": "SLOW_WALK", "direction": "front",
          "movement_direction": [1.0, 0.0, 0.0], "facing_direction": [1.0, 0.0, 0.0]}
    roll = {"root_pos": np.tile([0.0, 0.0, 0.7], (5, 1)), "n_steps_requested": 1000,
            "n_steps_effective": 1000, "episode_length": 1000, "terminated_cause": "horizon_cap",
            "env_step_count_at_term": 1000, "fall": False, "fall_step": None,
            "first_plan_open_loop": {"n_frames": 48, "vx": 0.678}}
    m = build_script_meta(point=pt, point_tag="p00_x", point_idx=0, script_idx=3, seed=3,
                          seed_base=0, dur_s=20.0, height=-1.0, spawn_x=-6.0, spawn_z=0.85,
                          ol={"n_frames": 48, "vx": 0.678}, rollout=roll,
                          realized=net_speed_stats(roll["root_pos"]), tstat={"n_tokens": 5})
    ck("meta_no_wall_s", "wall_s" not in m and "wall_s" not in m["realized"])
    ck("meta_required_keys",
       {"schema", "point_tag", "point_idx", "script_idx", "nominal", "seed", "seed_base", "dur_s",
        "n_steps_requested_full", "n_steps_effective_planned", "env_episode_length", "realized",
        "planner_open_loop", "planner_open_loop_first_plan", "token_stats", "flags", "spawn"} <= set(m),
       str(sorted(set(m) ^ {"schema"})))
    ck("meta_step_keys_semantics",
       m["n_steps_requested_full"] == 1000 and m["n_steps_effective_planned"] == 1000
       and m["realized"]["n_steps"] == 5, "authoritative count = realized.n_steps")
    ck("meta_nominal_block",
       m["nominal"]["target_vel"] == 0.6 and m["nominal"]["mode_id"] == 1
       and m["nominal"]["ctrl_hz"] == CTRL_HZ and m["nominal"]["replan_hz"] == 10.0)
    ck("meta_ratio_and_flags", m["realized"]["ratio_to_nominal"] == decay_ratio(
        m["realized"]["net_speed_xy"], 0.6) and m["flags"]["fall"] is False)
    ck("meta_json_serializable", isinstance(json.dumps(m, ensure_ascii=False), str))

    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, extra in checks:
        if not ok:
            print(f"  FAIL {name} {extra}")
    print(f"SELFTEST {npass}/{len(checks)} PASS")
    return 0 if npass == len(checks) else 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="D060 source-1 planner command->script generator")
    ap.add_argument("--target-vel", default="grid",
                    help="comma list, or 'grid' = {0,0.2,...,1.2}; -1 = mode-default sentinel")
    ap.add_argument("--mode", default="SLOW_WALK,RUN", help="comma list of names or ids (SLOW_WALK=1, RUN=3)")
    ap.add_argument("--direction", default="front", help="comma list of front|left|right")
    ap.add_argument("--point", action="append", default=None,
                    help="explicit point 'vel:MODE:direction' (repeatable); overrides the cartesian "
                         "product. Negative vel needs the equals form: --point=-1:RUN:front "
                         "(or use --target-vel -1 --mode RUN --direction front)")
    ap.add_argument("--n-scripts", type=int, default=10, help="scripts per command point (§5t needs >=10)")
    ap.add_argument("--dur-s", type=float, default=20.0, help="rollout duration per script (s @ 50 Hz)")
    ap.add_argument("--out-dir", default="data/d060/planner_scripts_smoke")
    ap.add_argument("--seed-base", type=int, default=0, help="planner random_seed = seed_base + script_idx")
    ap.add_argument("--spawn-x", type=float, default=-6.0)
    ap.add_argument("--spawn-z", type=float, default=0.85)
    ap.add_argument("--height", type=float, default=-1.0, help="planner height (-1 = default sentinel)")
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--apt-dir", default=APT_DIR_DEFAULT)
    ap.add_argument("--ros2-data", default=ROS2_DATA_DEFAULT)
    ap.add_argument("--planner-onnx", default=None, help="override planner ONNX path")
    ap.add_argument("--open-loop-point-probe", action="store_true", default=True,
                    help="record a standing-context planner ONNX probe per point (D033 protocol)")
    ap.add_argument("--no-open-loop-point-probe", dest="open_loop_point_probe", action="store_false")
    ap.add_argument("--venv-hint", default=".venv_mjlab")
    ap.add_argument("--dry-run", action="store_true", help="enumerate command points/seeds only")
    ap.add_argument("--selftest", action="store_true", help="pure numpy checks, no mujoco/ORT")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
