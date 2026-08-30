"""Closed-loop SONIC planner: 10 Hz re-planning + terrain-triggered gait switch.

Unlike `planner_sonic.py` (which runs the planner ONCE and replays the trajectory
open-loop), this drives the official closed loop: every 5 control steps (10 Hz at
50 Hz control) the kinematic planner is re-run with the LIVE 4-frame qpos context
and the current LocomotionMode, so the mode command can change mid-run.

Demonstrates the MQ08 finding that low-impact gaits (stealth) survive rough 0.08
where walk stalls: a roughness trigger (root-z oscillation over a sliding window)
switches walk -> stealth, and the switched run advances further than pure walk.

Usage (server, mjlab venv):
    python planner_closed_loop.py --scenario walk          # pure walk on rough
    python planner_closed_loop.py --scenario stealth       # pure stealth on rough
    python planner_closed_loop.py --scenario walk2stealth  # trigger-switched
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
PLANNER_ONNX = f"{REPO}/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
ENC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_encoder.onnx"
DEC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"
FLAT = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
ROUGH = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/scene_43dof_rough.xml"

BODY_IDX = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]
MODE = {"walk": 2, "stealth": 18, "crawl": 8}


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


def run(scenario: str, spawn_x: float = -6.0, n_steps: int = 300,
        trigger_std: float = 0.02, trigger_window: int = 25, seed: int = 0,
        scene: str = "rough"):
    sys.path.insert(0, "/home/cvgluser/ros2_data")
    sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
    sys.path.insert(0, REPO)
    import mujoco
    import onnxruntime as ort
    from apt_g1.envs.mujoco_g1_flat_env import (
        MujocoG1FlatEnv, SONIC_DEFAULT_ANGLES_MUJOCO, G1_MUJOCO_TO_ISAACLAB_DOF)
    from eval_distill import NoQuantDecoder

    scene_path = FLAT if scene == "flat" else ROUGH
    planner = ort.InferenceSession(PLANNER_ONNX, providers=["CPUExecutionProvider"])
    enc = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])
    iname = enc.get_inputs()[0].name
    env = MujocoG1FlatEnv(NoQuantDecoder(DEC_ONNX), REPO, robot_scene=scene_path,
                          use_elastic_band=False, stand_only=True)
    m2i = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)
    fk_data = mujoco.MjData(env.model)  # separate data: FK must not corrupt env state

    def standing():
        q = np.zeros(36, dtype=np.float32)
        q[0:3] = [0, 0, 0.76]; q[3:7] = [1, 0, 0, 0]; q[7:36] = SONIC_DEFAULT_ANGLES_MUJOCO
        return q

    def live_q36():
        q = np.zeros(36, dtype=np.float32)
        q[0:3] = env.data.qpos[0:3]
        q[3:7] = env.data.qpos[3:7]
        q[7:36] = env.data.qpos[env.body_qpos_adr]
        return q

    def fk(q36):
        q = np.zeros(env.model.nq)
        q[:3] = q36[:3]; q[3:7] = q36[3:7]; q[env.body_qpos_adr] = q36[7:36]
        fk_data.qpos[:] = q
        mujoco.mj_forward(env.model, fk_data)
        return fk_data.xquat[BODY_IDX].copy()

    def plan(mode):
        # The planner's 4-frame context is the CURRENT live state tiled 4x.
        # (A 4-frame 10Hz history buffer -- and a canonical/frozen root -- both
        # feed a mismatched context and drive a gradual height collapse; see the
        # diagnostics in the tracker MQ09 note.)
        ctx = np.tile(live_q36()[None, None], (1, 4, 1)).astype(np.float32)
        inp = {"context_mujoco_qpos": ctx,
               "target_vel": np.array([-1.0], dtype=np.float32),
               "mode": np.array([mode], dtype=np.int64),
               "movement_direction": np.array([[1.0, 0, 0]], dtype=np.float32),
               "facing_direction": np.array([[1.0, 0, 0]], dtype=np.float32),
               "random_seed": np.array([seed], dtype=np.int64),
               "has_specific_target": np.array([[0]], dtype=np.int64),
               "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
               "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
               "allowed_pred_num_tokens": np.ones((1, 11), dtype=np.int64),
               "height": np.array([-1.0], dtype=np.float32)}
        qpos_out, nframes_out = planner.run(None, inp)
        traj = qpos_out[0][:int(nframes_out[0])]
        jp = traj[:, 7:36][:, m2i]
        jv = np.vstack([np.zeros((1, 29)), np.diff(jp, axis=0) * 50.0])
        bq = np.array([fk(t) for t in traj])
        apply_delta = _qn(_qmul(_heading(np.array([1.0, 0, 0, 0])), _heading_inv(bq[0, 0])))
        return jp, jv, bq, apply_delta

    # spawn on rough and settle
    env.reset()
    env.data.qpos[0] = spawn_x
    env.data.qpos[2] = 0.85
    for _ in range(40):
        env._step_physics(SONIC_DEFAULT_ANGLES_MUJOCO.copy())

    jp = jv = bq = apply_delta = None
    cur_frame = 0
    cur_mode = MODE.get(scenario, MODE["walk"])
    switched_at = None
    xs, hs, zstd = [], [], []

    for step in range(n_steps):
        if step % 5 == 0:  # 10 Hz re-plan
            jp, jv, bq, apply_delta = plan(cur_mode)
            cur_frame = 0

        # roughness trigger: sliding-window root-z std
        if scenario in ("walk2stealth", "walk2crawl") and switched_at is None and len(hs) >= trigger_window:
            w = np.asarray(hs[-trigger_window:])
            if float(w.std()) > trigger_std:
                switched_at = step
                cur_mode = MODE["stealth"] if scenario == "walk2stealth" else MODE["crawl"]

        live = env.data.qpos[3:7].astype(np.float64)
        obs = np.zeros(1762, dtype=np.float32); obs[0] = 0.0; p = 4
        for f in range(10):
            idx = min(cur_frame + f * 5, len(jp) - 1); obs[p:p + 29] = jp[idx]; p += 29
        for f in range(10):
            idx = min(cur_frame + f * 5, len(jv) - 1); obs[p:p + 29] = jv[idx]; p += 29
        p += 17
        for f in range(10):
            idx = min(cur_frame + f * 5, len(bq) - 1)
            nr = _qn(_qmul(apply_delta, bq[idx, 0]))
            btr = _qn(_qmul(_qconj(live), nr))
            rot = _rotmat(btr); obs[p:p + 6] = rot[:, :2].flatten(); p += 6
        x_pre = float(env.data.qpos[0]); h_pre = float(env.data.qpos[2])
        tok = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
        _, _, term, _ = env.step({"token": tok, "aux": np.zeros(12, dtype=np.float32)})
        cur_frame += 1
        # record PRE-step state: env.step() resets on termination, corrupting qpos
        xs.append(x_pre); hs.append(h_pre)
        if len(hs) >= trigger_window:
            zstd.append(float(np.asarray(hs[-trigger_window:]).std()))
        if term:
            break

    h = np.asarray(hs)
    return dict(scenario=scenario, fall="term" if term else None, n=len(h),
                x0=round(xs[0], 2), x_end=round(xs[-1], 2),
                adv=round(xs[-1] - xs[0], 2),
                h_min=round(float(h.min()), 2), h_end=round(float(h[-1]), 2),
                switched_at=switched_at,
                zstd_peak=round(float(max(zstd)) if zstd else 0.0, 4),
                zstd_mean=round(float(np.mean(zstd)) if zstd else 0.0, 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["walk", "stealth", "crawl", "walk2stealth", "walk2crawl"], default="walk")
    ap.add_argument("--trigger-std", type=float, default=0.02)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--scene", choices=["flat", "rough"], default="rough")
    args = ap.parse_args()
    r = run(args.scenario, trigger_std=args.trigger_std, n_steps=args.n_steps, scene=args.scene)
    print(f"{r['scenario']:<14} fall={str(r['fall']):>5} n={r['n']:>4} "
          f"adv={r['adv']:>6} h_min={r['h_min']:>5} h_end={r['h_end']:>5} "
          f"switched_at={r['switched_at']} zstd_peak={r['zstd_peak']} zstd_mean={r['zstd_mean']}")
