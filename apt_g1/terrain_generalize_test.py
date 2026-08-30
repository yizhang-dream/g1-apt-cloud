"""Terrain-generalization test: flat vs rough 0.08, per LocomotionMode.

For each key mode this drives the full planner->encoder->decoder pipeline and
reports, on FLAT ground and on the rough (0.1 m cells, 0-0.075 m bumps) hfield:

  * fall      : step index of termination (None = survived the whole clip)
  * adv       : root x displacement over the clip (walking + any sliding)
  * h_rms     : RMS deviation of root z from the planner's planned root z (m)
  * jp_rms    : RMS joint-position tracking error vs the planned trajectory (rad)

h_rms / jp_rms are the real "does it track the flat-ground reference" signal:
a collapse shows up as a large h_rms even when the 0.2 m termination threshold
is never crossed. This is part (4) of the goal (flat -> rough generalization).

Usage (server, mjlab venv):
    python terrain_generalize_test.py
"""

from __future__ import annotations

import sys

import numpy as np

REPO = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl"
PLANNER_ONNX = f"{REPO}/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
ENC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_encoder.onnx"
DEC_ONNX = f"{REPO}/gear_sonic_deploy/policy/release/model_decoder.onnx"
FLAT = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
ROUGH = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/scene_43dof_rough.xml"

BODY_IDX = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]

MODES = {
    0: "idle",
    1: "slow_walk",
    2: "walk",
    3: "run",
    4: "squat",
    5: "kneel_two_legs",
    8: "crawl",
    18: "stealth_walk",
}


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


def track(mode: int, scene: str, spawn_x: float):
    sys.path.insert(0, "/home/cvgluser/ros2_data")
    sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
    sys.path.insert(0, REPO)
    import mujoco
    import onnxruntime as ort
    from apt_g1.envs.mujoco_g1_flat_env import (
        MujocoG1FlatEnv, SONIC_DEFAULT_ANGLES_MUJOCO, G1_MUJOCO_TO_ISAACLAB_DOF)
    from eval_distill import NoQuantDecoder

    planner = ort.InferenceSession(PLANNER_ONNX, providers=["CPUExecutionProvider"])
    enc = ort.InferenceSession(ENC_ONNX, providers=["CPUExecutionProvider"])
    iname = enc.get_inputs()[0].name
    env = MujocoG1FlatEnv(NoQuantDecoder(DEC_ONNX), REPO, robot_scene=scene,
                          use_elastic_band=False, stand_only=True)
    m2i = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)

    def standing():
        q = np.zeros(36, dtype=np.float32)
        q[0:3] = [0, 0, 0.76]; q[3:7] = [1, 0, 0, 0]; q[7:36] = SONIC_DEFAULT_ANGLES_MUJOCO
        return q

    def fk(q36):
        q = np.zeros(env.model.nq)
        q[:3] = q36[:3]; q[3:7] = q36[3:7]; q[env.body_qpos_adr] = q36[7:36]
        env.data.qpos[:] = q
        mujoco.mj_forward(env.model, env.data)
        return env.data.xquat[BODY_IDX].copy()

    inp = {"context_mujoco_qpos": np.tile(standing()[None, None], (1, 4, 1)).astype(np.float32),
           "target_vel": np.array([-1.0], dtype=np.float32), "mode": np.array([mode], dtype=np.int64),
           "movement_direction": np.array([[1.0, 0, 0]], dtype=np.float32),
           "facing_direction": np.array([[1.0, 0, 0]], dtype=np.float32),
           "random_seed": np.array([0], dtype=np.int64), "has_specific_target": np.array([[0]], dtype=np.int64),
           "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
           "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
           "allowed_pred_num_tokens": np.ones((1, 11), dtype=np.int64),
           "height": np.array([-1.0], dtype=np.float32)}
    qpos_out, nframes_out = planner.run(None, inp)
    traj = qpos_out[0][:int(nframes_out[0])]
    jp = traj[:, 7:36][:, m2i]                      # planned joints, IsaacLab order
    planned_root_z = traj[:, 2].astype(np.float64)  # planned root height
    jv = np.vstack([np.zeros((1, 29)), np.diff(jp, axis=0) * 50.0])
    bq = np.array([fk(t) for t in traj])
    apply_delta = _qn(_qmul(_heading(np.array([1.0, 0, 0, 0])), _heading_inv(bq[0, 0])))

    env.reset()
    env.data.qpos[0] = spawn_x
    env.data.qpos[2] = 0.85
    for _ in range(40):
        env._step_physics(SONIC_DEFAULT_ANGLES_MUJOCO.copy())

    xs, hs, jp_err, h_err, fall = [], [], [], [], None
    for t in range(len(traj)):
        live = env.data.qpos[3:7].astype(np.float64)
        obs = np.zeros(1762, dtype=np.float32); obs[0] = 0.0; p = 4
        for f in range(10):
            idx = min(t + f * 5, len(jp) - 1); obs[p:p + 29] = jp[idx]; p += 29
        for f in range(10):
            idx = min(t + f * 5, len(jv) - 1); obs[p:p + 29] = jv[idx]; p += 29
        p += 17
        for f in range(10):
            idx = min(t + f * 5, len(bq) - 1)
            nr = _qn(_qmul(apply_delta, bq[idx, 0]))
            btr = _qn(_qmul(_qconj(live), nr))
            rot = _rotmat(btr); obs[p:p + 6] = rot[:, :2].flatten(); p += 6
        tok = enc.run(None, {iname: obs[None]})[0][0].astype(np.float32)
        _, _, term, _ = env.step({"token": tok, "aux": np.zeros(12, dtype=np.float32)})
        actual_isaac = env.data.qpos[env.body_qpos_adr][m2i].astype(np.float64)
        xs.append(float(env.data.qpos[0])); hs.append(float(env.data.qpos[2]))
        jp_err.append(float(np.linalg.norm(actual_isaac - jp[t]) / np.sqrt(29.0)))
        h_err.append(float(env.data.qpos[2] - planned_root_z[t]))
        if term:
            fall = t; break
    return dict(fall=fall, n=len(hs), x0=round(xs[0], 2), x_end=round(xs[-1], 2),
                adv=round(xs[-1] - xs[0], 2),
                h_rms=round(float(np.sqrt(np.mean(np.square(h_err)))), 3),
                jp_rms=round(float(np.sqrt(np.mean(np.square(jp_err)))), 3),
                h_min=round(float(min(hs)), 2), h_end=round(float(hs[-1]), 2))


def _row(tag, r):
    return (f"{tag:<6}{str(r['fall']):>6}{r['n']:>5}{r['adv']:>7}{r['h_rms']:>8}"
            f"{r['jp_rms']:>8}{r['h_min']:>7}{r['h_end']:>7}")


if __name__ == "__main__":
    hdr = f"{'tag':<6}{'fall':>6}{'n':>5}{'adv':>7}{'h_rms':>8}{'jp_rms':>8}{'h_min':>7}{'h_end':>7}"
    for m, name in MODES.items():
        print(f"--- {name} (mode {m}) ---")
        print(hdr)
        try:
            print(_row("flat", track(m, FLAT, 0.0)))
        except Exception as e:
            print(f"flat    ERROR {type(e).__name__}: {e}")
        try:
            print(_row("rough", track(m, ROUGH, -6.0)))
        except Exception as e:
            print(f"rough   ERROR {type(e).__name__}: {e}")
