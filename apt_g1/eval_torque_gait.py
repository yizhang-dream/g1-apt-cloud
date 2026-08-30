"""Torque-level closed-loop smoke test (TO11): does the correct actuator torque
tau_clean (foot-gait full ID) + light PD drive the G1 to walk?

This is the decisive test for the moment-level pipeline: TO06 applied the WRONG
feedforward (tau_SRB = J^T f, the CONTACT force, not the motor torque).  Here we apply
the CORRECT feedforward tau_clean = M qdd + qfrc_bias - qfrc_constraint (computed in
foot_gait_id.py / train_torque_decoder_gait.py) as a phase-indexed lookup, plus a light
PD on the default stand pose for stabilization.

Control:  tau = kp_scale*KP*(q_default - q) - kd_scale*KD*qdot   (all 29 body DOF)
          + tau_clean(phi)                                      (sagittal leg joints only)

Run on the SERVER under .venv_mjlab.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

import mujoco

from foot_gait_id import (
    SCENE, DEFAULT_Q, LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH,
    RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH,
)
from train_torque_decoder_gait import compute_gait_full

# SONIC PD gains (MuJoCo order, 29 non-hand DOF) -- copied from mujoco_g1_flat_env.py
KP = np.array([
    99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
    99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
    40.17924, 28.50125, 28.50125,
    14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
    14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
], dtype=np.float64)
KD = np.array([
    6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
    6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
    2.55789, 1.81445, 1.81445,
    0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
    0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
], dtype=np.float64)
EFFORT = np.array([
    88, 88, 88, 139, 50, 50,
    88, 88, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
], dtype=np.float64)

SAG = [LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH, RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH]


def setup():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    qpos_adr, dof_adr, act_ids = [], [], []
    for act_id in range(model.nu):
        jid = model.actuator_trnid[act_id, 0]
        name = model.joint(jid).name
        if "hand" in name:
            continue
        act_ids.append(act_id)
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])
    return model, data, np.asarray(qpos_adr, int), np.asarray(dof_adr, int), np.asarray(act_ids, int)


def main(v=0.5, T=0.5, kp_scale=0.5, kd_scale=1.0, seconds=10.0, seed=0, aux=None,
         track_gait=False, stab_kp=0.0, stab_kd=0.0, stab_sign=1):
    np.random.seed(seed)
    model, data, qpos_adr, dof_adr, act_ids = setup()
    phi_arr, X, tau, Q, Qd = compute_gait_full(v, T=T)
    phases = np.linspace(0, 2 * np.pi, tau.shape[0], endpoint=False)
    # tau columns: [hip_L, knee_L, ankle_L, hip_R, knee_R, ankle_R]
    tau_cols = {LEFT_HIP_PITCH: tau[:, 0], LEFT_KNEE: tau[:, 1], LEFT_ANKLE_PITCH: tau[:, 2],
                RIGHT_HIP_PITCH: tau[:, 3], RIGHT_KNEE: tau[:, 4], RIGHT_ANKLE_PITCH: tau[:, 5]}

    # fixed aux offset (joint-angle offset added to the PD default pose, scaled 0.2 like the paper)
    q_des = DEFAULT_Q.copy()
    if aux is not None:
        for j, a in aux.items():
            q_des[j] += 0.2 * a

    data.qpos[0:3] = [0.0, 0.0, 0.76]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = DEFAULT_Q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    dt = 0.005
    decim = 4
    dt_ctrl = dt * decim
    n_steps = int(seconds / dt_ctrl)
    x0 = float(data.qpos[0])
    h_min = float(data.qpos[2])
    fall = None
    com_prev = None  # TO20: whole-body CoM x for the ankle-strategy feedback
    for step in range(n_steps):
        t = step * dt_ctrl
        phi = 2.0 * np.pi * (t % T / T)
        tau_ff = np.array([np.interp(phi, phases, tau_cols[j]) for j in SAG])
        # TO20: ankle-strategy CoM tracking on the STANCE ankle. The gait's
        # left leg is stance for phi in [0, pi), right for [pi, 2pi). The
        # plan's CoM is x = v*t (hx), so the feedback regulates the drift of
        # the actual CoM away from the planned one.
        tau_stab = 0.0
        if stab_kp > 0.0 or stab_kd > 0.0:
            mujoco.mj_forward(model, data)
            com_x = float(data.subtree_com[0][0])
            if com_prev is None:
                com_vx = 0.0
            else:
                com_vx = (com_x - com_prev) / dt_ctrl
            com_prev = com_x
            err_x = com_x - v * t
            err_vx = com_vx - v
            tau_stab = stab_sign * (stab_kp * err_x + stab_kd * err_vx)
        stance_ankle = LEFT_ANKLE_PITCH if phi < np.pi else RIGHT_ANKLE_PITCH
        if track_gait:
            # TO18: PD tracks the PLANNED gait trajectory (with velocity
            # feedforward) instead of the standing default -- feedback and
            # feedforward cooperate instead of fighting.
            q_gait = np.array([np.interp(phi, phases, Q[:, j]) for j in range(Q.shape[1])])
            qd_gait = np.array([np.interp(phi, phases, Qd[:, j]) for j in range(Qd.shape[1])])
            base_q = q_gait
        else:
            qd_gait = None
            base_q = q_des
        for _ in range(decim):
            q = data.qpos[qpos_adr]
            qd = data.qvel[dof_adr]
            torque = kp_scale * KP * (base_q - q) - kd_scale * KD * qd
            if qd_gait is not None:
                torque = torque + kd_scale * KD * qd_gait
            for k, j in enumerate(SAG):
                torque[j] += tau_ff[k]
            torque[stance_ankle] += tau_stab
            torque = np.clip(torque, -EFFORT, EFFORT)
            ctrl = np.zeros(model.nu)
            ctrl[act_ids] = torque
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        h_min = min(h_min, float(data.qpos[2]))
        if float(data.qpos[2]) < 0.2 or not np.all(np.isfinite(data.qpos)):
            fall = step * dt_ctrl
            break

    disp = float(data.qpos[0] - x0)
    n = n_steps if fall is None else max(1, step + 1)
    print(f"=== {'TO18 gait-tracking' if track_gait else 'TO11 default-PD'} closed loop "
          f"(v={v} T={T} kp_scale={kp_scale} kd_scale={kd_scale} {seconds}s seed={seed}) ===")
    print(f"  fall={fall}  h_min={h_min:.3f}  disp={disp:+.2f}m  vx={disp/(n*dt_ctrl):+.2f} m/s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--kp-scale", type=float, default=0.5)
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track-gait", action="store_true",
                    help="TO18: PD tracks the planned gait trajectory (not the default stand)")
    ap.add_argument("--stab-kp", type=float, default=0.0,
                    help="TO20: ankle-strategy CoM position gain (0 = off)")
    ap.add_argument("--stab-kd", type=float, default=0.0,
                    help="TO20: ankle-strategy CoM velocity gain")
    ap.add_argument("--stab-sign", type=int, default=1, choices=[1, -1])
    a = ap.parse_args()
    main(a.v, a.T, a.kp_scale, a.kd_scale, a.seconds, a.seed, track_gait=a.track_gait,
         stab_kp=a.stab_kp, stab_kd=a.stab_kd, stab_sign=a.stab_sign)
