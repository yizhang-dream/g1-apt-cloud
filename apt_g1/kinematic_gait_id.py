"""Kinematic gait + full inverse-dynamics torque (TO08, corrected).

Measures the FULL multi-body ID torque of a smooth leg-swing gait and compares it
against the SRB "massless leg" static moment arm.

Correct self-consistent actuator torque (MuJoCo equation M qdd + qfrc_bias = tau_act + J^T lambda):
    tau_clean = (M qdd + qfrc_bias) - qfrc_constraint

IMPORTANT: we compute M qdd + qfrc_bias MANUALLY (M = mj_fullM, qfrc_bias from
mj_forward) rather than calling mj_inverse.  Debugging (TO08) showed that MuJoCo's
mj_inverse returns WRONG results for this floating-base G1 model when BOTH legs are
driven with non-zero qacc (e.g. hip +154 Nm vs manual -3.0 Nm, a 50x error), while the
manual M@qacc + qfrc_bias matches the inertia matrix exactly.  The manual route is used
throughout here.

Gait: pelvis at 0.76 m, constant forward speed v, upright; hip_pitch/knee as smooth
sinusoids (180 deg out of phase), analytic qdd.  The swing leg is airborne -> qfrc_constraint
~ 0, so tau_clean ~ swing-leg inertia + gravity (the term the massless-leg SRB drops).

Run on the SERVER under .venv_mjlab (casadi + mujoco 3.5.0).
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

import mujoco

SCENE = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"

LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH = 0, 3, 4
RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH = 6, 9, 10

DEFAULT_Q = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)


def setup():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    qpos_adr, dof_adr = [], []
    for act_id in range(model.nu):
        jid = model.actuator_trnid[act_id, 0]
        name = model.joint(jid).name
        if "hand" in name:
            continue
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])
    qpos_adr = np.asarray(qpos_adr, int)
    dof_adr = np.asarray(dof_adr, int)
    return model, data, qpos_adr, dof_adr


def main(v=0.5, T=0.5, A_h=0.30, A_k=0.50, n=201):
    model, data, qpos_adr, dof_adr = setup()
    w = 2 * np.pi / T

    def gait(t):
        phi = w * t
        q = DEFAULT_Q.copy()
        qd = np.zeros_like(q)
        qdd = np.zeros_like(q)
        q[LEFT_HIP_PITCH] += A_h * np.sin(phi)
        qd[LEFT_HIP_PITCH] = A_h * w * np.cos(phi)
        qdd[LEFT_HIP_PITCH] = -A_h * w * w * np.sin(phi)
        q[LEFT_KNEE] += A_k * (1 - np.cos(phi)) / 2.0
        qd[LEFT_KNEE] = A_k * w * np.sin(phi) / 2.0
        qdd[LEFT_KNEE] = A_k * w * w * np.cos(phi) / 2.0
        q[RIGHT_HIP_PITCH] += A_h * np.sin(phi + np.pi)
        qd[RIGHT_HIP_PITCH] = A_h * w * np.cos(phi + np.pi)
        qdd[RIGHT_HIP_PITCH] = -A_h * w * w * np.sin(phi + np.pi)
        q[RIGHT_KNEE] += A_k * (1 - np.cos(phi + np.pi)) / 2.0
        qd[RIGHT_KNEE] = A_k * w * np.sin(phi + np.pi) / 2.0
        qdd[RIGHT_KNEE] = A_k * w * w * np.cos(phi + np.pi) / 2.0
        return q, qd, qdd

    clean_h, clean_k, clean_a = [], [], []
    inv_h, inv_k = [], []
    con_h, con_k = [], []
    ts = np.linspace(0, T, n, endpoint=False)
    for t in ts:
        q, qd, qdd = gait(t)
        data.qpos[0] = v * t
        data.qpos[1] = 0.0
        data.qpos[2] = 0.76
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[qpos_adr] = q
        data.qvel[:] = 0.0
        data.qvel[0] = v
        data.qvel[dof_adr] = qd
        mujoco.mj_forward(model, data)  # computes qfrc_bias (C qd + g) and qfrc_constraint
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)
        qacc_full = np.zeros(model.nv)
        qacc_full[dof_adr] = qdd
        unconstrained = M @ qacc_full + data.qfrc_bias       # M qdd + C qd + g
        con = data.qfrc_constraint[dof_adr].copy()
        clean = unconstrained[dof_adr] - con                 # actuator torque
        inv_h.append(unconstrained[LEFT_HIP_PITCH])
        inv_k.append(unconstrained[LEFT_KNEE])
        con_h.append(con[LEFT_HIP_PITCH]); con_k.append(con[LEFT_KNEE])
        clean_h.append(clean[LEFT_HIP_PITCH])
        clean_k.append(clean[LEFT_KNEE])
        clean_a.append(clean[LEFT_ANKLE_PITCH])
    clean_h = np.array(clean_h); clean_k = np.array(clean_k); clean_a = np.array(clean_a)
    inv_h = np.array(inv_h); inv_k = np.array(inv_k)
    con_h = np.array(con_h); con_k = np.array(con_k)

    def rng(a):
        return f"[{a.min():+.1f}, {a.max():+.1f}]"

    print(f"=== TO08 kinematic-gait full-ID (v={v} T={T} A_h={A_h} A_k={A_k}, {n} pts) ===")
    print(f"{'joint(L)':>10} | {'M qdd+bias':>12} | {'qfrc_constraint':>16} | {'tau_clean':>11}")
    print("-" * 58)
    print(f"{'hip_pitch':>10} | {rng(inv_h):>12} | {rng(con_h):>16} | {rng(clean_h):>11}")
    print(f"{'knee':>10} | {rng(inv_k):>12} | {rng(con_k):>16} | {rng(clean_k):>11}")
    print(f"{'ankle':>10} | {'(no qdd)':>12} | {'-':>16} | {rng(clean_a):>11}")
    print()
    print("execution envelope: hip +-88, knee +-139, ankle +-50 N*m")
    print("reading: tau_clean ~ swing-leg inertia + gravity (contact ~0 for airborne leg).")
    print("         this is the torque the massless-leg SRB DROPS entirely.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--A-h", type=float, default=0.30)
    ap.add_argument("--A-k", type=float, default=0.50)
    ap.add_argument("--n", type=int, default=201)
    a = ap.parse_args()
    main(a.v, a.T, a.A_h, a.A_k, a.n)
