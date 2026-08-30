"""TO07 probe: the CORRECT self-consistent actuator torque  tau_clean.

MuJoCo equation of motion (no external applied forces):
    M qdd + C qd + g(q)  =  tau_act  +  J^T lambda        (lambda = contact force)
hence the actuator torque that reproduces a given motion under ground contact is
    tau_act  =  (M qdd + C qd + g)  -  J^T lambda
             =  qfrc_inverse        -  qfrc_constraint

where
  * qfrc_inverse    = mj_inverse(model, data)   (free-space ID, no contact)
  * qfrc_constraint = mj_forward(model, data)   (contact forces in joint space)

This reconciles the two historical negative results:
  * SRB (TO06)  fed  tau = J^T f = qfrc_constraint only   -> missing gravity/inertia
        -> robot collapsed (torque only the "ground" part).
  * A-ID (dir A) fed  tau = qfrc_inverse (free-space) only -> missing the -contact
        -> robot "stood but did not advance" (feedforward cancels gravity+inertia,
           leaving no forward drive; the -J^T lambda term is what actually drives).

This script:
  1. STATIC sanity check: G1 in the SONIC default stand pose -> qfrc_inverse, qfrc_constraint,
     tau_clean.  tau_clean should be ~small (the ground holds the robot up) while the
     other two are ~ body_weight x lever (large, opposite).
  2. SRBD stance-walk: track the SRBD-derived sagittal joint trajectory (swing leg held
     fixed to dodge the IK singularity) -> report hip/knee tau_clean magnitude.

Run on the SERVER under .venv_mjlab (casadi + mujoco 3.5.0); NOT the isaac venv.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

import mujoco

from srb_to import solve, grf_amp, M, I, G, N
from srb_to_torque import ik, knee_pos

SCENE = "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"

# lower-body DOF indices in MuJoCo body order (XML actuator order, hands skipped)
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


def setup_model():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    act_ids, qpos_adr, dof_adr = [], [], []
    for act_id in range(model.nu):
        jid = model.actuator_trnid[act_id, 0]
        name = model.joint(jid).name
        if "hand" in name:
            continue
        act_ids.append(act_id)
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])
    act_ids = np.asarray(act_ids, int)
    qpos_adr = np.asarray(qpos_adr, int)
    dof_adr = np.asarray(dof_adr, int)
    return model, data, act_ids, qpos_adr, dof_adr


def static_check():
    model, data, act_ids, qpos_adr, dof_adr = setup_model()
    # stand pose: pelvis at 0.76m, default joints, qd=qdd=0
    q = DEFAULT_Q.copy()
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = q
    data.qvel[:] = 0.0
    mujoco.mj_inverse(model, data)  # qacc=0 -> qfrc_inverse = g(q)
    qfrc_inv = data.qfrc_inverse[dof_adr].copy()
    mujoco.mj_forward(model, data)  # resolves contact -> qfrc_constraint = J^T lambda
    qfrc_con = data.qfrc_constraint[dof_adr].copy()
    tau_clean = qfrc_inv - qfrc_con

    print("=== TO07 STATIC sanity check (default stand pose) ===")
    print(f"{'joint':>12} | {'qfrc_inverse (g)':>18} | {'qfrc_constraint (J^T l)':>22} | {'tau_clean':>14}")
    print("-" * 72)
    for name, idx in [("hip_pitch_L", LEFT_HIP_PITCH), ("knee_L", LEFT_KNEE),
                      ("ankle_L", LEFT_ANKLE_PITCH), ("hip_pitch_R", RIGHT_HIP_PITCH)]:
        print(f"{name:>12} | {qfrc_inv[idx]:>+18.1f} | {qfrc_con[idx]:>+22.1f} | {tau_clean[idx]:>+14.1f}")
    print()
    print("reading: tau_clean should be ~0 (ground holds the robot up);")
    print("         qfrc_inverse and qfrc_constraint should be large and ~opposite.")
    print()


def srbd_walk_check(v=1.0, T=0.5, d=1.0):
    model, data, act_ids, qpos_adr, dof_adr = setup_model()
    r = solve(v, T=T, d=d)
    z0, zd0, th0, thd0 = r["z0"], r["zd0"], r["th0"], r["thd0"]
    S = v * T
    L = S * d / 4.0
    A = grf_amp(d)
    state = np.array([0.0, z0, th0, zd0, thd0])  # x,z,th,zd,thd
    hs = d * T / 2.0 / N
    rows = []
    for foot_x, t0 in [(L, 0.0), (S / 2.0 + L, T / 2.0)]:
        for k in range(N + 1):
            t = t0 + k * hs
            x, z, th, zd, thd = state
            Fz = A * np.sin(2 * np.pi * (t - t0) / (d * T))
            rows.append((t, x, z, th, foot_x, Fz))
            zdd = Fz / M - G
            thdd = (foot_x - x) * Fz / I
            state = state + hs * np.array([v, zd, thd, zdd, thdd])

    # Build the sagittal joint trajectory: stance leg follows IK, swing leg held at default.
    # Only the LEFT leg is made the stance leg in the first half-cycle to keep it smooth.
    Q, srb_static_hip, srb_static_knee = [], [], []
    for i, (t, x, z, th, foot_x, Fz) in enumerate(rows[: len(rows) // 2]):
        th_h, th_k = ik(foot_x, 0.0, x, z)
        q = DEFAULT_Q.copy()
        q[LEFT_HIP_PITCH] = th_h
        q[LEFT_KNEE] = th_k
        Q.append(q)
        srb_static_hip.append(Fz * (foot_x - x))
        kx, kz = knee_pos(th_h, x, z)
        srb_static_knee.append(Fz * (foot_x - kx))
    Q = np.asarray(Q)

    dt = hs
    Qd = np.gradient(Q, dt, axis=0)
    Qdd = np.gradient(Qd, dt, axis=0)

    inv_h, inv_k, con_h, con_k, clean_h, clean_k = [], [], [], [], [], []
    for i in range(len(Q)):
        x, z, th = rows[i][1], rows[i][2], rows[i][3]
        data.qpos[0] = x
        data.qpos[1] = 0.0
        data.qpos[2] = z
        data.qpos[3:7] = [np.cos(th / 2.0), 0.0, np.sin(th / 2.0), 0.0]
        data.qpos[qpos_adr] = Q[i]
        data.qvel[:] = 0.0
        data.qvel[dof_adr] = Qd[i]
        data.qacc[:] = 0.0
        data.qacc[dof_adr] = Qdd[i]
        mujoco.mj_inverse(model, data)
        inv = data.qfrc_inverse[dof_adr].copy()
        mujoco.mj_forward(model, data)
        con = data.qfrc_constraint[dof_adr].copy()
        inv_h.append(inv[LEFT_HIP_PITCH]); con_h.append(con[LEFT_HIP_PITCH])
        inv_k.append(inv[LEFT_KNEE]);   con_k.append(con[LEFT_KNEE])
        clean_h.append((inv - con)[LEFT_HIP_PITCH])
        clean_k.append((inv - con)[LEFT_KNEE])
    inv_h = np.array(inv_h); con_h = np.array(con_h); clean_h = np.array(clean_h)
    inv_k = np.array(inv_k); con_k = np.array(con_k); clean_k = np.array(clean_k)
    srb_static_hip = np.array(srb_static_hip); srb_static_knee = np.array(srb_static_knee)

    def rng(a):
        return f"[{a.min():+.1f}, {a.max():+.1f}]"

    print(f"=== TO07 SRBD stance-walk check (v={v} T={T} d={d}) ===")
    print(f"{'joint':>10} | {'SRB static':>12} | {'qfrc_inverse':>13} | {'qfrc_constraint':>17} | {'tau_clean':>12}")
    print("-" * 76)
    print(f"{'hip_pitch':>10} | {rng(srb_static_hip):>12} | {rng(inv_h):>13} | {rng(con_h):>17} | {rng(clean_h):>12}")
    print(f"{'knee':>10} | {rng(srb_static_knee):>12} | {rng(inv_k):>13} | {rng(con_k):>17} | {rng(clean_k):>12}")
    print()
    print("execution envelope: hip +-88, knee +-139, ankle +-50 N*m")
    print("reading: tau_clean = qfrc_inverse - qfrc_constraint is the correct actuator torque.")
    print("         SRB static ~= qfrc_constraint (contact term only).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--d", type=float, default=1.0)
    ap.add_argument("--static-only", action="store_true")
    a = ap.parse_args()
    static_check()
    if not a.static_only:
        srbd_walk_check(a.v, a.T, a.d)
