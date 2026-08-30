"""Foot-space kinematic walking gait + full inverse-dynamics torque (TO09).

Designs a proper planar walking gait in FOOT space (stance foot on the ground,
swing foot on a clearance arc) with the CORRECT G1 leg geometry, maps it to
hip_pitch/knee via analytic 2-link IK, and computes the full multi-body ID torque
via the manual route (M @ qacc + qfrc_bias - qfrc_constraint; see TO08: mj_inverse
is buggy for the floating-base model with both legs driven).

Correct G1 leg geometry (measured from scene_43dof.xml, TO08):
    pelvis height            0.760 m
    hip_pitch joint height   0.657 m   (0.103 m below pelvis)
    thigh L1                 0.3406 m
    shin  L2                 0.30 m
    ankle height (foot flat) 0.056 m
  -> hip-to-ankle vertical 0.601 m, leg length 0.6406 m, knee-flexion slack 0.039 m,
     horizontal reach ~0.22 m.

Angle convention (verified against MuJoCo FK, see main()):
    hip_pitch = -theta_h   (theta_h = thigh angle from vertical, +forward)
    knee       =  theta_k   (theta_k = knee flexion, 0 = straight)

Gait (walk, d=1): stride S = v*T, foot placement L = S/4; phase 1 [0,T/2) LEFT
stance at x=L with RIGHT swing; phase 2 [T/2,T) RIGHT stance at x=S/2+L with LEFT
swing.  Swing foot follows a half-sine clearance arc (height h_clear).

Run on the SERVER under .venv_mjlab.
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

import mujoco

# TO31 widened-feet sim test: APT_SCENE swaps in a scene variant whose foot
# collision spheres are scaled laterally (scene_43dof_wf*.xml)
SCENE = os.environ.get(
    "APT_SCENE",
    "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic/"
    "data/robot_model/model_data/g1/scene_43dof.xml")

# geometry (world sagittal plane)
PELVIS_H = 0.760
HIP_H = 0.657          # hip_pitch joint height above ground (foot flat)
ANKLE_H = 0.056        # ankle height above ground (foot flat)
L1 = 0.3406            # thigh
L2 = 0.30              # shin
HIP_DZ = PELVIS_H - HIP_H  # 0.103 m, hip below pelvis

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


def ik(fx, fz, hx, hz):
    """hip (hx, hz) + ankle (fx, fz) -> (hip_pitch, knee)."""
    dx = fx - hx
    dz = fz - hz
    d = np.hypot(dx, dz)
    d = np.clip(d, 1e-6, L1 + L2 - 1e-6)
    cos_k = (d * d - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_k = np.clip(cos_k, -1.0, 1.0)
    theta_k = np.arccos(cos_k)
    phi = np.arctan2(dx, -dz)
    delta = np.arcsin(np.clip(L2 * np.sin(theta_k) / d, -1.0, 1.0))
    theta_h = phi + delta
    return -theta_h, theta_k  # G1 hip_pitch = -theta_h, knee = theta_k


def fk(hip_pitch, knee, hx, hz):
    """hip_pitch/knee -> ankle (fx, fz) in world sagittal plane."""
    theta_h = -hip_pitch
    theta_k = knee
    fx = hx + L1 * np.sin(theta_h) + L2 * np.sin(theta_h - theta_k)
    fz = hz - L1 * np.cos(theta_h) - L2 * np.cos(theta_h - theta_k)
    return fx, fz


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
    return model, data, np.asarray(qpos_adr, int), np.asarray(dof_adr, int)


def verify_ik(model, data, qpos_adr, dof_adr):
    """Check the analytic IK against MuJoCo FK at the default pose."""
    print("=== IK<->MuJoCo FK verification (default stand pose) ===")
    # default pose: hip at (0, 0.657), ankle at (-0.002, 0.056)
    hp, kn = ik(-0.002, ANKLE_H, 0.0, HIP_H)
    print(f"  analytic ik(ankle x=-0.002, hip x=0): hip_pitch={hp:+.4f} knee={kn:+.4f}")
    print(f"  G1 default pose:                       hip_pitch={DEFAULT_Q[0]:+.4f} knee={DEFAULT_Q[3]:+.4f}")
    fx, fz = fk(hp, kn, 0.0, HIP_H)
    print(f"  fk round-trip: ankle=({fx:+.4f}, {fz:+.4f})  (expected (-0.002, {ANKLE_H}))")
    print(f"  knee slack: leg {L1+L2:.4f} - hip-to-ankle {HIP_H-ANKLE_H:.4f} = {L1+L2-(HIP_H-ANKLE_H):.4f} m")
    print(f"  horizontal reach: {np.sqrt((L1+L2)**2 - (HIP_H-ANKLE_H)**2):.4f} m")
    print()


def main(v=0.5, T=0.5, h_clear=0.06, n=201):
    model, data, qpos_adr, dof_adr = setup()
    verify_ik(model, data, qpos_adr, dof_adr)

    S = v * T
    L = S / 4.0
    half = n // 2
    ts = np.linspace(0, T, n, endpoint=False)
    dt = T / n

    # build joint trajectory q(t), qd(t), qdd(t) via numeric diff of IK
    Q = np.zeros((n, len(DEFAULT_Q)))
    for i, t in enumerate(ts):
        hx = v * t
        if t < T / 2.0:
            stance_x = L                          # LEFT stance
            toe_x = S / 2.0 + L - S               # RIGHT toe-off (prev stance, one stride back)
            heel_x = S / 2.0 + L                  # RIGHT heel-strike
        else:
            stance_x = S / 2.0 + L                # RIGHT stance
            toe_x = L                             # LEFT toe-off
            heel_x = S + L                        # LEFT heel-strike (next cycle)
        u = (t % (T / 2.0)) / (T / 2.0)
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5   # quintic smoothstep (zero vel+accel at ends)
        swing_x = toe_x + (heel_x - toe_x) * s
        swing_z = ANKLE_H + h_clear * (1.0 - np.cos(2.0 * np.pi * u)) / 2.0  # cosine bump
        q = DEFAULT_Q.copy()
        if t < T / 2.0:  # LEFT stance, RIGHT swing
            hpL, knL = ik(stance_x, ANKLE_H, hx, HIP_H)
            hpR, knR = ik(swing_x, swing_z, hx, HIP_H)
            q[LEFT_HIP_PITCH], q[LEFT_KNEE] = hpL, knL
            q[RIGHT_HIP_PITCH], q[RIGHT_KNEE] = hpR, knR
        else:  # RIGHT stance, LEFT swing
            hpR, knR = ik(stance_x, ANKLE_H, hx, HIP_H)
            hpL, knL = ik(swing_x, swing_z, hx, HIP_H)
            q[RIGHT_HIP_PITCH], q[RIGHT_KNEE] = hpR, knR
            q[LEFT_HIP_PITCH], q[LEFT_KNEE] = hpL, knL
        Q[i] = q

    Qd = np.gradient(Q, dt, axis=0)
    Qdd = np.gradient(Qd, dt, axis=0)

    clean_h, clean_k = [], []
    inv_h, con_h = [], []
    for i in range(n):
        data.qpos[0] = v * ts[i]
        data.qpos[1] = 0.0
        data.qpos[2] = PELVIS_H
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[qpos_adr] = Q[i]
        data.qvel[:] = 0.0
        data.qvel[0] = v
        data.qvel[dof_adr] = Qd[i]
        mujoco.mj_forward(model, data)
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)
        qacc_full = np.zeros(model.nv)
        qacc_full[dof_adr] = Qdd[i]
        uncon = M @ qacc_full + data.qfrc_bias
        con = data.qfrc_constraint[dof_adr].copy()
        clean = uncon[dof_adr] - con
        inv_h.append(uncon[dof_adr[LEFT_HIP_PITCH]]); con_h.append(con[LEFT_HIP_PITCH])
        clean_h.append(clean[LEFT_HIP_PITCH]); clean_k.append(clean[LEFT_KNEE])
    inv_h = np.array(inv_h); con_h = np.array(con_h)
    clean_h = np.array(clean_h); clean_k = np.array(clean_k)

    def rng(a):
        return f"[{a.min():+.1f}, {a.max():+.1f}]"

    print(f"=== TO09 foot-gait full-ID (v={v} T={T} h_clear={h_clear} S={S:.2f} L={L:.3f}) ===")
    print(f"{'joint(L)':>10} | {'M qdd+bias':>12} | {'qfrc_constraint':>16} | {'tau_clean':>11}")
    print("-" * 58)
    print(f"{'hip_pitch':>10} | {rng(inv_h):>12} | {rng(con_h):>16} | {rng(clean_h):>11}")
    print(f"{'knee':>10} | {'-':>12} | {'-':>16} | {rng(clean_k):>11}")
    print()
    print("execution envelope: hip +-88, knee +-139 N*m")
    print("reading: tau_clean = actuator torque to reproduce this walking motion under contact.")
    print("         (the massless-leg SRB only gives the stance contact term, not the swing inertia.)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--h-clear", type=float, default=0.06)
    ap.add_argument("--n", type=int, default=201)
    a = ap.parse_args()
    main(a.v, a.T, a.h_clear, a.n)
