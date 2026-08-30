"""Planar (sagittal) 5-link biped model for the G1 + Lagrangian dynamics (TO12).

Builds the multi-body model the SRB was missing: a body (pelvis+torso+waist+arms) plus
two 2-link legs (thigh + shin) with the REAL G1 masses and lengths, then derives the
equations of motion M(q) qdd + C(q,qd) qd + g(q) = tau symbolically in CasADi.

This is the FOUNDATION for the full-body NMP: unlike the SRB (point mass + massless leg),
the mass matrix M(q) here includes the leg inertia, and g(q) includes the leg gravity --
the two things TO06/TO07 showed were missing.

Model (G1, sagittal plane; point masses at segment midpoints, body as a point mass with
a pitch inertia):
    q = [x, z, theta, hipL, kneeL, hipR, kneeR]   (7 DOF)
    (x,z) = hip reference point; theta = body pitch (from vertical, +forward);
    hipL/hipR = thigh angle from vertical (+forward); kneeL/kneeR = knee flexion (0=straight).

Segments (masses from scene_43dof.xml, summed over the leg chain):
    body  m_b  = pelvis+torso+waist+arms ~ 20.7 kg, pitch inertia I_b
    thigh m_t  = hip_pitch_link+hip_roll_link+hip_yaw_link = 4.572 kg, L_t = 0.3406 m
    shin  m_s  = knee_link+ankle_*_links = 2.614 kg, L_s = 0.30 m
    (total ~35 kg, matching the G1.)

Run on the SERVER under .venv_mjlab (casadi).
"""
from __future__ import annotations

import numpy as np
import casadi as ca

# ---- model parameters (G1, sagittal) ----
G = 9.81
M_B = 20.7      # body (pelvis 3.813 + torso 9.598 + waist ~0.29 + arms ~7)
I_B = 0.30      # body pitch inertia about the hip, kg*m^2 (torso diag ~0.12 + arms, rough)
H_B = 0.10      # body COM height above the hip reference, m
M_T = 4.572     # thigh (hip_pitch_link 1.35 + hip_roll 1.52 + hip_yaw 1.702)
L_T = 0.3406    # thigh length, m
M_S = 2.614     # shin (knee_link 1.932 + ankle 0.074 + foot 0.608)
L_S = 0.30      # shin length, m


def build_dynamics():
    q = ca.MX.sym("q", 7)
    qd = ca.MX.sym("qd", 7)
    x, z, th, hL, kL, hR, kR = q[0], q[1], q[2], q[3], q[4], q[5], q[6]

    # ---- COM positions (world) ----
    # body COM is ABOVE the hip reference (z + H_B cos th); the LEGS hang DOWN
    # (z - L cos(...)).  hip/knee angles are measured from the vertical-down leg.
    pb = ca.vertcat(x + H_B * ca.sin(th), z + H_B * ca.cos(th))
    # left thigh COM
    ptL = ca.vertcat(x + (L_T / 2) * ca.sin(th + hL), z - (L_T / 2) * ca.cos(th + hL))
    # left shin COM
    psL = ca.vertcat(x + L_T * ca.sin(th + hL) + (L_S / 2) * ca.sin(th + hL - kL),
                     z - L_T * ca.cos(th + hL) - (L_S / 2) * ca.cos(th + hL - kL))
    ptR = ca.vertcat(x + (L_T / 2) * ca.sin(th + hR), z - (L_T / 2) * ca.cos(th + hR))
    psR = ca.vertcat(x + L_T * ca.sin(th + hR) + (L_S / 2) * ca.sin(th + hR - kR),
                     z - L_T * ca.cos(th + hR) - (L_S / 2) * ca.cos(th + hR - kR))

    # ---- COM velocities (v = J(q) qd) ----
    vb = ca.jacobian(pb, q) @ qd
    vtL = ca.jacobian(ptL, q) @ qd
    vsL = ca.jacobian(psL, q) @ qd
    vtR = ca.jacobian(ptR, q) @ qd
    vsR = ca.jacobian(psR, q) @ qd

    # ---- kinetic + potential energy ----
    T = (0.5 * M_B * ca.dot(vb, vb) + 0.5 * I_B * qd[2]**2
         + 0.5 * M_T * (ca.dot(vtL, vtL) + ca.dot(vtR, vtR))
         + 0.5 * M_S * (ca.dot(vsL, vsL) + ca.dot(vsR, vsR)))
    V = G * (M_B * pb[1] + M_T * (ptL[1] + ptR[1]) + M_S * (psL[1] + psR[1]))

    # ---- mass matrix M(q) = d2T / dqd2 ----
    M = ca.jacobian(ca.jacobian(T, qd), qd)  # 7x7
    # ---- bias C qd + g (Coriolis + gravity) ----
    # Lagrange: d/dt(dT/dqd) - dT/dq + dV/dq = tau ; at qdd=0, d/dt(dT/dqd) = d(M qd)/dq * qd
    p = ca.jacobian(T, qd).T  # generalized momentum = M qd
    dp_dq = ca.jacobian(p, q)
    bias = dp_dq @ qd - ca.jacobian(T, q).T + ca.jacobian(V, q).T

    M_fn = ca.Function("M", [q], [M])
    bias_fn = ca.Function("bias", [q, qd], [bias])
    return M_fn, bias_fn


def verify(M_fn, bias_fn):
    q_stand = np.array([0.0, 0.64, 0.0, 0.0, 0.0, 0.0, 0.0])  # legs straight down
    M = np.array(M_fn(q_stand))
    print("=== TO12 planar 5-link model verification ===")
    print(f"total mass = {M_B + 2*M_T + 2*M_S:.2f} kg (G1 ~35 kg)")
    print(f"M(q_stand) diag = {np.round(np.diag(M), 4)}")
    print(f"M symmetric? {np.allclose(M, M.T, atol=1e-8)}")
    eig = np.linalg.eigvalsh(M)
    print(f"M eigenvalues min={eig.min():.4f} (should be >0)")
    # gravity torque at standing pose (bias at qd=0)
    g_stand = np.array(bias_fn(q_stand, np.zeros(7))).ravel()
    print(f"gravity torque g(q_stand) = {np.round(g_stand, 4)} N*m (should be ~0 for straight legs)")
    # also check a bent-leg pose has a nonzero gravity torque (leg mass pulls down)
    q_bent = np.array([0.0, 0.64, 0.1, 0.3, 0.5, -0.3, 0.5])
    g_bent = np.array(bias_fn(q_bent, np.zeros(7))).ravel()
    print(f"gravity torque g(q_bent)   = {np.round(g_bent, 4)} N*m (nonzero, leg mass present)")


if __name__ == "__main__":
    M_fn, bias_fn = build_dynamics()
    verify(M_fn, bias_fn)
