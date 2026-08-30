"""Planar biped walking NMP via direct collocation (TO13).

Optimizes a periodic walking gait for the 5-link planar biped (planar_biped_model.py)
using direct collocation + IPOPT.  The gait has two single-stance phases (LEFT stance,
then RIGHT stance) with the stance foot pinned (no slip) and the swing foot above the
ground; periodicity with leg-swap closes the loop.  The contact force is a decision
variable (with a friction cone).  The output is a self-consistent (q, qd, tau, f) over
one cycle -- the dynamically-consistent motion TO11 showed was missing.

State q = [x, z, theta, hipL, kneeL, hipR, kneeR]; control = [tau_hipL, tau_kneeL,
tau_hipR, tau_kneeR] + [f_sx, f_sz] (stance ground reaction).

Run on the SERVER under .venv_mjlab (casadi).
"""
from __future__ import annotations

import numpy as np
import casadi as ca

from planar_biped_model import G, M_B, I_B, H_B, M_T, L_T, M_S, L_S, build_dynamics


def foot_pos(q, side):
    """side=0 left, side=1 right -> (fx, fz) of the ankle in world sagittal plane."""
    x, z, th = q[0], q[1], q[2]
    hip = q[3 + 2 * side]
    knee = q[4 + 2 * side]
    return ca.vertcat(x + L_T * ca.sin(th + hip) + L_S * ca.sin(th + hip - knee),
                      z - L_T * ca.cos(th + hip) - L_S * ca.cos(th + hip - knee))


def main(T=0.5, S=0.25, N=20, mu=0.7, h_clear=0.04, w_tau=1e-2, w_rate=1e-2, v_target=0.5, h_ref=0.64):
    M_fn, bias_fn = build_dynamics()
    dt = T / N
    half = N // 2

    opti = ca.Opti()
    Q = opti.variable(7, N + 1)
    Qd = opti.variable(7, N + 1)
    TAU = opti.variable(4, N)
    F = opti.variable(2, N)  # stance contact force (sx, sz)

    # ---- foot jacobians as Functions of a symbolic q (can't jacobian an Opti slice) ----
    q_sym = ca.MX.sym("q", 7)
    J_fn = {}
    for side in [0, 1]:
        p_sym = foot_pos(q_sym, side)
        J_fn[side] = ca.Function(f"J_{side}", [q_sym], [ca.jacobian(p_sym, q_sym)])

    # ---- B matrix: leg torques act on q[3:7] ----
    B = ca.vertcat(ca.DM.zeros(3, 4), ca.DM.eye(4))

    cost = 0.0
    for k in range(N):
        side = 0 if k < half else 1  # 0 = LEFT stance, 1 = RIGHT stance
        # stance foot (the one on the ground), swing foot
        p_s = foot_pos(Q[:, k], side)
        J_s = J_fn[side](Q[:, k])
        p_sw = foot_pos(Q[:, k], 1 - side)
        # fixed stance x position: phase1 left foot at 0, phase2 right foot at S/2
        stance_x = 0.0 if side == 0 else S / 2.0

        M = M_fn(Q[:, k])
        bias = bias_fn(Q[:, k], Qd[:, k])
        qdd = ca.solve(M, B @ TAU[:, k] + J_s.T @ F[:, k] - bias)

        # dynamics residual (trapezoidal collocation)
        # q_{k+1} = q_k + dt/2 (qd_k + qd_{k+1})
        opti.subject_to(Q[:, k + 1] - Q[:, k] - (dt / 2) * (Qd[:, k] + Qd[:, k + 1]) == 0)
        # qd_{k+1} = qd_k + dt/2 (qdd_k + qdd_{k+1})
        M1 = M_fn(Q[:, k + 1])
        bias1 = bias_fn(Q[:, k + 1], Qd[:, k + 1])
        side1 = 0 if (k + 1) < half else 1
        J_s1 = J_fn[side1](Q[:, k + 1])
        qdd1 = ca.solve(M1, B @ TAU[:, min(k + 1, N - 1)] + J_s1.T @ F[:, min(k + 1, N - 1)] - bias1)
        opti.subject_to(Qd[:, k + 1] - Qd[:, k] - (dt / 2) * (qdd + qdd1) == 0)

        # stance foot: on ground, fixed x (no-slip velocity constraint is RELAXED here:
        # it made the phase-switch transition infeasible; re-add once a feasible gait exists)
        opti.subject_to(p_s[1] == 0.0)
        opti.subject_to(p_s[0] - stance_x == 0.0)
        # friction cone
        opti.subject_to(F[0, k] - mu * F[1, k] <= 0.0)
        opti.subject_to(-F[0, k] - mu * F[1, k] <= 0.0)
        opti.subject_to(F[1, k] >= 1e-3)
        # swing foot clearance (lower + upper bound to prevent a huge high kick)
        opti.subject_to(p_sw[1] >= h_clear)
        opti.subject_to(p_sw[1] <= 0.2)
        # keep the body UPRIGHT (prevent the degenerate "lying-down" gait)
        opti.subject_to(Q[2, k] >= -0.3)
        opti.subject_to(Q[2, k] <= 0.3)
        # torque limits (relaxed to 200 N*m for a first feasible gait; tighten later)
        opti.subject_to(TAU[0, k] >= -200); opti.subject_to(TAU[0, k] <= 200)
        opti.subject_to(TAU[1, k] >= -200); opti.subject_to(TAU[1, k] <= 200)
        opti.subject_to(TAU[2, k] >= -200); opti.subject_to(TAU[2, k] <= 200)
        opti.subject_to(TAU[3, k] >= -200); opti.subject_to(TAU[3, k] <= 200)

        cost += w_tau * ca.sumsqr(TAU[:, k])
        if k > 0:
            cost += w_rate * ca.sumsqr(TAU[:, k] - TAU[:, k - 1])  # smooth torque over time
        cost += (Qd[0, k] - v_target) ** 2
        cost += 2.0 * (Q[1, k] - h_ref) ** 2
        cost += 100.0 * Q[2, k] ** 2

    # ---- periodicity (leg swap + stride advance) ----
    # q_N = [q0.x + S, q0.z, q0.th, q0.hipR, q0.kneeR, q0.hipL, q0.kneeL]
    swap = ca.DM([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ])
    advance = ca.DM([S, 0, 0, 0, 0, 0, 0])
    opti.subject_to(Q[:, N] - (swap @ Q[:, 0] + advance) == 0)
    opti.subject_to(Qd[:, N] - swap @ Qd[:, 0] == 0)

    opti.minimize(cost)

    # ---- initial guess: standing pose with a slight leg split ----
    t = np.linspace(0, T, N + 1)
    q0 = np.zeros((7, N + 1))
    for k in range(N + 1):
        q0[:, k] = [S * k / N, h_ref, 0.0, 0.1, 0.05, -0.1, 0.05]
    opti.set_initial(Q, q0)
    opti.set_initial(Qd, np.zeros((7, N + 1)))
    opti.set_initial(TAU, np.zeros((4, N)))
    opti.set_initial(F, np.tile([0.0, M_B * G / 2], (N, 1)).T)

    opti.solver("ipopt", {"ipopt.print_level": 3, "print_time": 0, "ipopt.max_iter": 2000})
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"[NMP] solve failed: {e}")
        return None

    Qs = sol.value(Q)
    Qds = sol.value(Qd)
    TAUs = sol.value(TAU)
    Fs = sol.value(F)
    print(f"\n=== TO13 planar biped NMP (T={T} S={S} N={N}) ===")
    print(f"cost = {sol.value(cost):.4f}")
    print(f"tau ranges: hipL[{TAUs[0].min():.1f},{TAUs[0].max():.1f}] "
          f"kneeL[{TAUs[1].min():.1f},{TAUs[1].max():.1f}] "
          f"hipR[{TAUs[2].min():.1f},{TAUs[2].max():.1f}] kneeR[{TAUs[3].min():.1f},{TAUs[3].max():.1f}] N*m")
    print(f"z range [{Qs[1].min():.2f},{Qs[1].max():.2f}]  theta range [{Qs[2].min():.3f},{Qs[2].max():.3f}]")
    print(f"contact fz range [{Fs[1].min():.1f},{Fs[1].max():.1f}] N (fz>0 = push up)")
    # foot z range (should be ~0 for stance, >clearance for swing)
    def foot_z(q, side):
        hip = q[3 + 2 * side]; knee = q[4 + 2 * side]
        return q[1] - L_T * np.cos(q[2] + hip) - L_S * np.cos(q[2] + hip - knee)
    pLz = [foot_z(Qs[:, k], 0) for k in range(N + 1)]
    pRz = [foot_z(Qs[:, k], 1) for k in range(N + 1)]
    print(f"foot L z [{min(pLz):.3f},{max(pLz):.3f}]  foot R z [{min(pRz):.3f},{max(pRz):.3f}]")
    np.savez("/home/cvgluser/ros2_data/apt_g1/outputs/nmp_biped_gait.npz",
             q=Qs, qd=Qds, tau=TAUs, f=Fs, T=T, S=S)
    print("saved outputs/nmp_biped_gait.npz")
    return dict(q=Qs, qd=Qds, tau=TAUs, f=Fs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--S", type=float, default=0.25)
    ap.add_argument("--T", type=float, default=0.5)
    a = ap.parse_args()
    main(T=a.T, S=a.S, N=a.N)
