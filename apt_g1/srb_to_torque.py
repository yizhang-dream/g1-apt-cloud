"""G1 SRB TO -> 2-link leg IK -> joint torque -> (state, torque) data.

Extends srb_to.py: rolls out the periodic SRB solution (CoM + stance-foot
GRF), then for each stance step maps the foot position (via 2-link IK) to
hip/knee angles and computes the joint torques via the static moment arm
(tau = J^T f, i.e. force x lever arm about each joint).

The 2-link leg: thigh L1 + shin L2, hip at the CoM, foot on the ground during
stance.  Angles: theta_h (thigh from vertical, +forward), theta_k (knee
flexion, 0 = straight).  Torques are the pitch-axis torques of the stance leg.

Gait is (T, d): cycle time + duty factor (walk d=1, run d<1 with flight).
Torque is only defined during stance (foot on ground); flight has no GRF.
"""

from __future__ import annotations

import numpy as np

from srb_to import M, I, G, N, solve, grf_amp

L1 = 0.34          # thigh length, m (effective: hip->sole reach ~0.68m)
L2 = 0.34          # shin length, m
L_HEEL = 0.08      # ankle lever arm: CoP behind ankle at heel strike, m
L_TOE = 0.14       # ankle lever arm: CoP ahead of ankle at toe-off, m (foot ~0.22m)


def fk(theta_h, theta_k, hx, hz):
    """Forward kinematics: hip+knee angles -> foot (x, z)."""
    x = hx + L1 * np.sin(theta_h) + L2 * np.sin(theta_h - theta_k)
    z = hz - L1 * np.cos(theta_h) - L2 * np.cos(theta_h - theta_k)
    return x, z


def ik(fx, fz, hx, hz):
    """Inverse kinematics: foot + hip -> (theta_h, theta_k)."""
    dx, dz = fx - hx, fz - hz
    d = np.hypot(dx, dz)
    d = np.clip(d, 1e-6, L1 + L2 - 1e-6)
    cos_k = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_k = np.clip(cos_k, -1.0, 1.0)
    theta_k = np.arccos(cos_k)                      # knee flexion (0=straight)
    phi = np.arctan2(dx, -dz)                        # leg vector from vertical
    delta = np.arcsin(np.clip(L2 * np.sin(theta_k) / d, -1.0, 1.0))
    theta_h = phi + delta                           # thigh FORWARD of the leg vector
    return theta_h, theta_k


def knee_pos(theta_h, hx, hz):
    return hx + L1 * np.sin(theta_h), hz - L1 * np.cos(theta_h)


def roll_out(v, T=0.5, d=1.0):
    """Solve the SRB TO and roll out one full cycle -> per-step (com, foot, Fz)."""
    r = solve(v, T=T, d=d)
    z0, zd0, th0, thd0 = r["z0"], r["zd0"], r["th0"], r["thd0"]
    S = v * T
    L = S * d / 4.0
    A = grf_amp(d)
    state = np.array([0.0, z0, th0, zd0, thd0])     # x, z, th, zd, thd
    rows = []
    hs = d * T / 2.0 / N
    hf = (1.0 - d) * T / 2.0 / N
    for foot_x, t0 in [(L, 0.0), (S / 2.0 + L, T / 2.0)]:
        for k in range(N + 1):
            t = t0 + k * hs
            x, z, th, zd, thd = state
            Fz = A * np.sin(2 * np.pi * (t - t0) / (d * T))
            rows.append((t, x, z, th, foot_x, Fz))
            zdd = Fz / M - G
            thdd = (foot_x - x) * Fz / I
            state = state + hs * np.array([v, zd, thd, zdd, thdd])
        if d < 1.0:
            for k in range(N + 1):
                x, z, th, zd, thd = state
                state = state + hf * np.array([v, zd, thd, -G, 0.0])
    return rows


def main():
    v = 1.0
    for name, T, d in [("walk", 0.5, 1.0), ("run", 0.5, 0.5)]:
        rows = roll_out(v, T=T, d=d)
        print(f"[{name}] rollout: {len(rows)} stance steps over one cycle (v={v} m/s)")

        # verify IK<->FK round-trip on a few sample foot positions (foot on ground z=0)
        print("  IK<->FK round-trip check (foot on ground, hip at CoM height):")
        for (fx, fz) in [(0.0, 0.0), (0.2, 0.0), (0.15, 0.0), (-0.1, 0.0)]:
            th_h, th_k = ik(fx, fz, 0.0, 0.65)
            fx2, fz2 = fk(th_h, th_k, 0.0, 0.65)
            err = np.hypot(fx2 - fx, fz2 - fz)
            print(f"    foot=({fx:+.2f},{fz:+.2f}) -> th_h={th_h:+.3f} th_k={th_k:+.3f} "
                  f"-> recon=({fx2:+.3f},{fz2:+.3f}) err={err:.2e}")

        # per-step joint angle + torque (stance leg), hip at the CoM
        angs, taus, fzs = [], [], []
        for (t, x, z, th, foot_x, Fz) in rows:
            th_h, th_k = ik(foot_x, 0.0, x, z)
            kx, kz = knee_pos(th_h, x, z)
            tau_h = Fz * (foot_x - x)                       # moment about hip (Fz only, Fx=0)
            tau_k = Fz * (foot_x - kx)                      # moment about knee
            angs.append((th_h, th_k))
            taus.append((tau_h, tau_k))
            fzs.append(Fz)

        angs = np.array(angs); taus = np.array(taus); fzs = np.array(fzs)
        print(f"  joint angles: hip [{angs[:,0].min():+.3f}, {angs[:,0].max():+.3f}] rad, "
              f"knee [{angs[:,1].min():+.3f}, {angs[:,1].max():+.3f}] rad")
        print(f"  joint torques: hip [{taus[:,0].min():+.1f}, {taus[:,0].max():+.1f}] Nm, "
              f"knee [{taus[:,1].min():+.1f}, {taus[:,1].max():+.1f}] Nm")
        print(f"  GRF: min {fzs.min():.1f} N, max {fzs.max():.1f} N (never negative = no pull)")


if __name__ == "__main__":
    main()
