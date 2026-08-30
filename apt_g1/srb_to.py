"""G1 bipedal SRB trajectory optimization (CasADi) -- APT-RL "Impulse scale-based TO".

2D single rigid body q=(x,z,theta); vertical GRF as a half-sine pulse scaled to
balance weight over the stance phase (momentum conservation, paper Eq.1); shoot
one gait cycle and minimize the periodicity cost (paper Eq.2) over (z0, th0, thd0).

Gait is parameterized by (T, d): T = full gait-cycle time, d = duty factor
(stance fraction of each half-cycle).  d=1 -> walk (no flight); d<1 -> run
(with ballistic flight phase).  Stride S = v*T; the stance foot plants ahead of
the CoM by L = S*d/4 so the pitch moment arm changes sign through stance (foot
ahead -> CoM passes over -> foot behind), which is what makes the pitch periodic.

Horizontal propulsion F_x is NOT modeled (coasting F_x=0): a FIXED F_x sine
broke periodicity; the paper co-optimizes the F_x Bezier profile with the
initial condition (a bigger change, deferred).

Sweeps gaits x speeds and reports, per (gait, speed), the periodic initial
condition and physicality (min/max body height, min/max pitch, min GRF).
"""

from __future__ import annotations

import casadi as ca
import numpy as np

M = 36.165          # total mass, kg
I = 3.981           # pitch inertia about CoM, kg.m^2
G = 9.81            # gravity
LAM1 = 0.5          # Eq.2 pitch-rate periodicity weight
LAM2 = 0.0          # Eq.2 max-pitch regularization (0 for walk)
LAM3 = 1.0          # vertical-velocity periodicity weight (needed for flight gaits)
N = 200             # RK4 steps per phase


def grf_amp(d):
    """Vertical GRF amplitude balancing weight over stance fraction d of a half-cycle."""
    return M * G * np.pi / (2.0 * d)


def _rk4(y, t, h, ode):
    k1 = ode(y, t)
    k2 = ode(y + h / 2 * k1, t + h / 2)
    k3 = ode(y + h / 2 * k2, t + h / 2)
    k4 = ode(y + h * k3, t + h)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_step(x0, z0, th0, xd0, zd0, thd0, foot_x, t0, T, d):
    """Integrate one step (T/2): stance (d*T/2, GRF active) + flight ((1-d)*T/2, ballistic)."""
    A = grf_amp(d)
    y = ca.vertcat(x0, z0, th0, xd0, zd0, thd0)   # 6-dim: x,z,th,xd,zd,thd

    def stance_ode(yy, tt):
        x = yy[0]; xd = yy[3]; zd = yy[4]; thd = yy[5]
        Fz = A * ca.sin(2 * np.pi * (tt - t0) / (d * T))
        return ca.vertcat(xd, zd, thd, 0.0, Fz / M - G, (foot_x - x) * Fz / I)

    def flight_ode(yy, tt):
        xd = yy[3]; zd = yy[4]; thd = yy[5]
        return ca.vertcat(xd, zd, thd, 0.0, -G, 0.0)

    hs = d * T / 2.0 / N
    for k in range(N):
        y = _rk4(y, t0 + k * hs, hs, stance_ode)
    if d < 1.0:
        hf = (1.0 - d) * T / 2.0 / N
        tf = t0 + d * T / 2.0
        for k in range(N):
            y = _rk4(y, tf + k * hf, hf, flight_ode)
    return y


def solve(v, T=0.5, d=1.0):
    S = v * T
    L = S * d / 4.0
    A = grf_amp(d)
    z0 = ca.MX.sym("z0"); zd0 = ca.MX.sym("zd0"); th0 = ca.MX.sym("th0"); thd0 = ca.MX.sym("thd0")

    ymid = integrate_step(0.0, z0, th0, v, zd0, thd0, L, 0.0, T, d)
    x1 = ymid[0]; z1 = ymid[1]; th1 = ymid[2]; xd1 = ymid[3]; zd1 = ymid[4]; thd1 = ymid[5]
    yend = integrate_step(x1, z1, th1, xd1, zd1, thd1, S / 2.0 + L, T / 2.0, T, d)
    zT = yend[1]; thT = yend[2]; zdT = yend[4]; thdT = yend[5]

    cost = ((zT - z0) ** 2 + (thT - th0) ** 2 + LAM1 * (thdT - thd0) ** 2
            + LAM3 * (zdT - zd0) ** 2)
    nlp = {"x": ca.vertcat(z0, zd0, th0, thd0), "f": cost}
    solver = ca.nlpsol("solver", "ipopt", nlp, {"ipopt.print_level": 0, "print_time": 0})
    sol = solver(x0=ca.vertcat(0.65, 0.0, 0.0, 0.0))
    z0s, zd0s, th0s, thd0s = np.array(sol["x"]).ravel()

    def rollout_np(z0_, zd0_, th0_, thd0_):
        state = np.array([0.0, z0_, th0_, v, zd0_, thd0_])   # 6-dim, initial xd=v
        zs, ths, fzs = [], [], []
        hs = d * T / 2.0 / N
        hf = (1.0 - d) * T / 2.0 / N
        for foot, t0 in [(L, 0.0), (S / 2.0 + L, T / 2.0)]:
            for k in range(N + 1):
                t = t0 + k * hs
                x, z, th, xd, zd, thd = state
                Fz = A * np.sin(2 * np.pi * (t - t0) / (d * T))
                zs.append(z); ths.append(th); fzs.append(Fz)
                state = state + hs * np.array([xd, zd, thd, 0.0,
                                               Fz / M - G, (foot - x) * Fz / I])
            if d < 1.0:
                for k in range(N + 1):
                    x, z, th, xd, zd, thd = state
                    zs.append(z); ths.append(th); fzs.append(0.0)
                    state = state + hf * np.array([xd, zd, thd, 0.0, -G, 0.0])
        return np.array(zs), np.array(ths), np.array(fzs)

    zs, ths, fzs = rollout_np(z0s, zd0s, th0s, thd0s)
    return dict(v=v, T=T, d=d, z0=z0s, zd0=zd0s, th0=th0s, thd0=thd0s, cost=float(sol["f"]),
                z_min=zs.min(), z_max=zs.max(), th_min=ths.min(), th_max=ths.max(),
                Fz_min=fzs.min())


if __name__ == "__main__":
    gaits = [("walk", 0.5, 1.0), ("run", 0.5, 0.5)]
    speeds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"{'gait':>5}{'v(m/s)':>8}{'z0(m)':>8}{'zd0(m/s)':>10}{'thd0(rad/s)':>12}{'cost':>10}"
          f"{'z_min':>8}{'z_max':>8}{'th_amp(rad)':>12}{'Fz_min(N)':>10}")
    for name, T, d in gaits:
        for v in speeds:
            r = solve(v, T=T, d=d)
            th_amp = (r["th_max"] - r["th_min"]) / 2
            print(f"{name:>5}{r['v']:>8.1f}{r['z0']:>8.3f}{r['zd0']:>10.3f}{r['thd0']:>12.4f}{r['cost']:>10.1e}"
                  f"{r['z_min']:>8.3f}{r['z_max']:>8.3f}{th_amp:>12.3f}{r['Fz_min']:>10.1f}")
