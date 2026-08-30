"""Train a phase->torque decoder on the SRB-TO torque data, verify learnability
across gaits (walk d=1 vs run d=0.5) and speeds.

The SRB TO gives a self-consistent (feedforward) joint torque as a deterministic
periodic function of the gait phase, speed and duty factor.  This trains a small
MLP (sin(phi), cos(phi), speed, duty) -> (hip torque, knee torque, ankle torque)
and reports the reconstruction MAE in N*m -- the paper's torque decoder is exactly
this kind of latent/phase -> torque mapping, so a low MAE means the TO torque is
learnable (and we can later swap phase for a learned latent).  The ankle torque
uses a moving center-of-pressure (CoP slides heel->toe during stance) as its lever
arm, giving the plantarflexion sign change a real ankle has.

Also runs a cross-gait generalization check: train on walk only, test on run.

Compare: PD-torque labels gave MAE ~18.76 N*m (feedback, unpredictable);
ID torque ~4.13 N*m; the self-consistent TO torque should be far lower.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from srb_to import solve, grf_amp, M, I, G, N
from srb_to_torque import ik, knee_pos, L1, L2, L_HEEL, L_TOE


def gen_trajectory(v, T=0.5, d=1.0):
    """Stance-phase (state, torque) samples for one gait cycle at speed v."""
    r = solve(v, T=T, d=d)
    z0, zd0, th0, thd0 = r["z0"], r["zd0"], r["th0"], r["thd0"]
    S = v * T
    L = S * d / 4.0
    A = grf_amp(d)
    state = np.array([0.0, z0, th0, zd0, thd0])     # 5-dim: x, z, th, zd, thd
    X, Y = [], []
    hs = d * T / 2.0 / N
    hf = (1.0 - d) * T / 2.0 / N
    for foot_x, t0 in [(L, 0.0), (S / 2.0 + L, T / 2.0)]:
        for k in range(N + 1):
            t = t0 + k * hs
            x, z, th, zd, thd = state
            Fz = A * np.sin(2 * np.pi * (t - t0) / (d * T))
            th_h, th_k = ik(foot_x, 0.0, x, z)
            kx, kz = knee_pos(th_h, x, z)
            tau_h = Fz * (foot_x - x)          # hip torque (Fz only, Fx=0)
            tau_k = Fz * (foot_x - kx)         # knee torque
            u = (t - t0) / (d * T / 2.0)       # normalized stance phase [0,1]
            tau_a = Fz * (-L_HEEL + (L_HEEL + L_TOE) * u)   # ankle torque (CoP heel->toe)
            phi = 2 * np.pi * t / T
            X.append([np.sin(phi), np.cos(phi), v, d])
            Y.append([tau_h, tau_k, tau_a])
            zdd = Fz / M - G
            thdd = (foot_x - x) * Fz / I
            state = state + hs * np.array([v, zd, thd, zdd, thdd])
        # integrate the flight phase (no GRF -> no torque samples, but the CoM
        # must advance so the next stance starts from the right state)
        if d < 1.0:
            for k in range(N + 1):
                x, z, th, zd, thd = state
                state = state + hf * np.array([v, zd, thd, -G, 0.0])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def make_net():
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 3))


def train(net, Xt, Yt, iters=2000):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for it in range(iters):
        opt.zero_grad()
        loss = lossf(net(Xt), Yt)
        loss.backward()
        opt.step()
    return net


def mae_raw(net, X, Y, mean, std):
    with torch.no_grad():
        pred = (net(torch.tensor(X)).numpy()) * std + mean
    return np.abs(pred - Y).mean(0)


def main():
    speeds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    gaits = [("walk", 0.5, 1.0), ("run", 0.5, 0.5)]

    # ---- gather all data + per-gait split ----
    Xall, Yall, Xgait, Ygait = [], [], {}, {}
    for name, T, d in gaits:
        X, Y = [], []
        for v in speeds:
            xx, yy = gen_trajectory(v, T=T, d=d)
            X.append(xx); Y.append(yy)
        X = np.concatenate(X); Y = np.concatenate(Y)
        Xgait[name], Ygait[name] = X, Y
        Xall.append(X); Yall.append(Y)
    Xall = np.concatenate(Xall); Yall = np.concatenate(Yall)
    print(f"data: {len(Xall)} samples (walk {len(Xgait['walk'])}, run {len(Xgait['run'])}), "
          f"input dim 4, output dim 3 (hip/knee/ankle)")

    mean, std = Yall.mean(0), Yall.std(0) + 1e-6
    Yn = (Yall - mean) / std
    Xt, Yt = torch.tensor(Xall), torch.tensor(Yn)

    # ---- full-data reconstruction (walk+run) ----
    net = make_net()
    train(net, Xt, Yt)
    mae = mae_raw(net, Xall, Yall, mean, std)
    print(f"\n[full walk+run] reconstruction MAE (raw N*m): "
          f"hip={mae[0]:.3f}  knee={mae[1]:.3f}  ankle={mae[2]:.3f}")
    for name in ["walk", "run"]:
        mae_g = mae_raw(net, Xgait[name], Ygait[name], mean, std)
        print(f"  per-gait {name:>4}: hip={mae_g[0]:.3f}  knee={mae_g[1]:.3f}  ankle={mae_g[2]:.3f}")

    # ---- cross-gait generalization: train walk, test run ----
    Xw, Yw = Xgait["walk"], Ygait["walk"]
    mw, sw = Yw.mean(0), Yw.std(0) + 1e-6
    Xwt = torch.tensor(Xw); Ywt = torch.tensor((Yw - mw) / sw)
    net2 = make_net()
    train(net2, Xwt, Ywt)
    mae_x = mae_raw(net2, Xgait["run"], Ygait["run"], mw, sw)
    print(f"\n[cross-gait: train walk -> test run] MAE: "
          f"hip={mae_x[0]:.3f}  knee={mae_x[1]:.3f}  ankle={mae_x[2]:.3f}")

    print(f"\ntorque range: hip [{Yall[:,0].min():.1f},{Yall[:,0].max():.1f}] "
          f"knee [{Yall[:,1].min():.1f},{Yall[:,1].max():.1f}] "
          f"ankle [{Yall[:,2].min():.1f},{Yall[:,2].max():.1f}] N*m")


if __name__ == "__main__":
    main()
