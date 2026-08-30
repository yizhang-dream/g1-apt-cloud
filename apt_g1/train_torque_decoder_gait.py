"""Train a phase -> torque decoder on the foot-gait full-ID torque (TO10).

Reuses foot_gait_id.py's geometry + IK + full-ID torque (tau_clean = M qdd + qfrc_bias
- qfrc_constraint, the correct actuator torque).  Generates tau_clean for several walking
speeds, then trains a small MLP (sin phi, cos phi, v) -> 6-d torque (hip/knee/ankle x
left/right) and reports the reconstruction MAE in N*m.

This is the learnability gate for the moment-level pipeline: the paper's torque decoder
is exactly a phase/latent -> torque map; a low MAE means the self-consistent planning
torque is learnable and can later be swapped for a learned latent.  Compare: the SRB
torque decoder (train_torque_decoder_srb.py) got MAE 0.57-1.1 N*m on the (much simpler)
SRB static torque.

Run on the SERVER under .venv_mjlab (casadi + mujoco + torch).
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

import mujoco
import torch
import torch.nn as nn

from foot_gait_id import (
    SCENE, HIP_H, ANKLE_H, L1, L2, PELVIS_H, DEFAULT_Q,
    LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH,
    RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH,
    ik, setup,
)


def compute_gait_full(v, T=0.5, h_clear=0.06, n=201):
    """compute_gait_torques with the planned trajectory exposed.

    Returns (phi[n], X[n,3], tau[n,6], Q[n,29], Qd[n,29]): Q/Qd are the
    planned joint positions/velocities (29-dim MuJoCo body order) that tau
    was computed FROM -- TO18 uses them as the PD tracking target so the
    feedback no longer fights the feedforward.
    """
    model, data, qpos_adr, dof_adr = setup()
    S = v * T
    L = S / 4.0
    ts = np.linspace(0, T, n, endpoint=False)
    dt = T / n
    Q = np.zeros((n, len(DEFAULT_Q)))
    for i, t in enumerate(ts):
        hx = v * t
        if t < T / 2.0:
            stance_x = L
            toe_x = S / 2.0 + L - S
            heel_x = S / 2.0 + L
        else:
            stance_x = S / 2.0 + L
            toe_x = L
            heel_x = S + L
        u = (t % (T / 2.0)) / (T / 2.0)
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        swing_x = toe_x + (heel_x - toe_x) * s
        swing_z = ANKLE_H + h_clear * (1.0 - np.cos(2.0 * np.pi * u)) / 2.0
        q = DEFAULT_Q.copy()
        if t < T / 2.0:
            q[LEFT_HIP_PITCH], q[LEFT_KNEE] = ik(stance_x, ANKLE_H, hx, HIP_H)
            q[RIGHT_HIP_PITCH], q[RIGHT_KNEE] = ik(swing_x, swing_z, hx, HIP_H)
        else:
            q[RIGHT_HIP_PITCH], q[RIGHT_KNEE] = ik(stance_x, ANKLE_H, hx, HIP_H)
            q[LEFT_HIP_PITCH], q[LEFT_KNEE] = ik(swing_x, swing_z, hx, HIP_H)
        Q[i] = q

    Qd = np.gradient(Q, dt, axis=0)
    Qdd = np.gradient(Qd, dt, axis=0)

    cols = [LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH,
            RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH]
    tau = np.zeros((n, 6))
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
        con = data.qfrc_constraint[dof_adr]
        clean = uncon[dof_adr] - con
        tau[i] = [clean[c] for c in cols]

    phi = 2 * np.pi * ts / T
    # two-cycle data (phase wrapped twice gives more samples)
    X = np.stack([np.sin(phi), np.cos(phi), np.full_like(phi, v)], axis=1)
    return phi, X, tau, Q, Qd


def compute_gait_torques(v, T=0.5, h_clear=0.06, n=201):
    """Return (phi[2*n], X[(2*n,3)], Y[(2*n,6)]) for one speed.

    Y columns: [hip_L, knee_L, ankle_L, hip_R, knee_R, ankle_R] = tau_clean.
    """
    phi, X, tau, _Q, _Qd = compute_gait_full(v, T=T, h_clear=h_clear, n=n)
    return phi, X, tau


def main(speeds=(0.3, 0.5, 0.8), n=201):
    Xs, Ys = [], []
    for v in speeds:
        phi, X, Y = compute_gait_torques(v, n=n)
        Xs.append(X); Ys.append(Y)
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    print(f"dataset: {len(X)} samples, input (sin,cos,v), output 6-d torque (hip/knee/ankle x L/R)")

    mean, std = Y.mean(0), Y.std(0) + 1e-6
    Yn = (Y - mean) / std
    Xt = torch.tensor(X); Yt = torch.tensor(Yn)

    net = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 6))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for it in range(3000):
        opt.zero_grad()
        loss = lossf(net(Xt), Yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (net(Xt).numpy()) * std + mean
    mae = np.abs(pred - Y).mean(0)
    names = ["hip_L", "knee_L", "ankle_L", "hip_R", "knee_R", "ankle_R"]
    print(f"\n[full] reconstruction MAE (raw N*m):")
    for nm, m in zip(names, mae):
        print(f"  {nm:>8}: {m:.3f}")
    print(f"torque range: " + " ".join(f"{nm}[{Y[:,i].min():.0f},{Y[:,i].max():.0f}]" for i, nm in enumerate(names)))
    # save
    np.savez("/home/cvgluser/ros2_data/apt_g1/outputs/torque_gait_data.npz",
             X=X, Y=Y, names=np.array(names))
    print("\nsaved outputs/torque_gait_data.npz")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=201)
    a = ap.parse_args()
    main(n=a.n)
