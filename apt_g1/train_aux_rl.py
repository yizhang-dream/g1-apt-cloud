"""Minimal torque-level aux RL (TO17): does a LEARNED time-varying aux improve the walk?

The paper's aux mechanism:  tau = tau_dec + kp*(q_default + 0.2*aux - q) - kd*qdot,
where aux is an RL-learned 12-d joint offset.  TO16 showed a FIXED aux offset is neutral;
this trains a small policy  obs -> aux (time-varying) with a simple REINFORCE gradient to
answer whether the LEARNED aux helps (vs aux=0) in the torque-level pipeline.

Base feedforward tau_dec = the kinematic gait tau_clean (TO09/TO10, the best-performing
feedforward).  Reward = forward displacement over a bounded episode.

Run on the SERVER under .venv_mjlab (mujoco + torch).
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

from foot_gait_id import SCENE, DEFAULT_Q, LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH, \
    RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH
from train_torque_decoder_gait import compute_gait_torques

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
    88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50,
    88, 50, 50, 25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5,
], dtype=np.float64)
SAG = [LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH, RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH]
LOWER12 = list(range(12))  # aux acts on the 12 lower-body DOF


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


class Policy(nn.Module):
    def __init__(self, obs_dim=6, act_dim=12):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, act_dim))
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))  # std ~ 0.37

    def forward(self, obs):
        return self.net(obs)


def run_episode(model, data, qpos_adr, dof_adr, act_ids, tau_cols, phases, policy,
                T=0.5, kp_scale=0.5, kd_scale=1.0, max_steps=250, train=True, aux_scale=0.2):
    data.qpos[0:3] = [0.0, 0.0, 0.76]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = DEFAULT_Q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    dt_ctrl = 0.02
    x0 = float(data.qpos[0])
    logps, rewards = [], []
    for step in range(max_steps):
        phi = 2.0 * np.pi * ((step * dt_ctrl) % T / T)
        tau_ff = np.array([np.interp(phi, phases, tau_cols[j]) for j in SAG])
        # obs: [vx, vz, height, sin(phi), cos(phi), roll-gravity]
        vx = float(data.qvel[0])
        vz = float(data.qvel[2])
        h = float(data.qpos[2])
        obs = torch.tensor([vx, vz, h - 0.76, np.sin(phi), np.cos(phi), 0.0], dtype=torch.float32)
        if policy is None:
            aux_np = np.zeros(12)
        elif train:
            mean = policy(obs)
            std = torch.exp(policy.log_std)
            dist = torch.distributions.Normal(mean, std)
            aux = dist.rsample()
            logp = dist.log_prob(aux).sum()
            logps.append(logp)
            aux_np = aux.detach().numpy()
        else:
            aux = policy(obs)
            aux_np = aux.detach().numpy()
        q_des = DEFAULT_Q.copy()
        q_des[LOWER12] += aux_scale * np.clip(aux_np, -1, 1)
        for _ in range(4):
            q = data.qpos[qpos_adr]
            qd = data.qvel[dof_adr]
            torque = kp_scale * KP * (q_des - q) - kd_scale * KD * qd
            for k, j in enumerate(SAG):
                torque[j] += tau_ff[k]
            torque = np.clip(torque, -EFFORT, EFFORT)
            ctrl = np.zeros(model.nu)
            ctrl[act_ids] = torque
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        # reward = forward progress this step
        rewards.append(data.qvel[0] * dt_ctrl)
        if float(data.qpos[2]) < 0.2 or not np.all(np.isfinite(data.qpos)):
            break
    disp = float(data.qpos[0] - x0)
    return disp, logps, rewards


def main(episodes=2000, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model, data, qpos_adr, dof_adr, act_ids = setup()
    phi_arr, X, tau = compute_gait_torques(0.5, T=0.5)
    phases = np.linspace(0, 2 * np.pi, tau.shape[0], endpoint=False)
    tau_cols = {LEFT_HIP_PITCH: tau[:, 0], LEFT_KNEE: tau[:, 1], LEFT_ANKLE_PITCH: tau[:, 2],
                RIGHT_HIP_PITCH: tau[:, 3], RIGHT_KNEE: tau[:, 4], RIGHT_ANKLE_PITCH: tau[:, 5]}

    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline = 0.0
    for ep in range(episodes):
        disp, logps, rewards = run_episode(model, data, qpos_adr, dof_adr, act_ids,
                                           tau_cols, phases, policy, train=True)
        R = disp  # total return = displacement
        if logps:
            loss = -(sum(logps) * (R - baseline))
            opt.zero_grad(); loss.backward(); opt.step()
        baseline = 0.9 * baseline + 0.1 * R
        if (ep + 1) % 200 == 0:
            print(f"ep {ep+1:5d}  disp={disp:+.2f}  baseline={baseline:+.2f}", flush=True)

    # evaluate: aux=0 vs learned aux
    disp0 = run_episode(model, data, qpos_adr, dof_adr, act_ids, tau_cols, phases,
                        None, train=False)[0]
    disp_learned = run_episode(model, data, qpos_adr, dof_adr, act_ids, tau_cols, phases,
                               policy, train=False)[0]
    print(f"\n=== TO17 aux RL ({episodes} eps) ===")
    print(f"  aux=0 (tau_clean+PD):      disp={disp0:+.2f} m")
    print(f"  learned aux (tau+PD+aux):  disp={disp_learned:+.2f} m")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    a = ap.parse_args()
    main(episodes=a.episodes)
