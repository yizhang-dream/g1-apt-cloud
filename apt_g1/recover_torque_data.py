"""Recover PD torque labels for the recorded SONIC closed-loop data.

The paper's key asset is torque-annotated data (TO provides state + torque).
Our exp_all3 dataset only has (command, proprio, token).  The MuJoCo G1 env
drives joints with a PD law

    tau = kp * (q_des - q) - kd * qdot

so given the recorded token (-> q_des via the frozen SONIC decoder) and the
recorded proprio history (-> current q, qdot), the exact torque label can be
recomputed offline without re-simulation.  This script produces

    data/torque_data/input.npy   (N, 16)  = [v9-router phase (sin,cos), cmd(14)]
    data/torque_data/tau.npy     (N, 12)  = PD torque of the 12 lower joints
    data/torque_data/meta.json             = per-joint mean/std, group counts

which is the training set for the paper-style torque decoder
(phase + command -> torque), the missing piece of the hybrid control scheme
tau = tau_dec + kp*(q_default - q + a_scale*a_aux) - kd*qdot.
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import torch

LOCAL = r"C:\Users\zyz\Documents\gr00t\apt_g1"
REPO = r"D:\GR00T-WholeBodyControl"
sys.path.insert(0, os.path.dirname(LOCAL))
sys.path.insert(0, LOCAL)
sys.path.insert(0, REPO)

from apt_g1.envs.mujoco_g1_flat_env import (
    G1_ISAACLAB_TO_MUJOCO_DOF,
    MujocoG1FlatEnv,
)
from apt_g1.eval_distill import NoQuantDecoder
from rough_sweep import load_router

D = os.path.join(LOCAL, "data", "exp_all3")
OUT = os.path.join(LOCAL, "data", "torque_data")


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    cmd = np.load(os.path.join(D, "cmd.npy"))
    token = np.load(os.path.join(D, "token.npy"))
    proprio = np.load(os.path.join(D, "proprio.npy"))
    mode = np.load(os.path.join(D, "mode.npy"))
    speed = np.load(os.path.join(D, "speed.npy"))
    angle_bin = np.load(os.path.join(D, "angle_bin.npy"))
    N = len(token)
    print("rows", N, flush=True)

    # constants + env (for motor limits / default angles)
    decoder = NoQuantDecoder(os.path.join(LOCAL, "model_decoder.onnx"))
    env = MujocoG1FlatEnv(decoder, REPO, use_elastic_band=False, stand_only=True)
    env.command = np.zeros(3, dtype=np.float32)
    mapping = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF, dtype=int)
    kp = env.kp[:12]
    kd = env.kd[:12]

    # group rows (for both phase labeling and stride sampling)
    pm, ps, nets, protos, gmap = load_router("distill_v9")
    group_counts = {}
    missing = 0
    rows_by_group = {}
    for i in range(N):
        key = (int(mode[i]), round(float(speed[i]), 2), int(angle_bin[i]))
        gi = gmap.get(key)
        if gi is None:
            missing += 1
            continue
        rows_by_group.setdefault(gi, []).append(i)
        group_counts[gi] = group_counts.get(gi, 0) + 1
    print("missing groups:", missing, flush=True)

    # stride-sample each group to ~700 rows (torque is a periodic fn of phase)
    idx = []
    gid_of = {}
    for gi, rows in rows_by_group.items():
        stride = max(1, len(rows) // 700)
        sel = rows[::stride]
        for i in sel:
            idx.append(i)
            gid_of[i] = gi
    idx = np.asarray(idx)
    n_sel = len(idx)
    print("sampled rows:", n_sel, flush=True)

    sc_all = np.zeros((n_sel, 2), dtype=np.float32)
    out_tau = np.zeros((n_sel, 12), dtype=np.float32)
    out_qdes = np.zeros((n_sel, 12), dtype=np.float32)
    out_jpos = np.zeros((n_sel, 12), dtype=np.float32)
    out_jvel = np.zeros((n_sel, 12), dtype=np.float32)
    out_cmd = cmd[idx]
    out_gid = np.asarray([gid_of[i] for i in idx], dtype=np.int64)

    # per-row decode (ONNX model is fixed batch=1)
    for k, i in enumerate(idx):
        hist = {
            "base_angular_velocity": proprio[i, 0:30].reshape(10, 3),
            "body_joint_positions": proprio[i, 30:320].reshape(10, 29),
            "body_joint_velocities": proprio[i, 320:610].reshape(10, 29),
            "last_actions": proprio[i, 610:900].reshape(10, 29),
            "gravity_dir": proprio[i, 900:930].reshape(10, 3),
        }
        env.history = hist
        q_des, _ = env._decode_body_action(token[i], np.zeros(12, dtype=np.float32))
        jpos = hist["body_joint_positions"][-1]
        jvel = hist["body_joint_velocities"][-1]
        out_tau[k] = kp * (q_des[:12] - jpos[:12]) - kd * jvel[:12]
        out_qdes[k] = q_des[:12]
        out_jpos[k] = jpos[:12]
        out_jvel[k] = jvel[:12]
        if k % 2000 == 0:
            print("row", k, flush=True)

    # v9 router phase per sampled group (vectorized per group)
    for gi, rows in rows_by_group.items():
        sel = np.asarray([k for k in range(n_sel) if out_gid[k] == gi])
        if len(sel) == 0:
            continue
        rows_i = idx[sel]
        x = np.concatenate(
            [(proprio[rows_i] - pm) / ps, cmd[rows_i]], axis=1
        ).astype(np.float32)
        with torch.no_grad():
            sc_all[sel] = (
                nets[gi](torch.from_numpy(x))[0].numpy().astype(np.float32)
            )

    os.makedirs(OUT, exist_ok=True)
    tau_mean = out_tau.mean(axis=0)
    tau_std = out_tau.std(axis=0) + 1e-6
    np.save(os.path.join(OUT, "input.npy"), np.concatenate([sc_all, out_cmd], axis=1))
    np.save(os.path.join(OUT, "tau.npy"), out_tau)
    np.save(os.path.join(OUT, "tau_norm_mean.npy"), tau_mean)
    np.save(os.path.join(OUT, "tau_norm_std.npy"), tau_std)
    np.save(os.path.join(OUT, "group_ids.npy"), out_gid)
    np.save(os.path.join(OUT, "q_des.npy"), out_qdes)
    meta = {
        "n": int(n_sel),
        "tau_mean": tau_mean.tolist(),
        "tau_std": tau_std.tolist(),
        "group_counts": {str(k): v for k, v in sorted(group_counts.items())},
        "input_dim": 16,
        "output_dim": 12,
        "note": "tau = kp*(q_des - q) - kd*qdot, first 12 MuJoCo-order joints",
    }
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print("saved", OUT, "rows", n_sel, flush=True)


if __name__ == "__main__":
    main()
