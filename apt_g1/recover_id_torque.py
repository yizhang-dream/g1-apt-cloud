"""Recover inverse-dynamics torque labels by replaying the phase routers.

Unlike the PD-torque labels (feedback error, unpredictable from phase), the
inverse-dynamics torque of the replayed gait is the *motion-required*
feedforward -- the paper's TO-torque analogue.  For each control step of a
replayed stable command:

    qacc = (qvel_after - qvel_before) / dt
    data.qacc = qacc; mj_inverse -> qfrc_inverse[12 lower dofs] = tau_id

and the v9 router phase (sin/cos) + command feature are recorded.  Outputs:
data/torque_id/{input.npy (N,16), tau.npy (N,12), meta.json}.

Commands covered (stable groups from the flat battery):
walk fwd (2,-1,4), walk bin1, slow 0.6 fwd, slow 0.2 bins 0/1/2/6, idle,
jump -- the full stable flat repertoire.
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

import mujoco

from rough_sweep import load_router, feat_for, make_env
from apt_g1.eval_distill import hist_to_proprio

SEEDS = [0, 1, 2]


def bin_angle(b):
    return b * np.pi / 4.0 - np.pi


SCEN = [
    ("walk_fwd", 2, -1.0, 4, 2000),
    ("walk_bin1", 2, -1.0, 1, 2000),
    ("slow06_fwd", 1, 0.6, 4, 2000),
    ("slow02_bin0", 1, 0.2, 0, 1500),
    ("slow02_bin1", 1, 0.2, 1, 1500),
    ("slow02_bin2", 1, 0.2, 2, 1500),
    ("slow02_bin6", 1, 0.2, 6, 1500),
    ("idle", 0, -1.0, 4, 1500),
    ("jump", 17, -1.0, 4, 1500),
]


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    name = "distill_v9"
    pm, ps, nets, protos, gmap = load_router(name)
    env = make_env(0.0)
    dofs = env.body_dof_adr[:12]

    outs = []
    for tag, m, s, b, n_steps in SCEN:
        gi = gmap[(m, s, b)]
        B = len(protos[gi])
        feat = feat_for(
            dict(
                mode=m,
                speed=s,
                mdir=[float(np.cos(bin_angle(b))), float(np.sin(bin_angle(b))), 0.0],
                fdir=[float(np.cos(bin_angle(b))), float(np.sin(bin_angle(b))), 0.0],
            )
        )
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            env.reset()
            env.data.qpos[2] = 0.76
            env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(
                np.float32
            )
            env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
            mujoco.mj_forward(env.model, env.data)
            env._reset_history()
            env._fill_history_from_state()
            sc_prev = None
            got = 0
            for t in range(n_steps):
                prop = hist_to_proprio(env._get_sonic_history())
                x = np.concatenate([(prop - pm) / ps, feat]).astype(np.float32)
                with torch.no_grad():
                    sc = nets[gi](torch.from_numpy(x[None]))[0].numpy().astype(np.float32)
                if sc_prev is not None:
                    sc = 0.3 * sc_prev + 0.7 * sc
                sc_prev = sc
                phi = float(np.arctan2(sc[0], sc[1]))
                token = protos[gi][int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)]
                qvel_old = env.data.qvel.copy()
                obs, reward, terminated, info = env.step(
                    {"token": token, "aux": np.zeros(12, dtype=np.float32)}
                )
                qacc = (env.data.qvel - qvel_old) / 0.02
                env.data.qacc[:] = qacc
                mujoco.mj_inverse(env.model, env.data)
                tau_id = env.data.qfrc_inverse[dofs].copy()
                outs.append(
                    (
                        sc.astype(np.float32),
                        feat.astype(np.float32),
                        tau_id.astype(np.float32),
                        m,
                        s,
                        b,
                    )
                )
                got += 1
                if terminated:
                    break
            print(tag, "seed", seed, "steps", got, flush=True)
    x = np.stack([o[0] for o in outs])
    c = np.stack([o[1] for o in outs])
    tau = np.stack([o[2] for o in outs])
    # clip impact spikes per joint at the 99th percentile (1st/99th)
    lo = np.percentile(tau, 1.0, axis=0)
    hi = np.percentile(tau, 99.0, axis=0)
    tau = np.clip(tau, lo[None, :], hi[None, :])
    meta_rows = [(o[3], o[4], o[5]) for o in outs]
    out = os.path.join(LOCAL, "data", "torque_id")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "input.npy"), np.concatenate([x, c], axis=1))
    np.save(os.path.join(out, "tau.npy"), tau)
    tau_mean = tau.mean(axis=0)
    tau_std = tau.std(axis=0) + 1e-6
    np.save(os.path.join(out, "tau_norm_mean.npy"), tau_mean)
    np.save(os.path.join(out, "tau_norm_std.npy"), tau_std)
    json.dump(
        {
            "n": len(outs),
            "scen": SCEN,
            "tau_mean": tau_mean.tolist(),
            "tau_std": tau_std.tolist(),
            "input_dim": 16,
            "output_dim": 12,
            "note": "ID torque via mj_inverse on replayed router gaits",
            "clip": {"pct": [1.0, 99.0], "lo": lo.tolist(), "hi": hi.tolist()},
        },
        open(os.path.join(out, "meta.json"), "w"),
        indent=1,
    )
    print("saved", out, len(outs), "rows")


if __name__ == "__main__":
    main()
