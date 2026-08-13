"""Headless MuJoCo rough-terrain robustness sweep for the phase routers.

Sweeps hfield amplitude (+-amp) x router (v6/v9) x seed for walk-forward,
plus an idle (zero-token standing) control per amplitude.  Records fall step,
minimum pelvis height, and horizontal displacement.  Results are saved to
``outputs/rough_mujoco_sweep.json`` for the MuJoCo-platform wrap-up.

Run on the local Windows machine (Python 3.13 + mujoco 3.11, CPU).
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

LOCAL = r"C:\Users\zyz\Documents\gr00t\apt_g1"
REPO = r"D:\GR00T-WholeBodyControl"
sys.path.insert(0, os.path.dirname(LOCAL))
sys.path.insert(0, LOCAL)
sys.path.insert(0, REPO)

import mujoco

import make_rough_xml as mrx
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from apt_g1.eval_distill import NoQuantDecoder, hist_to_proprio

DEV = "cpu"
STEPS = 1200  # 24 s @ 50 Hz
AMPS = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
ROUTERS = ["distill_v9", "distill_final"]  # distill_final == v6
SEEDS = [0, 1, 2]


class PhaseNet(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


D = os.path.join(LOCAL, "data", "exp_all3")
cmd = np.load(os.path.join(D, "cmd.npy"))
modes_list = np.load(os.path.join(D, "meta_modes.npy"))


def load_router(odir_name):
    odir = os.path.join(LOCAL, "outputs", odir_name)
    norm = np.load(os.path.join(odir, "phase_norm.npz"))
    meta = json.load(open(os.path.join(odir, "phase_meta.json")))
    nets, protos = {}, {}
    for gi, md in meta.items():
        if gi.startswith("_"):
            continue
        gi = int(gi)
        net = PhaseNet(930 + cmd.shape[1])
        net.load_state_dict(
            torch.load(os.path.join(odir, f"phase_g{gi}.pt"), map_location=DEV)
        )
        net.eval()
        nets[gi] = net
        protos[gi] = np.load(os.path.join(odir, f"proto_g{gi}.npy"))
    gmap = {
        tuple(md["group"]): int(gi)
        for gi, md in meta.items()
        if not gi.startswith("_")
    }
    return (
        norm["pmean"].ravel(),
        norm["pstd"].ravel(),
        nets,
        protos,
        gmap,
    )


def angle_bin_of(a):
    return int(np.floor((a + np.pi) / (2 * np.pi) * 8)) % 8


def feat_for(c):
    oh = np.zeros(len(modes_list), dtype=np.float32)
    oh[int(np.where(modes_list == int(c["mode"]))[0][0])] = 1
    return np.concatenate(
        [
            oh,
            np.array(c["mdir"], dtype=np.float32),
            np.array(c["fdir"], dtype=np.float32),
            np.array([c["speed"], -1.0, 1.0], dtype=np.float32),
        ]
    ).astype(np.float32)


def make_env(amp):
    """Create an env on the requested terrain; amp=0 uses the flat scene."""
    decoder = NoQuantDecoder(os.path.join(LOCAL, "model_decoder.onnx"))
    if amp <= 0.0:
        scene = os.path.join(
            REPO, "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
        )
    else:
        mrx.build(amp=amp, seed=0)
        scene = mrx.OUT
    env = MujocoG1FlatEnv(
        decoder,
        REPO,
        robot_scene=scene,
        use_elastic_band=False,
        stand_only=True,
    )
    env.command = np.zeros(3, dtype=np.float32)
    return env


def terrain_z(amp, x, y):
    if amp <= 0.0:
        return 0.0
    h = np.load(os.path.join(LOCAL, "outputs", "rough_h.npy"))
    n = h.shape[0]
    res = 40.0 / n
    i = int(np.clip(round(x / res + n / 2), 0, n - 1))
    j = int(np.clip(round(y / res + n / 2), 0, n - 1))
    return float(h[i, j])


def run_one(env, amp, token_fn, seed, steps):
    rng = np.random.default_rng(seed)
    env.reset()
    env.data.qpos[2] = terrain_z(amp, 0.0, 0.0) + 0.76
    env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(
        np.float32
    )
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history()
    env._fill_history_from_state()
    x0 = float(env.data.qpos[0])
    y0 = float(env.data.qpos[1])
    heights = []
    xs, ys = [], []
    fall = None
    for t in range(steps):
        # record position BEFORE the step: the env auto-resets on termination
        # and zeroes qpos, so post-termination positions are meaningless.
        xs.append(float(env.data.qpos[0]))
        ys.append(float(env.data.qpos[1]))
        token = token_fn(env, t, seed)
        obs, reward, terminated, info = env.step(
            {"token": token, "aux": np.zeros(12, dtype=np.float32)}
        )
        heights.append(float(env.data.qpos[2]))
        if terminated:
            fall = t
            break
    n_steps = steps if fall is None else fall
    if fall is None:
        dx = float(env.data.qpos[0] - x0)
        dy = float(env.data.qpos[1] - y0)
    else:
        dx = float(xs[fall] - x0)
        dy = float(ys[fall] - y0)
    return {
        "fall": fall,
        "h_min": round(float(min(heights)), 3),
        "dx": round(dx, 2),
        "dy": round(dy, 2),
        "disp": round(float(np.hypot(dx, dy)), 2),
        "vx_est": round(float(np.hypot(dx, dy) / max(1e-6, n_steps * 0.02)), 3),
    }


def main():
    routers = {name: load_router(name) for name in ROUTERS}
    walk_feat = feat_for(
        dict(mode=2, speed=-1.0, mdir=[1.0, 0.0, 0.0], fdir=[1.0, 0.0, 0.0])
    )
    results = {}
    for amp in AMPS:
        env = make_env(amp)
        results[str(amp)] = {}
        for seed in SEEDS:
            r = run_one(
                env,
                amp,
                lambda e, t, s: np.zeros(64, dtype=np.float32),
                seed,
                STEPS,
            )
            results[str(amp)][f"idle_s{seed}"] = r
            print(f"amp={amp} idle seed={seed}: {r}", flush=True)
        for rname, (pm, ps, nets, protos, gmap) in routers.items():
            gi = gmap[(2, -1.0, angle_bin_of(0.0))]
            B = len(protos[gi])
            sc_prev = {}

            def token_fn(
                e,
                t,
                seed,
                _pm=pm,
                _ps=ps,
                _nets=nets,
                _protos=protos,
                _gi=gi,
                _B=B,
            ):
                prop = hist_to_proprio(e._get_sonic_history())
                x = np.concatenate([(prop - _pm) / _ps, walk_feat]).astype(np.float32)
                with torch.no_grad():
                    sc = _nets[_gi](torch.from_numpy(x[None]))[0].numpy().astype(
                        np.float32
                    )
                prev = sc_prev.get(seed)
                if prev is not None:
                    sc = 0.3 * prev + 0.7 * sc
                sc_prev[seed] = sc
                phi = float(np.arctan2(sc[0], sc[1]))
                b = int(np.floor((phi + np.pi) / (2 * np.pi) * _B) % _B)
                return _protos[_gi][b]

            for si, seed in enumerate(SEEDS):
                r = run_one(env, amp, token_fn, seed, STEPS)
                results[str(amp)][f"{rname}_s{si}"] = r
                print(f"amp={amp} {rname} seed={seed}: {r}", flush=True)
        json.dump(
            results,
            open(os.path.join(LOCAL, "outputs", "rough_mujoco_sweep.json"), "w"),
            indent=1,
        )
    print("saved outputs/rough_mujoco_sweep.json")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
