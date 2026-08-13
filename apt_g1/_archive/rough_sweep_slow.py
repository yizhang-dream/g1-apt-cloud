"""Slow-walk rough-terrain robustness for the v9 router (MuJoCo local).

Same harness as ``rough_sweep.py`` but for the SLOW_WALK command group
(mode 1, speed 0.2, forward), at amps 0.04 / 0.06 / 0.08.  Question: does the
lower-speed gait survive rougher terrain than the walk group?
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

import make_rough_xml as mrx
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from apt_g1.eval_distill import NoQuantDecoder, hist_to_proprio
from rough_sweep import load_router, feat_for, angle_bin_of, terrain_z, run_one, make_env

AMPS = [0.04, 0.06, 0.08]
SEEDS = [0, 1, 2]
STEPS = 1200


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    pm, ps, nets, protos, gmap = load_router("distill_v9")
    slow_feat = feat_for(
        dict(mode=1, speed=0.2, mdir=[1.0, 0.0, 0.0], fdir=[1.0, 0.0, 0.0])
    )
    gi = gmap[(1, 0.2, angle_bin_of(0.0))]
    B = len(protos[gi])
    results = {}
    for amp in AMPS:
        env = make_env(amp)
        results[str(amp)] = {}
        sc_prev = {}

        def token_fn(e, t, seed, _pm=pm, _ps=ps, _nets=nets, _protos=protos, _gi=gi, _B=B):
            prop = hist_to_proprio(e._get_sonic_history())
            x = np.concatenate([(prop - _pm) / _ps, slow_feat]).astype(np.float32)
            with torch.no_grad():
                sc = _nets[_gi](torch.from_numpy(x[None]))[0].numpy().astype(np.float32)
            prev = sc_prev.get(seed)
            if prev is not None:
                sc = 0.3 * prev + 0.7 * sc
            sc_prev[seed] = sc
            phi = float(np.arctan2(sc[0], sc[1]))
            b = int(np.floor((phi + np.pi) / (2 * np.pi) * _B) % _B)
            return _protos[_gi][b]

        for si, seed in enumerate(SEEDS):
            r = run_one(env, amp, token_fn, seed, STEPS)
            results[str(amp)][f"slow_s{si}"] = r
            print(f"amp={amp} slow seed={seed}: {r}", flush=True)
        json.dump(
            results,
            open(os.path.join(LOCAL, "outputs", "rough_mujoco_slow.json"), "w"),
            indent=1,
        )
    print("saved outputs/rough_mujoco_slow.json")


if __name__ == "__main__":
    main()
