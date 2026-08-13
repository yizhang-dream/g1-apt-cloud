"""Perturbation ceiling: closed-loop survival with oracle tokens + k dims off by 1 level."""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/cvgluser/ros2_data')
sys.path.insert(0, '/home/cvgluser/ros2_data/apt_g1')
sys.path.insert(0, '/home/cvgluser/ros2_data/GR00T-WholeBodyControl')
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder, SCEN, build_cmd_feature, run_rollout

def main():
    steps = 300
    D = '/home/cvgluser/ros2_data/apt_g1/data/exp1'
    modes = np.load(D + '/meta_modes.npy')
    token = np.load(D + '/token.npy'); mode = np.load(D + '/mode.npy'); speed = np.load(D + '/speed.npy'); cmd = np.load(D + '/cmd.npy')
    mdir = cmd[:, 4:7]
    repo = '/home/cvgluser/ros2_data/GR00T-WholeBodyControl'
    decoder = NoQuantDecoder(repo + '/gear_sonic_deploy/policy/release/model_decoder.onnx')
    env = MujocoG1FlatEnv(decoder, repo, use_elastic_band=False, stand_only=True)
    env.command = np.zeros(3, dtype=np.float32)

    oracle = {}
    for name, c in SCEN:
        sel = (mode == c['mode']) & (np.abs(speed - c['speed']) < 1e-6)
        if name == 'backward':
            sel &= mdir[:, 0] < 0
        oracle[name] = token[np.where(sel)[0]].astype(np.float32)

    out = {}
    ks = [0, 1, 2, 4, 8]
    rng = np.random.default_rng(0)
    for name, c in SCEN:
        seq = oracle[name]
        cmd_feat = build_cmd_feature(c, modes)
        out[name] = {}
        for k in ks:
            def src_fn(t):
                tok = seq[t % len(seq)].copy()
                if k:
                    dims = rng.choice(64, k, replace=False)
                    signs = rng.choice([-1, 1], k).astype(np.float32)
                    tok[dims] = np.clip(tok[dims] + signs * (1/16), -1, 1)
                return tok
            r = run_rollout(env, 'oracle', cmd_feat, steps, oracle=seq, seed=0)
            # re-run with perturbation via monkey-patched token source
            r = run_rollout(env, 'custom', cmd_feat, steps, oracle=seq, seed=0)
            # custom run with perturbed tokens
            env.reset()
            heights, vxs = [], []
            fall = None
            for t in range(steps):
                tok = src_fn(t)
                obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
                heights.append(float(env.data.qpos[2])); vxs.append(float(env._get_base_linear_velocity()[0]))
                if terminated:
                    fall = t; break
            out[name][k] = dict(fall_step=fall,
                                h_mean=round(float(np.mean(heights)), 3),
                                h_min=round(float(np.min(heights)), 3),
                                vx=round(float(np.mean(vxs)), 3))
            print(f'{name:10s} k={k} fall={fall} h_min={out[name][k]["h_min"]} vx={out[name][k]["vx"]}', flush=True)
    json.dump(out, open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/perturb.json', 'w'), indent=1)

if __name__ == '__main__':
    main()