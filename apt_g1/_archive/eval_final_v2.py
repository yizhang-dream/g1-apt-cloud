"""Prototype v2: nearest-row prototypes, B=64 bins; eval slow_walk/jump/walk at 1000 steps x 3 seeds."""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, '/home/cvgluser/ros2_data')
sys.path.insert(0, '/home/cvgluser/ros2_data/apt_g1')
sys.path.insert(0, '/home/cvgluser/ros2_data/GR00T-WholeBodyControl')
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder, hist_to_proprio

class PhaseNet(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, 2))
    def forward(self, x):
        return self.net(x)

D = '/home/cvgluser/ros2_data/apt_g1/data/exp1'
cmd = np.load(D + '/cmd.npy'); token = np.load(D + '/token.npy'); mode = np.load(D + '/mode.npy'); speed = np.load(D + '/speed.npy')
mdir = cmd[:, 4:7]; modes_list = np.load(D + '/meta_modes.npy')
val = np.zeros(len(token), dtype=bool); val[15606:17938] = True; val[18722:20308] = True
norm = np.load('/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_norm.npz')
meta = json.load(open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_meta.json'))
pmean = norm['pmean'].ravel(); pstd = norm['pstd'].ravel()
B = 64

nets = {}; protos = {}
for gi, md in meta.items():
    gi = int(gi); g = tuple(md['group'])
    net = PhaseNet(930 + 13).cuda()
    net.load_state_dict(torch.load(f'/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_g{gi}.pt', map_location='cuda'))
    net.eval(); nets[gi] = net
    rows = np.where((mode == g[0]) & (np.abs(speed - g[1]) < 1e-6) & ~val)[0]
    if g[2] == 1: rows = rows[mdir[rows, 0] > 0]
    elif g[2] == -1: rows = rows[mdir[rows, 0] < 0]
    T = token[rows]; mu = np.array(md['mu'], dtype=np.float32); V2 = np.array(md['V2'], dtype=np.float32)
    proj = (T - mu) @ V2.T
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    bi = np.floor((phi + np.pi) / (2 * np.pi) * B).astype(int) % B
    proto = np.zeros((B, 64), dtype=np.float32)
    centers = np.linspace(-np.pi, np.pi, B, endpoint=False)
    for b in range(B):
        sel = np.where(bi == b)[0]
        if len(sel) == 0:
            proto[b] = proto[b-1] if b > 0 else T[0]
            continue
        # nearest row to bin center in circular phase
        d = np.abs((phi[sel] - centers[b] + np.pi) % (2*np.pi) - np.pi)
        proto[b] = T[sel[int(np.argmin(d))]]
    protos[gi] = np.clip(np.round(proto * 16) / 16, -1, 1)

env = MujocoG1FlatEnv(NoQuantDecoder('/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx'),
                      '/home/cvgluser/ros2_data/GR00T-WholeBodyControl', use_elastic_band=False, stand_only=True)
env.command = np.zeros(3, dtype=np.float32)

def feat_for(c):
    oh = np.zeros(len(modes_list), dtype=np.float32); oh[int(np.where(modes_list == int(c['mode']))[0][0])] = 1
    return np.concatenate([oh, np.array(c['mdir'], dtype=np.float32), np.array(c['fdir'], dtype=np.float32),
                           np.array([c['speed'], -1.0, 1.0], dtype=np.float32)]).astype(np.float32)

def run(c, gi, steps, seed):
    import mujoco
    env.reset()
    rng = np.random.default_rng(seed)
    env.data.qpos[2] += float(rng.normal(0, 0.005))
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(np.float32)
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history(); env._fill_history_from_state()
    heights, vxs = [], []
    fall = None
    disp = 0.0
    for t in range(steps):
        prop = hist_to_proprio(env._get_sonic_history())
        x = np.concatenate([((prop - pmean) / pstd), feat_for(c)]).astype(np.float32)
        with torch.no_grad():
            sc = nets[gi](torch.from_numpy(x[None]).cuda())[0].cpu().numpy()
        phi = float(np.arctan2(sc[0], sc[1]))
        b = int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)
        tok = protos[gi][b]
        obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
        vx = float(env._get_base_linear_velocity()[0])
        heights.append(float(env.data.qpos[2])); vxs.append(vx); disp += vx * 0.02
        if terminated:
            fall = t; break
    return dict(fall_step=fall, completed=fall is not None and fall >= steps - 1, h_mean=round(float(np.mean(heights)), 3),
                h_min=round(float(min(heights)), 3), vx=round(float(np.mean(vxs)), 3), disp=round(float(disp), 2))

scen = [('idle', dict(mode=0, speed=-1.0, mdir=[0,0,0], fdir=[1,0,0])),
        ('slow_walk', dict(mode=1, speed=0.2, mdir=[1,0,0], fdir=[1,0,0])),
        ('walk', dict(mode=2, speed=-1.0, mdir=[1,0,0], fdir=[1,0,0])),
        ('jump', dict(mode=17, speed=-1.0, mdir=[1,0,0], fdir=[1,0,0]))]
gmap = {}
for gi, md in meta.items():
    g = tuple(md['group'])
    for name, c in scen:
        if int(c['mode']) == g[0] and abs(c['speed'] - g[1]) < 1e-6:
            gmap[name] = int(gi)
out = {}
for name, c in scen:
    gi = gmap[name]
    out[name] = {}
    for seed in [0, 1, 2]:
        r = run(c, gi, 1000, seed)
        out[name][f'seed{seed}'] = r
        print(f'{name:10s} seed{seed} completed={r["completed"]} h_min={r["h_min"]} vx={r["vx"]} disp={r["disp"]}', flush=True)
json.dump(out, open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/eval_phase_final_v2.json', 'w'), indent=1)
print('saved')