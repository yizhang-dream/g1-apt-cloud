"""Final phase-router battery: 1000 steps x 3 seeds per mode + command-switch episode."""
import json, os, sys, time
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
cmd = np.load(D + '/cmd.npy'); modes_list = np.load(D + '/meta_modes.npy')
norm = np.load('/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_norm.npz')
meta = json.load(open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_meta.json'))
pmean = norm['pmean'].ravel(); pstd = norm['pstd'].ravel()
B = 40

nets = {}
protos = {}
for gi, md in meta.items():
    gi = int(gi)
    net = PhaseNet(930 + 13).cuda()
    net.load_state_dict(torch.load(f'/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_g{gi}.pt', map_location='cuda'))
    net.eval()
    nets[gi] = net
    # rebuild prototypes from PCA meta + train rows
    rows = np.load(f'/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_rows_g{gi}.npy') if os.path.exists(f'/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_rows_g{gi}.npy') else None

env = MujocoG1FlatEnv(NoQuantDecoder('/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx'),
                      '/home/cvgluser/ros2_data/GR00T-WholeBodyControl', use_elastic_band=False, stand_only=True)
env.command = np.zeros(3, dtype=np.float32)

def feat_for(c):
    oh = np.zeros(len(modes_list), dtype=np.float32); oh[int(np.where(modes_list == int(c['mode']))[0][0])] = 1
    return np.concatenate([oh, np.array(c['mdir'], dtype=np.float32), np.array(c['fdir'], dtype=np.float32),
                           np.array([c['speed'], -1.0, 1.0], dtype=np.float32)]).astype(np.float32)

def reset_with_jitter(seed):
    env.reset()
    rng = np.random.default_rng(seed)
    env.data.qpos[2] += float(rng.normal(0, 0.005))
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(np.float32)
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco = __import__('mujoco')
    mujoco.mj_forward(env.model, env.data)
    env._reset_history()
    env._fill_history_from_state()

def run_router(c, gi, steps, seed):
    net = nets[gi]; md = meta[str(gi)]
    mu = np.array(md['mu'], dtype=np.float32); V2 = np.array(md['V2'], dtype=np.float32)
    reset_with_jitter(seed)
    heights, vxs = [], []
    fall = None
    x0 = float(env.data.xpos[env.model.body('pelvis').id][0])
    feat = feat_for(c)
    for t in range(steps):
        prop = hist_to_proprio(env._get_sonic_history())
        x = np.concatenate([((prop - pmean) / pstd), feat]).astype(np.float32)
        with torch.no_grad():
            sc = net(torch.from_numpy(x[None]).cuda())[0].cpu().numpy()
        phi = float(np.arctan2(sc[0], sc[1]))
        b = int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)
        # prototype from saved prototypes (recompute quickly from meta is not available; load from npy saved in training)
        tok = protos_global[gi][b]
        obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
        heights.append(float(env.data.qpos[2])); vxs.append(float(env._get_base_linear_velocity()[0]))
        if terminated:
            fall = t; break
    return dict(fall_step=fall, h_mean=round(float(np.mean(heights)), 3), h_min=round(float(min(heights)), 3),
                vx=round(float(np.mean(vxs)), 3), x_end=round(float(env.data.xpos[env.model.body('pelvis').id][0]) - x0, 2))

# load prototypes (recompute from dataset + meta)
D2 = '/home/cvgluser/ros2_data/apt_g1/data/exp1'
token = np.load(D2 + '/token.npy'); mode = np.load(D2 + '/mode.npy'); speed = np.load(D2 + '/speed.npy'); mdir = cmd[:, 4:7]
val = np.zeros(len(token), dtype=bool); val[15606:17938] = True; val[18722:20308] = True
protos_global = {}
for gi, md in meta.items():
    gi = int(gi); g = tuple(md['group'])
    rows = np.where((mode == g[0]) & (np.abs(speed - g[1]) < 1e-6) & ~val)[0]
    if g[2] == 1:
        rows = rows[mdir[rows, 0] > 0]
    elif g[2] == -1:
        rows = rows[mdir[rows, 0] < 0]
    T = token[rows]; mu = np.array(md['mu'], dtype=np.float32); V2 = np.array(md['V2'], dtype=np.float32)
    proj = (T - mu) @ V2.T
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    bi = np.floor((phi + np.pi) / (2 * np.pi) * B).astype(int) % B
    proto = np.zeros((B, 64), dtype=np.float32); cnt = np.zeros(B, dtype=np.float32)
    for k in range(len(rows)):
        proto[bi[k]] += T[k]; cnt[bi[k]] += 1
    protos_global[gi] = np.clip(np.round((proto / np.maximum(cnt, 1)[:, None]) * 16) / 16, -1, 1)

gmap = {}
for gi, md in meta.items():
    gi = int(gi)
    g = tuple(md['group'])
    for name, c in [('idle', dict(mode=0, speed=-1.0, mdir=[0,0,0], fdir=[1,0,0])),
                    ('slow_walk', dict(mode=1, speed=0.2, mdir=[1,0,0], fdir=[1,0,0])),
                    ('walk', dict(mode=2, speed=-1.0, mdir=[1,0,0], fdir=[1,0,0])),
                    ('jump', dict(mode=17, speed=-1.0, mdir=[1,0,0], fdir=[1,0,0]))]:
        if int(c['mode']) == g[0] and abs(c['speed'] - g[1]) < 1e-6:
            gmap[name] = gi

out = {}
for name, c in [('idle', dict(mode=0, speed=-1.0, mdir=[0,0,0], fdir=[1,0,0])),
                ('slow_walk', dict(mode=1, speed=0.2, mdir=[1,0,0], fdir=[1,0,0])),
                ('walk', dict(mode=2, speed=-1.0, mdir=[1,0,0], fdir=[1,0,0])),
                ('jump', dict(mode=17, speed=-1.0, mdir=[1,0,0], fdir=[1,0,0]))]:
    gi = gmap[name]
    out[name] = {}
    for seed in [0, 1, 2]:
        r = run_router(c, gi, 1000, seed)
        out[name][f'seed{seed}'] = r
        print(f'{name:10s} seed{seed} fall={r["fall_step"]} h_min={r["h_min"]} vx={r["vx"]} x_end={r["x_end"]}', flush=True)
json.dump(out, open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/eval_phase_final.json', 'w'), indent=1)

# command-switch episode
def run_switch(schedule, seed=0):
    reset_with_jitter(seed)
    heights, vxs = [], []
    fall = None
    for name, secs in schedule:
        c = dict(name=name)
        if name == 'idle': c = dict(mode=0, speed=-1.0, mdir=[0,0,0], fdir=[1,0,0])
        elif name == 'slow_walk': c = dict(mode=1, speed=0.2, mdir=[1,0,0], fdir=[1,0,0])
        elif name == 'walk': c = dict(mode=2, speed=-1.0, mdir=[1,0,0], fdir=[1,0,0])
        elif name == 'jump': c = dict(mode=17, speed=-1.0, mdir=[1,0,0], fdir=[1,0,0])
        gi = gmap[name]
        steps = int(secs * 50)
        for t in range(steps):
            prop = hist_to_proprio(env._get_sonic_history())
            x = np.concatenate([((prop - pmean) / pstd), feat_for(c)]).astype(np.float32)
            with torch.no_grad():
                sc = nets[gi](torch.from_numpy(x[None]).cuda())[0].cpu().numpy()
            phi = float(np.arctan2(sc[0], sc[1]))
            b = int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)
            tok = protos_global[gi][b]
            obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
            heights.append(float(env.data.qpos[2])); vxs.append(float(env._get_base_linear_velocity()[0]))
            if terminated:
                fall = len(heights) - 1
                return dict(fall_step=fall, h_min=round(float(min(heights)), 3), vx=round(float(np.mean(vxs)), 3))
    return dict(fall_step=None, h_min=round(float(min(heights)), 3), vx=round(float(np.mean(vxs)), 3))

sched = [('idle', 5), ('walk', 10), ('idle', 5), ('slow_walk', 10), ('jump', 5), ('idle', 5)]
sw = run_switch(sched, seed=4)
print('switch episode:', sw, flush=True)
json.dump({'switch': sw, 'schedule': sched}, open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/eval_switch.json', 'w'), indent=1)
print('done')