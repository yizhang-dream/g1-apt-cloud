"""Proto variants for marginal groups: mean vs nearest-row vs median, B=40/64/80."""
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
    def forward(self, x): return self.net(x)

D = '/home/cvgluser/ros2_data/apt_g1/data/exp_all'
cmd = np.load(D+'/cmd.npy'); modes_list = np.load(D+'/meta_modes.npy')
token = np.load(D+'/token.npy'); mode = np.load(D+'/mode.npy'); speed = np.load(D+'/speed.npy'); ab = np.load(D+'/angle_bin.npy')
odir = '/home/cvgluser/ros2_data/apt_g1/outputs/distill_v5'
norm = np.load(odir+'/phase_norm.npz')
meta = json.load(open(odir+'/phase_meta.json'))
pmean = norm['pmean'].ravel(); pstd = norm['pstd'].ravel()

nets = {}
for gi, md in meta.items():
    gi = int(gi)
    net = PhaseNet(930 + cmd.shape[1]).cuda()
    net.load_state_dict(torch.load(f'{odir}/phase_g{gi}.pt', map_location='cuda'))
    net.eval(); nets[gi] = net
gmap = {tuple(md['group']): int(gi) for gi, md in meta.items()}

def build_proto(gi, B, how):
    md = meta[str(gi)]; g = tuple(md['group'])
    rows = np.where((mode == g[0]) & (np.abs(speed - g[1]) < 1e-6) & (ab == g[2]))[0]
    # apply v5 row restrictions for slow (only used if requested)
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
        if how == 'mean':
            proto[b] = T[sel].mean(0)
        elif how == 'median':
            proto[b] = np.median(T[sel], axis=0)
        else:  # nearest
            d = np.abs((phi[sel] - centers[b] + np.pi) % (2*np.pi) - np.pi)
            proto[b] = T[sel[int(np.argmin(d))]]
    return np.clip(np.round(proto * 16) / 16, -1, 1).astype(np.float32)

env = MujocoG1FlatEnv(NoQuantDecoder('/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx'),
                      '/home/cvgluser/ros2_data/GR00T-WholeBodyControl', use_elastic_band=False, stand_only=True)
env.command = np.zeros(3, dtype=np.float32)

def angle_bin_of(a):
    return int(np.floor((a + np.pi) / (2 * np.pi) * 8)) % 8

def feat_for(c):
    oh = np.zeros(len(modes_list), dtype=np.float32)
    oh[int(np.where(modes_list == int(c['mode']))[0][0])] = 1
    return np.concatenate([oh, np.array(c['mdir'], dtype=np.float32), np.array(c['fdir'], dtype=np.float32),
                           np.array([c['speed'], -1.0, 1.0], dtype=np.float32)]).astype(np.float32)

def run(c, gi, proto, steps, seed, ema=0.3):
    import mujoco
    env.reset(); rng = np.random.default_rng(seed)
    env.data.qpos[2] += float(rng.normal(0, 0.005))
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(np.float32)
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history(); env._fill_history_from_state()
    heights, vxs, vys = [], [], []
    fall = None; sc_prev = None
    B = len(proto)
    for t in range(steps):
        prop = hist_to_proprio(env._get_sonic_history())
        x = np.concatenate([((prop - pmean) / pstd), feat_for(c)]).astype(np.float32)
        with torch.no_grad():
            sc = nets[gi](torch.from_numpy(x[None]).cuda())[0].cpu().numpy().astype(np.float32)
        if ema > 0 and sc_prev is not None:
            sc = ema * sc_prev + (1 - ema) * sc
        sc_prev = sc
        phi = float(np.arctan2(sc[0], sc[1]))
        b = int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)
        tok = proto[b]
        obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
        v = env._get_base_linear_velocity()
        vxs.append(float(v[0])); vys.append(float(v[1])); heights.append(float(env.data.qpos[2]))
        if terminated:
            fall = t; break
    vxs = np.array(vxs); vys = np.array(vys); heights = np.array(heights)
    spd = np.sqrt(vxs**2 + vys**2)
    return dict(fall_step=fall, completed=fall is not None and fall >= steps - 1,
                h_min=round(float(heights.min()), 3), vx=round(float(vxs.mean()), 3),
                vy=round(float(vys.mean()), 3), path=round(float(spd.sum()*0.02), 2))

scen = {
    'jump': (17, -1.0, 0.0),
    'turn_right': (1, 0.2, np.pi/3),
    'turn_left': (1, 0.2, -np.pi/3),
    'strafe_left': (1, 0.2, -np.pi/2),
    'stealth': (18, -1.0, -np.pi/6),
}
for how in ['mean', 'nearest', 'median']:
    for B in [40, 64]:
        print(f'=== {how} B={B} ===', flush=True)
        for name, (m, sp, a) in scen.items():
            gi = gmap.get((int(m), round(float(sp), 2), angle_bin_of(a)))
            if gi is None:
                continue
            proto = build_proto(gi, B, how)
            c = dict(mode=m, speed=sp, mdir=[float(np.cos(a)), float(np.sin(a)), 0.0], fdir=[float(np.cos(a)), float(np.sin(a)), 0.0])
            comp = 0; paths = []; mins = []
            for seed in [0, 1, 2]:
                r = run(c, gi, proto, 1000, seed)
                comp += 1 if r['completed'] else 0
                paths.append(r['path']); mins.append(r['h_min'])
            print(f'  {name:12s} completed {comp}/3 path_mean {np.mean(paths):.1f} h_min_avg {np.mean(mins):.3f}', flush=True)