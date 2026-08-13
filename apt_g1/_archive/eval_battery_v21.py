"""Battery v2.1: loaded prototypes + 2D metrics."""
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

D = '/home/cvgluser/ros2_data/apt_g1/data/exp_all'
cmd = np.load(D + '/cmd.npy'); modes_list = np.load(D + '/meta_modes.npy')
odir = '/home/cvgluser/ros2_data/apt_g1/outputs/distill_v2'
norm = np.load(odir + '/phase_norm.npz')
meta = json.load(open(odir + '/phase_meta.json'))
pmean = norm['pmean'].ravel(); pstd = norm['pstd'].ravel()
B = 40

nets = {}; protos = {}
for gi, md in meta.items():
    gi = int(gi)
    net = PhaseNet(930 + cmd.shape[1]).cuda()
    net.load_state_dict(torch.load(f'{odir}/phase_g{gi}.pt', map_location='cuda'))
    net.eval(); nets[gi] = net
    protos[gi] = np.load(f'{odir}/proto_g{gi}.npy')

gmap = {}
for gi, md in meta.items():
    gmap[tuple(md['group'])] = int(gi)

def angle_bin_of(a):
    NB = 8
    return int(np.floor((a + np.pi) / (2 * np.pi) * NB)) % NB

def feat_for(c):
    oh = np.zeros(len(modes_list), dtype=np.float32)
    oh[int(np.where(modes_list == int(c['mode']))[0][0])] = 1
    return np.concatenate([oh, np.array(c['mdir'], dtype=np.float32), np.array(c['fdir'], dtype=np.float32),
                           np.array([c['speed'], -1.0, 1.0], dtype=np.float32)]).astype(np.float32)

env = MujocoG1FlatEnv(NoQuantDecoder('/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx'),
                      '/home/cvgluser/ros2_data/GR00T-WholeBodyControl', use_elastic_band=False, stand_only=True)
env.command = np.zeros(3, dtype=np.float32)

def yaw_of():
    q = env.data.xquat[env.model.body('pelvis').id]
    return float(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))

def run(c, gi, steps, seed, ema=0.3):
    import mujoco
    env.reset()
    rng = np.random.default_rng(seed)
    env.data.qpos[2] += float(rng.normal(0, 0.005))
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(np.float32)
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history(); env._fill_history_from_state()
    heights, vxs, vys = [], [], []
    fall = None
    sc_prev = None
    yaw0 = yaw_of()
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
        tok = protos[gi][b]
        obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
        v = env._get_base_linear_velocity()
        vxs.append(float(v[0])); vys.append(float(v[1]))
        heights.append(float(env.data.qpos[2]))
        if terminated:
            fall = t; break
    vxs = np.array(vxs); vys = np.array(vys); heights = np.array(heights)
    spd = np.sqrt(vxs ** 2 + vys ** 2)
    return dict(fall_step=fall, completed=fall is not None and fall >= steps - 1,
                h_mean=round(float(heights.mean()), 3), h_min=round(float(heights.min()), 3),
                vx=round(float(vxs.mean()), 3), vy=round(float(vys.mean()), 3),
                speed=round(float(spd.mean()), 3), path=round(float(spd.sum() * 0.02), 2),
                yaw_change=round(float(yaw_of() - yaw0), 3))

scen = [
    ('idle', 0, -1.0, 0.0),
    ('slow_fwd', 1, 0.2, 0.0),
    ('slow_back', 1, 0.2, np.pi),
    ('walk_fwd', 2, -1.0, 0.0),
    ('walk_back', 2, -1.0, np.pi),
    ('jump', 17, -1.0, 0.0),
    ('turn_right', 1, 0.2, np.pi / 3),
    ('turn_left', 1, 0.2, -np.pi / 3),
    ('strafe_right', 1, 0.2, np.pi / 2),
    ('strafe_left', 1, 0.2, -np.pi / 2),
    ('stealth', 18, -1.0, -np.pi / 6),
]
out = {}
for name, m, sp, a in scen:
    abin = angle_bin_of(a)
    gi = gmap.get((int(m), round(float(sp), 2), abin))
    if gi is None:
        print(f'{name:12s} NO GROUP mode={m} speed={sp} bin={abin}', flush=True)
        out[name] = {'note': 'no router'}
        continue
    c = dict(mode=m, speed=sp, mdir=[float(np.cos(a)), float(np.sin(a)), 0.0], fdir=[1.0, 0.0, 0.0])
    out[name] = {}
    for seed in [0, 1, 2]:
        r = run(c, gi, 1000, seed)
        out[name][f'seed{seed}'] = r
        print(f'{name:12s} seed{seed} done={r["completed"]} h_min={r["h_min"]} vx={r["vx"]} vy={r["vy"]} path={r["path"]} yaw={r["yaw_change"]}', flush=True)
json.dump(out, open(f'{odir}/eval_battery_v21.json', 'w'), indent=1)

sched = [('idle', 5), ('walk_fwd', 10), ('idle', 5), ('slow_back', 10), ('turn_left', 8), ('slow_fwd', 10), ('jump', 5), ('idle', 5)]
def run_switch(sched, seed=4):
    import mujoco
    env.reset(); rng = np.random.default_rng(seed)
    env.data.qpos[2] += float(rng.normal(0, 0.005))
    env.data.qpos[env.body_qpos_adr] += rng.normal(0, 0.01, env.num_body).astype(np.float32)
    env.data.qvel[:] = rng.normal(0, 0.02, env.model.nv).astype(np.float32)
    mujoco.mj_forward(env.model, env.data)
    env._reset_history(); env._fill_history_from_state()
    heights, vxs, vys = [], [], []
    sc_prev = None
    for name, secs in sched:
        m, sp, a = next(x for x in scen if x[0] == name)[1:4]
        gi = gmap.get((int(m), round(float(sp), 2), angle_bin_of(a)))
        if gi is None:
            return dict(fall_step=len(heights), note=f'no group {name}')
        c = dict(mode=m, speed=sp, mdir=[float(np.cos(a)), float(np.sin(a)), 0.0], fdir=[1.0, 0.0, 0.0])
        for t in range(int(secs * 50)):
            prop = hist_to_proprio(env._get_sonic_history())
            x = np.concatenate([((prop - pmean) / pstd), feat_for(c)]).astype(np.float32)
            with torch.no_grad():
                sc = nets[gi](torch.from_numpy(x[None]).cuda())[0].cpu().numpy().astype(np.float32)
            if sc_prev is not None:
                sc = 0.3 * sc_prev + 0.7 * sc
            sc_prev = sc
            phi = float(np.arctan2(sc[0], sc[1]))
            b = int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)
            tok = protos[gi][b]
            obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
            v = env._get_base_linear_velocity()
            vxs.append(float(v[0])); vys.append(float(v[1]))
            heights.append(float(env.data.qpos[2]))
            if terminated:
                return dict(fall_step=len(heights) - 1, h_min=round(float(min(heights)), 3),
                            vx=round(float(np.mean(vxs)), 3), vy=round(float(np.mean(vys)), 3))
    return dict(fall_step=None, h_min=round(float(min(heights)), 3), vx=round(float(np.mean(vxs)), 3), vy=round(float(np.mean(vys)), 3))
sw = run_switch(sched)
print('switch:', sw, flush=True)
json.dump({'switch': sw, 'schedule': sched}, open(f'{odir}/eval_switch_v21.json', 'w'), indent=1)
print('done')