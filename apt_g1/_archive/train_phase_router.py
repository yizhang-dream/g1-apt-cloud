"""Phase-regression router: circular gait phase from token PCA; regress (sin,cos) from proprio; eval closed loop."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

D = '/home/cvgluser/ros2_data/apt_g1/data/exp1'
proprio = np.load(D + '/proprio.npy'); cmd = np.load(D + '/cmd.npy'); token = np.load(D + '/token.npy')
mode = np.load(D + '/mode.npy'); speed = np.load(D + '/speed.npy'); modes_list = np.load(D + '/meta_modes.npy')
mdir = cmd[:, 4:7]
n = len(proprio)
val = np.zeros(n, dtype=bool); val[15606:17938] = True; val[18722:20308] = True
tr = ~val

def dsign(x):
    return 1 if x > 1e-6 else (-1 if x < -1e-6 else 0)

groups = {}
for i in range(n):
    if tr[i]:
        g = (int(mode[i]), round(float(speed[i]), 2), dsign(float(mdir[i, 0])))
        groups.setdefault(g, []).append(i)
groups = {g: np.array(v) for g, v in groups.items()}
print('groups:', {str(k): len(v) for k, v in groups.items()})

tr_idx = np.where(tr)[0]
pmean = proprio[tr_idx].mean(0, keepdims=True).astype(np.float32)
pstd = proprio[tr_idx].std(0, keepdims=True).astype(np.float32) + 1e-6
Pmat = ((proprio - pmean) / pstd).astype(np.float32)
X = np.concatenate([Pmat, cmd.astype(np.float32)], axis=1).astype(np.float32)

class PhaseNet(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, 2))
    def forward(self, x):
        return self.net(x)

B = 40
routers = {}
protos = {}
meta = {}
for gi, (g, rows) in enumerate(groups.items()):
    m, sp, ds = g
    if len(rows) < 400:
        continue
    T = token[rows]
    mu = T.mean(0)
    Tc = T - mu
    _, _, Vt = np.linalg.svd(Tc, full_matrices=False)
    V2 = Vt[:2]  # 2 x 64
    proj = Tc @ V2.T  # T x 2
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    # bins
    bi = np.floor((phi + np.pi) / (2 * np.pi) * B).astype(int) % B
    proto = np.zeros((B, 64), dtype=np.float32)
    cnt = np.zeros(B, dtype=np.float32)
    for k in range(len(rows)):
        proto[bi[k]] += T[k]
        cnt[bi[k]] += 1
    proto = np.clip(np.round((proto / np.maximum(cnt, 1)[:, None]) * 16) / 16, -1, 1)
    y = np.stack([np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)
    ntr = int(len(rows) * 0.8)
    ds = TensorDataset(torch.from_numpy(X[rows[:ntr]]), torch.from_numpy(y[:ntr]))
    ld = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    net = PhaseNet(930 + 13).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    best = 1e9
    for ep in range(50):
        net.train(); tl = 0.0; tb = 0
        for x, yy in ld:
            x = x.cuda(non_blocking=True); yy = yy.cuda(non_blocking=True)
            opt.zero_grad(); loss = lossf(net(x), yy); loss.backward(); opt.step()
            tl += loss.item() * len(yy); tb += len(yy)
        net.eval()
        with torch.no_grad():
            pv = net(torch.from_numpy(X[rows[ntr:]]).cuda()).cpu().numpy()
        err = float(np.mean(1 - (pv[:, 0] * y[ntr:, 0] + pv[:, 1] * y[ntr:, 1])))
        if err < best:
            best = err
        if (ep + 1) % 10 == 0:
            print(f'  group {g} ep {ep+1} loss {tl/tb:.4f} ang_err {err:.4f}', flush=True)
    routers[gi] = net
    protos[gi] = proto
    meta[gi] = dict(group=list(g), rows=len(rows), mu=mu.tolist(), V2=V2.tolist(), n_bins=B)
    torch.save(net.state_dict(), f'/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_g{gi}.pt')
    print('group', g, 'best ang err', round(best, 4), flush=True)
np.savez('/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_norm.npz', pmean=pmean, pstd=pstd)
json.dump(meta, open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/phase_meta.json', 'w'), indent=1)

# eval
sys.path.insert(0, '/home/cvgluser/ros2_data')
sys.path.insert(0, '/home/cvgluser/ros2_data/apt_g1')
sys.path.insert(0, '/home/cvgluser/ros2_data/GR00T-WholeBodyControl')
from apt_g1.envs.mujoco_g1_flat_env import MujocoG1FlatEnv
from eval_distill import NoQuantDecoder, hist_to_proprio
env = MujocoG1FlatEnv(NoQuantDecoder('/home/cvgluser/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx'),
                      '/home/cvgluser/ros2_data/GR00T-WholeBodyControl', use_elastic_band=False, stand_only=True)
env.command = np.zeros(3, dtype=np.float32)
scen = {
    'idle': dict(mode=0, speed=-1.0, mdir=[0, 0, 0], fdir=[1, 0, 0]),
    'slow_walk': dict(mode=1, speed=0.2, mdir=[1, 0, 0], fdir=[1, 0, 0]),
    'walk': dict(mode=2, speed=-1.0, mdir=[1, 0, 0], fdir=[1, 0, 0]),
    'jump': dict(mode=17, speed=-1.0, mdir=[1, 0, 0], fdir=[1, 0, 0]),
}
gmap = {}
for gi, md in meta.items():
    g = tuple(md['group'])
    for name, c in scen.items():
        if int(c['mode']) == g[0] and abs(c['speed'] - g[1]) < 1e-6 and dsign(c['mdir'][0]) == g[2]:
            gmap[name] = gi
print('gmap', gmap)
out = {}
for name, c in scen.items():
    gi = gmap.get(name)
    if gi is None:
        print(name, 'no router'); continue
    net = routers[gi].eval()
    md = meta[gi]; proto = protos[gi]
    oh = np.zeros(len(modes_list), dtype=np.float32); oh[int(np.where(modes_list == int(c['mode']))[0][0])] = 1
    feat = np.concatenate([oh, np.array(c['mdir'], dtype=np.float32), np.array(c['fdir'], dtype=np.float32),
                           np.array([c['speed'], -1.0, 1.0], dtype=np.float32)]).astype(np.float32)
    env.reset()
    heights, vxs = [], []
    fall = None
    for t in range(600):
        prop = hist_to_proprio(env._get_sonic_history())
        x = np.concatenate([((prop - pmean.ravel()) / pstd.ravel()), feat]).astype(np.float32)
        with torch.no_grad():
            sc = net(torch.from_numpy(x[None]).cuda())[0].cpu().numpy()
        phi = float(np.arctan2(sc[0], sc[1]))
        b = int(np.floor((phi + np.pi) / (2 * np.pi) * B) % B)
        tok = proto[b]
        obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
        heights.append(float(env.data.qpos[2])); vxs.append(float(env._get_base_linear_velocity()[0]))
        if terminated:
            fall = t; break
    out[name] = dict(fall_step=fall, h_min=round(float(min(heights)), 3), vx=round(float(np.mean(vxs)), 3),
                     h_mean=round(float(np.mean(heights)), 3))
    print(f'{name:10s} fall={fall} h_min={out[name]["h_min"]} vx={out[name]["vx"]}', flush=True)
json.dump(out, open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/eval_phase_router.json', 'w'), indent=1)
print('done')