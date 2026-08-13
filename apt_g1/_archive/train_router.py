"""Phase-router distillation v2: per command-group period/prototypes/router + closed-loop eval."""
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
gid_of = np.full(n, -1, dtype=int)
for i in range(n):
    if tr[i]:
        g = (int(mode[i]), round(float(speed[i]), 2), dsign(float(mdir[i, 0])))
        if g not in groups:
            groups[g] = []
        groups[g].append(i)
        gid_of[i] = len(groups) - 1
print('groups:', {str(k): len(v) for k, v in sorted(groups.items(), key=lambda x: x[1][0])})

def period_of(rows):
    toks = token[rows]
    L = len(toks)
    best = (0.0, 1)
    for P in range(20, 201):
        if L - P < 200:
            continue
        a = toks[:L-P]; b = toks[P:]
        c = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        if c > best[0]:
            best = (c, P)
    return best

class Router(nn.Module):
    def __init__(self, d_in, n_out, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, n_out))
    def forward(self, x):
        return self.net(x)

tr_idx = np.where(tr)[0]
pmean = proprio[tr_idx].mean(0, keepdims=True).astype(np.float32)
pstd = proprio[tr_idx].std(0, keepdims=True).astype(np.float32) + 1e-6
Pmat = ((proprio - pmean) / pstd).astype(np.float32)
X = np.concatenate([Pmat, cmd.astype(np.float32)], axis=1).astype(np.float32)

routers = {}
protos = {}
periods = {}
for gi, (g, rows) in enumerate(sorted(groups.items(), key=lambda x: x[1][0])):
    rows = np.array(rows)
    if len(rows) < 400:
        continue
    m, sp, ds = g
    corr, P = period_of(rows)
    periods[gi] = int(P)
    print('group', g, 'rows', len(rows), 'bestP', P, 'corr', round(corr, 3), flush=True)
    if corr < 0.15:
        print('  low periodicity, using P=1')
        P = 1
    # phase labels: within each contiguous block, index mod P
    labels = np.zeros(len(rows), dtype=np.int64)
    blk = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
    for b in blk:
        idx = np.searchsorted(rows, b)
        for k in range(len(b)):
            labels[idx[k]] = k % P
    protos[gi] = np.zeros((P, 64), dtype=np.float32)
    cnt = np.zeros(P, dtype=np.float32)
    for k in range(len(rows)):
        protos[gi][labels[k]] += token[rows[k]]
        cnt[labels[k]] += 1
    protos[gi] = np.clip(np.round((protos[gi] / np.maximum(cnt, 1)[:, None]) * 16) / 16, -1, 1)
    # router: train on 80% rows (time-ordered), val 20%
    ntr = int(len(rows) * 0.8)
    tr_rows, va_rows = rows[:ntr], rows[ntr:]
    ds = TensorDataset(torch.from_numpy(X[tr_rows]), torch.from_numpy(labels[:ntr]))
    ld = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    router = Router(930 + 13, P).cuda()
    opt = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss()
    best_acc = 0
    for ep in range(40):
        router.train(); tl = 0.0; tb = 0
        for x, yb in ld:
            x = x.cuda(non_blocking=True); yb = yb.cuda(non_blocking=True)
            opt.zero_grad(); loss = lossf(router(x), yb); loss.backward(); opt.step()
            tl += loss.item() * len(yb); tb += len(yb)
        router.eval()
        with torch.no_grad():
            pred = np.concatenate([router(x.cuda(non_blocking=True)).cpu().numpy() for x, yb in ld])
        acc = float((pred.argmax(1) == labels[:ntr]).mean())
        if len(va_rows):
            with torch.no_grad():
                vp = router(torch.from_numpy(X[va_rows]).cuda()).cpu().numpy()
            vacc = float((vp.argmax(1) == labels[ntr:]).mean())
        else:
            vacc = acc
        if vacc > best_acc:
            best_acc = vacc
        if (ep + 1) % 10 == 0:
            print(f'  ep {ep+1} loss {tl/tb:.4f} acc {acc:.3f} val_acc {vacc:.3f}', flush=True)
    routers[gi] = router
    torch.save(router.state_dict(), f'/home/cvgluser/ros2_data/apt_g1/outputs/distill/router_g{gi}.pt')
    print('  best val acc', round(best_acc, 3), flush=True)

np.savez('/home/cvgluser/ros2_data/apt_g1/outputs/distill/router_norm.npz', pmean=pmean, pstd=pstd)
json.dump({'periods': {str(k): v for k, v in periods.items()},
           'groups': {str(k): list(g) for k, g in enumerate(sorted(groups.items(), key=lambda x: x[1][0]))}},
          open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/router_meta.json', 'w'), indent=1)
torch.save({gi: r.state_dict() for gi, r in routers.items()}, '/home/cvgluser/ros2_data/apt_g1/outputs/distill/routers_all.pt')
print('saved routers for groups', list(routers.keys()))

# ---- closed-loop eval ----
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
# map scenario to group index
gmap = {}
sorted_groups = sorted(groups.items(), key=lambda x: x[1][0])
for gi, (g, rows) in enumerate(sorted_groups):
    m, sp, ds = g
    for name, c in scen.items():
        if int(c['mode']) == m and abs(c['speed'] - sp) < 1e-6 and dsign(c['mdir'][0]) == ds:
            gmap[name] = gi
print('gmap', gmap)
out = {}
for name, c in scen.items():
    gi = gmap.get(name)
    if gi is None or gi not in routers:
        out[name] = {'note': 'no router', 'gi': gi}
        print(name, 'no router', gi, flush=True)
        continue
    router = routers[gi].eval()
    P = periods[gi]
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
            logits = router(torch.from_numpy(x[None]).cuda())[0]
            b = int(logits.argmax().item())
        tok = protos[gi][b]
        obs, reward, terminated, info = env.step({'token': tok, 'aux': np.zeros(12, dtype=np.float32)})
        heights.append(float(env.data.qpos[2])); vxs.append(float(env._get_base_linear_velocity()[0]))
        if terminated:
            fall = t; break
    out[name] = dict(fall_step=fall, h_min=round(float(min(heights)), 3), vx=round(float(np.mean(vxs)), 3),
                     h_mean=round(float(np.mean(heights)), 3))
    print(f'{name:10s} fall={fall} h_min={out[name]["h_min"]} vx={out[name]["vx"]}', flush=True)
json.dump(out, open('/home/cvgluser/ros2_data/apt_g1/outputs/distill/eval_router.json', 'w'), indent=1)
print('done')