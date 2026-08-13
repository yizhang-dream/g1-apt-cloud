"""v2.3: density-based outlier filter per group before PCA/prototypes."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def load_csv(path, dtype, skip=0):
    with open(path) as f:
        lines = f.read().splitlines()
    rows = []
    for ln in lines[skip:]:
        if not ln.strip():
            continue
        parts = ln.split(',')
        if parts and parts[-1] == '':
            parts = parts[:-1]
        rows.append([float(x) for x in parts])
    return np.array(rows, dtype=dtype)

def build(pi_path, cmd_path, tok_path):
    PI = load_csv(pi_path, np.float32)
    CMD = load_csv(cmd_path, np.float64)
    tok = load_csv(tok_path, np.float64, skip=1)
    N, M = len(PI), len(CMD)
    j = np.floor(np.arange(N) * M / N).astype(int)
    cm = CMD[j]
    mode = cm[:, 7].astype(np.int32)
    mdir = cm[:, 8:11].astype(np.float32)
    fdir = cm[:, 11:14].astype(np.float32)
    speed = cm[:, 14].astype(np.float32)
    height = cm[:, 15].astype(np.float32)
    planner_en = cm[:, 5].astype(np.float32)
    return PI[:, 64:].copy(), PI[:, :64].copy(), mode, speed, mdir, fdir, height, planner_en

p1 = build('/tmp/exp1/policy_input.csv', '/tmp/exp1/commands.csv', '/tmp/exp1/logs/token_state.csv')
p2 = build('/tmp/exp2/policy_input.csv', '/tmp/exp2/commands.csv', '/tmp/exp2/logs/token_state.csv')
proprio = np.concatenate([p1[0], p2[0]]).astype(np.float32)
token = np.concatenate([p1[1], p2[1]]).astype(np.float32)
mode = np.concatenate([p1[2], p2[2]]).astype(np.int32)
speed = np.concatenate([p1[3], p2[3]]).astype(np.float32)
mdir = np.concatenate([p1[4], p2[4]]).astype(np.float32)
fdir = np.concatenate([p1[5], p2[5]]).astype(np.float32)
height = np.concatenate([p1[6], p2[6]]).astype(np.float32)
planner_en = np.concatenate([p1[7], p2[7]]).astype(np.float32)
modes = np.unique(mode)
mode_oh = np.zeros((len(mode), len(modes)), dtype=np.float32)
for k, m in enumerate(modes):
    mode_oh[:, k] = (mode == m)
cmd = np.concatenate([mode_oh, mdir, fdir, speed[:, None], height[:, None], planner_en[:, None]], axis=1).astype(np.float32)
angle = np.arctan2(mdir[:, 1], mdir[:, 0])
NB = 8
angle_bin = (np.floor((angle + np.pi) / (2 * np.pi) * NB).astype(int)) % NB
D = '/home/cvgluser/ros2_data/apt_g1/data/exp_all'
np.save(D + '/proprio.npy', proprio); np.save(D + '/cmd.npy', cmd); np.save(D + '/token.npy', token)
np.save(D + '/mode.npy', mode); np.save(D + '/speed.npy', speed); np.save(D + '/angle_bin.npy', angle_bin)
np.save(D + '/meta_modes.npy', modes)

class PhaseNet(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, 2))
    def forward(self, x):
        return self.net(x)

pmean = proprio.mean(0, keepdims=True).astype(np.float32)
pstd = proprio.std(0, keepdims=True).astype(np.float32) + 1e-6
P = ((proprio - pmean) / pstd).astype(np.float32)
X = np.concatenate([P, cmd], axis=1).astype(np.float32)
B = 40

groups = {}
for i in range(len(proprio)):
    g = (int(mode[i]), round(float(speed[i]), 2), int(angle_bin[i]))
    groups.setdefault(g, []).append(i)
groups = {g: np.array(v) for g, v in groups.items()}

def density_filter(rows, T, sample_step=4):
    S = rows[::sample_step]
    Ts = T[::sample_step]
    dmin = np.empty(len(rows), dtype=np.float32)
    chunk = 2000
    for k in range(0, len(rows), chunk):
        A = T[k:k+chunk]
        dots = A @ Ts.T
        # d2 = |a|^2 + |s|^2 - 2 dots
        d2 = (A ** 2).sum(1, keepdims=True) + (Ts ** 2).sum(1)[None, :] - 2 * dots
        np.maximum(d2, 0, out=d2)
        dmin[k:k+chunk] = np.sqrt(d2.min(1))
    thr = 2.5 * float(np.median(dmin))
    return rows[dmin <= thr], (dmin <= thr).mean()

odir = '/home/cvgluser/ros2_data/apt_g1/outputs/distill_v3'
os.makedirs(odir, exist_ok=True)
meta = {}
nets = {}
t0 = time.time()
for gi, (g, rows_all) in enumerate(groups.items()):
    if len(rows_all) < 100:
        continue
    rows_f, kept = density_filter(rows_all, token[rows_all])
    if len(rows_f) < 100:
        print('skip group', g, len(rows_all), 'after filter', len(rows_f))
        continue
    rows = rows_f
    T = token[rows]
    mu = T.mean(0)
    Tc = T - mu
    _, _, Vt = np.linalg.svd(Tc, full_matrices=False)
    V2 = Vt[:2]
    proj = Tc @ V2.T
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    bi = np.floor((phi + np.pi) / (2 * np.pi) * B).astype(int) % B
    proto = np.zeros((B, 64), dtype=np.float32)
    cnt = np.zeros(B, dtype=np.float32)
    for k in range(len(rows)):
        proto[bi[k]] += T[k]
        cnt[bi[k]] += 1
    proto = np.clip(np.round((proto / np.maximum(cnt, 1)[:, None]) * 16) / 16, -1, 1)
    np.save(f'{odir}/proto_g{gi}.npy', proto)
    y = np.stack([np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)
    ntr = int(len(rows) * 0.8)
    ds = TensorDataset(torch.from_numpy(X[rows[:ntr]]), torch.from_numpy(y[:ntr]))
    ld = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    net = PhaseNet(930 + cmd.shape[1]).cuda()
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
        if len(rows) - ntr >= 20:
            with torch.no_grad():
                pv = net(torch.from_numpy(X[rows[ntr:]]).cuda()).cpu().numpy()
            err = float(np.mean(1 - (pv[:, 0] * y[ntr:, 0] + pv[:, 1] * y[ntr:, 1])))
        else:
            err = float(tl / tb)
        if err < best:
            best = err
    nets[gi] = net
    meta[str(gi)] = dict(group=list(g), rows_all=int(len(rows_all)), rows=int(len(rows)),
                         mu=mu.tolist(), V2=V2.tolist(), n_bins=B, kept=float(kept))
    torch.save(net.state_dict(), f'{odir}/phase_g{gi}.pt')
    print(f'group {g} rows {len(rows)}/{len(rows_all)} kept {kept:.2f} ang_err {best:.4f}', flush=True)
np.savez(f'{odir}/phase_norm.npz', pmean=pmean, pstd=pstd)
json.dump(meta, open(f'{odir}/phase_meta.json', 'w'), indent=1)
torch.save({str(gi): net.state_dict() for gi, net in nets.items()}, f'{odir}/routers_all.pt')
print('done', len(nets), 'routers', round(time.time()-t0, 1), 's')