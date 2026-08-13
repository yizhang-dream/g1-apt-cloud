"""v5: slow_fwd group = exp1 first slow-walk phase only (rows 1831..8250)."""
import json, os, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

D = '/home/cvgluser/ros2_data/apt_g1/data/exp_all'
proprio = np.load(D+'/proprio.npy'); cmd = np.load(D+'/cmd.npy'); token = np.load(D+'/token.npy')
mode = np.load(D+'/mode.npy'); speed = np.load(D+'/speed.npy'); ab = np.load(D+'/angle_bin.npy')
odir3 = '/home/cvgluser/ros2_data/apt_g1/outputs/distill_v3'
odir5 = '/home/cvgluser/ros2_data/apt_g1/outputs/distill_v5'
os.makedirs(odir5, exist_ok=True)
meta3 = json.load(open(odir3+'/phase_meta.json'))
norm = np.load(odir3+'/phase_norm.npz')
gi_slow = None
for gi, md in meta3.items():
    if tuple(md['group']) == (1, 0.2, 4):
        gi_slow = int(gi); break
rows_slow = np.where((mode == 1) & (np.abs(speed - 0.2) < 1e-6) & (ab == 4) & (np.arange(len(mode)) >= 1831) & (np.arange(len(mode)) < 8250))[0]
print('slow rows (phase1 only)', len(rows_slow))
T = token[rows_slow]
mu = T.mean(0); Tc = T - mu
_, _, Vt = np.linalg.svd(Tc, full_matrices=False)
V2 = Vt[:2]
proj = Tc @ V2.T
phi = np.arctan2(proj[:, 1], proj[:, 0])
B = 40
bi = np.floor((phi + np.pi) / (2 * np.pi) * B).astype(int) % B
proto = np.zeros((B, 64), dtype=np.float32); cnt = np.zeros(B, dtype=np.float32)
for k in range(len(rows_slow)):
    proto[bi[k]] += T[k]; cnt[bi[k]] += 1
proto = np.clip(np.round((proto / np.maximum(cnt, 1)[:, None]) * 16) / 16, -1, 1)
np.save(f'{odir5}/proto_g{gi_slow}.npy', proto)

class PhaseNet(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, 2))
    def forward(self, x): return self.net(x)

pmean = norm['pmean']; pstd = norm['pstd']
X = np.concatenate([((proprio - pmean) / pstd).astype(np.float32), cmd], axis=1).astype(np.float32)
y = np.stack([np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)
ntr = int(len(rows_slow) * 0.8)
ds = TensorDataset(torch.from_numpy(X[rows_slow[:ntr]]), torch.from_numpy(y[:ntr]))
ld = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
net = PhaseNet(930 + cmd.shape[1]).cuda()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
lossf = nn.MSELoss()
best = 1e9
for ep in range(60):
    net.train(); tl = 0.0; tb = 0
    for x, yy in ld:
        x = x.cuda(non_blocking=True); yy = yy.cuda(non_blocking=True)
        opt.zero_grad(); loss = lossf(net(x), yy); loss.backward(); opt.step()
        tl += loss.item() * len(yy); tb += len(yy)
    net.eval()
    with torch.no_grad():
        pv = net(torch.from_numpy(X[rows_slow[ntr:]]).cuda()).cpu().numpy()
    err = float(np.mean(1 - (pv[:, 0] * y[ntr:, 0] + pv[:, 1] * y[ntr:, 1])))
    if err < best:
        best = err
        torch.save(net.state_dict(), f'{odir5}/phase_g{gi_slow}.pt')
    if (ep + 1) % 10 == 0:
        print(f'ep {ep+1} loss {tl/tb:.5f} ang_err {err:.4f}', flush=True)
print('best', best)
meta5 = dict(meta3)
meta5[str(gi_slow)] = dict(group=[1, 0.2, 4], rows_all=int(len(rows_slow)), rows=int(len(rows_slow)),
                           mu=mu.tolist(), V2=V2.tolist(), n_bins=B, kept=1.0, note='exp1 phase1 only')
json.dump(meta5, open(f'{odir5}/phase_meta.json', 'w'), indent=1)
for gi, md in meta3.items():
    gi = int(gi)
    if gi == gi_slow:
        continue
    shutil.copy(f'{odir3}/phase_g{gi}.pt', f'{odir5}/phase_g{gi}.pt')
    shutil.copy(f'{odir3}/proto_g{gi}.npy', f'{odir5}/proto_g{gi}.npy')
shutil.copy(f'{odir3}/phase_norm.npz', f'{odir5}/phase_norm.npz')
print('v5 ready')