"""Relabel dataset with kNN-smoothed tokens, then train MLP. Eval closed loop."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

D = '/home/cvgluser/ros2_data/apt_g1/data/exp1'
proprio = np.load(D + '/proprio.npy'); cmd = np.load(D + '/cmd.npy'); token = np.load(D + '/token.npy')
mode = np.load(D + '/mode.npy'); speed = np.load(D + '/speed.npy')
mdir = cmd[:, 4:7]
n = len(proprio)
val = np.zeros(n, dtype=bool); val[15606:17938] = True; val[18722:20308] = True
tr = ~val
tr_idx = np.where(tr)[0]
pmean = proprio[tr_idx].mean(0, keepdims=True).astype(np.float32)
pstd = proprio[tr_idx].std(0, keepdims=True).astype(np.float32) + 1e-6
P = ((proprio - pmean) / pstd).astype(np.float32)
t0 = time.time()
# kNN-smoothed labels on train rows (k=5, within same mode+speed+dir)
Y2 = token.copy()
for m in np.unique(mode):
    for s in np.unique(speed[mode == m]):
        for sign in [-1, 0, 1]:
            sel = np.where((mode == m) & (np.abs(speed - s) < 1e-6))[0]
            if sign == 1:
                sel = sel[mdir[sel, 0] > 0]
            elif sign == -1:
                sel = sel[mdir[sel, 0] < 0]
            else:
                sel = sel[mdir[sel, 0] == 0]
            sel = sel[tr[sel]]
            if len(sel) < 6:
                continue
            C = P[sel]
            dots = C @ C.T
            # for each row, top-6 including self
            nb = np.argsort(-dots, axis=1)[:, :6]
            Y2[sel] = np.round(token[sel][nb].mean(axis=1) * 16) / 16
print('relabel done', round(time.time() - t0, 1), 's')

class MLP(nn.Module):
    def __init__(self, d_in, d_out=64, hidden=1024, layers=3, drop=0.15):
        super().__init__()
        seq = [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop)]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop)]
        seq += [nn.Linear(hidden, d_out)]
        self.net = nn.Sequential(*seq)
    def forward(self, x):
        return self.net(x)

X = np.concatenate([P, cmd.astype(np.float32)], axis=1).astype(np.float32)
model = MLP(930 + 13).cuda()
tr_ds = TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y2[tr]))
va_ds = TensorDataset(torch.from_numpy(X[val]), torch.from_numpy(Y2[val]))
tr_ld = DataLoader(tr_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
va_ld = DataLoader(va_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
lossf = nn.MSELoss()
best = None
for ep in range(60):
    model.train(); tl = 0.0; tb = 0
    for x, y in tr_ld:
        x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
        opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
        tl += loss.item() * len(y); tb += len(y)
    sched.step()
    model.eval(); preds = []
    with torch.no_grad():
        for x, y in va_ld:
            preds.append(model(x.cuda(non_blocking=True)).cpu().numpy())
    pred = np.concatenate(preds)
    mse = float(((pred - Y2[val]) ** 2).mean())
    q = np.round(pred * 16) / 16
    per = float((q == Y2[val]).mean()); full = float(np.all(q == Y2[val], axis=1).mean())
    if best is None or mse < best[1]:
        best = (ep, mse, per, full)
        torch.save(model.state_dict(), '/home/cvgluser/ros2_data/apt_g1/outputs/distill/model_knn_mlp.pt')
        np.savez('/home/cvgluser/ros2_data/apt_g1/outputs/distill/norm_knn_mlp.npz', pmean=pmean, pstd=pstd)
    if (ep + 1) % 10 == 0:
        print(f'ep {ep+1} mse {mse:.5f} per_dim {per:.3f} full {full:.3f}', flush=True)
print('best', best)