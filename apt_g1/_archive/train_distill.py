"""Distill (command + proprio) -> SONIC token regression.

Data: apt_g1/data/exp1/{proprio,cmd,token,mode,speed,meta_modes}.npy
Val split: backward-walk phase + final slow-walk phase (contiguous, driver-time aligned).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self, d_in, d_out=64, hidden=1024, layers=2, drop=0.1):
        super().__init__()
        seq = [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop)]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop)]
        seq += [nn.Linear(hidden, d_out)]
        self.net = nn.Sequential(*seq)
    def forward(self, x):
        return self.net(x)

class GRUEncoder(nn.Module):
    def __init__(self, d_cmd, frame_dim=93, hidden=256, layers=2, d_out=64, drop=0.1):
        super().__init__()
        self.gru = nn.GRU(frame_dim, hidden, layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(hidden + d_cmd, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, d_out))
    def forward(self, proprio, cmd):
        out, _ = self.gru(proprio)
        return self.head(torch.cat([out[:, -1], cmd], dim=-1))

def metrics(pred, target):
    pred = np.asarray(pred); target = np.asarray(target)
    mse = float(((pred - target) ** 2).mean())
    q_pred = np.round(pred * 16) / 16
    per_dim = float((q_pred == target).mean())
    full = float(np.all(q_pred == target, axis=1).mean())
    return mse, np.sqrt(mse), per_dim, full

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/home/cvgluser/ros2_data/apt_g1/data/exp1')
    ap.add_argument('--out-dir', default='/home/cvgluser/ros2_data/apt_g1/outputs/distill')
    ap.add_argument('--model', choices=['mlp', 'gru'], required=True)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    D = args.data_dir
    proprio = np.load(D + '/proprio.npy')
    cmd = np.load(D + '/cmd.npy')
    token = np.load(D + '/token.npy')
    print('data', proprio.shape, cmd.shape, token.shape)

    n = len(proprio)
    # contiguous val: backward phase + final slow-walk phase (driver-time aligned)
    val_mask = np.zeros(n, dtype=bool)
    val_mask[15606:17938] = True   # backward slow walk + trailing idle
    val_mask[18722:20308] = True   # final slow walk + trailing idle
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    print('train', len(tr_idx), 'val', len(va_idx))

    pmean = proprio[tr_idx].mean(0, keepdims=True).astype(np.float32)
    pstd = proprio[tr_idx].std(0, keepdims=True).astype(np.float32) + 1e-6
    Xp = (proprio - pmean) / pstd
    Xc = cmd.astype(np.float32)
    Y = token.astype(np.float32)

    Xp_tr = torch.from_numpy(Xp[tr_idx]); Xc_tr = torch.from_numpy(Xc[tr_idx]); Y_tr = torch.from_numpy(Y[tr_idx])
    Xp_va = torch.from_numpy(Xp[va_idx]); Xc_va = torch.from_numpy(Xc[va_idx]); Y_va = torch.from_numpy(Y[va_idx])
    tr_ds = TensorDataset(Xp_tr, Xc_tr, Y_tr)
    va_ds = TensorDataset(Xp_va, Xc_va, Y_va)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    if args.model == 'mlp':
        model = MLP(930 + Xc.shape[1]).cuda()
    else:
        model = GRUEncoder(Xc.shape[1], frame_dim=93).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()

    best = None
    patience = 12
    wait = 0
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl = 0.0; tb = 0
        for xp, xc, y in tr_ld:
            xp = xp.cuda(non_blocking=True); xc = xc.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            if args.model == 'gru':
                xp = xp.view(-1, 10, 93)
            opt.zero_grad()
            out = model(xp, xc) if args.model == 'gru' else model(torch.cat([xp, xc], dim=-1))
            loss = lossf(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y); tb += len(y)
        sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xp, xc, y in va_ld:
                xp = xp.cuda(non_blocking=True); xc = xc.cuda(non_blocking=True)
                if args.model == 'gru':
                    xp = xp.view(-1, 10, 93)
                out = model(xp, xc) if args.model == 'gru' else model(torch.cat([xp, xc], dim=-1))
                preds.append(out.cpu().numpy())
        pred = np.concatenate(preds)
        mse, rmse, per_dim, full = metrics(pred, Y[va_idx])
        history.append(dict(ep=ep + 1, train_mse=tl / tb, val_mse=mse, val_rmse=rmse, per_dim=per_dim, full=full))
        if best is None or mse < best['val_mse']:
            best = dict(ep=ep + 1, val_mse=mse, val_rmse=rmse, per_dim=per_dim, full=full)
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{args.out_dir}/model_{args.model}.pt')
            np.savez(f'{args.out_dir}/norm_{args.model}.npz', pmean=pmean, pstd=pstd)
            wait = 0
        else:
            wait += 1
        if (ep + 1) % 5 == 0 or wait >= patience:
            print(f'ep {ep+1} train_mse {tl/tb:.5f} val_mse {mse:.5f} rmse {rmse:.4f} per_dim {per_dim:.3f} full {full:.3f} (best {best["val_mse"]:.5f})', flush=True)
        if wait >= patience:
            print('early stop at', ep + 1)
            break
    print('best', best)
    json.dump({'best': best, 'history': history, 'args': vars(args)}, open(f'{args.out_dir}/metrics_{args.model}.json', 'w'), indent=1)
    print('time_s', round(time.time() - t0, 1))

if __name__ == '__main__':
    main()