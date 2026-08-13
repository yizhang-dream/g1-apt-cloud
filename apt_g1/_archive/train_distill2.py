"""Distill variants: plain deep MLP, AR (prev-token) MLP, deep AR."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MLPDeep(nn.Module):
    def __init__(self, d_in, d_out=64, hidden=1024, layers=3, drop=0.1):
        super().__init__()
        seq = [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop)]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop)]
        seq += [nn.Linear(hidden, d_out)]
        self.net = nn.Sequential(*seq)
    def forward(self, x):
        return self.net(x)

def metrics(pred, target):
    pred = np.asarray(pred); target = np.asarray(target)
    mse = float(((pred - target) ** 2).mean())
    q = np.round(pred * 16) / 16
    return mse, float(np.sqrt(((pred - target) ** 2).mean())), float((q == target).mean()), float(np.all(q == target, axis=1).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['deep', 'ar', 'ar_deep'], required=True)
    ap.add_argument('--data-dir', default='/home/cvgluser/ros2_data/apt_g1/data/exp1')
    ap.add_argument('--out-dir', default='/home/cvgluser/ros2_data/apt_g1/outputs/distill')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=8e-4)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    D = args.data_dir
    proprio = np.load(D + '/proprio.npy'); cmd = np.load(D + '/cmd.npy'); token = np.load(D + '/token.npy')
    n = len(proprio)
    val = np.zeros(n, dtype=bool)
    val[15606:17938] = True; val[18722:20308] = True
    tr = ~val
    pmean = proprio[tr].mean(0, keepdims=True).astype(np.float32)
    pstd = proprio[tr].std(0, keepdims=True).astype(np.float32) + 1e-6
    Xp = (proprio - pmean) / pstd
    Xc = cmd.astype(np.float32)
    Y = token.astype(np.float32)
    prev = np.zeros_like(Y)
    prev[1:] = Y[:-1]

    use_ar = args.arch.startswith('ar')
    if use_ar:
        X = np.concatenate([Xp, Xc, prev], axis=1).astype(np.float32)
    else:
        X = np.concatenate([Xp, Xc], axis=1).astype(np.float32)

    layers = 3 if args.arch in ('ar_deep', 'deep') else 2
    model = MLPDeep(X.shape[1], hidden=1024, layers=layers).cuda()
    tr_ds = TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]))
    va_ds = TensorDataset(torch.from_numpy(X[val]), torch.from_numpy(Y[val]))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()
    best = None; wait = 0; t0 = time.time()
    for ep in range(args.epochs):
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
        mse, rmse, per_dim, full = metrics(pred, Y[val])
        if best is None or mse < best['val_mse']:
            best = dict(ep=ep + 1, val_mse=mse, val_rmse=rmse, per_dim=per_dim, full=full)
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{args.out_dir}/model_{args.arch}.pt')
            np.savez(f'{args.out_dir}/norm_{args.arch}.npz', pmean=pmean, pstd=pstd)
            wait = 0
        else:
            wait += 1
        if (ep + 1) % 5 == 0 or wait >= 12:
            print(f'{args.arch} ep {ep+1} train {tl/tb:.5f} val_mse {mse:.5f} rmse {rmse:.4f} per_dim {per_dim:.3f} full {full:.3f}', flush=True)
        if wait >= 12:
            print('early stop', ep + 1); break
    print(args.arch, 'best', best)
    json.dump({'best': best, 'args': vars(args)}, open(f'{args.out_dir}/metrics_{args.arch}.json', 'w'), indent=1)
    print('time_s', round(time.time() - t0, 1))

if __name__ == '__main__':
    main()