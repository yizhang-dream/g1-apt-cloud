"""Variant training: AR-delta MLP, transformer, transformer-AR-delta, with prev-noise + proprio aug."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MLPBlock(nn.Module):
    def __init__(self, d_in, d_out, hidden=1024, layers=3, drop=0.15):
        super().__init__()
        seq = [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop)]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop)]
        seq += [nn.Linear(hidden, d_out)]
        self.net = nn.Sequential(*seq)
    def forward(self, x):
        return self.net(x)

class TF(nn.Module):
    def __init__(self, d_cmd, frame_dim=93, d_model=256, nhead=4, layers=2, d_out=64, drop=0.15, ar=False):
        super().__init__()
        self.ar = ar
        self.frame_proj = nn.Linear(frame_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 10, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                         dropout=drop, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        d_in = d_model + d_cmd + (64 if ar else 0)
        self.head = MLPBlock(d_in, d_out, hidden=512, layers=2, drop=drop)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, proprio, cmd, prev=None):
        x = self.frame_proj(proprio) + self.pos
        x = self.encoder(x)
        h = x[:, -1]
        h = self.ln(h)
        if self.ar:
            h = torch.cat([h, cmd, prev], dim=-1)
        else:
            h = torch.cat([h, cmd], dim=-1)
        return self.head(h)

def metrics(pred, target):
    pred = np.asarray(pred); target = np.asarray(target)
    mse = float(((pred - target) ** 2).mean())
    q = np.round(pred * 16) / 16
    return mse, float(np.sqrt(((pred - target) ** 2).mean())), float((q == target).mean()), float(np.all(q == target, axis=1).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['ar_delta', 'tf', 'tf_ar'], required=True)
    ap.add_argument('--data-dir', default='/home/cvgluser/ros2_data/apt_g1/data/exp1')
    ap.add_argument('--out-dir', default='/home/cvgluser/ros2_data/apt_g1/outputs/distill')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    D = args.data_dir
    proprio = np.load(D + '/proprio.npy'); cmd = np.load(D + '/cmd.npy'); token = np.load(D + '/token.npy')
    n = len(proprio)
    val = np.zeros(n, dtype=bool); val[15606:17938] = True; val[18722:20308] = True
    tr = ~val
    pmean = proprio[tr].mean(0, keepdims=True).astype(np.float32)
    pstd = proprio[tr].std(0, keepdims=True).astype(np.float32) + 1e-6
    Xp = ((proprio - pmean) / pstd).astype(np.float32)
    Xc = cmd.astype(np.float32)
    Y = token.astype(np.float32)
    prev = np.zeros_like(Y); prev[1:] = Y[:-1]
    delta = np.zeros_like(Y); delta[1:] = Y[1:] - Y[:-1]
    use_ar = args.arch.startswith('tf_ar') or args.arch == 'ar_delta'
    pred_delta = args.arch == 'ar_delta' or args.arch == 'tf_ar'
    target = delta if pred_delta else Y

    def make_model():
        if args.arch == 'ar_delta':
            return MLPBlock(930 + 13 + 64, 64, layers=3)
        return TF(13, ar=use_ar)

    model = make_model().cuda()
    tr_ds = TensorDataset(torch.from_numpy(Xp[tr]), torch.from_numpy(Xc[tr]),
                          torch.from_numpy(prev[tr]), torch.from_numpy(target[tr]))
    va_ds = TensorDataset(torch.from_numpy(Xp[val]), torch.from_numpy(Xc[val]),
                          torch.from_numpy(prev[val]), torch.from_numpy(target[val]))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()
    best = None; wait = 0; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tl = 0.0; tb = 0
        for xp, xc, pv, y in tr_ld:
            xp = xp.cuda(non_blocking=True); xc = xc.cuda(non_blocking=True)
            pv = pv.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            if args.arch == 'ar_delta':
                pv_n = pv + torch.randn_like(pv) * 0.03
                xin = torch.cat([xp, xc, pv_n], dim=-1)
                out = model(xin)
            else:
                xp = xp.view(-1, 10, 93)
                xp = xp + torch.randn_like(xp) * 0.05
                out = model(xp, xc, pv + torch.randn_like(pv) * 0.03 if use_ar else None)
            opt.zero_grad(); loss = lossf(out, y); loss.backward(); opt.step()
            tl += loss.item() * len(y); tb += len(y)
        sched.step()
        model.eval(); preds = []
        with torch.no_grad():
            for xp, xc, pv, y in va_ld:
                xp = xp.cuda(non_blocking=True); xc = xc.cuda(non_blocking=True); pv = pv.cuda(non_blocking=True)
                if args.arch == 'ar_delta':
                    out = model(torch.cat([xp, xc, pv], dim=-1))
                else:
                    xp = xp.view(-1, 10, 93)
                    out = model(xp, xc, pv if use_ar else None)
                preds.append(out.cpu().numpy())
        out_arr = np.concatenate(preds)
        if pred_delta:
            pred = out_arr + Y[val]
        else:
            pred = out_arr
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
            print(f'{args.arch} ep {ep+1} train {tl/tb:.6f} val_mse {mse:.6f} rmse {rmse:.4f} per_dim {per_dim:.3f} full {full:.3f}', flush=True)
        if wait >= 12:
            print('early stop', ep + 1); break
    print(args.arch, 'best', best)
    json.dump({'best': best, 'args': vars(args)}, open(f'{args.out_dir}/metrics_{args.arch}.json', 'w'), indent=1)
    print('time_s', round(time.time() - t0, 1))

if __name__ == '__main__':
    main()