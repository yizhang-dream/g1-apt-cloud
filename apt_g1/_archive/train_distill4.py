"""Free-running (schedule sampling) fine-tune of ar_delta to fix exposure bias."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/home/cvgluser/ros2_data/apt_g1/data/exp1')
    ap.add_argument('--out-dir', default='/home/cvgluser/ros2_data/apt_g1/outputs/distill')
    ap.add_argument('--init', default='/home/cvgluser/ros2_data/apt_g1/outputs/distill/model_ar_delta.pt')
    ap.add_argument('--norm', default='/home/cvgluser/ros2_data/apt_g1/outputs/distill/norm_ar_delta.npz')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--p-teacher', type=float, default=0.3)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    D = args.data_dir
    proprio = np.load(D + '/proprio.npy'); cmd = np.load(D + '/cmd.npy'); token = np.load(D + '/token.npy')
    n = len(proprio)
    val = np.zeros(n, dtype=bool); val[15606:17938] = True; val[18722:20308] = True
    tr = ~val
    norm = np.load(args.norm)
    Xp = ((proprio - norm['pmean']) / norm['pstd']).astype(np.float32)
    Xc = cmd.astype(np.float32); Y = token.astype(np.float32)

    def blocks_for(mask):
        rows = np.where(mask)[0]
        blk = []
        i = 0
        while i < len(rows):
            j = i + 1
            while j < len(rows) and rows[j] == rows[j-1] + 1:
                j += 1
            if j - i >= 8:
                blk.append(rows[i:j])
            i = j
        return blk

    tr_blk = blocks_for(tr)
    va_blk = blocks_for(val)
    print('train blocks:', [len(b) for b in tr_blk], 'val blocks:', [len(b) for b in va_blk])

    from train_distill3 import MLPBlock
    model = MLPBlock(930 + 13 + 64, 64, layers=3).cuda()
    model.load_state_dict(torch.load(args.init, map_location='cuda'))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.MSELoss()

    def free_run(blocks, teacher_p):
        model.eval()
        tot = 0.0; cnt = 0
        with torch.no_grad():
            for b in blocks:
                prev = torch.from_numpy(Y[b[0]]).cuda()
                for k in range(1, len(b)):
                    i = b[k]
                    if teacher_p > 0 and np.random.rand() < teacher_p:
                        prev_in = torch.from_numpy(Y[i-1]).cuda()
                    else:
                        prev_in = prev
                    x = torch.from_numpy(np.concatenate([Xp[i], Xc[i]])[None]).cuda()
                    d = model(torch.cat([x, prev_in[None]], dim=-1))[0]
                    pred = prev_in + d
                    tot += float(((pred - torch.from_numpy(Y[i]).cuda()) ** 2).mean())
                    cnt += 1
                    prev = torch.clamp(torch.round(pred * 16) / 16, -1.0, 1.0)
        return tot / cnt

    t0 = time.time()
    for ep in range(args.epochs):
        np.random.shuffle(tr_blk)
        model.train()
        tl = 0.0; cnt = 0
        for b in tr_blk:
            prev = torch.from_numpy(Y[b[0]]).cuda()
            for k in range(1, len(b)):
                i = b[k]
                if np.random.rand() < args.p_teacher:
                    prev_in = torch.from_numpy(Y[i-1]).cuda()
                else:
                    prev_in = prev.detach()
                x = torch.from_numpy(np.concatenate([Xp[i], Xc[i]])[None]).cuda()
                d = model(torch.cat([x, prev_in[None]], dim=-1))[0]
                tgt = torch.from_numpy(Y[i]).cuda() - prev_in
                loss = lossf(d, tgt)
                loss.backward()
                tl += loss.item(); cnt += 1
                with torch.no_grad():
                    pred = prev_in + d
                    prev = torch.clamp(torch.round(pred * 16) / 16, -1.0, 1.0)
                if cnt % 64 == 0:
                    opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        vm = free_run(va_blk, 0.0)
        print(f'ep {ep+1} train {tl/cnt:.6f} val_free_run_mse {vm:.6f} time {round(time.time()-t0,1)}s', flush=True)
        torch.save(model.state_dict(), f'{args.out_dir}/model_ar_delta_fr.pt')
    print('done', round(time.time()-t0,1))

if __name__ == '__main__':
    main()