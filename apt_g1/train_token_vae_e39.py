"""E39: dual (direction + speed) disentangled DirSpeedPhaseTokenVAE.

E38 (dir-disentangled VAE + higher speed incentive) only got vx 0.399 (+8%
vs E37) while straightness dropped 0.97->0.81 and disp fell 21.55->19.4m:
direction disentanglement is NOT the key to the fast bin (0.535).

E39 hypothesis: SPEED is also entangled in z — the policy cannot freely reach
a fast gait because the "fast" region of z carries speed-related side effects
(the same failure mode E36 had for direction, now for speed).

E39 keeps the SAME DirSpeedPhaseTokenVAE architecture (so vae.pt loads into
apt_flat_env.py unchanged via strict=False), but trains with TWO adversarial
heads:
  - dir_head:   z -> 8 direction bins, encoder minimizes its accuracy (E37)
  - speed_head: z -> 3 speed bins,  encoder minimizes its accuracy (NEW)
The decoder is then forced to use psi_bin for direction AND v_bin for speed;
z only encodes gait style/residual. Both heads are discarded after training.

Loss: rec + beta*kl - adv_dir*CE(dir_head(z), db) - adv_spd*CE(speed_head(z), vb)

Saved: vae.pt (state_dict only), pca.npz, z_walk.npy, vbin_meta.json,
dbin_meta.json, meta.json (adds dir_head_acc + speed_head_acc + adv weights).

【2026-09-05 性能补丁】训练语义严格不变（batch 组成 / 90-10 切分 / 对抗头 3 步
循环 / 损失与指标计算式 / 随机种子 / epoch 数 / 模型结构 / 保存路径均未改动），
仅加速小 kernel 发射路径：
  1) 全显存数据：数据集整体常驻 GPU，DataLoader 改 num_workers=0、pin_memory=False
     （GPU 张量不能进 worker 子进程；shuffle 随机序列与种子不变 → batch 索引不变）
  2) torch.compile：VAE 主模型 + 两个对抗头（--compile-mode none 关闭）
  3) fused 优化器：AdamW/Adam 加 fused kernel（--no-fused-opt 关闭）
旧行为逃生旗标：`--cpu-data --compile-mode none --no-fused-opt`
测试安全旗标：`--epochs`（缩短训期）/ `--out-dir`（输出重定向沙箱，避免覆盖生产
vae.pt），默认值与原硬编码一致。
另增 [SPEED] 每 epoch 墙钟/吞吐日志与训练结束总结行，不影响任何数值。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DirSpeedPhaseTokenVAE(nn.Module):
    """Identical architecture to train_token_vae_e37.DirSpeedPhaseTokenVAE."""

    def __init__(self, token_dim: int = 64, window: int = 10, latent_dim: int = 16,
                 hidden_dim: int = 256, phase_dim: int = 2,
                 n_vbins: int = 3, n_dbins: int = 8):
        super().__init__()
        self.token_dim = token_dim
        self.window = window
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        self.n_vbins = n_vbins
        self.n_dbins = n_dbins
        self.speed_embed = nn.Embedding(n_vbins, 8)
        self.dir_embed = nn.Embedding(n_dbins, 8)
        flat = token_dim * window
        self.encoder = nn.Sequential(
            nn.Linear(flat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + phase_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.reshape(x.shape[0], -1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, phase: torch.Tensor,
               vb: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
        se = self.speed_embed(vb)
        de = self.dir_embed(db)
        return self.decoder(torch.cat([z, phase, se, de], dim=-1))

    def forward(self, x: torch.Tensor, phase: torch.Tensor,
                vb: torch.Tensor, db: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, phase, vb, db)
        return recon, mu, logvar, z


def build_windows(tok: np.ndarray, window: int) -> np.ndarray:
    n, d = tok.shape
    pad = np.zeros((window - 1, d), dtype=np.float32)
    ext = np.vstack([pad, tok])
    return np.stack([ext[i:i + window].reshape(-1) for i in range(n)])


def walk_phase_rate(tok: np.ndarray, mode: np.ndarray):
    walk = tok[mode == 2]
    pmean = walk.mean(0)
    c = walk - pmean
    cov = c.T @ c / len(c)
    ev, V = np.linalg.eigh(cov)
    V2 = V[:, ::-1][:, :2].astype(np.float32)
    proj = (tok - pmean) @ V2
    phi = np.arctan2(proj[:, 1], proj[:, 0]).astype(np.float32)
    rate = np.zeros(len(tok), dtype=np.float32)
    dphi = np.diff(phi)
    dphi = np.mod(dphi + np.pi, 2.0 * np.pi) - np.pi
    rate[1:] = np.abs(dphi)
    rate[0] = rate[1]
    return pmean.astype(np.float32), V2, rate, phi


def _build_optimizer(cls, params, use_fused: bool, **kw):
    """构造优化器；use_fused 时启用 fused kernel，老 torch 不支持则回退普通构造。"""
    if not use_fused:
        return cls(params, **kw)
    try:
        return cls(params, fused=torch.cuda.is_available(), **kw)
    except TypeError:
        print("[SPEED] fused optimizer unsupported by this torch -> eager fallback", flush=True)
        return cls(params, **kw)


def main():
    ap = argparse.ArgumentParser(
        description="E39 dual-disentangled token VAE (perf patch 2026-09-05)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--data-on-gpu", dest="data_on_gpu", action="store_true",
                   default=True, help="数据集常驻显存（默认开启）")
    g.add_argument("--cpu-data", dest="data_on_gpu", action="store_false",
                   help="数据留在 CPU（旧行为逃生旗标）")
    ap.add_argument("--compile-mode", choices=["none", "default", "reduce-overhead"],
                    default="default", help="torch.compile 模式；none 关闭编译")
    ap.add_argument("--no-fused-opt", action="store_true",
                    help="关闭 fused 优化器（旧行为逃生旗标）")
    ap.add_argument("--epochs", type=int, default=30,
                    help="训练 epoch 数（默认 30，与原硬编码一致）")
    ap.add_argument("--out-dir", type=str, default="outputs/token_vae_e39",
                    help="输出目录（默认 outputs/token_vae_e39；测试可重定向沙箱避免覆盖生产 vae.pt）")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    adv_dir = 3.0
    adv_spd = 3.0
    data_dir = "data/exp_all3"
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    tok = np.load(os.path.join(data_dir, "token.npy")).astype(np.float32)
    mode = np.load(os.path.join(data_dir, "mode.npy"))
    angle_bin = np.load(os.path.join(data_dir, "angle_bin.npy")).astype(np.int64)
    print("tokens", tok.shape, tok.min(), tok.max())

    pmean, V2, rate, phi = walk_phase_rate(tok, mode)
    rw = rate[mode == 2]
    edges = np.quantile(rw, [1.0 / 3.0, 2.0 / 3.0])
    vb = np.clip(np.digitize(rate, edges), 0, 2).astype(np.int64)
    db = angle_bin
    print("v-bin counts", np.bincount(vb))
    print("d-bin counts", np.bincount(db, minlength=8))
    np.savez(os.path.join(out_dir, "pca.npz"), pmean=pmean, V2=V2,
             rate=float(np.abs(np.diff(phi[mode == 2])).mean()))
    with open(os.path.join(out_dir, "vbin_meta.json"), "w") as f:
        json.dump({"n_bins": 3, "edges": [float(e) for e in edges],
                   "bin_counts": [int(c) for c in np.bincount(vb)]}, f, indent=1)
    with open(os.path.join(out_dir, "dbin_meta.json"), "w") as f:
        json.dump({"n_bins": 8,
                   "bin_counts": [int(c) for c in np.bincount(db, minlength=8)],
                   "bin4_is_forward": True}, f, indent=1)

    phase2 = np.stack([np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)
    window, latent_dim, hidden = 10, 16, 256
    x = build_windows(tok, window)
    y = tok
    n = len(tok)
    ntr = int(n * 0.9)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr_idx, va_idx = perm[:ntr], perm[ntr:]
    # 性能补丁：数据整体搬显存（GPU 张量不能被 worker 子进程 pickle，
    # 故 num_workers=0/pin_memory=False；shuffle 随机序列与种子不变 → batch 索引不变）
    data_on_gpu = args.data_on_gpu
    if data_on_gpu and not torch.cuda.is_available():
        print("[SPEED] CUDA unavailable -> keep dataset on CPU", flush=True)
        data_on_gpu = False

    def _dev(t: torch.Tensor) -> torch.Tensor:
        return t.cuda() if data_on_gpu else t

    ds_tr = TensorDataset(_dev(torch.from_numpy(x[tr_idx])), _dev(torch.from_numpy(y[tr_idx])),
                          _dev(torch.from_numpy(phase2[tr_idx])), _dev(torch.from_numpy(vb[tr_idx])),
                          _dev(torch.from_numpy(db[tr_idx])))
    ds_va = TensorDataset(_dev(torch.from_numpy(x[va_idx])), _dev(torch.from_numpy(y[va_idx])),
                          _dev(torch.from_numpy(phase2[va_idx])), _dev(torch.from_numpy(vb[va_idx])),
                          _dev(torch.from_numpy(db[va_idx])))
    if data_on_gpu:
        dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=0, pin_memory=False)
        dl_va = DataLoader(ds_va, batch_size=1024, num_workers=0, pin_memory=False)
        print("[SPEED] dataset on GPU (num_workers=0, pin_memory=False)", flush=True)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=1024, num_workers=4, pin_memory=True)

    model = DirSpeedPhaseTokenVAE(window=window, latent_dim=latent_dim,
                                  hidden_dim=hidden, n_vbins=3, n_dbins=8).cuda()
    # adversarial heads: z -> 8 (direction) and z -> 3 (speed), discarded after training
    dir_head = nn.Sequential(
        nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 8),
    ).cuda()
    speed_head = nn.Sequential(
        nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 3),
    ).cuda()
    # 性能补丁：torch.compile 只包装训练循环里的前向调用（fwd_*），
    # 保存 state_dict / 最终 encode 统计仍用原始模块，vae.pt 格式完全不变
    fwd_model, fwd_dir_head, fwd_speed_head = model, dir_head, speed_head
    if args.compile_mode != "none":
        _ckw = {"mode": args.compile_mode} if args.compile_mode != "default" else {}
        fwd_model = torch.compile(model, **_ckw)
        fwd_dir_head = torch.compile(dir_head, **_ckw)
        fwd_speed_head = torch.compile(speed_head, **_ckw)
        print(f"[SPEED] torch.compile enabled (mode={args.compile_mode})", flush=True)
    opt = _build_optimizer(torch.optim.AdamW, model.parameters(),
                           use_fused=not args.no_fused_opt, lr=1e-3, weight_decay=1e-5)
    dir_opt = _build_optimizer(torch.optim.Adam, dir_head.parameters(),
                               use_fused=not args.no_fused_opt, lr=1e-3)
    speed_opt = _build_optimizer(torch.optim.Adam, speed_head.parameters(),
                                 use_fused=not args.no_fused_opt, lr=1e-3)
    if not args.no_fused_opt and torch.cuda.is_available():
        print("[SPEED] fused optimizers enabled (AdamW/Adam fused=True)", flush=True)
    # class-balanced adversarial CE for both heads (bin 4 = 70% of dir data;
    # speed bins ~1/3 each but weights keep the head honest if skewed)
    dcounts = np.bincount(db, minlength=8).astype(np.float32)
    dcls_w = (dcounts.sum() / (8.0 * dcounts + 1e-6))
    dcls_w = torch.from_numpy(dcls_w).cuda()
    vcounts = np.bincount(vb, minlength=3).astype(np.float32)
    vcls_w = (vcounts.sum() / (3.0 * vcounts + 1e-6))
    vcls_w = torch.from_numpy(vcls_w).cuda()
    beta = 0.1
    epochs = args.epochs
    best = float("inf")
    ce = nn.functional.cross_entropy
    t_train_start = time.perf_counter()
    total_train_samples = 0
    for ep in range(epochs):
        model.train()
        ep_t0 = time.perf_counter()
        tot, cnt, dir_acc, dir_cnt, spd_acc, spd_cnt = 0.0, 0, 0.0, 0, 0.0, 0
        n_iter = 0
        for xb, yb, pb, vbb, dbb in dl_tr:
            xb, yb, pb, vbb, dbb = (t.cuda() for t in (xb, yb, pb, vbb, dbb))
            # (1) train both heads (several steps) on detached z
            with torch.no_grad():
                mu0, lv0 = model.encode(xb)
                z0 = model.reparameterize(mu0, lv0).detach()
            for _ in range(3):
                dloss = ce(fwd_dir_head(z0), dbb, weight=dcls_w)
                dir_opt.zero_grad(); dloss.backward(); dir_opt.step()
                sloss = ce(fwd_speed_head(z0), vbb, weight=vcls_w)
                speed_opt.zero_grad(); sloss.backward(); speed_opt.step()
            # (2) train VAE: recon + KL - adv_dir*CE(dir) - adv_spd*CE(speed)
            recon, mu, logvar, z = fwd_model(xb, pb, vbb, dbb)
            rec = nn.functional.mse_loss(recon, yb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            adv_d = ce(fwd_dir_head(z), dbb, weight=dcls_w)
            adv_s = ce(fwd_speed_head(z), vbb, weight=vcls_w)
            loss = rec + beta * kl - adv_dir * adv_d - adv_spd * adv_s
            opt.zero_grad(); loss.backward(); opt.step()
            tot += rec.item() * len(xb); cnt += len(xb)
            with torch.no_grad():
                dir_acc += (fwd_dir_head(z).argmax(1) == dbb).float().sum().item()
                dir_cnt += len(xb)
                spd_acc += (fwd_speed_head(z).argmax(1) == vbb).float().sum().item()
                spd_cnt += len(xb)
            n_iter += 1
        model.eval()
        va_rec, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb, pb, vbb, dbb in dl_va:
                xb, yb, pb, vbb, dbb = (t.cuda() for t in (xb, yb, pb, vbb, dbb))
                recon, mu, logvar, _ = fwd_model(xb, pb, vbb, dbb)
                va_rec += nn.functional.mse_loss(recon, yb).item() * len(xb)
                va_n += len(xb)
        va_mse = va_rec / va_n
        ep_dt = time.perf_counter() - ep_t0
        total_train_samples += cnt
        print(f"ep {ep+1}/{epochs} tr_rec_mse={tot/cnt:.5f} va_mse={va_mse:.5f} "
              f"kl={kl.item():.4f} dir_acc={dir_acc/dir_cnt:.3f} "
              f"spd_acc={spd_acc/spd_cnt:.3f}", flush=True)
        print(f"[SPEED] epoch={ep+1}/{epochs} wall={ep_dt:.1f}s iters={n_iter} "
              f"it_s={n_iter / ep_dt:.2f} throughput={cnt / ep_dt:.0f} samples/s", flush=True)
        if va_mse < best:
            best = va_mse
            torch.save(model.state_dict(), os.path.join(out_dir, "vae.pt"))

    _total_dt = time.perf_counter() - t_train_start
    print(f"[SPEED] total={_total_dt:.1f}s epoch_avg={_total_dt / epochs:.2f}s "
          f"throughput={total_train_samples / _total_dt:.0f}samples/s", flush=True)

    with torch.no_grad():
        idx = np.where(mode == 2)[0]
        xw = torch.from_numpy(x[idx]).cuda()
        pw = torch.from_numpy(phase2[idx]).cuda()
        zw = model.encode(xw)[0].cpu().numpy()
    np.save(os.path.join(out_dir, "z_walk.npy"), zw.mean(0).astype(np.float32))
    print("val recon MAE:", float(np.sqrt(best)))
    # final in-game adversarial accs (random baselines: dir 1/8, speed 1/3)
    with torch.no_grad():
        mu_all, _ = model.encode(torch.from_numpy(x[va_idx]).cuda())
        fin_dir = (dir_head(mu_all).argmax(1).cpu() == torch.from_numpy(db[va_idx])).float().mean().item()
        fin_spd = (speed_head(mu_all).argmax(1).cpu() == torch.from_numpy(vb[va_idx])).float().mean().item()
    print(f"final dir_head acc (want ~0.125): {fin_dir:.3f}")
    print(f"final speed_head acc (want ~0.333): {fin_spd:.3f}")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"window": window, "latent_dim": latent_dim, "hidden": hidden,
                   "token_dim": 64, "phase_dim": 2, "n_vbins": 3, "n_dbins": 8,
                   "adv_dir": adv_dir, "adv_spd": adv_spd, "val_mse": best,
                   "val_mae": float(np.sqrt(best)),
                   "dir_head_acc": fin_dir, "speed_head_acc": fin_spd}, f, indent=1)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
