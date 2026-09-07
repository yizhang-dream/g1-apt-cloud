"""D046 v2: E39 双解耦 VAE 的 v2 fork——+mode(SONIC 子集) 条件轴 + UL 正交轴。

对 train_token_vae_e39.py（E39 canonical，md5 56676924e3dd73a0f8d0ba5474a42545）
的最小 diff（逐行可审计）：
  1. DirSpeedPhaseTokenVAE 增可选 mode_embed(n_modes,8) / ul_embed(3,8)，与
     speed/dir embedding 同维(8)拼接进 decoder 条件；n_modes=0 且 use_ul=False 时
     模块不创建，state_dict/前向与原版逐位一致（向后兼容铁律，--parity-check）。
  2. 可选 ul_head（z->3，对抗 CE，adv_ul 默认 3.0，类平衡）；训练后即弃；
     owner 更正（2026-09-07）后 head acc 仅为描述性统计，PASS 判据走条件通路
     可控性探针（probe_vae_v2.py J2'/J5/J6）。
  3. 数据通路：--inputs-dir 指 build_b4lite_vae_inputs_v2.py 产物
     {token,mode_id,ul,angle_bin,segment_bounds,window_keep_mask}.npy；
     窗口按 segment_bounds 切（不跨段，段首 window-1 帧不成窗——v1 同款）；
     window_keep_mask 实现 intermediate 材料 0.3 权重（窗口级确定性过滤）；
     dev 固定用 <inputs-dir>/dev/（池构建时冻结的演员桶划分，无随机帧划分）。
  4. phase/v-bin 沿 E39 canonical：walk_phase_rate 投影基取 --walk-mode-idx
     （v2 = WALK 的 embed idx）帧，v-bin 三分位自派生（dev 复用 train edges）。
  5. CLI 增 --inputs-dir/--use-ul/--ul-labels/--walk-mode-idx/--adv-ul/
     --ckpt-every/--beta/--parity-check；不带 --inputs-dir 时数据路径与行为
     = 原版（data/exp_all3，随机 90/10 帧划分，无条件轴）。

Loss: rec + beta*kl - adv_dir*CE(dir_head(z),db) - adv_spd*CE(speed_head(z),vb)
      [- adv_ul*CE(ul_head(z),ub) when --use-ul]

Usage (server, lab-ts 3060, venv_isaac):
  python train_token_vae_e39_v2.py --inputs-dir .../vae_inputs_v2 --use-ul \
      --epochs 100 --ckpt-every 10 --out-dir .../vae_v2/run1
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

from train_token_vae_e39 import walk_phase_rate  # E39 canonical 投影机制复用


def phi_rate_from(tok: np.ndarray, pmean: np.ndarray,
                  V2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用（train 冻结的）phase 投影基对任意 split 求 phi/rate。

    与 walk_phase_rate 内部公式逐位一致；dev/tfam 必须复用 train 投影基——
    phase 是条件方案的一部分，per-split 自算 PCA 属条件口径漂移（D046v2 修正：
    dev 子集可能不含 WALK 帧，per-split PCA 会得到空集 NaN）。
    """
    proj = (tok - pmean) @ V2
    phi = np.arctan2(proj[:, 1], proj[:, 0]).astype(np.float32)
    rate = np.zeros(len(tok), dtype=np.float32)
    dphi = np.diff(phi)
    dphi = np.mod(dphi + np.pi, 2.0 * np.pi) - np.pi
    rate[1:] = np.abs(dphi)
    rate[0] = rate[1]
    return rate, phi


class DirSpeedPhaseTokenVAE(nn.Module):
    """E39 架构 + 可选 mode/ul 条件 embedding（缺省关闭 = 与 E39 逐位一致）。"""

    def __init__(self, token_dim: int = 64, window: int = 10, latent_dim: int = 16,
                 hidden_dim: int = 256, phase_dim: int = 2,
                 n_vbins: int = 3, n_dbins: int = 8,
                 n_modes: int = 0, use_ul: bool = False):
        super().__init__()
        self.token_dim = token_dim
        self.window = window
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        self.n_vbins = n_vbins
        self.n_dbins = n_dbins
        self.n_modes = n_modes
        self.use_ul = use_ul
        self.speed_embed = nn.Embedding(n_vbins, 8)
        self.dir_embed = nn.Embedding(n_dbins, 8)
        # --- D046v2 diff: 可选条件轴（缺省不创建模块，保证零 diff 兼容） ---
        self.mode_embed = nn.Embedding(n_modes, 8) if n_modes > 0 else None
        self.ul_embed = nn.Embedding(3, 8) if use_ul else None
        cond_dim = phase_dim + 16 + (8 if self.mode_embed is not None else 0) \
            + (8 if self.ul_embed is not None else 0)
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
            nn.Linear(latent_dim + cond_dim, hidden_dim),
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
               vb: torch.Tensor, db: torch.Tensor,
               mb: torch.Tensor | None = None,
               ub: torch.Tensor | None = None) -> torch.Tensor:
        parts = [z, phase, self.speed_embed(vb), self.dir_embed(db)]
        if self.mode_embed is not None and mb is not None:
            parts.append(self.mode_embed(mb))
        if self.ul_embed is not None and ub is not None:
            parts.append(self.ul_embed(ub))
        return self.decoder(torch.cat(parts, dim=-1))

    def forward(self, x: torch.Tensor, phase: torch.Tensor,
                vb: torch.Tensor, db: torch.Tensor,
                mb: torch.Tensor | None = None,
                ub: torch.Tensor | None = None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, phase, vb, db, mb, ub)
        return recon, mu, logvar, z


def build_windows_bounded(tok: np.ndarray, bounds: np.ndarray,
                          window: int) -> tuple[np.ndarray, np.ndarray]:
    """段界窗口（不跨段）：返回 (x[N,window*d], y[N,d])，y=窗末帧 token。"""
    xs, ys = [], []
    d = tok.shape[1]
    for a, b in bounds:
        n = int(b - a)
        if n < window:
            continue
        seg = tok[a:b]
        sw = np.lib.stride_tricks.sliding_window_view(seg, window, axis=0)
        xs.append(sw.transpose(0, 2, 1).reshape(n - window + 1, d * window))
        ys.append(seg[window - 1:])
    return (np.concatenate(xs).astype(np.float32),
            np.concatenate(ys).astype(np.float32))


def per_frame_labels(arr: np.ndarray, bounds: np.ndarray,
                     window: int) -> np.ndarray:
    """逐窗标签 = 窗末帧标签（窗口枚举与 build_windows_bounded 一致）。"""
    outs = []
    for a, b in bounds:
        n = int(b - a)
        if n < window:
            continue
        outs.append(arr[a + window - 1:b])
    return np.concatenate(outs)


def build_windows_orig(tok: np.ndarray, window: int) -> np.ndarray:
    """原版 build_windows（全序列滑动窗，首部零填充）——无 --inputs-dir 时行为。"""
    n, d = tok.shape
    pad = np.zeros((window - 1, d), dtype=np.float32)
    ext = np.vstack([pad, tok])
    return np.stack([ext[i:i + window].reshape(-1) for i in range(n)])


def _build_optimizer(cls, params, **kw):
    return cls(params, **kw)


def parity_check() -> None:
    """零 UL/mode 前向对账：本类默认构造 vs E39 原类，逐位一致。"""
    import train_token_vae_e39 as e39
    torch.manual_seed(0)
    a = DirSpeedPhaseTokenVAE()
    sb = a.state_dict()
    torch.manual_seed(0)
    b = e39.DirSpeedPhaseTokenVAE()
    b.load_state_dict(sb, strict=True)
    x = torch.randn(4, 10, 64)
    ph = torch.randn(4, 2)
    vbv = torch.randint(0, 3, (4,))
    dbv = torch.randint(0, 8, (4,))
    torch.manual_seed(42)
    ra, ma, la, _ = a(x, ph, vbv, dbv)
    torch.manual_seed(42)
    rb, mb_, lb, _ = b(x, ph, vbv, dbv)
    assert torch.equal(ra, rb) and torch.equal(ma, mb_) and torch.equal(la, lb), \
        "parity FAIL: forward outputs differ"
    ka = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    kb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert ka == kb, "parity FAIL: state_dict keys/shapes differ"
    print("[parity] PASS: default ctor == E39 original (state_dict + forward bit-exact)")


def main():
    ap = argparse.ArgumentParser(
        description="D046 v2 token VAE (E39 + mode/UL conditional axes)")
    ap.add_argument("--inputs-dir", type=str, default=None,
                    help="vae_inputs_v2 目录；缺省 = 原版行为（data/exp_all3，无条件轴）")
    ap.add_argument("--use-ul", action="store_true", default=False,
                    help="启用 UL 正交轴（ul_embed + ul_head）")
    ap.add_argument("--ul-labels", type=str, default=None,
                    help="覆盖 ul.npy 路径（缺省 <inputs-dir>/ul.npy）")
    ap.add_argument("--walk-mode-idx", type=int, default=-1,
                    help="phase/v-bin 投影基的 mode embed idx（v2=WALK；缺省 -1 = 原版 mode==2）")
    ap.add_argument("--adv-ul", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ckpt-every", type=int, default=0, help="每 N epoch 存 ckpt（0=关）")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--data-on-gpu", action="store_true", default=False)
    ap.add_argument("--parity-check", action="store_true", default=False,
                    help="仅跑零 UL/mode 前向对账后退出")
    ap.add_argument("--out-dir", type=str, default="outputs/token_vae_e39")
    args = ap.parse_args()
    if args.parity_check:
        parity_check()
        return
    torch.manual_seed(0)
    np.random.seed(0)
    adv_dir = 3.0
    adv_spd = 3.0
    use_inputs = args.inputs_dir is not None
    use_ul = bool(args.use_ul) and use_inputs  # UL 轴仅 inputs 通路有效
    data_dir = args.inputs_dir if use_inputs else "data/exp_all3"
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    tok = np.load(os.path.join(data_dir, "token.npy")).astype(np.float32)
    angle_bin = np.load(os.path.join(data_dir, "angle_bin.npy")).astype(np.int64)
    if use_inputs:
        mode_id = np.load(os.path.join(data_dir, "mode_id.npy")).astype(np.int64)
        bounds = np.load(os.path.join(data_dir, "segment_bounds.npy"))
        ul = np.load(args.ul_labels or os.path.join(data_dir, "ul.npy")).astype(np.int64)
        km_path = os.path.join(data_dir, "window_keep_mask.npy")
        keep_mask = np.load(km_path) if os.path.isfile(km_path) else None
        n_modes = int(mode_id.max()) + 1
        walk_idx = args.walk_mode_idx if args.walk_mode_idx >= 0 else None
        assert len(mode_id) == len(tok) == len(ul) == len(angle_bin)
    else:
        mode_id = np.load(os.path.join(data_dir, "mode.npy"))
        bounds, ul, keep_mask = None, None, None
        n_modes, walk_idx = 0, None
        ul = np.zeros(len(tok), dtype=np.int64)
    print("tokens", tok.shape, tok.min(), tok.max())

    # phase/v-bin（E39 canonical：WALK 帧投影 + 三分位 rate bins）
    base_mode = ((mode_id == walk_idx).astype(int) * 2 if walk_idx is not None
                 else np.asarray(mode_id))
    pmean, V2, rate, phi = walk_phase_rate(tok, base_mode)
    edges = np.quantile(rate[base_mode == 2], [1.0 / 3.0, 2.0 / 3.0])
    vb = np.clip(np.digitize(rate, edges), 0, 2).astype(np.int64)
    db = angle_bin
    print("v-bin counts", np.bincount(vb, minlength=3))
    print("d-bin counts", np.bincount(db, minlength=8))
    if use_inputs:
        print("mode counts", np.bincount(mode_id, minlength=n_modes))
        print("ul counts", np.bincount(ul, minlength=3))
    np.savez(os.path.join(out_dir, "pca.npz"), pmean=pmean, V2=V2,
             rate=float(np.abs(np.diff(phi[base_mode == 2])).mean()))
    with open(os.path.join(out_dir, "vbin_meta.json"), "w") as f:
        json.dump({"n_bins": 3, "edges": [float(e) for e in edges],
                   "bin_counts": [int(c) for c in np.bincount(vb, minlength=3)],
                   "walk_base": "mode==2 (original)" if walk_idx is None
                                else f"embed_idx={walk_idx} (WALK)"}, f, indent=1)
    with open(os.path.join(out_dir, "dbin_meta.json"), "w") as f:
        json.dump({"n_bins": 8,
                   "bin_counts": [int(c) for c in np.bincount(db, minlength=8)]},
                  f, indent=1)

    window, latent_dim, hidden = 10, 16, 256
    phase2 = np.stack([np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)

    if use_inputs:
        x, y = build_windows_bounded(tok, bounds, window)
        n_drop = len(tok) - len(x)
        mb = per_frame_labels(mode_id, bounds, window)
        ub = per_frame_labels(ul, bounds, window)
        pb = per_frame_labels(phase2, bounds, window)
        vbw = per_frame_labels(vb, bounds, window)
        dbw = per_frame_labels(db, bounds, window)
        assert len(x) == len(mb) == len(ub) == len(pb) == len(vbw) == len(dbw)
        if keep_mask is not None:
            assert len(keep_mask) == len(x), \
                f"keep_mask {len(keep_mask)} != windows {len(x)}"
            x, y, pb, vbw, dbw, mb, ub = (v[keep_mask] for v in
                                          (x, y, pb, vbw, dbw, mb, ub))
        print(f"windows {len(x)} (start/cross-segment windows dropped: {n_drop}"
              f"{'; keep-mask applied' if keep_mask is not None else ''})")
        # dev = 冻结划分（inputs-dir/dev/），v-bin 复用 train edges
        dev_dir = os.path.join(data_dir, "dev")
        dtok = np.load(os.path.join(dev_dir, "token.npy")).astype(np.float32)
        dbnd = np.load(os.path.join(dev_dir, "segment_bounds.npy"))
        dmode = np.load(os.path.join(dev_dir, "mode_id.npy")).astype(np.int64)
        # --ul-labels 只覆盖 train 侧；dev 恒读 inputs-dir/dev/ul.npy，
        # 否则 train 长度的标签会在 dev 段界上切出错误标签且不报错
        dul = (np.load(os.path.join(dev_dir, "ul.npy")).astype(np.int64)
               ) if use_ul else np.zeros(len(dtok), dtype=np.int64)
        dangle = np.load(os.path.join(dev_dir, "angle_bin.npy")).astype(np.int64)
        dx, dy = build_windows_bounded(dtok, dbnd, window)
        dmb = per_frame_labels(dmode, dbnd, window)
        dub = per_frame_labels(dul, dbnd, window)
        # dev 复用 train 的 phase 投影基（pmean/V2），不自算 PCA
        drate, dphi = phi_rate_from(dtok, pmean, V2)
        dphase2 = np.stack([np.sin(dphi), np.cos(dphi)], axis=1).astype(np.float32)
        dpb = per_frame_labels(dphase2, dbnd, window)
        dvbw = per_frame_labels(
            np.clip(np.digitize(drate, edges), 0, 2).astype(np.int64), dbnd, window)
        ddbw = per_frame_labels(dangle, dbnd, window)
        va_tensors = (dx, dy, dpb, dvbw, ddbw, dmb, dub)
    else:
        # 原版路径：全序列滑动窗 + 随机 90/10 帧划分（E39 canonical 行为）
        y = tok
        x = build_windows_orig(tok, window)
        n = len(tok)
        ntr = int(n * 0.9)
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        tr_idx, va_idx = perm[:ntr], perm[ntr:]
        mb = np.zeros(n, dtype=np.int64)
        ub = np.zeros(n, dtype=np.int64)
        pb, vbw, dbw = phase2, vb, db
        va_tensors = (x[va_idx], y[va_idx], pb[va_idx], vbw[va_idx],
                      dbw[va_idx], mb[va_idx], ub[va_idx])

    data_on_gpu = args.data_on_gpu and torch.cuda.is_available()

    def _dev(t: torch.Tensor) -> torch.Tensor:
        return t.cuda() if data_on_gpu else t

    def mk_ds(*arrays, shuffle: bool):
        ts = [_dev(torch.from_numpy(np.ascontiguousarray(a))) for a in arrays]
        return DataLoader(TensorDataset(*ts), batch_size=512 if shuffle else 1024,
                          shuffle=shuffle, num_workers=0, pin_memory=False)

    if use_inputs:
        dl_tr = mk_ds(x, y, pb, vbw, dbw, mb, ub, shuffle=True)
    else:
        dl_tr = mk_ds(x[tr_idx], y[tr_idx], pb[tr_idx], vbw[tr_idx],
                      dbw[tr_idx], mb[tr_idx], ub[tr_idx], shuffle=True)
    dl_va = mk_ds(*va_tensors, shuffle=False)

    model = DirSpeedPhaseTokenVAE(window=window, latent_dim=latent_dim,
                                  hidden_dim=hidden, n_vbins=3, n_dbins=8,
                                  n_modes=n_modes, use_ul=use_ul).cuda()
    dir_head = nn.Sequential(
        nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 8)).cuda()
    speed_head = nn.Sequential(
        nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 3)).cuda()
    ul_head = (nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(),
                             nn.Linear(64, 3)).cuda()) if use_ul else None
    opt = _build_optimizer(torch.optim.AdamW, model.parameters(),
                           lr=1e-3, weight_decay=1e-5)
    dir_opt = _build_optimizer(torch.optim.Adam, dir_head.parameters(), lr=1e-3)
    speed_opt = _build_optimizer(torch.optim.Adam, speed_head.parameters(), lr=1e-3)
    ul_opt = (_build_optimizer(torch.optim.Adam, ul_head.parameters(), lr=1e-3)
              if use_ul else None)
    dcounts = np.bincount(np.asarray(dbw).reshape(-1), minlength=8).astype(np.float32)
    dcls_w = torch.from_numpy(dcounts.sum() / (8.0 * dcounts + 1e-6)).cuda()
    vcounts = np.bincount(np.asarray(vbw).reshape(-1), minlength=3).astype(np.float32)
    vcls_w = torch.from_numpy(vcounts.sum() / (3.0 * vcounts + 1e-6)).cuda()
    ucls_w = None
    if use_ul:
        ucounts = np.bincount(np.asarray(ub).reshape(-1), minlength=3).astype(np.float32)
        ucls_w = torch.from_numpy(ucounts.sum() / (3.0 * ucounts + 1e-6)).cuda()
    beta = args.beta
    epochs = args.epochs
    best = float("inf")
    ce = nn.functional.cross_entropy
    t_train_start = time.perf_counter()
    total_train_samples = 0
    curves = []
    for ep in range(epochs):
        model.train()
        ep_t0 = time.perf_counter()
        tot, cnt = 0.0, 0
        dir_acc = dir_cnt = spd_acc = spd_cnt = 0
        ul_acc = ul_cnt = 0
        n_iter = 0
        for batch in dl_tr:
            if use_ul:
                xb, yb, pb, vbb, dbb, mbb, ubb = (t.cuda() for t in batch)
            else:
                xb, yb, pb, vbb, dbb, mbb, _ = (t.cuda() for t in batch)
                ubb = None
            with torch.no_grad():
                mu0, lv0 = model.encode(xb)
                z0 = model.reparameterize(mu0, lv0).detach()
            for _ in range(3):
                dloss = ce(dir_head(z0), dbb, weight=dcls_w)
                dir_opt.zero_grad(); dloss.backward(); dir_opt.step()
                sloss = ce(speed_head(z0), vbb, weight=vcls_w)
                speed_opt.zero_grad(); sloss.backward(); speed_opt.step()
                if use_ul:
                    uloss = ce(ul_head(z0), ubb, weight=ucls_w)
                    ul_opt.zero_grad(); uloss.backward(); ul_opt.step()
            recon, mu, logvar, z = model(xb, pb, vbb, dbb, mbb, ubb)
            rec = nn.functional.mse_loss(recon, yb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            adv_d = ce(dir_head(z), dbb, weight=dcls_w)
            adv_s = ce(speed_head(z), vbb, weight=vcls_w)
            loss = rec + beta * kl - adv_dir * adv_d - adv_spd * adv_s
            if use_ul:
                adv_u = ce(ul_head(z), ubb, weight=ucls_w)
                loss = loss - args.adv_ul * adv_u
            opt.zero_grad(); loss.backward(); opt.step()
            tot += rec.item() * len(xb); cnt += len(xb)
            with torch.no_grad():
                dir_acc += (dir_head(z).argmax(1) == dbb).sum().item()
                dir_cnt += len(xb)
                spd_acc += (speed_head(z).argmax(1) == vbb).sum().item()
                spd_cnt += len(xb)
                if use_ul:
                    ul_acc += (ul_head(z).argmax(1) == ubb).sum().item()
                    ul_cnt += len(xb)
            n_iter += 1
        model.eval()
        va_rec, va_n = 0.0, 0
        with torch.no_grad():
            for batch in dl_va:
                if use_ul:
                    xb, yb, pb, vbb, dbb, mbb, ubb = (t.cuda() for t in batch)
                else:
                    xb, yb, pb, vbb, dbb, mbb, _ = (t.cuda() for t in batch)
                    ubb = None
                recon, mu, logvar, _ = model(xb, pb, vbb, dbb, mbb, ubb)
                va_rec += nn.functional.mse_loss(recon, yb).item() * len(xb)
                va_n += len(xb)
        va_mse = va_rec / va_n
        ep_dt = time.perf_counter() - ep_t0
        total_train_samples += cnt
        tr_rec = tot / cnt
        ul_str = f" ul_acc={ul_acc / max(ul_cnt, 1):.3f}" if use_ul else ""
        print(f"ep {ep+1}/{epochs} tr_rec_mse={tr_rec:.5f} va_mse={va_mse:.5f} "
              f"kl={kl.item():.4f} dir_acc={dir_acc/max(dir_cnt,1):.3f} "
              f"spd_acc={spd_acc/max(spd_cnt,1):.3f}{ul_str}", flush=True)
        print(f"[SPEED] epoch={ep+1}/{epochs} wall={ep_dt:.1f}s iters={n_iter} "
              f"throughput={cnt / ep_dt:.0f} samples/s", flush=True)
        curves.append({"epoch": ep + 1, "train_mse": tr_rec, "dev_mse": va_mse,
                       "dir_acc": dir_acc / max(dir_cnt, 1),
                       "spd_acc": spd_acc / max(spd_cnt, 1),
                       "ul_acc": (ul_acc / max(ul_cnt, 1)) if use_ul else None})
        if args.ckpt_every and (ep + 1) % args.ckpt_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"vae_ep{ep+1:03d}.pt"))
            heads = {"dir_head": dir_head.state_dict(),
                     "speed_head": speed_head.state_dict()}
            if use_ul:
                heads["ul_head"] = ul_head.state_dict()
            torch.save(heads, os.path.join(out_dir, f"heads_ep{ep+1:03d}.pt"))
        if va_mse < best:
            best = va_mse
            torch.save(model.state_dict(), os.path.join(out_dir, "vae.pt"))

    _total_dt = time.perf_counter() - t_train_start
    print(f"[SPEED] total={_total_dt:.1f}s epoch_avg={_total_dt / epochs:.2f}s "
          f"throughput={total_train_samples / _total_dt:.0f}samples/s", flush=True)

    # z_walk：WALK 条件窗的 mu 均值（v2 用 train 窗的 WALK 子集；原版 mode==2 同义）
    with torch.no_grad():
        if use_inputs:
            if walk_idx is None:
                raise ValueError(
                    "use_inputs 且 walk_idx=None：z_walk 无法定位 WALK 窗"
                    "（mb==None 恒 False 会静默落盘 NaN 均值），检查 --n-modes 配置")
            wmask = (mb == walk_idx)
        else:
            wmask = (np.asarray(mode_id) == 2)
        if not wmask.any():
            raise ValueError("z_walk: WALK 掩码为空，拒绝对空数组取均值落盘 NaN")
        xw = torch.from_numpy(x[wmask]).cuda()
        zw = model.encode(xw)[0].cpu().numpy()
    np.save(os.path.join(out_dir, "z_walk.npy"), zw.mean(0).astype(np.float32))
    # 实为 RMSE（sqrt of best val MSE）；json 键 val_mae 为历史口径，不改
    print("val recon RMSE (json key val_mae):", float(np.sqrt(best)))

    # 终态 head acc（描述性统计；owner 更正后不作为 PASS 判据）
    with torch.no_grad():
        accs = {"dir": [0.0, 0], "spd": [0.0, 0], "ul": [0.0, 0]}
        for batch in dl_va:
            if use_ul:
                xb, yb, pb, vbb, dbb, mbb, ubb = (t.cuda() for t in batch)
            else:
                xb, yb, pb, vbb, dbb, mbb, _ = (t.cuda() for t in batch)
                ubb = None
            mu, _ = model.encode(xb)
            accs["dir"][0] += (dir_head(mu).argmax(1) == dbb).sum().item()
            accs["spd"][0] += (speed_head(mu).argmax(1) == vbb).sum().item()
            accs["dir"][1] += len(xb); accs["spd"][1] += len(xb)
            if use_ul:
                accs["ul"][0] += (ul_head(mu).argmax(1) == ubb).sum().item()
                accs["ul"][1] += len(xb)
    fin_dir = accs["dir"][0] / max(accs["dir"][1], 1)
    fin_spd = accs["spd"][0] / max(accs["spd"][1], 1)
    fin_ul = (accs["ul"][0] / max(accs["ul"][1], 1)) if use_ul else None
    print(f"final dir_head acc (random 0.125, descriptive): {fin_dir:.3f}")
    print(f"final speed_head acc (random 0.333, descriptive): {fin_spd:.3f}")
    if use_ul:
        print(f"final ul_head acc (random 0.333, descriptive): {fin_ul:.3f}")
    meta = {"window": window, "latent_dim": latent_dim, "hidden": hidden,
            "token_dim": 64, "phase_dim": 2, "n_vbins": 3, "n_dbins": 8,
            "n_modes": n_modes, "use_ul": use_ul,
            "adv_dir": adv_dir, "adv_spd": adv_spd,
            "adv_ul": args.adv_ul if use_ul else None,
            "inputs_dir": args.inputs_dir,
            "walk_embed_idx": walk_idx,
            "final_snapshot": f"vae_ep{epochs:03d}.pt" if args.ckpt_every else "vae.pt",
            "val_mse": best, "val_mae": float(np.sqrt(best)),
            "dir_head_acc": fin_dir, "speed_head_acc": fin_spd,
            "ul_head_acc": fin_ul,
            "train_wall_sec": round(_total_dt, 1),
            "curves": curves}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
