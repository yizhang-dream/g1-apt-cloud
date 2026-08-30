"""E37: direction-disentangled DirSpeedPhaseTokenVAE (clean fast bin).

E36 (E35 VAE + speed incentive anti-stop 0.4/progress 0.5) pushed vx 0.295->0.372
but re-introduced E31-style drift (straightness 0.93->0.69). Root cause: the
fast speed-bin's latent z still implicitly encodes heading — the E35 decoder's
psi_bin conditioning is not enough to make direction purely a command axis.

E37 keeps the SAME DirSpeedPhaseTokenVAE architecture (so the saved vae.pt loads
unchanged into apt_flat_env.py via strict=False), but trains it with an
adversarial direction-disentanglement loss: a direction classifier (dir_head)
is trained to predict psi_bin from z, while the encoder is trained to MINIMIZE
that classifier's ability (maximize its CE), so z becomes direction-invariant.
The decoder is then forced to use psi_bin for direction, and z only encodes
gait style/speed. dir_head is discarded after training (not saved).

Saved: vae.pt (DirSpeedPhaseTokenVAE state_dict only), pca.npz, z_walk.npy,
vbin_meta.json, dbin_meta.json, meta.json (adds dir_accuracy + adv weight).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DirSpeedPhaseTokenVAE(nn.Module):
    """Identical architecture to train_token_vae_e35.DirSpeedPhaseTokenVAE."""

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


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    adv_weight = 3.0
    data_dir = "data/exp_all3"
    out_dir = "outputs/token_vae_e37"
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
    ds_tr = TensorDataset(torch.from_numpy(x[tr_idx]), torch.from_numpy(y[tr_idx]),
                          torch.from_numpy(phase2[tr_idx]), torch.from_numpy(vb[tr_idx]),
                          torch.from_numpy(db[tr_idx]))
    ds_va = TensorDataset(torch.from_numpy(x[va_idx]), torch.from_numpy(y[va_idx]),
                          torch.from_numpy(phase2[va_idx]), torch.from_numpy(vb[va_idx]),
                          torch.from_numpy(db[va_idx]))
    dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=1024, num_workers=4, pin_memory=True)

    model = DirSpeedPhaseTokenVAE(window=window, latent_dim=latent_dim,
                                  hidden_dim=hidden, n_vbins=3, n_dbins=8).cuda()
    # adversarial direction classifier: z -> 8 logits (discarded after training)
    dir_head = nn.Sequential(
        nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 8),
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    dir_opt = torch.optim.Adam(dir_head.parameters(), lr=1e-3)
    # class-balanced adversarial CE (bin 4 = 70% of data; without weights the
    # head collapses to predicting the majority class and disentanglement fails)
    counts = np.bincount(db, minlength=8).astype(np.float32)
    cls_w = (counts.sum() / (8.0 * counts + 1e-6))
    cls_w = torch.from_numpy(cls_w).cuda()
    beta = 0.1
    epochs = 30
    best = float("inf")
    ce = nn.functional.cross_entropy
    for ep in range(epochs):
        model.train()
        tot, cnt, dir_acc, dir_cnt = 0.0, 0, 0.0, 0
        for xb, yb, pb, vbb, dbb in dl_tr:
            xb, yb, pb, vbb, dbb = (t.cuda() for t in (xb, yb, pb, vbb, dbb))
            # (1) train dir_head (several steps) to predict direction from detached z
            with torch.no_grad():
                mu0, lv0 = model.encode(xb)
                z0 = model.reparameterize(mu0, lv0).detach()
            for _ in range(3):
                dir_loss = ce(dir_head(z0), dbb, weight=cls_w)
                dir_opt.zero_grad(); dir_loss.backward(); dir_opt.step()
            # (2) train VAE: recon + KL - adv_weight * CE(dir_head(z), db)
            recon, mu, logvar, z = model(xb, pb, vbb, dbb)
            rec = nn.functional.mse_loss(recon, yb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            adv = ce(dir_head(z), dbb, weight=cls_w)
            loss = rec + beta * kl - adv_weight * adv
            opt.zero_grad(); loss.backward(); opt.step()
            tot += rec.item() * len(xb); cnt += len(xb)
            with torch.no_grad():
                dir_acc += (dir_head(z).argmax(1) == dbb).float().sum().item()
                dir_cnt += len(xb)
        model.eval()
        va_rec, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb, pb, vbb, dbb in dl_va:
                xb, yb, pb, vbb, dbb = (t.cuda() for t in (xb, yb, pb, vbb, dbb))
                recon, mu, logvar, _ = model(xb, pb, vbb, dbb)
                va_rec += nn.functional.mse_loss(recon, yb).item() * len(xb)
                va_n += len(xb)
        va_mse = va_rec / va_n
        print(f"ep {ep+1}/{epochs} tr_rec_mse={tot/cnt:.5f} va_mse={va_mse:.5f} "
              f"kl={kl.item():.4f} dir_acc={dir_acc/dir_cnt:.3f}", flush=True)
        if va_mse < best:
            best = va_mse
            torch.save(model.state_dict(), os.path.join(out_dir, "vae.pt"))

    with torch.no_grad():
        idx = np.where(mode == 2)[0]
        xw = torch.from_numpy(x[idx]).cuda()
        pw = torch.from_numpy(phase2[idx]).cuda()
        zw = model.encode(xw)[0].cpu().numpy()
    np.save(os.path.join(out_dir, "z_walk.npy"), zw.mean(0).astype(np.float32))
    print("val recon MAE:", float(np.sqrt(best)))
    # final disentanglement probe: random baseline = 1/8 = 0.125
    with torch.no_grad():
        mu_all, _ = model.encode(torch.from_numpy(x[va_idx]).cuda())
        fin_acc = (dir_head(mu_all).argmax(1).cpu() == torch.from_numpy(db[va_idx])).float().mean().item()
    print("final dir_head acc (want ~0.125):", fin_acc)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"window": window, "latent_dim": latent_dim, "hidden": hidden,
                   "token_dim": 64, "phase_dim": 2, "n_vbins": 3, "n_dbins": 8,
                   "adv_weight": adv_weight, "val_mse": best,
                   "val_mae": float(np.sqrt(best)), "dir_head_acc": fin_acc}, f, indent=1)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
