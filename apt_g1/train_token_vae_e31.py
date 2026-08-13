"""E31: speed-conditioned phase token VAE over SONIC FSQ tokens.

Extends the E27 PhaseTokenVAE decoder with a commanded-speed condition so the
latent manifold itself encodes gait speed (the E28-E30 finding: the frozen
walk manifold has a ~0.35 m/s fidelity ceiling; a speed axis should let RL
pick faster skills instead of a single slow one).

Decoder: D(z, sin(phi), cos(phi), v_bin) -> token, where v_bin is a learned
embedding of one of NBIN speed bins derived from the walk data's measured
phase-advance rate (rad/step): slow / medium / fast.

Encoder stays as E27: causal window (10 x 64) -> z (16-d).
Saves: pca.npz, z_walk.npy, meta.json (E27-compatible) + vbin_meta.json
(bin edges) + vae.pt with the new decoder.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SpeedPhaseTokenVAE(nn.Module):
    """E27 PhaseTokenVAE + speed-bin conditioning on the decoder."""

    def __init__(self, token_dim: int = 64, window: int = 10, latent_dim: int = 16,
                 hidden_dim: int = 256, phase_dim: int = 2, n_bins: int = 3):
        super().__init__()
        self.token_dim = token_dim
        self.window = window
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        self.n_bins = n_bins
        self.speed_embed = nn.Embedding(n_bins, 8)
        flat = token_dim * window
        self.encoder = nn.Sequential(
            nn.Linear(flat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        # decoder: (z + phase 2 + speed 8) -> token
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + phase_dim + 8, hidden_dim),
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

    def decode(self, z: torch.Tensor, phase: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
        se = self.speed_embed(vb)
        return self.decoder(torch.cat([z, phase, se], dim=-1))

    def forward(self, x: torch.Tensor, phase: torch.Tensor, vb: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, phase, vb)
        return recon, mu, logvar, z


def build_windows(tok: np.ndarray, window: int) -> np.ndarray:
    n, d = tok.shape
    pad = np.zeros((window - 1, d), dtype=np.float32)
    ext = np.vstack([pad, tok])
    return np.stack([ext[i:i + window].reshape(-1) for i in range(n)])


def walk_phase_rate(tok: np.ndarray, mode: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pca_mean, V2, per-row phase-rate) for walk-mode rows."""
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
    data_dir = "data/exp_all3"
    out_dir = "outputs/token_vae_e31"
    os.makedirs(out_dir, exist_ok=True)

    tok = np.load(os.path.join(data_dir, "token.npy")).astype(np.float32)
    mode = np.load(os.path.join(data_dir, "mode.npy"))
    print("tokens", tok.shape, tok.min(), tok.max())

    pmean, V2, rate, phi = walk_phase_rate(tok, mode)
    # speed bins from the walk phase-rate percentiles (rad/step)
    rw = rate[mode == 2]
    edges = np.quantile(rw, [1.0 / 3.0, 2.0 / 3.0])
    print("rate quartiles", [round(float(q), 4) for q in np.quantile(rw, [0.0, 0.33, 0.67, 1.0])])
    vb = np.clip(np.digitize(rate, edges), 0, 2).astype(np.int64)
    print("bin counts", np.bincount(vb))
    np.savez(
        os.path.join(out_dir, "pca.npz"), pmean=pmean, V2=V2,
        rate=float(np.abs(np.diff(phi[mode == 2])).mean()),
    )
    with open(os.path.join(out_dir, "vbin_meta.json"), "w") as f:
        json.dump({"n_bins": 3, "edges": [float(e) for e in edges],
                   "bin_counts": [int(c) for c in np.bincount(vb)]}, f, indent=1)

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
                          torch.from_numpy(phase2[tr_idx]), torch.from_numpy(vb[tr_idx]))
    ds_va = TensorDataset(torch.from_numpy(x[va_idx]), torch.from_numpy(y[va_idx]),
                          torch.from_numpy(phase2[va_idx]), torch.from_numpy(vb[va_idx]))
    dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=1024, num_workers=4, pin_memory=True)

    model = SpeedPhaseTokenVAE(window=window, latent_dim=latent_dim, hidden_dim=hidden).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    beta = 0.1
    epochs = 30
    best = float("inf")
    for ep in range(epochs):
        model.train()
        tot, cnt = 0.0, 0
        for xb, yb, pb, vb_b in dl_tr:
            xb, yb, pb, vb_b = xb.cuda(), yb.cuda(), pb.cuda(), vb_b.cuda()
            recon, mu, logvar, _ = model(xb, pb, vb_b)
            rec = nn.functional.mse_loss(recon, yb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += rec.item() * len(xb); cnt += len(xb)
        model.eval()
        va_rec, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb, pb, vb_b in dl_va:
                xb, yb, pb, vb_b = xb.cuda(), yb.cuda(), pb.cuda(), vb_b.cuda()
                recon, mu, logvar, _ = model(xb, pb, vb_b)
                va_rec += nn.functional.mse_loss(recon, yb).item() * len(xb)
                va_n += len(xb)
        va_mse = va_rec / va_n
        print(f"ep {ep+1}/{epochs} tr_rec_mse={tot/cnt:.5f} va_mse={va_mse:.5f} kl={kl.item():.4f}",
              flush=True)
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
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"window": window, "latent_dim": latent_dim, "hidden": hidden,
                   "token_dim": 64, "phase_dim": 2, "n_bins": 3,
                   "val_mse": best, "val_mae": float(np.sqrt(best))}, f, indent=1)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
