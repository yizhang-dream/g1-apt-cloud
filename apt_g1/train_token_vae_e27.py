"""E27: phase-conditioned token VAE over SONIC FSQ tokens.

Encoder: causal window of the last W tokens ending at t -> z_t (16-d).
Decoder: D(z, sin(phi), cos(phi)) -> token_t, where phi is the walk-cycle
phase from a 2-PC PCA fit on walk (mode 2) tokens. The decoder is frozen for
RL (E27): the policy outputs z (a skill), the env advances a phase clock at
the measured walk cadence, and D(z, phi) produces the cyclic token sequence
that the frozen SONIC decoder turns into joint targets. This is the paper's
TVAE structure (one latent + decoder generates the motion) without a behavior
prior: the gait clock is a fixed constant, and z must be found by RL.

Also saves:
- pca.npz   : PCA mean/V2 for the walk-cycle phase
- z_walk.npy: mean encoded latent of walk windows (latent warm-start init)
- meta.json : phase rate (rad/step) measured on walk data
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PhaseTokenVAE(nn.Module):
    def __init__(self, token_dim: int = 64, window: int = 10, latent_dim: int = 16,
                 hidden_dim: int = 256, phase_dim: int = 2):
        super().__init__()
        self.token_dim = token_dim
        self.window = window
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
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
            nn.Linear(latent_dim + phase_dim, hidden_dim),
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

    def decode(self, z: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, phase], dim=-1))

    def forward(self, x: torch.Tensor, phase: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, phase)
        return recon, mu, logvar, z


def build_windows(tok: np.ndarray, window: int) -> np.ndarray:
    """Causal windows: x[t] = tok[t-window+1:t+1] (zeros before t=0)."""
    n, d = tok.shape
    pad = np.zeros((window - 1, d), dtype=np.float32)
    ext = np.vstack([pad, tok])
    return np.stack([ext[i:i + window].reshape(-1) for i in range(n)])


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    data_dir = "data/exp_all3"
    out_dir = "outputs/token_vae_e27"
    os.makedirs(out_dir, exist_ok=True)

    tok = np.load(os.path.join(data_dir, "token.npy")).astype(np.float32)
    mode = np.load(os.path.join(data_dir, "mode.npy"))
    print("tokens", tok.shape, tok.min(), tok.max())

    # ---- walk-cycle phase from a 2-PC PCA fit on walk (mode 2) tokens ----
    walk = tok[mode == 2]
    pmean = walk.mean(0)
    c = walk - pmean
    cov = c.T @ c / len(c)
    ev, V = np.linalg.eigh(cov)
    V2 = V[:, ::-1][:, :2].astype(np.float32)
    proj = (tok - pmean) @ V2
    phi = np.arctan2(proj[:, 1], proj[:, 0]).astype(np.float32)
    # measured phase advance rate on walk data (rad per 50 Hz control step)
    idx = np.where(mode == 2)[0]
    dphi = np.diff(phi[idx])
    dphi = np.mod(dphi + np.pi, 2.0 * np.pi) - np.pi
    rate = float(np.abs(dphi).mean())
    print("walk-phase PCA var ratio", (ev[::-1][:2] / ev.sum()).round(3),
          "rate rad/step", round(rate, 4))
    np.savez(os.path.join(out_dir, "pca.npz"), pmean=pmean, V2=V2, rate=rate)

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
                          torch.from_numpy(phase2[tr_idx]))
    ds_va = TensorDataset(torch.from_numpy(x[va_idx]), torch.from_numpy(y[va_idx]),
                          torch.from_numpy(phase2[va_idx]))
    dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=1024, num_workers=4, pin_memory=True)

    model = PhaseTokenVAE(window=window, latent_dim=latent_dim, hidden_dim=hidden).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    beta = 0.1  # paper TVAE KL weight
    epochs = 30
    best = float("inf")
    for ep in range(epochs):
        model.train()
        tot, cnt = 0.0, 0
        for xb, yb, pb in dl_tr:
            xb, yb, pb = xb.cuda(), yb.cuda(), pb.cuda()
            recon, mu, logvar, _ = model(xb, pb)
            rec = nn.functional.mse_loss(recon, yb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += rec.item() * len(xb)
            cnt += len(xb)
        model.eval()
        va_rec, va_n = 0.0, 0
        zs = []
        with torch.no_grad():
            for xb, yb, pb in dl_va:
                xb, yb, pb = xb.cuda(), yb.cuda(), pb.cuda()
                recon, mu, logvar, z = model(xb, pb)
                va_rec += nn.functional.mse_loss(recon, yb).item() * len(xb)
                va_n += len(xb)
                zs.append(mu.cpu().numpy())
        va_mse = va_rec / va_n
        print(f"ep {ep+1}/{epochs} tr_rec_mse={tot/cnt:.5f} va_mse={va_mse:.5f} kl={kl.item():.4f}",
              flush=True)
        if va_mse < best:
            best = va_mse
            torch.save(model.state_dict(), os.path.join(out_dir, "vae.pt"))

    # z_walk: mean encoded latent of walk windows (warm-start init)
    with torch.no_grad():
        xw = torch.from_numpy(x[idx]).cuda()
        pw = torch.from_numpy(phase2[idx]).cuda()
        zw = model.encode(xw)[0].cpu().numpy()
    np.save(os.path.join(out_dir, "z_walk.npy"), zw.mean(0).astype(np.float32))
    print("val recon MAE:", float(np.sqrt(best)))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"window": window, "latent_dim": latent_dim, "hidden": hidden,
                   "token_dim": 64, "phase_dim": 2, "val_mse": best,
                   "val_mae": float(np.sqrt(best)), "phase_rate": rate}, f, indent=1)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
