"""Train a small VAE on SONIC FSQ tokens."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from apt_g1.sonic.token_vae import TokenVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="outputs/flat_g1/reference_tokens_all.npy")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-coef", type=float, default=1e-3)
    parser.add_argument("--save", default="outputs/flat_g1/token_vae.pt")
    args = parser.parse_args()

    tokens = np.load(args.tokens).astype(np.float32)
    data = torch.from_numpy(tokens)
    model = TokenVAE(token_dim=64, latent_dim=args.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    for epoch in range(args.epochs):
        total_loss = 0.0
        for (batch,) in loader:
            recon, mu, logvar, _ = model(batch)
            recon_loss = torch.nn.functional.mse_loss(recon, batch)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            loss = recon_loss + args.kl_coef * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1}/{args.epochs}, loss={total_loss / len(loader):.6f}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save)
    with torch.no_grad():
        recon = model(data)[0]
        err = torch.nn.functional.mse_loss(recon, data).item()
    print(f"saved {args.save}, final recon mse={err:.6f}")


if __name__ == "__main__":
    main()
