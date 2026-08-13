"""Train a temporal VAE over SONIC token windows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from apt_g1.sonic.token_seq_vae import TokenSeqVAE


def build_windows(tokens: np.ndarray, labels: np.ndarray, window: int, stride: int):
    xs, ys = [], []
    for name in np.unique(labels):
        seq = tokens[labels == name]
        for start in range(0, max(1, len(seq) - window + 1), stride):
            xs.append(seq[start : start + window])
            ys.append(name)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="outputs/flat_g1/reference_tokens_all.npy")
    parser.add_argument("--labels", default="outputs/flat_g1/reference_token_labels.npy")
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-coef", type=float, default=1e-3)
    parser.add_argument("--save", default="outputs/flat_g1/token_seq_vae.pt")
    args = parser.parse_args()

    tokens = np.load(args.tokens).astype(np.float32)
    labels = np.load(args.labels)
    xs, ys = build_windows(tokens, labels, args.window, args.stride)
    data = torch.from_numpy(xs)
    print(f"windows: {xs.shape}")

    model = TokenSeqVAE(
        token_dim=args.token_dim, latent_dim=args.latent_dim, window=args.window
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
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
        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1}/{args.epochs}, loss={total_loss / len(loader):.6f}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save)
    with torch.no_grad():
        recon = model(data)[0]
        err = torch.nn.functional.mse_loss(recon, data).item()
        mu_all = model.encode(data)[0].numpy()
    print(f"saved {args.save}, final recon mse={err:.6f}")

    means = defaultdict(list)
    for y, z in zip(ys, mu_all):
        means[y].append(z)
    for name, zs in means.items():
        mean = np.mean(zs, axis=0).astype(np.float32)
        path = Path(args.save).parent / f"{Path(name).stem}_seq_latent_mean{args.latent_dim}.npy"
        np.save(path, mean)
        print(f"latent mean {name}: {path}")


if __name__ == "__main__":
    main()
