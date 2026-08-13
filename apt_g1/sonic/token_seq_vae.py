"""Temporal VAE over short windows of SONIC FSQ tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenSeqVAE(nn.Module):
    def __init__(
        self,
        token_dim: int = 64,
        latent_dim: int = 16,
        window: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.window = window
        flat_dim = token_dim * window
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_dim),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.reshape(x.shape[0], -1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out.reshape(z.shape[0], self.window, self.token_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
