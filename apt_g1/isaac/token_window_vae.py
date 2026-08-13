"""E27: phase-conditioned token VAE (frozen decoder for RL).

z_t (16-d) encodes the causal window of the last 10 SONIC tokens ending at t;
the decoder D(z, sin(phi), cos(phi)) -> token_t keeps RL outputs on the SONIC
token manifold while the env's phase clock drives the gait cycle.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PhaseTokenVAE(nn.Module):
    def __init__(self, token_dim: int = 64, window: int = 10, latent_dim: int = 16,
                 hidden_dim: int = 256, phase_dim: int = 2):
        super().__init__()
        self.token_dim = token_dim
        self.window = window
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + phase_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
            nn.Tanh(),
        )

    def decode(self, z: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, phase], dim=-1))
