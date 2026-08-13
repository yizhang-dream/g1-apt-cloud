"""E27/E31: phase- (and speed-) conditioned token VAE (frozen decoder for RL).

E27: z_t (16-d) encodes the causal window of the last 10 SONIC tokens ending at
t; the decoder D(z, sin(phi), cos(phi)) -> token_t keeps RL outputs on the
SONIC token manifold while the env's phase clock drives the gait cycle.

E31: SpeedPhaseTokenVAE adds a speed-bin embedding so the decoder is
D(z, sin(phi), cos(phi), v_bin) -> token — the manifold itself encodes gait
speed (slow/mid/fast walk cadences), letting RL pick faster skills.
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


class SpeedPhaseTokenVAE(nn.Module):
    """E31: PhaseTokenVAE with a speed-bin embedding on the decoder.

    Decoder: D(z, sin(phi), cos(phi), v_bin) -> token. v_bin is a discrete
    speed condition (trained on walk phase-rate thirds); the env maps the
    commanded vx to a bin so the manifold itself encodes gait speed.
    """

    def __init__(self, token_dim: int = 64, window: int = 10, latent_dim: int = 16,
                 hidden_dim: int = 256, phase_dim: int = 2, n_bins: int = 3):
        super().__init__()
        self.token_dim = token_dim
        self.window = window
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        self.n_bins = n_bins
        self.speed_embed = nn.Embedding(n_bins, 8)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + phase_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
            nn.Tanh(),
        )

    def decode(self, z: torch.Tensor, phase: torch.Tensor,
               v_bin: torch.Tensor) -> torch.Tensor:
        se = self.speed_embed(v_bin)
        return self.decoder(torch.cat([z, phase, se], dim=-1))


class DirSpeedPhaseTokenVAE(nn.Module):
    """E35: SpeedPhaseTokenVAE + heading-bin embedding on the decoder.

    Decoder: D(z, sin(phi), cos(phi), v_bin, psi_bin) -> token. psi_bin is an
    8-bin heading (bin 4 = +x forward, same formula as build_exp3_dataset).
    Direction is fully determined by the command condition so the policy's z
    only picks speed — fixing E31's systematic yaw drift (the fast skill's
    implicit turning bias is no longer needed).
    """

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
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + phase_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
            nn.Tanh(),
        )

    def decode(self, z: torch.Tensor, phase: torch.Tensor,
               v_bin: torch.Tensor, d_bin: torch.Tensor) -> torch.Tensor:
        se = self.speed_embed(v_bin)
        de = self.dir_embed(d_bin)
        return self.decoder(torch.cat([z, phase, se, de], dim=-1))
