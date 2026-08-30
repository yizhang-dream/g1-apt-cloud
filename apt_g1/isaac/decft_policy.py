"""E44: decoder fine-tuning policy (PPO trains the frozen SONIC decoder).

Architecture: obs -> encoder -> z (16-d, E39 warm-start) -> frozen
DirSpeedPhaseTokenVAE (E39) -> token -> FSQ -> TRAINABLE SONIC decoder MLP ->
mu (29-d normalized joint targets). The Gaussian action is defined in
joint-target space: a = mu + sigma*eps, so PPO's score-function gradient
reaches the decoder weights (the decoder acts as the policy's mean network).
A frozen copy of the official decoder (decoder_ref) provides an L2 drift
regularizer so token semantics are not destroyed.

obs layout (decft mode, mirror AptFlatG1Env):
  [0:91]   base parts (lin_vel3 ang_vel3 grav3 jpos29 jvel29 cmds3
           last_phase2 last_aux12 mode_oh5 gate_tick1 root_z1)
  [91:1021] 930-d proprio history (ang_vel30 jpos290 jvel290 last_act290 grav30)
  [1021:1023] sin/cos of the walk-clock phase
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from apt_g1.isaac.sonic_decoder_torch import SonicTorchDecoder
from apt_g1.isaac.token_window_vae import DirSpeedPhaseTokenVAE

# decft obs layout constants (must mirror AptFlatG1Env decft mode)
_BASE_DIM = 91
_HIST_DIM = 930
_PHASE_DIM = 2
OBS_DIM = _BASE_DIM + _HIST_DIM + _PHASE_DIM  # 1023
_CMD_SLICE = slice(3 + 3 + 3 + 29 + 29, 3 + 3 + 3 + 29 + 29 + 3)  # 67:70
_HIST_SLICE = slice(_BASE_DIM, _BASE_DIM + _HIST_DIM)
_PHASE_SLICE = slice(_BASE_DIM + _HIST_DIM, OBS_DIM)


class DecFtPolicy(nn.Module):
    """E44: latent -> frozen VAE -> token -> trainable SONIC decoder -> action.

    `decoder_ft = True` tells PPOTrainer.update to route the aux branch
    through policy.action_mean(z, obs) so decoder weights receive gradients.
    """

    decoder_ft = True

    def __init__(
        self,
        obs_dim: int,
        vae_path: str,
        decoder_path: str,
        hidden_dim: int = 256,
        phase_init_std: float = -4.0,
        aux_init_std: float = -2.0,
        vx_max: float = 0.8,
        n_vbins: int = 3,
        n_dbins: int = 8,
        device: str = "cuda:0",
        freeze_decoder: bool = False,  # E44 two-phase: phase 1 freezes decoder
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = 16
        self.aux_dim = 29
        self.vx_max = vx_max
        self.n_vbins = n_vbins
        self.n_dbins = n_dbins

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # E39-compatible key names (phase_mean/phase_log_std) for warm-start
        self.phase_mean = nn.Linear(hidden_dim, self.latent_dim)
        self.phase_log_std = nn.Parameter(
            torch.full((self.latent_dim,), phase_init_std)
        )
        self.aux_log_std = nn.Parameter(torch.full((self.aux_dim,), aux_init_std))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # frozen E39 dual-disentangle VAE (token manifold prior)
        vae = DirSpeedPhaseTokenVAE(n_vbins=n_vbins, n_dbins=n_dbins)
        vae.load_state_dict(
            torch.load(vae_path, map_location="cpu"), strict=False
        )
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        self.vae = vae

        # SONIC decoder (official weights init). E44 two-phase: phase 1 keeps it
        # frozen so the z-head learns against the OFFICIAL decoder first, then
        # phase 2 unfreezes it (v2 drift constraints) for the 0.06->0.08 course.
        dec = SonicTorchDecoder(decoder_path, device=device)
        if freeze_decoder:
            dec.eval()
            for p in dec.parameters():
                p.requires_grad_(False)
        else:
            dec.train()  # no dropout layers; just marks the module trainable
        self.decoder = dec
        # frozen official copy for drift regularization
        self.decoder_ref = copy.deepcopy(dec)
        self.decoder_ref.eval()
        for p in self.decoder_ref.parameters():
            p.requires_grad_(False)

        self._vx_edges = torch.linspace(0.0, vx_max, n_vbins + 1)[1:-1]
        self.to(device)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def forward_actor(self, obs: torch.Tensor) -> dict:
        x = self.encoder(obs)
        return {
            # placeholder; the decft trainer branch replaces aux_mean via
            # action_mean() so this never drives the distribution
            "aux_mean": torch.zeros(
                obs.shape[0], self.aux_dim, device=obs.device
            ),
            "aux_log_std": self.aux_log_std.expand(obs.shape[0], self.aux_dim),
            "phase_mean": self.phase_mean(x),
            "phase_log_std": self.phase_log_std.expand(
                obs.shape[0], self.latent_dim
            ),
        }

    def action_mean(
        self, z: torch.Tensor, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable decoder chain: z -> token -> FSQ -> decoder -> mu.

        Returns (mu (N,29), drift_reg (scalar MSE vs official decoder)).
        """
        cmd = obs[:, _CMD_SLICE]
        sc = obs[:, _PHASE_SLICE]
        hist = obs[:, _HIST_SLICE]
        edges = self._vx_edges.to(obs.device)
        vb = torch.bucketize(cmd[:, 0], edges).clamp(0, self.n_vbins - 1)
        ang = torch.atan2(cmd[:, 1], cmd[:, 0])
        db = torch.floor((ang + np.pi) / (2.0 * np.pi) * self.n_dbins).long() % self.n_dbins
        token = self.vae.decode(z, sc, vb, db)
        tok_q = self.decoder.quantize_tokens(token)
        dec_in = torch.cat([tok_q, hist], dim=1)
        mu = self.decoder.net(dec_in)
        with torch.no_grad():
            mu_ref = self.decoder_ref.net(dec_in.detach())
        reg = nn.functional.mse_loss(mu, mu_ref)
        return mu, reg

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        p = self.forward_actor(obs)
        pd = Normal(p["phase_mean"], p["phase_log_std"].exp())
        z = pd.mean if deterministic else pd.sample()
        mu, _ = self.action_mean(z, obs)
        ad = Normal(mu, p["aux_log_std"].exp())
        a = ad.mean if deterministic else ad.sample()
        log_prob = pd.log_prob(z).sum(-1) + ad.log_prob(a).sum(-1)
        entropy = pd.entropy().sum(-1) + ad.entropy().sum(-1)
        return {"phase": z, "aux": a}, log_prob, entropy, self.get_value(obs), p
