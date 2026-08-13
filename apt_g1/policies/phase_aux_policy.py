"""APT-style policy: RL selects the phase (latent action) + auxiliary action.

The phase is normalized and converted to a phase-bin prototype token by the
env (the frozen distilled prior); the auxiliary action corrects joint targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class PhaseAuxPolicy(nn.Module):
    def __init__(self, obs_dim: int, aux_dim: int = 12, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.aux_dim = aux_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.phase_mean = nn.Linear(hidden_dim, 2)
        self.phase_log_std = nn.Parameter(torch.full((2,), -4.0))
        self.aux_mean = nn.Linear(hidden_dim, aux_dim)
        self.aux_log_std = nn.Parameter(torch.full((aux_dim,), -4.0))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor(self, obs):
        x = self.encoder(obs)
        return {
            "phase_mean": self.phase_mean(x),
            "phase_log_std": self.phase_log_std.expand_as(self.phase_mean(x)),
            "aux_mean": self.aux_mean(x),
            "aux_log_std": self.aux_log_std.expand_as(self.aux_mean(x)),
        }

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, deterministic=False):
        p = self.forward_actor(obs)
        pd = Normal(p["phase_mean"], p["phase_log_std"].exp())
        ad = Normal(p["aux_mean"], p["aux_log_std"].exp())
        phase = pd.mean if deterministic else pd.sample()
        aux = ad.mean if deterministic else ad.sample()
        action = {"phase": phase, "aux": aux}
        log_prob = pd.log_prob(phase).sum(-1) + ad.log_prob(aux).sum(-1)
        return action, log_prob, self.get_value(obs)

    def evaluate_actions(self, obs, actions):
        p = self.forward_actor(obs)
        pd = Normal(p["phase_mean"], p["phase_log_std"].exp())
        ad = Normal(p["aux_mean"], p["aux_log_std"].exp())
        log_prob = pd.log_prob(actions["phase"]).sum(-1) + ad.log_prob(actions["aux"]).sum(-1)
        entropy = pd.entropy().sum(-1) + ad.entropy().sum(-1)
        return log_prob, entropy, self.get_value(obs)
