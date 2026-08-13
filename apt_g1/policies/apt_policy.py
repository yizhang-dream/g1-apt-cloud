"""APT policy: SONIC token + auxiliary action + optional skill selection."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class APTPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        token_dim: int = 64,
        aux_dim: int = 12,
        hand_dim: int = 2,
        num_skills: int = 2,
        hidden_dim: int = 256,
        use_skill_selection: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.token_dim = token_dim
        self.aux_dim = aux_dim
        self.hand_dim = hand_dim
        self.num_skills = num_skills
        self.use_skill_selection = use_skill_selection

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.token_mean = nn.Linear(hidden_dim, token_dim)
        self.token_log_std = nn.Parameter(torch.full((token_dim,), -4.0))

        self.aux_mean = nn.Linear(hidden_dim, aux_dim)
        self.aux_log_std = nn.Parameter(torch.full((aux_dim,), -4.0))

        self.hand_mean = nn.Linear(hidden_dim, hand_dim)
        self.hand_log_std = nn.Parameter(torch.full((hand_dim,), -4.0))

        self.skill_logits = nn.Linear(hidden_dim, num_skills)

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.encoder(obs)
        return {
            "token_mean": self.token_mean(x),
            "token_log_std": self.token_log_std.expand_as(self.token_mean(x)),
            "aux_mean": self.aux_mean(x),
            "aux_log_std": self.aux_log_std.expand_as(self.aux_mean(x)),
            "hand_mean": self.hand_mean(x),
            "hand_log_std": self.hand_log_std.expand_as(self.hand_mean(x)),
            "skill_logits": self.skill_logits(x),
        }

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        params = self.forward_actor(obs)
        token_dist = Normal(params["token_mean"], params["token_log_std"].exp())
        aux_dist = Normal(params["aux_mean"], params["aux_log_std"].exp())
        hand_dist = Normal(params["hand_mean"], params["hand_log_std"].exp())
        skill_dist = Categorical(logits=params["skill_logits"])

        if deterministic:
            token = token_dist.mean
            aux = aux_dist.mean
            hand = hand_dist.mean
            skill = skill_dist.probs.argmax(dim=-1)
        else:
            token = token_dist.sample()
            aux = aux_dist.sample()
            hand = hand_dist.sample()
            skill = skill_dist.sample()

        action = {
            "token": token,
            "aux": aux,
            "hand": hand,
            "skill": skill,
        }
        log_prob = (
            token_dist.log_prob(token).sum(-1)
            + aux_dist.log_prob(aux).sum(-1)
            + hand_dist.log_prob(hand).sum(-1)
            + skill_dist.log_prob(skill)
        )
        return action, log_prob, self.get_value(obs)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.forward_actor(obs)
        token_dist = Normal(params["token_mean"], params["token_log_std"].exp())
        aux_dist = Normal(params["aux_mean"], params["aux_log_std"].exp())
        hand_dist = Normal(params["hand_mean"], params["hand_log_std"].exp())
        skill_dist = Categorical(logits=params["skill_logits"])

        log_prob = (
            token_dist.log_prob(actions["token"]).sum(-1)
            + aux_dist.log_prob(actions["aux"]).sum(-1)
            + hand_dist.log_prob(actions["hand"]).sum(-1)
            + skill_dist.log_prob(actions["skill"])
        )
        entropy = (
            token_dist.entropy().sum(-1)
            + aux_dist.entropy().sum(-1)
            + hand_dist.entropy().sum(-1)
            + skill_dist.entropy()
        )
        return log_prob, entropy, self.get_value(obs)
