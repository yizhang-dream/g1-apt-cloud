"""Vectorized PPO for the APT Isaac env with paper-style training extras.

Implements the RL-stage mechanisms from APT-RL that were missing in the
MuJoCo attempts:

- latent action KL regularization w.r.t. N(0, I) (paper coefficient 2.5e-6)
- latent-space exploration bonus that decays to zero over training
  (paper: "exploration bonus that progressively decayed to zero")
- standard PPO (clip, GAE, entropy) on the phase latent + aux action
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class AptPPOPolicy(nn.Module):
    """MLP actor-critic: obs -> phase(2) + aux(12) [+ gate logits] + value."""

    def __init__(
        self,
        obs_dim: int,
        aux_dim: int = 12,
        gate_k: int = 0,
        hidden_dim: int = 256,
        phase_init_std: float = -4.0,
        aux_init_std: float = -4.0,
        use_phase: bool = True,
        latent_dim: int = 0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.aux_dim = aux_dim
        self.gate_k = gate_k
        self.use_phase = use_phase
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        if use_phase:
            self.phase_mean = nn.Linear(hidden_dim, 2)
            self.phase_log_std = nn.Parameter(torch.full((2,), phase_init_std))
        if latent_dim > 0:
            # E27 latent head: canonical "phase_mean/phase_log_std" keys so the
            # trainer's latent-KL / warmstart machinery works unchanged.
            self.phase_mean = nn.Linear(hidden_dim, latent_dim)
            self.phase_log_std = nn.Parameter(
                torch.full((latent_dim,), phase_init_std)
            )
        self.aux_mean = nn.Linear(hidden_dim, aux_dim)
        self.aux_log_std = nn.Parameter(torch.full((aux_dim,), aux_init_std))
        if gate_k > 0:
            self.gate_logits = nn.Linear(hidden_dim, gate_k)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor(self, obs: torch.Tensor) -> dict:
        x = self.encoder(obs)
        out = {
            "aux_mean": self.aux_mean(x),
            "aux_log_std": self.aux_log_std.expand_as(self.aux_mean(x)),
        }
        if self.use_phase or self.latent_dim > 0:
            out["phase_mean"] = self.phase_mean(x)
            out["phase_log_std"] = self.phase_log_std.expand_as(self.phase_mean(x))
        if self.gate_k > 0:
            out["gate_logits"] = self.gate_logits(x)
        return out

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Return dict of sampled actions, log_prob, entropy, value."""
        p = self.forward_actor(obs)
        ad = Normal(p["aux_mean"], p["aux_log_std"].exp())
        aux = ad.mean if deterministic else ad.sample()
        out = {"aux": aux}
        log_prob = ad.log_prob(aux).sum(-1)
        entropy = ad.entropy().sum(-1)
        if self.use_phase or self.latent_dim > 0:
            pd = Normal(p["phase_mean"], p["phase_log_std"].exp())
            phase = pd.mean if deterministic else pd.sample()
            out["phase"] = phase
            if self.latent_dim > 0:
                out["latent"] = phase
            log_prob = log_prob + pd.log_prob(phase).sum(-1)
            entropy = entropy + pd.entropy().sum(-1)
        if self.gate_k > 0:
            gd = Categorical(logits=p["gate_logits"])
            gate = gd.sample() if not deterministic else gd.probs.argmax(-1)
            out["gate"] = gate
            log_prob = log_prob + gd.log_prob(gate)
            entropy = entropy + gd.entropy()
        return out, log_prob, entropy, self.get_value(obs), p


def kl_normal_std_normal(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """D_KL(N(mean, diag sigma^2) || N(0, I)), summed over latent dims."""
    var = torch.exp(2.0 * log_std)
    return 0.5 * (mean.pow(2) + var - 2.0 * log_std - 1.0).sum(-1)


class PPOTrainer:
    def __init__(
        self,
        policy: AptPPOPolicy,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        num_epochs: int = 5,
        minibatch_size: int = 512,
        entropy_coef: float = 0.001,
        latent_kl_coef: float = 2.5e-6,
        latent_expl_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        max_iters: int = 500,
        value_coef: float = 0.5,
        device: str = "cuda:0",
    ):
        self.policy = policy.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.latent_kl_coef = latent_kl_coef
        self.latent_expl_coef = latent_expl_coef
        self.max_grad_norm = max_grad_norm
        self.max_iters = max_iters
        self.value_coef = value_coef
        self.it = 0

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        truncated: torch.Tensor,
        last_value: torch.Tensor,
    ) -> torch.Tensor:
        """rewards/values/dones/truncated: (T, N); returns advantages (T, N)."""
        T, N = rewards.shape
        adv = torch.zeros_like(rewards)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            cont = (~truncated[t]).float()
            delta = rewards[t] + self.gamma * next_value * cont - values[t]
            gae = delta + self.gamma * self.lam * cont * gae
            adv[t] = gae
        return adv

    def update(
        self,
        rollout: dict,
        phase_labels: torch.Tensor | None = None,
        phase_warm_coef: float = 0.0,
    ) -> dict:
        """rollout: obs (T,N,D), phase/aux/gate, logp, value, reward, done, trunc."""
        T, N = rollout["obs"].shape[:2]
        adv = self.compute_gae(
            rollout["reward"], rollout["value"], rollout["done"], rollout["trunc"], rollout["last_value"]
        )
        returns = adv + rollout["value"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = rollout["obs"].reshape(T * N, -1)
        phase = (
            rollout["phase"].reshape(T * N, -1)
            if rollout.get("phase") is not None
            else None
        )
        aux = rollout["aux"].reshape(T * N, -1)
        logp_old = rollout["logp"].reshape(-1)
        adv_f = adv.reshape(-1)
        ret_f = returns.reshape(-1)
        val_f = rollout["value"].reshape(-1)
        gate = rollout.get("gate")
        if gate is not None:
            gate = gate.reshape(-1)
        if phase_labels is not None:
            phase_labels = phase_labels.reshape(-1, phase_labels.shape[-1])

        # decaying latent exploration coefficient (paper: exploration -> exploitation)
        expl_coef = self.latent_expl_coef * max(0.0, 1.0 - self.it / max(1, self.max_iters))

        idx = torch.randperm(T * N, device=self.device)
        losses = []
        for start in range(0, T * N, self.minibatch_size):
            mb = idx[start : start + self.minibatch_size]
            p = self.policy.forward_actor(obs[mb])
            ad = Normal(p["aux_mean"], p["aux_log_std"].exp())
            lp = ad.log_prob(aux[mb]).sum(-1)
            if phase is not None:
                pd = Normal(p["phase_mean"], p["phase_log_std"].exp())
                lp = lp + pd.log_prob(phase[mb]).sum(-1)
            if gate is not None:
                gd = Categorical(logits=p["gate_logits"])
                lp = lp + gd.log_prob(gate[mb])
            ratio = (lp - logp_old[mb]).exp()
            surr1 = ratio * adv_f[mb]
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_f[mb]
            ploss = -torch.min(surr1, surr2).mean()
            vloss = torch.nn.functional.mse_loss(self.policy.get_value(obs[mb]), ret_f[mb])
            ent = ad.entropy().sum(-1)
            if phase is not None:
                ent = ent + pd.entropy().sum(-1)
            if gate is not None:
                ent = ent + gd.entropy()
            loss = (
                ploss
                + self.value_coef * vloss
                - self.entropy_coef * ent.mean()
            )
            if phase is not None:
                kl = kl_normal_std_normal(p["phase_mean"], p["phase_log_std"])
                loss = (
                    loss
                    - expl_coef * pd.entropy().sum(-1).mean()
                    + self.latent_kl_coef * kl.mean()
                )
            if phase_labels is not None:
                loss = (
                    loss
                    + phase_warm_coef
                    * torch.nn.functional.mse_loss(p["phase_mean"], phase_labels[mb])
                )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            losses.append(
                {
                    "loss": loss.item(),
                    "ploss": ploss.item(),
                    "vloss": vloss.item(),
                    "ent": ent.mean().item(),
                    "kl": kl.mean().item() if phase is not None else 0.0,
                    "expl": expl_coef,
                }
            )
        self.it += 1
        for d in losses:
            d.setdefault("kl", 0.0)
        agg = {k: float(np.mean([d[k] for d in losses])) for k in losses[0]}
        return agg
