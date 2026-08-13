"""Vanilla RL baseline env (no SONIC prior) for the APT comparison.

Same physics/reward as ``AptFlatG1Env`` but the policy directly outputs 29-d
normalized joint position targets:

    q_des = sonic_default + clip(action) * sonic_scale

No phase router, no SONIC decoder, no auxiliary split. This is the paper's
"vanilla RL" ablation on flat ground.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils import configclass

from apt_g1.isaac.apt_flat_env import (
    AptFlatG1Env,
    AptFlatG1EnvCfg,
    _sonic_default_isaac,
    _sonic_scale_isaac,
)


@configclass
class AptFlatG1VanillaEnvCfg(AptFlatG1EnvCfg):
    action_space: int = 29
    observation_space: int = 100
    use_sonic_prior: bool = False


class AptFlatG1VanillaEnv(AptFlatG1Env):
    cfg: AptFlatG1VanillaEnvCfg

    def __init__(self, cfg: AptFlatG1VanillaEnvCfg, render_mode=None, **kwargs):
        self._sonic_default = _sonic_default_isaac()
        self._sonic_scale = _sonic_scale_isaac()
        super().__init__(cfg, render_mode, **kwargs)

    def _compute_q_des(self, phase: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        # aux here carries the full 29-d normalized joint action
        action = torch.clamp(aux, -1.0, 1.0)
        default = self._sonic_default_t
        scale = self._sonic_scale_t
        return default + action * scale

    def _pre_physics_step(self, actions: torch.Tensor):
        self._sample_disturbance()
        self._update_gate()
        self._q_des = self._compute_q_des(
            torch.zeros(self.num_envs, 2, device=self.device), actions
        ).detach()
        self._last_phase = torch.zeros(self.num_envs, 2, device=self.device)
        self._last_aux = actions.detach()[:, :12]
        self._actions = actions.clone()

    def _get_observations(self) -> dict:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        gravity = self.robot.data.projected_gravity_b
        jpos_rel = self.robot.data.joint_pos[:, self._body_idx] - self._sonic_default_t
        jvel = self.robot.data.joint_vel[:, self._body_idx]
        last_actions = (self._q_des - self._sonic_default_t) / self._sonic_scale_t
        obs = torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                gravity,
                jpos_rel,
                jvel,
                self._commands,
                last_actions,
                self.robot.data.root_pos_w[:, 2:3],
            ],
            dim=-1,
        )
        assert obs.shape[1] == self.cfg.observation_space, obs.shape
        return {"policy": obs}
