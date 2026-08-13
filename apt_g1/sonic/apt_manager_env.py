"""SONIC manager-env adapter with APT auxiliary action support.

The stock `ManagerEnvWrapper` supports `use_student_direct_latent`: the policy
outputs a pre-quantization latent, the wrapper quantizes it with FSQ, and the
ATM decoder produces body actions. This adapter adds the APT auxiliary action
after the decoder output:

    body_action = SonicDecoder(token) + aux_scale * aux
"""

from __future__ import annotations

from typing import Any

import torch

from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper


class APTManagerEnvWrapper(ManagerEnvWrapper):
    def __init__(self, env, config, aux_dim: int = 12, aux_scale: float = 0.2):
        super().__init__(env, config)
        self.apt_aux_dim = aux_dim
        self.apt_aux_scale = aux_scale
        self.apt_aux: torch.Tensor | None = None

    def _decode_direct_latent(self, full_latent, atm_obs_dict):
        body_actions = super()._decode_direct_latent(full_latent, atm_obs_dict)
        if self.apt_aux is not None:
            body_actions[..., : self.apt_aux.shape[-1]] += (
                self.apt_aux_scale * self.apt_aux
            )
        return body_actions

    def step(self, actions: dict[str, Any]):
        self.apt_aux = actions.get("apt_aux", None)
        return super().step(actions)
