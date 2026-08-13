"""G1 flat-ground environment adapters for the APT policy.

The wrapper converts APT actions (SONIC token + auxiliary action) into the full
G1 action vector expected by the simulator:

    lower_body_target = SonicDecoder(token) + aux_scale * aux
    full_action[arms] = fixed arm pose
    full_action[lower_body] = lower_body_target
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class APTFlatG1Env:
    def __init__(
        self,
        env: Any,
        sonic_decoder,
        lower_body_indices: list[int],
        num_skills: int,
        aux_scale: float,
        arm_joint_indices: list[int] | None = None,
        arm_default_pose: np.ndarray | None = None,
    ):
        self.env = env
        self.sonic_decoder = sonic_decoder
        self.lower_body_indices = np.asarray(lower_body_indices, dtype=int)
        self.num_skills = num_skills
        self.aux_scale = aux_scale
        self.arm_joint_indices = arm_joint_indices
        self.arm_default_pose = arm_default_pose

    def reset(self):
        return self.env.reset()

    def build_full_action(self, token, aux, proprioception=None) -> torch.Tensor:
        if proprioception is None:
            proprioception = torch.zeros(
                token.shape[0], 1, 1, dtype=torch.float32, device=token.device
            )
        decoded = self.sonic_decoder.decode(token, proprioception)
        # decoded shape: (batch, seq_len, action_dim)
        lower_target = decoded[:, 0, self.lower_body_indices]
        full_action = torch.zeros(
            decoded.shape[0], decoded.shape[-1], dtype=torch.float32, device=decoded.device
        )
        full_action[:, self.lower_body_indices] = lower_target + self.aux_scale * aux
        if self.arm_joint_indices is not None and self.arm_default_pose is not None:
            full_action[:, self.arm_joint_indices] = torch.as_tensor(
                self.arm_default_pose, dtype=torch.float32, device=decoded.device
            )
        return full_action

    def step(self, action: dict[str, np.ndarray]):
        full_action = self.build_full_action(
            action["token"], action["aux"], action.get("proprioception")
        )
        return self.env.step(full_action)


class IsaacLabG1FlatEnv:
    """Adapter for `Isaac-Velocity-Flat-G1-v0` from Isaac Lab."""

    def __init__(
        self,
        task_name: str = "Isaac-Velocity-Flat-G1-v0",
        num_envs: int = 1,
        device: str = "cuda:0",
    ):
        import gymnasium as gym

        import isaaclab_tasks.manager_based.locomotion.velocity.config.g1  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        self.task_name = task_name
        self.device = device
        env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        self.env = gym.make(task_name, cfg=env_cfg)

    def reset(self):
        obs, _ = self.env.reset()
        return obs["policy"]

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs["policy"], reward, terminated, truncated, info


class DummyG1FlatEnv:
    """A tiny deterministic environment for smoke-testing the training loop."""

    def __init__(self, action_dim: int = 29, obs_dim: int = 64, num_envs: int = 1):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_envs = num_envs
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

    def step(self, action: np.ndarray):
        self.t += 1
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        reward = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        info = {}
        return obs, reward, done, info
