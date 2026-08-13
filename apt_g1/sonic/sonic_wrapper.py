"""GEAR-SONIC encoder/decoder wrappers.

The SONIC training stack already implements a "student direct latent" path:

- policy outputs a pre-quantization 64-d latent
- the latent is quantized by FSQ
- the ATM decoder turns it into G1 joint actions

This wrapper loads the same `UniversalTokenModule` from the exported SONIC
model config and reuses that decoding path, so the APT policy only has to output
the latent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class SonicOnnxDecoder:
    """Direct ONNX wrapper for the released SONIC decoder.

    The released decoder expects one concatenated observation vector:

        obs_dict = [
            token_state                         64
            his_base_angular_velocity_10frame   30
            his_body_joint_positions_10frame   290
            his_body_joint_velocities_10frame  290
            his_last_actions_10frame           290
            his_gravity_dir_10frame             30
        ]                                     = 994

    and outputs a 29-dimensional body action.
    """

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        self.onnx_path = str(onnx_path)
        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_dim = self.session.get_inputs()[0].shape[1]
        self.output_name = self.session.get_outputs()[0].name
        self.token_dim = 64
        self.fsq_num_tokens = 2
        self.fsq_token_dim = 32
        self.fsq_offset = np.float32(0.032237)
        self.fsq_scale = np.float32(15.515501)
        self.fsq_half = np.float32(0.5)
        self.fsq_denom = np.float32(16.0)

    def quantize_token(self, token: np.ndarray) -> np.ndarray:
        """FSQ quantization matching the exported SONIC encoder graph.

        quantized = round(scale * tanh(latent + offset) - 0.5) / 16
        """
        latent = np.asarray(token, dtype=np.float32).reshape(
            -1, self.fsq_num_tokens, self.fsq_token_dim
        )
        x = np.tanh(latent + self.fsq_offset)
        quantized = np.round(self.fsq_scale * x - self.fsq_half) / self.fsq_denom
        return quantized.reshape(-1).astype(np.float32)

    def build_decoder_obs(
        self,
        token: np.ndarray,
        history: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Concatenate token and 10-frame history in the released order."""
        token = np.asarray(token, dtype=np.float32).reshape(-1)
        parts = [
            token,
            np.asarray(history["base_angular_velocity"], dtype=np.float32).reshape(-1),
            np.asarray(history["body_joint_positions"], dtype=np.float32).reshape(-1),
            np.asarray(history["body_joint_velocities"], dtype=np.float32).reshape(-1),
            np.asarray(history["last_actions"], dtype=np.float32).reshape(-1),
            np.asarray(history["gravity_dir"], dtype=np.float32).reshape(-1),
        ]
        obs = np.concatenate(parts)
        if obs.shape[0] != self.input_dim:
            raise ValueError(
                f"Decoder input dim mismatch: expected {self.input_dim}, got {obs.shape[0]}"
            )
        return obs[None, :]

    def decode(
        self,
        token: np.ndarray,
        history: dict[str, np.ndarray],
    ) -> np.ndarray:
        token = self.quantize_token(token)
        obs = self.build_decoder_obs(token, history)
        return self.session.run([self.output_name], {self.input_name: obs})[0]


class SonicDecoder:
    def __init__(
        self,
        model_config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        decoder_name: str = "g1_dyn",
        token_dim: int = 64,
    ):
        self.model_config_path = Path(model_config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.decoder_name = decoder_name
        self.token_dim = token_dim
        self.action_transform_module: Any = None

    def load(self):
        from omegaconf import OmegaConf

        from gear_sonic.trl.utils.common import custom_instantiate

        exported_config = OmegaConf.load(self.model_config_path)
        env_config = exported_config["env_config"]
        algo_config = exported_config["algo_config"]

        self.action_transform_module = custom_instantiate(
            algo_config.actor,
            env_config=env_config,
            algo_config=algo_config,
            _resolve=False,
        ).to(self.device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.action_transform_module.load_state_dict(checkpoint["policy_state_dict"])
        self.action_transform_module.eval()
        for param in self.action_transform_module.parameters():
            param.requires_grad_(False)

    @property
    def atm(self):
        return self.action_transform_module.actor_module

    def decode(
        self,
        full_latent: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a pre-quantization latent into (batch, seq, action_dim)."""
        atm = self.atm
        if proprioception.dim() == 2:
            proprioception = proprioception.unsqueeze(1)

        latent_reshaped = full_latent.view(-1, atm.max_num_tokens, atm.token_dim)
        if atm.quantizer is not None:
            quantized_codes, _ = atm.quantizer(latent_reshaped)
            tokens_for_decode = quantized_codes.contiguous()
        else:
            tokens_for_decode = latent_reshaped

        tokens_reshaped = tokens_for_decode.unsqueeze(1)
        tokens_flattened = tokens_for_decode.view(full_latent.shape[0], -1).unsqueeze(1)
        decode_input = {
            "token": tokens_reshaped,
            "token_flattened": tokens_flattened,
            "proprioception": proprioception,
        }
        decoded = atm.decode(self.decoder_name, decode_input)
        return decoded.get("meta_action", decoded.get("action"))


class SonicEncoder:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        token_dim: int = 64,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.token_dim = token_dim

    def load(self):
        if self.checkpoint_path is None:
            raise ValueError("SonicEncoder requires a checkpoint path.")
        raise NotImplementedError("SONIC encoder loading must be wired to your checkpoint.")

    def encode(self, reference_motion: np.ndarray) -> np.ndarray:
        """Map a motion reference to a (batch, token_dim) FSQ token."""
        raise NotImplementedError("SONIC encoder inference is not wired yet.")
