"""Batched SONIC decoder wrapper for Isaac Lab (ONNX Runtime).

The released decoder maps a 994-d vector
(64-d FSQ token + 10-frame proprioception history) to 29-d body joint
actions (normalized, IsaacLab order). This mirrors the per-env
``apt_g1.sonic.sonic_wrapper.SonicOnnxDecoder`` but processes a full batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


class SonicBatchedDecoder:
    def __init__(
        self,
        onnx_path: Union[str, Path],
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        self.onnx_path = str(onnx_path)
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
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
        assert self.input_dim == 994, f"unexpected decoder input dim {self.input_dim}"

    def quantize_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """FSQ quantization matching the exported SONIC encoder graph (batched)."""
        latent = np.asarray(tokens, dtype=np.float32).reshape(
            -1, self.fsq_num_tokens, self.fsq_token_dim
        )
        x = np.tanh(latent + self.fsq_offset)
        quantized = np.round(self.fsq_scale * x - self.fsq_half) / self.fsq_denom
        return quantized.reshape(-1, 64).astype(np.float32)

    def build_decoder_obs(
        self,
        tokens: np.ndarray,
        ang_vel: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        last_actions: np.ndarray,
        gravity_dir: np.ndarray,
    ) -> np.ndarray:
        """Build (N, 994) decoder obs from 10-frame history arrays.

        Each history array is (N, 10, D) with time axis oldest -> newest.
        """
        tokens = np.asarray(tokens, dtype=np.float32).reshape(-1, 64)
        parts = [
            tokens,
            np.asarray(ang_vel, dtype=np.float32).reshape(-1, 30),
            np.asarray(joint_pos, dtype=np.float32).reshape(-1, 290),
            np.asarray(joint_vel, dtype=np.float32).reshape(-1, 290),
            np.asarray(last_actions, dtype=np.float32).reshape(-1, 290),
            np.asarray(gravity_dir, dtype=np.float32).reshape(-1, 30),
        ]
        obs = np.concatenate(parts, axis=1).astype(np.float32)
        if obs.shape[1] != self.input_dim:
            raise ValueError(
                f"Decoder input dim mismatch: expected {self.input_dim}, got {obs.shape[1]}"
            )
        return obs

    def decode(
        self,
        tokens: np.ndarray,
        ang_vel: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        last_actions: np.ndarray,
        gravity_dir: np.ndarray,
    ) -> np.ndarray:
        """Return (N, 29) normalized joint actions in IsaacLab order."""
        tokens = self.quantize_tokens(tokens)
        obs = self.build_decoder_obs(
            tokens, ang_vel, joint_pos, joint_vel, last_actions, gravity_dir
        )
        return self.session.run([self.output_name], {self.input_name: obs})[0]
