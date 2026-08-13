"""Pure-torch reimplementation of the released SONIC decoder.

The exported ONNX decoder is a 6-layer MLP with SiLU activations
(x * sigmoid(x)) mapping 994-d (token + 10-frame proprio) to 29-d joint
actions. The ONNX graph is exported with static batch size 1, so for batched
Isaac training we extract the weights and run the same MLP natively in torch
(dynamic batch, GPU).
"""

from __future__ import annotations

import numpy as np
import onnx
import torch
import torch.nn as nn


class SonicTorchDecoder(nn.Module):
    def __init__(self, onnx_path: str, device: str = "cuda:0"):
        super().__init__()
        m = onnx.load(onnx_path)
        inits = {i.name: i for i in m.graph.initializer}

        def arr(name: str) -> np.ndarray:
            i = inits[name]
            import onnx.numpy_helper

            return onnx.numpy_helper.to_array(i)

        # weight names in order (ONNX MatMul: x @ W)
        w_names = [
            "onnx::MatMul_136",
            "onnx::MatMul_137",
            "onnx::MatMul_138",
            "onnx::MatMul_139",
            "onnx::MatMul_140",
            "onnx::MatMul_141",
            "onnx::MatMul_142",
        ]
        b_names = [
            "module.decoders.g1_dyn.module.0.bias",
            "module.decoders.g1_dyn.module.2.bias",
            "module.decoders.g1_dyn.module.4.bias",
            "module.decoders.g1_dyn.module.6.bias",
            "module.decoders.g1_dyn.module.8.bias",
            "module.decoders.g1_dyn.module.10.bias",
            "module.decoders.g1_dyn.module.12.bias",
        ]
        layers = []
        for i, (wn, bn) in enumerate(zip(w_names, b_names)):
            w = torch.from_numpy(arr(wn).T.copy()).float()  # (out, in)
            b = torch.from_numpy(arr(bn).copy()).float()
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            layers[-1].weight.data.copy_(w)
            layers[-1].bias.data.copy_(b)
            if i < len(w_names) - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.fsq_offset = 0.032237
        self.fsq_scale = 15.515501
        self.fsq_half = 0.5
        self.fsq_denom = 16.0
        self.to(device)
        self.eval()

    def quantize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """FSQ quantization matching the exported SONIC encoder graph (batched)."""
        latent = tokens.reshape(-1, 2, 32)
        x = torch.tanh(latent + self.fsq_offset)
        quantized = torch.round(self.fsq_scale * x - self.fsq_half) / self.fsq_denom
        return quantized.reshape(-1, 64)

    def build_decoder_obs(
        self,
        tokens: torch.Tensor,
        ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        last_actions: torch.Tensor,
        gravity_dir: torch.Tensor,
    ) -> torch.Tensor:
        """Build (N, 994) decoder obs from 10-frame history arrays (oldest->newest)."""
        parts = [
            tokens.reshape(-1, 64),
            ang_vel.reshape(-1, 30),
            joint_pos.reshape(-1, 290),
            joint_vel.reshape(-1, 290),
            last_actions.reshape(-1, 290),
            gravity_dir.reshape(-1, 30),
        ]
        return torch.cat(parts, dim=1)

    def decode(
        self,
        tokens: torch.Tensor,
        ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        last_actions: torch.Tensor,
        gravity_dir: torch.Tensor,
    ) -> torch.Tensor:
        """Return (N, 29) normalized joint actions in IsaacLab order."""
        tokens_q = self.quantize_tokens(tokens)
        obs = self.build_decoder_obs(tokens_q, ang_vel, joint_pos, joint_vel, last_actions, gravity_dir)
        return self.net(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (N, 994) float32 -> (N, 29). Assumes tokens already quantized."""
        return self.net(obs)
