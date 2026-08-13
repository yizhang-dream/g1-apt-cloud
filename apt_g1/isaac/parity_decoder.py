"""Numerical parity check: ONNX decoder (batch 1) vs torch MLP decoder."""

import numpy as np
import torch

from apt_g1.isaac.sonic_decoder_isaac import SonicBatchedDecoder
from apt_g1.isaac.sonic_decoder_torch import SonicTorchDecoder

P = (
    "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
    "gear_sonic_deploy/policy/release/model_decoder.onnx"
)

rng = np.random.default_rng(0)
obs = rng.normal(0.0, 1.0, (4, 994)).astype(np.float32)

ort = SonicBatchedDecoder(P, providers=["CPUExecutionProvider"])
outs = []
for i in range(4):
    o = ort.decode(
        obs[i : i + 1, :64],
        obs[i : i + 1, 64:94].reshape(1, 10, 3),
        obs[i : i + 1, 94:384].reshape(1, 10, 29),
        obs[i : i + 1, 384:674].reshape(1, 10, 29),
        obs[i : i + 1, 674:964].reshape(1, 10, 29),
        obs[i : i + 1, 964:994].reshape(1, 10, 3),
    )
    outs.append(o[0])
out_ort = np.stack(outs)
print("ort out", out_ort.shape, out_ort[0, :4])

torch_dec = SonicTorchDecoder(P, device="cpu")
with torch.no_grad():
    out_torch = torch_dec(torch.from_numpy(obs))
print("torch out", out_torch.shape, out_torch[0, :4].numpy())

max_abs = float(np.abs(out_ort - out_torch.numpy()).max())
print("max_abs_diff", max_abs)
assert max_abs < 1e-3, "parity FAIL"
print("PARITY OK")
