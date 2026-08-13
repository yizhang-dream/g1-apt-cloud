"""Compare onnx2torch conversion against the ONNX decoder output."""

import numpy as np
import onnxruntime as ort
import torch
from onnx2torch import convert

P = (
    "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
    "gear_sonic_deploy/policy/release/model_decoder.onnx"
)
rng = np.random.default_rng(0)
obs = rng.normal(0.0, 1.0, (4, 994)).astype(np.float32)

sess = ort.InferenceSession(P, providers=["CPUExecutionProvider"])
outs = []
for i in range(4):
    outs.append(sess.run(["action"], {"obs_dict": obs[i : i + 1]})[0][0])
out_ort = np.stack(outs)

model = convert(P).eval()
with torch.no_grad():
    out_torch = model(torch.from_numpy(obs)).numpy()
    out_torch_b1 = model(torch.from_numpy(obs[:1])).numpy()

print("ort  :", np.round(out_ort[0, :5], 5))
print("torch:", np.round(out_torch[0, :5], 5))
print("max_abs_diff batch4:", float(np.abs(out_ort - out_torch).max()))
print("max_abs_diff batch1:", float(np.abs(out_ort[:1] - out_torch_b1).max()))
print("model:", type(model))
