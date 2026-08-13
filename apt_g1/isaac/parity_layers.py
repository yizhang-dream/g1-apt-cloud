"""Debug: compare torch decoder layer-by-layer against ORT intermediates."""

import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime as ort
import torch

P = (
    "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
    "gear_sonic_deploy/policy/release/model_decoder.onnx"
)
m = onnx.load(P)
inits = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}

rng = np.random.default_rng(0)
obs = rng.normal(0.0, 1.0, (1, 994)).astype(np.float32)

sess = ort.InferenceSession(P, providers=["CPUExecutionProvider"])
names = [
    "/g1_dyn/module/module.0/Add_output_0",
    "/g1_dyn/module/module.2/Add_output_0",
    "/g1_dyn/module/module.12/Add_output_0",
    "action",
]
res = sess.run(names, {"obs_dict": obs})
for name, arr in zip(names, res):
    print(name, arr.shape, np.round(arr.reshape(-1)[:5], 4))

# torch/numpy replication
x = obs.reshape(1, 994)
w = ["onnx::MatMul_136", "onnx::MatMul_137", "onnx::MatMul_138",
     "onnx::MatMul_139", "onnx::MatMul_140", "onnx::MatMul_141", "onnx::MatMul_142"]
b = ["module.decoders.g1_dyn.module.0.bias", "module.decoders.g1_dyn.module.2.bias",
     "module.decoders.g1_dyn.module.4.bias", "module.decoders.g1_dyn.module.6.bias",
     "module.decoders.g1_dyn.module.8.bias", "module.decoders.g1_dyn.module.10.bias",
     "module.decoders.g1_dyn.module.12.bias"]
print("layer0 numpy:", np.round((x @ inits[w[0]] + inits[b[0]]).reshape(-1)[:5], 4))
print("ort layer0  :", np.round(res[1].reshape(-1)[:5], 4))

# torch linear check
lin0 = torch.nn.Linear(994, 2048)
with torch.no_grad():
    lin0.weight.copy_(torch.from_numpy(inits[w[0]].T.copy()))
    lin0.bias.copy_(torch.from_numpy(inits[b[0]].copy()))
    out0 = lin0(torch.from_numpy(x))
print("torch layer0:", np.round(out0.numpy().reshape(-1)[:5], 4))
