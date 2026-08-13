"""Inspect the released SONIC decoder ONNX graph (ops, shapes, weights)."""

import onnx

p = (
    "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
    "gear_sonic_deploy/policy/release/model_decoder.onnx"
)
m = onnx.load(p)
print("opset:", [(o.domain, o.version) for o in m.opset_import])
print("inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print("outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
print("initializer names/sizes:")
for init in m.graph.initializer:
    print(" ", init.name, list(init.dims))
print("nodes:")
for n in m.graph.node:
    print(" ", n.op_type, list(n.input), "->", list(n.output))

import onnx.numpy_helper

init_map = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
print("slice-related constants:")
for n in m.graph.node:
    if n.op_type in ("Slice", "Unsqueeze", "Concat"):
        vals = []
        for inp in n.input:
            if inp in init_map:
                vals.append((inp, init_map[inp].tolist()))
            else:
                vals.append((inp, None))
        print(" ", n.op_type, vals, "->", n.output)
print("constant node values:")
import onnx.numpy_helper as nh
for n in m.graph.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.t:
                print(" ", n.output[0], nh.to_array(a.t).tolist())