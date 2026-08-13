"""Minimal import check for the Isaac venv."""

import onnxruntime as ort

print("ort", ort.__version__, ort.get_available_providers())

import isaaclab  # noqa: E402

print("isaaclab", getattr(isaaclab, "__version__", "?"))
print("isaacsim import ok")
