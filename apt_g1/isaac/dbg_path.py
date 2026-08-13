"""Diagnose sys.path / PYTHONPATH before and after AppLauncher."""

import os
import sys

print("before: PYTHONPATH=", os.environ.get("PYTHONPATH"))
print("before: sys.path[0:4]=", sys.path[0:4])
try:
    import apt_g1

    print("apt_g1 import before AppLauncher: OK", apt_g1.__file__)
except Exception as e:  # noqa: BLE001
    print("apt_g1 import before AppLauncher: FAIL", e)

from isaaclab.app import AppLauncher  # noqa: E402

launcher_args = type(
    "A", (), {"headless": True, "num_envs": 1, "env_spacing": 4.0, "output_dir": "/tmp/dbg"}
)()
app = AppLauncher(launcher_args)

print("after: PYTHONPATH=", os.environ.get("PYTHONPATH"))
print("after: sys.path[0:6]=", sys.path[0:6])
try:
    import apt_g1

    print("apt_g1 import after AppLauncher: OK", apt_g1.__file__)
except Exception as e:  # noqa: BLE001
    print("apt_g1 import after AppLauncher: FAIL", e)

app.app.close()
