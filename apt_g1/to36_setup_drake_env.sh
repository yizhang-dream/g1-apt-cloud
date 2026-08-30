#!/usr/bin/env bash
# TO36 D1 前置：服务器建 .venv_drake（无 sudo / ensurepip 缺失 → --without-pip + get-pip 引导，
# 与 .venv_isaac 同惯例，见 HANDOFF/04_SERVER_GUIDE.md §「无 sudo」节）。
# 用法：bash to36_setup_drake_env.sh
set -euo pipefail

cd /home/cvgluser/ros2_data
echo "[env] python3 = $(python3 --version 2>&1)"
# Drake 官方 wheel 只覆盖部分 CPython 版本（3.10/3.11/3.12 视版本）；
# 若 pip install drake 找不到 wheel，改用 .venv_isaac 的解释器版本建 venv，
# 或 pip install drake==<匹配版本>（见设计文档 §5 风险表）。

if [ ! -d .venv_drake ]; then
  python3 -m venv --without-pip .venv_drake
  source .venv_drake/bin/activate
  curl -sS https://bootstrap.pypa.io/pip/get-pip.py -o /tmp/get-pip.py
  python3 /tmp/get-pip.py
else
  source .venv_drake/bin/activate
fi

pip install -U pip
pip install drake numpy
python3 - <<'EOF'
import pydrake
from pydrake.all import MultibodyPlant, Parser, DirectCollocation
from pydrake.solvers import IpoptSolver
print("[env] pydrake import OK:", pydrake.__version__ if hasattr(pydrake, "__version__") else "n/a")
print("[env] IPOPT available:", IpoptSolver().available())
EOF
echo "[env] .venv_drake 就绪。跑 D1："
echo "  source /home/cvgluser/ros2_data/.venv_drake/bin/activate"
echo "  python3 apt_g1/to36_leg_to_drake.py load"
