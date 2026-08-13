# Isaac Lab APT 训练栈（G1 平坦地面）

## 环境

- 服务器：`cvgluser@10.16.52.225`（RTX 3060 12GB，驱动 595.84）
- venv：`~/ros2_data/.venv_isaac`（Python 3.10.20，uv 创建）
- 版本：Isaac Lab 2.1.0 + Isaac Sim 4.5（pip 安装）

## 安装（已踩坑，按序执行）

```bash
uv venv --seed .venv_isaac --python 3.10
source .venv_isaac/bin/activate
pip install -U pip
pip install "setuptools<81" wheel
pip install --no-build-isolation flatdict==4.0.1   # setuptools>=81 无 pkg_resources
pip install "isaaclab[isaacsim,all]==2.1.0" --extra-index-url https://pypi.nvidia.com
pip install onnxruntime-gpu==1.23.2
```

运行前必须：

```bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONPATH=/home/cvgluser/ros2_data:/home/cvgluser/ros2_data/apt_g1:/home/cvgluser/ros2_data/GR00T-WholeBodyControl
cd /home/cvgluser/ros2_data/GR00T-WholeBodyControl   # 资产相对路径
```

## 运行

冒烟测试：

```bash
python -u /home/cvgluser/ros2_data/apt_g1/isaac/smoke_isaac.py --num-envs 4 --steps 50
```

训练 aux（冻结相位路由器先验 + PPO）：

```bash
python -u /home/cvgluser/ros2_data/apt_g1/isaac/train_apt_isaac.py \
  --num-envs 64 --iters 600 --rollout 24 --vx-max 0.8 --use-2hz-gate 1 \
  --latent-kl 2.5e-6 --latent-expl 0.01 --entropy 0.001 \
  --out /home/cvgluser/ros2_data/apt_g1/outputs/isaac_e13_gate_fix
```

vanilla 基线（无先验）：

```bash
python -u /home/cvgluser/ros2_data/apt_g1/isaac/train_apt_isaac.py \
  --env vanilla --num-envs 64 --iters 800 --rollout 24 --vx-max 0.8 \
  --out /home/cvgluser/ros2_data/apt_g1/outputs/isaac_e9_vanilla
```

评测（A/B/C/D，`eval_fast.py` 只跑指定项）：

```bash
python -u /home/cvgluser/ros2_data/apt_g1/isaac/eval_fast.py \
  --checkpoint .../policy_it_500.pt --tests A,B,C,D --out .../eval.json
```

## 关键参数

- `--use-2hz-gate`：1=决策绑定门控（E13 修正版），0=关闭。
- `--aux-scale`：aux 关节修正幅度（默认 0.2，论文值）。
- `--aux-l2/--aux-rate`：aux 正则（平坦地面建议 0）。
- `--vel-sigma2/--yaw-sigma2`：速度/偏航奖励宽度（默认 0.25；越紧越逼跟踪）。
- `--phase-mode`：策略直接选相位（latent 动作；需配 `--phase-warmstart-iters`）。
- `--disturbance-prob --disturbance-ramp-iters`：扰动课程（MuJoCo C2 语义：
  每回合概率调度一次 200–500N 推力）。

## 已知注意点

- 训练/评测脚本用 `os._exit(0)` 退出（`simulation_app.close()` 在该环境会挂死）。
- SONIC 解码器已转成 torch 动态 batch（ONNX 是固定 batch=1）。
- 评测必须把 `cfg.scene.num_envs=1` 设在 env 创建前（否则跑 64 envs 浪费）。
- 所有 checkpoint 建议取中段（300–500 iters），后期 PPO 有退化倾向。
