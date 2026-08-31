# 服务器使用指南

## 1. 连接

```text
命令:    ssh lab-ts        # 本机 ~/.ssh/config 别名（Tailscale：100.112.92.62）
```

- **与旧地址 `cvgluser@10.16.52.225` 是同一台机器**（2026-08-27 比对三把 sshd
  host key 指纹完全一致，同一系统安装）：机器从校园网搬到家用路由网络
  （192.168.31.x），改走 Tailscale 访问。`~/ros2_data` 及全部 canonical 产物不受影响。
- 硬件即原训练机：Legion 刃 7000P，RTX 3060 12GB；跑大 batch Isaac 训练前核对显存。
- gotcha：包装脚本在 `/tmp/run_apt_isaac.sh`，**机器重启即丢失**；重建方法见 §3。

## 2. 目录布局（/home/cvgluser/ros2_data）

```text
ros2_data/
├── apt_g1/                  # 主实验代码（= 本地 C:\...\gr00t\apt_g1 的服务端版本）
│   ├── data/exp_all3/       # 主数据集
│   ├── isaac/               # Isaac env / PPO / train / eval
│   ├── outputs/             # 评测 JSON + 训练日志 + token_vae_e27
│   ├── sonic/ encoder/ envs/ policies/ configs/
│   └── train_*.py / eval_*.py / build_*.py ...
├── GR00T-WholeBodyControl/  # 宿主仓库（含 outputs/ 下全部 Isaac checkpoint）
├── unitree_rl_mjlab/        # 官方从零配方（对照）
├── .venv_isaac/             # Isaac Lab 训练环境（主）
├── .venv_mjlab/             # mjlab 环境（对照）
├── xr_teleoperate/ Humanoid/ groot_transfer_bundle_20260722/
├── proj2605.md cmd/ gr00t_env.bash ...
```

## 3. 运行方式（Isaac）

包装脚本 `/tmp/run_apt_isaac.sh`：

```bash
#!/bin/bash
source /home/cvgluser/ros2_data/.venv_isaac/bin/activate
export PYTHONPATH=/home/cvgluser/ros2_data:/home/cvgluser/ros2_data/apt_g1:/home/cvgluser/ros2_data/GR00T-WholeBodyControl
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
cd /home/cvgluser/ros2_data/GR00T-WholeBodyControl   # 重要：cwd=仓库根
exec python "$@"
```

后台启动模板（注意 `</dev/null & disown` 防止 ssh 挂起）：

```bash
cd /home/cvgluser/ros2_data && \
nohup bash /tmp/run_apt_isaac.sh <script> <args...> \
  > /home/cvgluser/ros2_data/apt_g1/outputs/<run>.log 2>&1 < /dev/null & disown; echo OK
```

## 4. 常用命令模板

### 训练（示例：E27 latent）

```bash
cd /home/cvgluser/ros2_data && nohup bash /tmp/run_apt_isaac.sh \
  /home/cvgluser/ros2_data/apt_g1/isaac/train_apt_isaac.py \
  --num-envs 64 --iters 800 --rollout 24 --vx-max 0.8 --use-2hz-gate 1 \
  --latent-mode --latent-warmstart-iters 200 --phase-warmstart-coef 10.0 \
  --latent-kl 2.5e-6 --latent-expl 0.01 --entropy 0.001 \
  --anti-stop 1.0 --anti-stop-thresh 0.1 --progress-scale 0.3 --seed 0 \
  --out outputs/isaac_e27_latent \
  > /home/cvgluser/ros2_data/apt_g1/outputs/e27_train.log 2>&1 < /dev/null & disown; echo OK
```

checkpoint 落在 `GR00T-WholeBodyControl/outputs/<out>/policy_it_*.pt` +
`policy_final.pt`（每 50 iters 存一次）。

### 评测（示例：E27 A/B/C/D）

```bash
cd /home/cvgluser/ros2_data && nohup bash /tmp/run_apt_isaac.sh \
  /home/cvgluser/ros2_data/apt_g1/isaac/eval_apt_isaac.py \
  --checkpoint /home/cvgluser/ros2_data/GR00T-WholeBodyControl/outputs/isaac_e27_latent/policy_final.pt \
  --tests A,B,C,D --latent-mode \
  --out /home/cvgluser/ros2_data/apt_g1/outputs/isaac_eval_e27.json \
  > /home/cvgluser/ros2_data/apt_g1/outputs/e27_eval.log 2>&1 < /dev/null & disown; echo OK
```

关键 flag：`--phase-mode` / `--phase-anchor` / `--latent-mode` /
`--aux-scale`（eval 用） / `--terrain rough --terrain-noise 0.06
--terrain-seed 0` / `--phase-zero`（消融） / `--aux-zero`（消融）。

### 恢复 mjlab 从零训练（对照基线）

```bash
cd /home/cvgluser/ros2_data/unitree_rl_mjlab && \
nohup /home/cvgluser/ros2_data/.venv_mjlab/bin/python scripts/train.py \
  Unitree-G1-Flat --env.scene.num-envs=1024 --agent.max_iterations=5000 \
  --agent.logger tensorboard --video False --agent.resume \
  > /home/cvgluser/ros2_data/apt_g1/outputs/mjlab_train_g1_flat.log 2>&1 < /dev/null & disown; echo OK
```

## 5. 常见坑（重要）

1. **PowerShell 引号**：本地是 PowerShell，向 ssh 传含引号/括号的命令会被
   本地解析破坏。对策：用 base64 管道
   `echo <b64> | base64 -d | python -`，或避免内嵌复杂引号。
2. **ssh 挂起**：`nohup ... > log 2>&1 < /dev/null & disown` 仍可能让 ssh
   等 20-30s 超时；进程其实已启动，用第二条 ssh 验证即可。
3. **无 sudo / ensurepip 缺失**：建 venv 用
   `python3 -m venv --without-pip` + `curl get-pip.py` 引导。
4. **版本配对（mjlab）**：mujoco-warp 3.5.0 ↔ mujoco 3.5.0（3.11 缺
   mjENBL_MULTICCD）；mjlab 1.2.0 ↔ warp-lang 1.12.0（1.16 缺 wp.context）；
   logger 默认 wandb 需 `--agent.logger tensorboard`。
5. **cwd**：run_apt_isaac.sh 会把 cwd 设到 `GR00T-WholeBodyControl`，所以
   相对 `outputs/...` 会落到那里；VAE 等纯 torch 脚本直接在
   `~/ros2_data/apt_g1` 下跑。
6. **评测脚本**：`--tests A` 只过滤输出字典；旧版本仍会跑完全部 rollout
   （新版已修成真正跳过）。phase/latent 模式下 `aux` 与 `noaux` 键结果相同
   （latent 无 aux 通道），评测时间翻倍属正常。
7. **云上 Isaac 训练（2026-08-31 已全链打通，走 flux task 而非开发机）**：
   **pybind `Unable to cast <class 'list'>` 的真根因**（dev 实例 pp1–pp8
   二分定位）：train 头部注入 `sys.path.insert(0, Path(__file__).parent)`
   把 **PosixPath 对象**塞进 sys.path，kit 启动时把 sys.path 列表传给
   C++ 转换失败——`str()` 包装修复（此前「镜像系统性问题/需要平台介入」
   的归因不成立）。云训练官方正路 = `flux task`（git 方式 codeType=2，
   公开仓；startScript 必须 `gm-run <repo-dir>/script.py` 相对
   /workspace/isaaclab）而非开发机 SSH。已修的兼容问题清单：镜像
   IsaacLab 2.0.0 无 `effort_limit_sim`（用 `effort_limit`）、apt_g1/encoder
   包与 G1 URDF mesh（67 STL）需入仓、`ASSET_DIR` 须绝对路径（kit 启动后
   chdir）。ckpt 平台自动发现 output/**.pt，`flux task model list` 可取；
   任务 route 算力 ESKU000004 A10=¥4.01/时。任务 JSON 模板见
   `gr00t/tmp/task_to39*.json`。

## 6. 环境版本

- `.venv_isaac`：Python 3.10、torch 2.5.1、isaaclab 2.1.0、warp-lang 1.16.0、
  onnx2torch 1.15.5
- `.venv_mjlab`：torch 2.13.0+cu130、mjlab 1.2.0、mujoco-warp 3.5.0、
  mujoco 3.5.0、warp-lang 1.12.0、rsl-rl-lib 5.0.1
- GPU：NVIDIA RTX 3060 12GB（驱动 595.84，CUDA 13.2）
- 网络：服务器可访问 pypi / github（已实测）
