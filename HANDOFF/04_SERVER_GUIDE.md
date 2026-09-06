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
   **单仓多端（08-31）**：GitHub `yizhang-dream/g1-apt-cloud` = 本地 gr00t
   仓强推的统一历史；lab-ts 克隆在 `~/ros2_data/g1-apt-cloud-sync`
   （user.name/email 已配，可 push）；关键产物放 `apt_g1/outputs/sync/`
   （白名单跟踪），/personal 不可信。

## 6. 环境版本

- `.venv_isaac`：Python 3.10、torch 2.5.1、isaaclab 2.1.0、warp-lang 1.16.0、
  onnx2torch 1.15.5
- `.venv_mjlab`：torch 2.13.0+cu130、mjlab 1.2.0、mujoco-warp 3.5.0、
  mujoco 3.5.0、warp-lang 1.12.0、rsl-rl-lib 5.0.1
- GPU：NVIDIA RTX 3060 12GB（驱动 595.84，CUDA 13.2）
- 网络：服务器可访问 pypi / github（已实测）

## 7. CVGL 集群（算力主平台，2026-09-06 接入闭合）

lab-ts 之外的第二计算平台，定位 = 算力主平台（owner 09-06 指令）；lab-ts
保留给确定性复现/hash 对照（同 seed 跨硬件浮点不可逐位比，判读基准 run
必须同硬件）。接入已全链闭合：迁移、Vulkan 冒烟、eval 数值对照均通过。

- **登录**：`ssh cvgl` → 登录节点 `cvglloginnode`，用户 **zyz**（注意与
  lab-ts 的 cvgluser 不同名，路径坑已由容器软链绕过，见下）。Determined
  调度，det master = `10.0.1.66`，集群名 CVGL。
- **det 认证**：密码可从登录节点 `~/.docker/config.json` 的 harbor 凭据
  base64 解出（`zyz:...`，det 与 harbor 共用账号）；`det user login`
  一次即持久化 token 于 `~/.det/auth.yaml`。
- **磁盘**：登录节点根盘 384G 仅剩 ~30G（docker 盘小，**拉不动官方
  isaac 容器**——这是迁移定调 venv 直搬的原因）；`/home/zyz` 本身是 NAS
  挂载（`nas.cvgl.lab:/mnt/Peter/Workspace/zyz`，6.9T 可用 6.8T），数据
  直接放 `~/gr00t/`；agent 节点同一 NAS 挂在 `/workspace/zyz/`，det
  bind_mount 后容器内路径 = `/run/determined/workdir/home/`。备选
  `/UNSAFE_SSD4`（12T，可写）。
- **资源池**（8 agent × 8 slots）：`32c64t_256_3090`(node01)、
  `48c96t_512_4090`(node02)、`48c96t_512_3090`(node03/04)、
  `64c128t_512_4090`(node05)、`128c256t_1536_4090`(node06/07)、
  `128c256t_1536_6000Ada`(node08, RTX 6000 Ada 48G)。
  **资源使用纪律（owner 2026-09-06 指令）：不用 6000Ada 池（node08
  留给他用）；尽量用弱卡——优先 3090 池（`32c64t_256_3090` /
  `48c96t_512_3090`），不足再上 4090 池。**
- **数据布局（前会话已搬，勿重复 rsync）**：`~/gr00t/` 下 .venv_isaac
  21G + GR00T-WholeBodyControl 20G（含 outputs 7G）+ apt_g1 64G（含
  data 63G）+ uv_python 77M ≈ 105G；`token_stats_e49.npz` 已在。
  增量同步在 lab-ts 上直接 `rsync -a --info=progress2 ~/ros2_data/...
  cvgl:gr00t/...`（Tailscale 直连 `10.0.1.67:22332`，实测 29.3 MB/s，
  ~50G 量级约 30 min）。
- **venv 路径设计**：容器内 pyvenv.cfg/home 与 bin/python 软链已改写为
  `/run/determined/workdir/home/gr00t/uv_python/...`；镜像内建软链
  `/home/cvgluser/ros2_data → /run/determined/workdir/home/gr00t`，
  使 lab-ts 风格绝对路径在容器内直接解析。**宿主机登录节点上 venv 不可
  用**（软链指向容器路径，设计使然，勿在登录节点跑）。
- **镜像**：`harbor.cvgl.lab/library/zyz-apt-gr00t:u2204-isaac-pip-v3`
  （810MB，已推 harbor）；容器内 torch 2.5.1+cu124 CUDA OK。
- **冒烟状态**：Vulkan 通过（Isaac 日志 "Graphics API: Vulkan"，
  driver 590.48.01）；已知无害噪音 = GLFW headless 告警、iray 缺
  libGLU.so.1（只影响 iray 渲染，物理 eval 不受影响；要 3D 视频需在
  镜像补 libglu1-mesa）。
- **eval 数值对照（闭合）**：4090 跑 `isaac_e49a_fix2_s0/policy_it_50.pt`
  3 seeds：vx 0.696/0.688/0.698、disp 1.463/2.038/0.552、零摔；lab-ts
  基线 vx 0.703/0.684/0.686、disp 0.515/2.074/0.744——vx 差 ±0.01，
  disp 在 lab-ts 自身种子方差内（与 tracker E.md it_50 口径一致）。
  完整日志：cvgl `~/gr00t/smoke_logs/{venv,diag,simapp,eval_cluster}.log`。
- **任务模板**：本地仓 `tmp/cvgl_diag.yaml` / `cvgl_eval4.yaml` /
  `cvgl_eval5.yaml`（eval 用 `128c256t_1536_4090` 池）。
- **训练吞吐**：截至 09-06 尚无真训练数据（只有 eval），首次训练发射
  后回填 step/s 与 3060 对比。
