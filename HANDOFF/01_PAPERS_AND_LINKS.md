# 论文与仓库地址

## 1. 主论文：APT-RL

- 标题：Agile perceptive multiskill locomotion for quadrupedal robots in
  the wild（APT-RL: action pretrained transformer-based reinforcement learning）
- 作者：Jun-Gill Kang, Jaehyun Park, Tae-Gyu Song, Joon-Ha Kim, Seungwoo Hong,
  Hae-Won Park 等（KT / 首尔国立）
- 期刊：Science Robotics 11(116), eadz7397（2026）
- DOI：10.1126/scirobotics.adz7397
- 官网：https://www.science.org/doi/10.1126/scirobotics.adz7397
- 本地全文：`C:\Users\zyz\Documents\gr00t\tmp\pdfs\paper.txt`（纯文本）、
  `C:\Users\zyz\Documents\gr00t\tmp\pdfs\scirobotics.adz7397.pdf`
- 我们要复现的机制：TO 轨迹数据 → TVAE 潜空间（z∈ℝ16，KL 0.1）→ RL latent
  动作 + aux(12) + gait 选择 + 2Hz 门控 + 特权地图 → 感知蒸馏；混合控制
  τ=τ_dec+PD。

## 2. SONIC / GR00T 栈

- GEAR-SONIC：NVIDIA GEAR Lab 的运动跟踪基础模型 + 数据/遥操作/推理工具链。
  角色：官方闭环数据源 + 冻结 token 解码器（64-d FSQ token → 29 维关节目标）。
  - 仓库：https://github.com/NVIDIA/GEAR-SONIC
  - 本项目代码：`D:\GR00T-WholeBodyControl\gear_sonic\`
- GR00T-WholeBodyControl：本项目的宿主仓库（三子系统：decoupled_wbc /
  gear_sonic / gear_sonic_deploy）。
  - 仓库：https://github.com/NVIDIA/GR00T-WholeBodyControl
  - ElasticBand 实现参考：`D:\GR00T-WholeBodyControl\gear_sonic\utils\mujoco_sim\unitree_sdk2py_bridge.py`
- GR00T N1.5/N1.6（Decoupled WBC 背后的控制器）：背景知识，本项目未直接
  使用其权重。

## 3. 官方 RL 配方（方向 C 对照）

- unitree_rl_mjlab：Unitree 官方开源 RL 训练仓库（MuJoCo 后端）。
  - 仓库：https://github.com/unitreerobotics/unitree_rl_mjlab
  - 服务器路径：`/home/cvgluser/ros2_data/unitree_rl_mjlab/`
  - 训练命令：`python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096`
- mjlab：Isaac Lab API 的 MuJoCo-Warp 实现（NVIDIA/Google DeepMind）。
  - 仓库：https://github.com/mujocolab/mjlab
  - PyPI：https://pypi.org/project/mjlab/（本项目用 1.2.0）
- mujoco-warp：GPU 并行 MuJoCo。
  - 仓库：https://github.com/google-deepmind/mujoco_warp
- Isaac Lab：本项目主 RL 框架。
  - 仓库：https://github.com/isaac-sim/IsaacLab

## 4. 学习资料（阶段 0）

- 学习计划目录：`D:\GR00T-LearningPlan\`
  - `GR00T前置学习路线.md`、`前置知识启动计划-详细版.md`、`README.md`
  - 01-foundation ~ 08-wu-cvgl-humanoid（GR00T 栈分模块学习笔记）
  - 07-aptrl-sonic（APT-RL × SONIC 结合方向）

## 5. 本项目所有日志所在

`C:\Users\zyz\Documents\gr00t\refine-logs\`：

- `FINAL_REPORT.md`（2026-08-12 综合报告）
- `STAGE_SUMMARY_2026-08-13.md`（阶段总结：优先级链 + 方向 A/B/C）
- `EXPERIMENT_TRACKER.md`（全部 Run/Data/实验行）
- `MUJOCO_APT_LOG.md`、`DISTILL_EXPERIMENT.md`、`ROOT_CAUSE.md`
- `DATA_GENERALIZATION_LOG.md`（数据/泛化/三方向 12-22 节）
- `ISAAC_APT_LOG.md`（Isaac E1-E27 全记录）
- `HUMAN_READABLE_COMPLETE_REPORT.md`、`EXPERIMENT_PLAN.md`
- `APT_PROJECT_SUMMARY.md`（总览，含指针）
