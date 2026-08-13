# APT-RL × SONIC × Unitree G1 复现项目

> 在 Unitree G1 人形上，用 NVIDIA GEAR-SONIC 官方 token 数据 + 冻结解码器，
> 复现 Science Robotics 2026 的 APT-RL（动作预训练 Transformer 强化学习）管线，
> 并回答"蒸馏先验是否必要、论文机制在我们的替代管道里是否成立"。

**状态（2026-08-13）**：阶段性结论已得出（E1–E27 + 三方向 A/B/C）。
完整交接、实验史、结论速查见 [`HANDOFF/`](HANDOFF/README.md) ——**新读者先读它**。

---

## 仓库布局

```
gr00t/
├── HANDOFF/              # 交接包（先读 README.md）：项目总览/论文/实验史/产出索引/服务器指南
├── apt_g1/               # 主实验代码
│   ├── SCRIPT_MAP.md     # ← 每个脚本的用途与分类（CANONICAL / ARCHIVE / FORK）
│   ├── configs/          # MuJoCo 平坦地主线的 23 个 yaml 配置
│   ├── data/             # 数据集（gitignored；canonical = exp_all3，68,093 步）
│   ├── outputs/          # 评测 JSON/模型/视频（gitignored；见 outputs/README.md）
│   ├── encoder/ envs/ policies/ sonic/   # 库模块（相位路由器、G1 环境、策略头、SONIC 封装）
│   ├── isaac/            # Isaac Lab 训练栈（E1–E27 的 env/PPO/train/eval）
│   ├── _archive/         # 被取代/已判死路的脚本（保留可恢复，47 个）
│   └── *.py              # 现行入口与工具（29 个，见 SCRIPT_MAP）
├── refine-logs/          # 全部实验日志与总结（EXPERIMENT_TRACKER / FINAL_REPORT / *_LOG）
├── docs/                 # NVIDIA GR00T 离线文档 + 路线图
├── review-stage/         # 自动评审记录
├── tmp/                  # scratch（SDK 参考拷贝、论文 PDF；gitignored）
├── requirements.txt      # 依赖版本（服务器两 venv 实测）
└── .gitignore
```

## 从哪开始

1. **`HANDOFF/README.md`** —— 项目一句话、完整思路时间线、10 条核心结论、未决事项。
2. **`apt_g1/SCRIPT_MAP.md`** —— 想找某个脚本干什么，先查这里。
3. **`refine-logs/`** —— 逐实验细节（`EXPERIMENT_TRACKER.md` 汇总全部 Run/Data 行；
   `ISAAC_APT_LOG.md` 是 E1–E27 全记录）。

## 复现入口（在服务器上跑，非本机）

服务器：`ssh cvgluser@10.16.52.225`（`~/ros2_data`）。包装脚本 `/tmp/run_apt_isaac.sh`
会 source `.venv_isaac` 并设好 PYTHONPATH。具体命令模板见
`HANDOFF/04_SERVER_GUIDE.md` §4（训练 / 评测 / mjlab 对照恢复）。

关键现役入口：
- Isaac 训练/评测：`apt_g1/isaac/train_apt_isaac.py`、`apt_g1/isaac/eval_apt_isaac.py`
- v9 相位路由器：`apt_g1/train_phase_router_v9.py`
- E27 token VAE：`apt_g1/train_token_vae_e27.py`
- 力矩解码器（方向 A）：`apt_g1/recover_id_torque.py` + `apt_g1/train_torque_decoder.py`

## 环境约定

- **本机（Windows）无训练 venv**，仅作代码检视/整理。训练在服务器。
- 两个服务器 venv 版本见 `requirements.txt`（`.venv_isaac` 主；`.venv_mjlab` 对照）。
- `apt_g1/data/` 与 `apt_g1/outputs/` 被 gitignore（重型产物留在磁盘，不进版本控制）。
  clone 后这两个目录为空；canonical 产物在服务器 `~/ros2_data`，索引见
  `HANDOFF/03_OUTPUTS_INDEX.md`。

## 未决事项（摘要）

- mjlab 从零训练暂停于 ~900/5000 iters（checkpoint 已存，resume 命令见 HANDOFF/04）。
- 自洽 TO/逆动力学"规划"力矩数据：复现论文 aux 正向价值的最后前置条件，当前不可得。
- `apt_g1/isaac/server_*.py` 三件与非 server 版同源但已分叉演进，哪套为"正统"未判定。
