# AGENTS.md — 本仓库工作须知

> 面向在此仓库工作的 AI 代理。项目全部文档为中文，新文档/日志请保持中文。

## 项目是什么

在 Unitree G1 人形上，用 NVIDIA GEAR-SONIC 官方 token 数据 + 冻结解码器，复现
Science Robotics 2026 的 APT-RL（动作预训练 Transformer 强化学习）管线的**研究型
实验仓库**。产出是实验结论（含大量负结果），不是可复用的软件库。项目已阶段性闭环
（E1–E47 + MQ/TO 系列），详见 `HANDOFF/00_FINAL_SUMMARY.md`。

## 环境约定（最重要）

- **本机（Windows）没有任何训练 venv**，仅做代码检视/文档整理/轻量分析。
  **不要在本机安装依赖或尝试运行训练/评测**；`requirements.txt` 记录的是服务器
  两个 venv 的版本（`.venv_isaac` 主 / `.venv_mjlab` 对照），不是本地安装清单。
- 训练/评测全部在服务器：`ssh lab-ts`（本机 ~/.ssh/config 别名，走 Tailscale；
  该机器与旧地址 `10.16.52.225` 同一台，仅换了网络，详见
  `HANDOFF/04_SERVER_GUIDE.md` §1），目录 `~/ros2_data`，经包装脚本
  `/tmp/run_apt_isaac.sh`（source venv + PYTHONPATH + cwd=GR00T-WholeBodyControl）。
  命令模板、目录布局见同文件后续章节。
- `apt_g1/data/` 与 `apt_g1/outputs/` 被 gitignore；clone 后为空，canonical 产物
  在服务器，索引见 `HANDOFF/03_OUTPUTS_INDEX.md`。仓库还全局忽略
  mp4/png/npy/pt/onnx 等重型产物，不要试图提交它们。

## 改代码前必读（按序）

1. `HANDOFF/README.md` — 项目定位、"什么算我们的贡献"口径、11+ 条核心结论。
2. `apt_g1/SCRIPT_MAP.md` — **每个脚本的角色与分类**（CANONICAL / ARCHIVE /
   FORK / DEV / MODULE）。找脚本、判断某脚本是否现役，先查这里。
3. `refine-logs/EXPERIMENT_TRACKER.md` — 全部实验 Run/Data 行的汇总台账；
   `HANDOFF/02_EXPERIMENT_HISTORY.md` 是实验史叙事版。

## 修改规则

- **新脚本必须登记进 `apt_g1/SCRIPT_MAP.md`**（角色 + 用途 + 对应实验号），
  否则后续无法判断其地位。
- 被取代的脚本**移入 `apt_g1/_archive/` 并在 SCRIPT_MAP 标注归档原因**，不要删除。
- 实验有结果（含负结果）必须记入 `refine-logs/tracker/` 对应系列文件
  （R/D/E/MQ/TO，数据唯一事实源；`EXPERIMENT_TRACKER.md` 仅为总索引，
  记完同步其行数统计），
  重大节点同步更新 `HANDOFF/` 相关文档。
- 实验记录**新文档（阶段叶子/专题日志/调研/收束报告等）先挂
  `refine-logs/README.md` 的扇出树、再写内容**（层级/命名/状态规则见该页），
  提交前 `python refine-logs/tools/tree_check.py` 全绿（挂树/实存/链接三项）；
  新文档头部带【层位】导航条并回链父域。Run 行仍只追加进 tracker 系列文件，
  不新开文件。
- 实验号是全仓库通用语言（E27、E39、MQ09、TO06……），新实验接着编号，
  结论表述要与 HANDOFF/README.md §3 的既有口径一致（例如"0.08 悬崖是蒸馏
  路径边界，非解码器本身"这类已修正归因不要写回旧说法）。
- 提交信息惯例：`feat(e35): ...`、`chore: ...`（experiment 号入 scope，中英混排可）。
- `apt_g1/isaac/server_*.py` 三件是与非 server 版同源但已分叉的 FORK，改动时
  注意两边不要误合并。

## 服务器操作 gotchas

- 后台启动用 `nohup ... > log 2>&1 < /dev/null & disown`；ssh 仍可能等 20–30s
  超时，进程其实已启动，用第二条 ssh 验证。
- 本地是 PowerShell/Git Bash，向 ssh 传含引号/括号的命令会被本地解析破坏；
  复杂命令用 base64 管道（`echo <b64> | base64 -d | python -`）。
- Isaac Sim viewport/hydra 渲染在服务器段错误；3D 视频用 MuJoCo offscreen
  （`apt_g1/replay_render_mujoco.py`，`MUJOCO_GL=egl`）。

## 科研 skill 路由（2026-08-27 配置）

四维矩阵（数据管理 / 项目管理 / 思路创新与衔接 / 内容产出）见用户级 skill
`gr00t-research-ops`（`~/.zcode/skills/`），后端 codex/researchclaw/evo 均已
冒烟通过。核心分工：复盘补产出走 `autoresearch`；新实验设计走
`agent-codex-autoscirub`（rubric 防设计漂移）；文献衔接走 `agent-researchclaw`；
跨会话结论先写 evo（group `gr00t-apt`）、动手前先检索。论文写作类 skill
暂不启用。

## 目录速查

```
HANDOFF/      交接包（README → 00_FINAL_SUMMARY → 02/03/04）
apt_g1/       主实验代码；SCRIPT_MAP.md 是索引；_archive/ 是坟场
  isaac/      Isaac Lab 训练栈（env/PPO/train/eval，E1–E47）
  configs/    MuJoCo 平坦地主线的 yaml 配置
refine-logs/  全部实验日志（README.md 是扇出树根地图，树完整性由
              tools/tree_check.py 强制；tracker/ 五系列文件为事实源总台账，
              EXPERIMENT_TRACKER 为索引；stages/ 阶段叶子；FINAL_REPORT 收束定稿）
docs/         NVIDIA GR00T 离线文档 + 路线图（只读参考）
review-stage/ 自动评审记录；tmp/ scratch（gitignored）
```

文档钻取层级（L0 结论卡 → L4 原始产物，每层越往下越细）：总图见
`HANDOFF/README.md` §0；各层文档顶部有【层位】导航条，补新文档时保持。
实验记录文档同时挂 `refine-logs/README.md` 扇出树（树深度 ≠ 层位编号 L0–L4，
两套口径的区分见该页扇出规则）。
