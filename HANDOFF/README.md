# APT-RL × SONIC × G1 实验交接包（总览）

> 生成日期：2026-08-13。本文件夹是给"接手此项目的另一个 AI / 未来的我"
> 的完整交接。先读本文件，再按需读 `01~04` 子文档。

## 1. 项目一句话

在 Unitree G1 人形上，用 NVIDIA GEAR-SONIC 的官方 token 数据 + 冻结解码器，
复现 Science Robotics 2026 的 APT-RL（动作预训练 Transformer 强化学习）
论文管线（TO 数据 → 潜空间 → RL 增强 → 地形泛化），并回答"蒸馏先验是否
必要、论文机制在我们的替代管道里是否成立"。

## 2. 完整思路时间线

### 阶段 0：学习（D:\GR00T-LearningPlan）
- 按 `GR00T前置学习路线.md` 学了 GR00T 栈：01-foundation → 02-decoupled-wbc
  → 03-gear-sonic → 04-motionbricks → 05-deployment → 06-integration →
  07-aptrl-sonic → 08-wu-cvgl-humanoid。
- 读了 APT-RL 论文原文（`gr00t\tmp\pdfs\paper.txt` / `scirobotics.adz7397.pdf`）。
- 动手起点：用户想用 GR00T/SONIC 资产在 G1 上复现 APT-RL 的机制。

### 阶段 1：MuJoCo 两条线（R/D 系列，结论：基础设施不足）
- 控制器线（R001-R015，单进程 MuJoCo PPO）：7 个 RL 变体全部劣于 aux=0
  基线 → 判定单进程 PPO 样本效率不足，终止（详见 `refine-logs/MUJOCO_APT_LOG.md`、
  `EXPERIMENT_TRACKER.md` R 系列、`ROOT_CAUSE.md`）。
- 数据线（D001-D028）：官方闭环数据采集（D001）、BC 回归（D003，开环 val
  MSE 0.0012 但闭环 20-30 倍复合误差）、闭合周期数据（闭合误差 0.00000 但
  无益）、walk 方向 oracle（D028：官方 token 开环回放 6 新方向 95-270 步内
  全倒）。
- 关键发现：SONIC 是 200ms 闭环跟踪器，长效动作靠 planner 重规划；去掉后
  教师本身做不了 walk+转向。

### 阶段 2：Isaac 机制矩阵（E1-E14，平坦地面）
- 自建 Isaac Lab DirectRLEnv：G1 + 冻结相位路由器 + torch SONIC 解码器
  （ONNX→torch），64 envs，0.7s/iter。
- E9/E11 vanilla RL（直出 29 维关节）：800/2000 iters 均 0/3 立即倒 →
  蒸馏先验必要。
- aux 在平坦无正向价值（超速+偏航漂移）；E13 修正门控 2Hz 是唯一正向机制
  （位移 3-11m → 34-37m）。

### 阶段 3：地形/感知/动作通道（E15-E21）
- 特权 elevation 地图进 aux 无价值（E16）；gate 学会选先验最优组但 aux 仍
  破坏（E20c：0.06 上 gate-only 3/3 前进 38-43m）；离散台阶/小障碍先验
  可过，垫脚石/连续粗糙 >0.06 是盲区。

### 阶段 4：用户优先级链（2026-08-12 夜 ~ 08-13 凌晨）
- 优先级 1（数据来源定论）：官方长效数据只能来自 planner 闭环重规划或真
  TO/ID 力矩数据；官方 token 开环扩充被判死。
- 优先级 2（平坦命令完备性）：显式回退表 → 24/24 命令 3/3×20s 无跌倒；
  60s+ 马拉松 2/3（walk_back 段长跑后脆弱）。
- 优先级 3（Isaac aux 判据）：E22a/E22b 均不达标（A 34-36m / 0.9-4.3m vs
  阈值 42.9m）→ 定论：位置 token + aux 关节偏移管道下 aux 无正向价值。

### 阶段 5：三方向（2026-08-13）
- 方向 A 力矩级数据：ID 力矩解码器 val MAE 4.13；混合 ×0.2/0.3 平坦 3/3
  与基线持平，但论文式纯 ID 前馈 2.5s 倒、粗糙无增益 → 缺自洽 TO/ID 规划
  数据。
- 方向 B 连续潜空间：相位插值读取无损（MuJoCo 3/3）；E23 自由相位坍缩、
  E24 +压力恢复鲁棒但慢、E25 相位锚定消融证明 aux 是唯一破坏源（aux 归零
  49m）、E26 纯相位偏移全项达标（46m）但地形 0.08 无增益。
- 方向 C 千级并行：Isaac 128 envs 仅 +61% 吞吐（瓶颈在更新/同步，不在显存）；
  官方正解 mjlab（MuJoCo-Warp）1024 envs 冒烟通过（1.03s/iter，0.84GB）。

### 阶段 6：E27 缺失实验（2026-08-13，用户指定优先）
- 用户指出"自主学习、不通过 Sonic 学行为、但让 Sonic 动"这一格从未干净
  试过。实现：相位条件化 token VAE（z∈ℝ16，D(z,φ)→token）+ Isaac latent
  模式（固定步态时钟 0.121 rad/步）+ z_walk warm-start。
- 结果：A 3/3 走 19.1m（vx 0.32）、B 12/12、C 3/3、D 3/3。对比 E9/E11
  vanilla（直出关节 0/3 立即倒）→ **Sonic token 流形本身就是关键先验**。

## 3. 核心结论速查（10 条）

1. 蒸馏先验（冻结相位路由器 + SONIC 解码器）在当前设备/数据条件下是最优
   可用控制器：平坦全项通过，rough 0.06 3/3。
2. 同等预算 vanilla RL（直出关节）无法行走（E9/E11 0/3）。
3. aux 关节偏移通道在 SONIC 位置 token 管道下永远是破坏源（E2-E22b、E25
   消融全部一致）；复现论文 aux 正向价值需力矩级解码器 + 自洽 TO/ID 数据。
4. 2Hz 门控是唯一被验证为正的论文机制（E13）。
5. gate 选择（anti-stop 压力下）能收敛到选先验最优组，无损但不增益（E20c）。
6. 连续潜空间（相位插值/锚定）无损但机制中性；E26 全项达标但 A 仅 ≈97%
   基线，地形 0.08 无增益。
7. 相位条件化 token VAE + 步态时钟 + z_walk warm-start 让"无行为先验"
   RL 能学会行走（E27：19m），速度减半是先验缺失的样本效率代价。
8. 力矩级数据（ID）改善前馈但单独不能解锁论文式控制。
9. 千级并行正解 = mjlab（MuJoCo-Warp）官方配方，独立于蒸馏管道；3060
   12GB 可跑 1024 envs（0.84GB），4096 默认需 24GB GPU。
10. "蒸馏是否必要"的最终答案：蒸馏先验 + 回退表是可用下限；官方从零配方
    是外部对照（mjlab 已跑 ~900 iters，跌倒率 0.08-0.17，可 resume）。

## 4. 交接快速路径

按此顺序读：
1. `02_EXPERIMENT_HISTORY.md` —— 实验历史与结论
2. `03_OUTPUTS_INDEX.md` —— 所有产出在哪里（本地 + 服务器）
3. `04_SERVER_GUIDE.md` —— 服务器怎么用、代码怎么放
4. `01_PAPERS_AND_LINKS.md` —— 论文与仓库地址
5. `C:\Users\zyz\Documents\gr00t\refine-logs\STAGE_SUMMARY_2026-08-13.md` +
   `FINAL_REPORT.md` —— 官方总结

## 5. 未决事项

- 文档 "G 200 T"（用户 2026-08-11 提到，内含"另一个 codex 的上下文"）：
  始终未找到，待用户提供路径；可能是第二篇论文/另一份上下文。
- mjlab 从零训练暂停于 ~900/5000 iters（跌倒率 0.08-0.17，checkpoint 已存），
  恢复命令见 `04_SERVER_GUIDE.md`。
- 自洽 TO/逆动力学"规划"力矩数据：复现论文 aux 正向价值的最后前置条件，
  当前实验室内不可得。
- 本地无 3D 渲染视频产出环境（无显示服务器）；如需视频需本地有显示环境。
