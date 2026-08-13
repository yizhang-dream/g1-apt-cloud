# Isaac Lab APT 阶段运行记录（2026-08-12 起）

## 背景与切换依据

MuJoCo 两条线均已按论文判据走完并留档（见 `MUJOCO_APT_LOG.md`）：

- 控制器线：7 个 RL 变体全部劣于 aux=0 基线，原因=单进程 MuJoCo PPO 样本效率不足（缺并行向量化 RL 基础设施）。
- 数据线：闭合周期判据（闭合误差 0.00000）达成但无益，原因=漂移根源在控制器层缺闭环修正。

按目标约定，两条线结束后切换到 Isaac Lab。论文方法对照（2026-08-12 拆解）确认 Isaac 阶段要补齐的
**RL 阶段缺失机制**：

1. 并行向量化 RL（Isaac Lab / Isaac Sim 4.5，数千 env 级）。
2. latent KL 正则（论文系数 2.5e-6，soft 约束，不强制 encoder 正则）。
3. latent 探索奖励逐步衰减到 0（exploration -> exploitation）。
4. 2 Hz 步态门控：决策 2 Hz、保持 0.5 s、门控状态与 2 Hz 布尔信号进入观测。
5. 命令区间拉宽（论文 vx ∈ [-1,7]；先做 vx 0–0.8 与 aux=0 基线对比，再逐步拉宽）。

刻意替换（不算遗漏）：TO 数据 -> SONIC 官方闭环数据；TVAE -> 相位路由器；力矩解码器 -> SONIC ONNX 解码器。
感知蒸馏（depth+LIDAR teacher->student）留到地形/真机阶段。

## 环境信息

- 服务器：`cvgluser@10.16.52.225`，RTX 3060 12GB，驱动 595.84，EGL/Vulkan 就绪。
- venv：`~/ros2_data/.venv_isaac`（Python 3.10.20 via uv，pip 26.2.1，setuptools 80.10.2）。
- 目标版本：Isaac Lab 2.1.0 + Isaac Sim 4.5（与 gear_sonic 的 `train_agent_trl.py` 默认 `isaaclab.app.AppLauncher` 路径兼容）。

## 安装踩坑（均已解决并留档）

1. 系统 python3 缺 venv/ensurepip -> 改用 uv 托管 Python 3.10 + `uv venv --seed`。
2. `flatdict==4.0.1` sdist 构建失败：setuptools>=81 移除 pkg_resources ->
   先 `pip install "setuptools<81" wheel`，再 `pip install --no-build-isolation flatdict==4.0.1`，
   之后主安装（默认 build isolation）直接复用已装 flatdict，不再触发构建。
3. `--no-build-isolation` 跑主安装会缺 poetry-core（另一个 sdist 依赖）-> 不要全局关隔离；
   只对 flatdict 关隔离预装。
4. 安装脚本必须 LF 行尾（PowerShell here-string 的 CRLF 会让 bash 脚本里的命令/重定向带上 `\r`）。

安装命令（成功版 `install_isaac5.sh`）：

```bash
source .venv_isaac/bin/activate
pip install -U pip
pip install "setuptools<81" wheel
pip install --no-build-isolation flatdict==4.0.1
pip install "isaaclab[isaacsim,all]==2.1.0" --extra-index-url https://pypi.nvidia.com
```

## Isaac env 设计（`apt_g1/isaac/`）

| 文件 | 内容 |
|---|---|
| `apt_flat_env.py` | DirectRLEnv 子类：G1(29-DOF) + 平面地形；冻结相位路由器 -> 64-d token -> SONIC ONNX -> 29-d 关节目标；aux(12) 叠加在下肢；2 Hz 门控保持 + 反馈观测；扰动；速度跟踪奖励；A/B/C/D 可评测 |
| `batched_router.py` | 批量相位路由器（per-group MLP 掩码前向 + EMA），与 MuJoCo 单步版输出一致 |
| `sonic_decoder_isaac.py` | 批量 SONIC ONNX 解码（994-d 输入，FSQ 量化，CUDA/CPU provider） |
| `ppo_core.py` | 向量化 PPO：GAE/clip/entropy + latent KL(2.5e-6) + latent 探索奖励衰减 |
| `train_apt_isaac.py` | 训练入口（AppLauncher + rollout + update + ckpt） |
| `eval_apt_isaac.py` | A/B/C/D 评测（对齐 MuJoCo `eval_apt_aux.py` 口径：3 seeds、jitter、500N 脉冲、68s 切换、jump mode 17） |

关键对齐点（与 MuJoCo `mujoco_g1_flat_env.py` 逐条核对）：

- 10 帧 history 布局：ang_vel(3)、joint_pos_rel(29, SONIC 序)、joint_vel(29)、last_actions(29)、gravity(3)，旧->新。
- decoder obs：64 + 30 + 290 + 290 + 290 + 30 = 994。
- q_des = sonic_default_isaac + action*sonic_scale_isaac；aux 叠加到 12 个下肢关节（hip/knee/ankle）。
- 终止：root z < 0.2 或非有限；奖励与 MuJoCo 相同五项（track_xy/yaw/upright/height/stillness）+ 提前终止 -10。
- 扰动：pelvis 体坐标系外力（evaluation 中世界系方向先转体坐标系），一个控制步内保持。

## 实验矩阵（计划）

| # | 变体 | 目的 | 状态 |
|---|---|---|---|
| E1 | aux=0 基线（Isaac 并行版） | 验证 Isaac 下冻结路由器闭环与 MuJoCo 一致 | 待跑 |
| E2 | aux PPO（latent KL + 探索衰减 + 2Hz 门控，vx 0–0.8） | 控制器线主实验 | 待跑 |
| E3 | phase+aux 联合 RL（policy 选相位） | 论文 latent action 直接类比 | 待跑 |
| E4 | E2 + 扰动课程 | 鲁棒性 | 待跑 |
| E5 | E2 + 命令区间拉宽（vx 0–2 / -1–1） | 论文命令覆盖 | 待跑 |
| E6 | 2 Hz 门控消融（关掉 hold/反馈） | 门控机制贡献 | 待跑 |

每条线评测：A 60s 行走（vx=0.8）、B 500N×4 方向扰动 45s、C 68s 命令切换、D 跳跃 20s，
与 aux=0 对比；结果写回本文件与 `EXPERIMENT_TRACKER.md`。

## E1：Isaac aux=0 基线 A/B/C/D 评测（2026-08-12，已完成）

- 环境：`AptFlatG1Env`，num_envs=1 逐项评测，3 seeds，jitter 对齐 MuJoCo。
- 结果文件：`apt_g1/outputs/isaac_eval_noaux.json`。

| 测试 | Isaac noaux | MuJoCo aux=0（对照） | 说明 |
|---|---|---|---|
| A 60s 行走 vx=0.8 | **3/3**（vx≈0.79，位移 47.2–47.4m） | 3/3（~51m） | 同量级 |
| B 500N×4 方向扰动 45s | **12/12** | 10/12 | Isaac 更稳 |
| C 68s 命令切换 | **3/3** | 2/3 | Isaac 更稳 |
| D 跳跃 20s | **3/3**（h_min 0.638） | 2/3 | Isaac 更稳 |

- 附注：E1 的 `aux` 行是随机初始化策略的 aux（无 checkpoint），仅供灵敏度参考：
  A 3/3 但位移降到 27.6m（随机 aux 造成偏航漂移），B/C/D 仍 3/3。
  说明 aux 通道确实影响行为，E2 训练后的 aux 才是真正对照。

## E2：aux PPO 训练（进行中）

- 命令：`train_apt_isaac.py --num-envs 64 --iters 500 --rollout 24 --vx-max 0.8
  --use-2hz-gate 1 --latent-kl 2.5e-6 --latent-expl 0.01 --entropy 0.001 --seed 0`
- 机制：冻结相位路由器 token 先验 + PPO 训练 aux；含论文 latent KL、探索衰减、
  2Hz 门控保持与反馈观测。
- 速度：0.7s/iter（64 env × 24 步），远快于 MuJoCo 单进程。

### E2 结果（aux PPO 500 iters，无扰动，已完成）

- 训练：`outputs/isaac_e2_aux_v1/`（policy_final.pt + train_log.json），500 iters 全程 fall=0，收益平稳 ~1.8。
- 评测：`outputs/isaac_eval_e2_aux.json`

| 测试 | aux（trained） | noaux（对照） | 结论 |
|---|---|---|---|
| A 60s 行走 | 3/3，vx 0.60–0.65，**位移 3–11m** | 3/3，vx 0.79，位移 47m | aux 导致严重偏航漂移 |
| B 500N 扰动 | 12/12（h_min 0.694–0.719） | 12/12（h_min 0.727–0.748） | 存活持平，aux 姿态略低 |
| C 68s 切换 | 3/3 | 3/3 | 持平 |
| D 跳跃 | 3/3（vx 0.57–0.58） | 3/3（vx 0.68） | aux 降低跳跃速度 |

**结论（与 MuJoCo 控制器线互相印证）**：无扰动压力时，PPO 学到的 aux 不是增益修正，
而是把先验行为带偏（位移 47m→3–11m、跳跃速度下降）。存活率全部达标，说明 Isaac 并行 RL
基础设施确实解决了 MuJoCo 的“学不到稳定闭环”问题，但平坦地形 + 低扰动下 aux 没有正向收益。

### E4：aux + 扰动课程（进行中）

- 命令：`--disturbance-prob 0.35 --disturbance-ramp-iters 200 --iters 800`，
  扰动在前 200 iters 从 0 线性升到 0.35。
- 目的：给 aux 提供“需要修正”的梯度压力，检验论文中 aux 的扰动恢复价值。

### E4 结果（aux + 扰动课程，已完成；注意有 GPU 争用）

- 训练：`outputs/isaac_e4_aux_dist/`，800 iters，扰动 0→0.35 ramp 200 iters。
- **坑**：E2/E4 训练进程在 `simulation_app.close()` 上挂死，后续评测与其争抢 GPU；
  已修复为 `os._exit(0)`。E2/E4 数值因此略受噪声影响，但结论稳定复现。
- it_600 快速 A 评测：aux 3/3 但 vx 0.40–0.44、位移 1.96–3.55m（漂移比 E2 更重）；
  noaux 对照 47.7m。
- 结论：加扰动压力后 aux 并未收敛成“只在扰动时修正”，反而更依赖旋转逃逸。
  根因假设：奖励缺少 aux 幅度/速率正则，且 track_yaw 的 σ²=0.25 过宽，
  慢速旋转几乎无代价。

### E7：aux + 扰动 + aux 正则 + 紧偏航（进行中）

- 新增 cfg：`aux_l2_scale=0.01`、`aux_rate_scale=0.005`、`yaw_sigma2=0.1`。
- 命令：`--disturbance-prob 0.35 --disturbance-ramp-iters 200 --iters 800
  --aux-l2 0.01 --aux-rate 0.005 --yaw-sigma2 0.1`
- 目的：让 aux 在“不必要时保持小”，只在扰动/切换时发力。

### E7 作废说明 + E7b（修正扰动语义）

- E7（旧语义）作废：扰动误实现为“每控制步 35% 概率推力”，远严于 MuJoCo C2
  （每回合 35% 概率调度一次 200–500N 单步推力）；收益被压到 0.9–1.2，不具可比性。
- **已修正** `_sample_disturbance`：reset 时按概率调度一次推力（步骤 50..episode-150，
  幅值 uniform 200–500N，方向 ±x/±y 随机），与 MuJoCo C2 语义一致。
- E7b：`--disturbance-prob 0.35 --disturbance-ramp-iters 200 --aux-l2 0.01
  --aux-rate 0.005 --yaw-sigma2 0.1 --iters 800`，输出 `outputs/isaac_e7b_aux_reg/`。

### E3 系列：phase+aux 联合 RL（策略直接选相位 latent）

| 变体 | 配置 | 结果 |
|---|---|---|
| E3 | phase+aux，无 EMA，800 iters | 收益后期 1.2、KL 爆到 10；策略漂移 |
| E3b | + 相位 EMA 0.3（同路由器） | vx≈0.04 站立偷懒；收益 1.6 但没走 |
| E3c | + EMA + 紧速度奖励 vel_sigma2=0.05 | 收益 1.2、vx≈0.07；仍不走路 |

**机制性结论**：离散相位 bin 选择 + MLP PPO 无法学会“推进步态相位”——恒定相位=静态姿势。
路由器是监督回归学到的振荡器，RL 在无结构化 latent 空间上难以重新发现它；
这正解释了论文为何用 TVAE 潜空间（解码器流形天然平滑、时序连贯）。SONIC FSQ token 空间
没有这个结构，用“选 token”替代“选 latent”会丢掉最关键的先验性质。

### E6：2Hz 门控消融（aux-only，gate off，进行中）

- 与 E2 同配置但 `--use-2hz-gate 0`；目的：检验门控保持/反馈观测的贡献。

### E6 结果（2Hz 门控消融，aux-only gate OFF）

- 训练：`outputs/isaac_e6_nogate/`，600 iters，收益 ~1.7–1.9 稳定。
- A 60s 行走：aux 3/3，vx 0.90–0.93（超速），位移 16–32m（seed 相关漂移）；
  noaux 3/3 47m。
- **对比 E2（gate ON）**：位移 16–32m vs 3–11m、vx 0.92 vs 0.63——关掉自由运行的
  2Hz tick 后 aux 的破坏性大幅下降。
- **机制结论**：我们 naive 实现的自由运行 2Hz 布尔信号（每 25 步 tick）被策略当作
  时钟源，诱发周期性 aux 干扰；论文的 2Hz 信号绑定步态决策，与我们的实现不等价。
  这属于实现层面的偏差，不是论文机制本身无效。

### E6 全量结果（gate-off aux，已完成）

| 测试 | E6 aux | noaux |
|---|---|---|
| A 60s | 3/3，vx 0.90–0.93（超速），位移 16–32m | 3/3，vx 0.79，47m |
| B 500N | 12/12（h_min 0.694–0.713） | 12/12（0.736–0.748） |
| C 68s | 3/3 | 3/3 |
| D 跳跃 | 3/3，vx 0.75–0.76（**快于 noaux 0.68**） | 3/3，vx 0.68 |

- 结论：gate-off aux 全项存活；跳跃速度为正收益；直线行走超速且偏航漂移（净位移降）。
- 结果文件：`outputs/isaac_eval_e6_bcd.json`。

### E8：相位 warm-start + RL（进行中）

- `--phase-mode --phase-warmstart-iters 150 --phase-warmstart-coef 10.0
  --use-2hz-gate 0 --iters 800`。
- warm-start：前 150 iters 给策略相位头加 MSE 监督（标签=冻结路由器相位输出），
  系数线性衰减；让策略从“能走”开始再 RL 微调。
- 早期信号：vx 0.20→0.28 爬升（对比 E3b/c 的 0.04），walk 先验被成功注入。

### E9：vanilla RL 基线（无 SONIC 先验，800 iters，已完成）

- `--env vanilla`：29 维关节位置动作直出，同一奖励/物理/PPO/算力。
- 训练：收益 ~0.78–0.94（先验版 ~1.8），摔倒率 1.5–2.1%，vx ~0.55–0.65。
- **A 60s 行走评测：aux 3/3 与 noaux 3/3 全部立即摔倒（h_min≈0.20，位移≈0）**。
- 结论：同等预算下无先验 RL 完全无法获得稳定行走；**蒸馏先验的必要性得到直接证据**。
- 注：评测脚本此前误用 64 envs（cfg 在 env 创建后才设 num_envs=1），结果只读 env0 仍有效，
  但浪费 64 倍时间；已修复，后续评测为单 env 快速模式。

### E10：aux 轻正则、无门控（600 iters，已完成）

- `--aux-l2 0.005 --aux-rate 0.002 --use-2hz-gate 0`，评测 it_500。
- A：aux 3/3，vx 0.91–0.94（超速），位移 2.9–6.5m；noaux 47m。
- 结论：即使加轻正则，aux 仍超速+偏航漂移；漂移是该奖励/动作空间下 aux 的稳健行为，
  与正则强度关系不大。

### E11：vanilla 长训 2000 iters（进行中）

- 目的：样本效率曲线——给 vanilla 2.5 倍算力，看能否达到先验版 800 iters 的水平。

### E11 结果（vanilla 2000 iters，已完成）

- 训练：收益 ~0.75–0.92 平台化（与 800 iters 持平），摔倒率 1.2–2.3%，vx ~0.5–0.7。
- **A 60s 评测：6/6 立即摔倒（h_min≈0.20，位移≈0）**。
- 结论：2.5 倍算力下 vanilla RL 仍无法获得稳定 60s 行走；先验的样本效率优势是数量级性的。

### E8 全量（phase warm-start it_300）B/C/D 评测（进行中）

---

## Isaac 阶段总结（2026-08-12）

### 实验矩阵

| # | 变体 | 训练 | A 60s 行走 | B 扰动 | C 切换 | D 跳跃 |
|---|---|---|---|---|---|---|
| E1 | 冻结路由器 aux=0（基线） | - | 3/3，47m | 12/12 | 3/3 | 3/3 |
| E2 | aux PPO，gate ON，无扰动 | 500it | 3/3，3–11m | 12/12 | 3/3 | 3/3（vx 0.57） |
| E4 | aux+扰动课程（旧语义，作废参考） | 800it | it_600：3/3，2–4m | - | - | - |
| E6 | aux PPO，gate OFF | 600it | 3/3，16–32m | 12/12 | 3/3 | 3/3（vx 0.76） |
| E7b | aux+扰动(修正语义)+正则 | 800it | it_600：**0/3 全倒** | - | - | - |
| E10 | aux 轻正则，gate OFF | 600it | 3/3，3–7m | - | - | - |
| E3/E3b/E3c | phase+aux 联合 RL | 800it | 不走路（vx≈0.04–0.1） | - | - | - |
| E8 | phase warm-start + RL（it_300） | 800it | 3/3，vx 0.99，7–19m | B/C/D 评测中 | | |
| E9 | vanilla RL（无先验） | 800it | **0/3 立即摔倒** | - | - | - |
| E11 | vanilla RL 长训 | 2000it | **0/3 立即摔倒** | - | - | - |

### 论文机制逐项状态（Isaac 已补齐）

| 论文机制 | 实现 | 结果 |
|---|---|---|
| 并行向量化 RL | Isaac Lab 64 envs，0.7s/iter | ✓ 基础设施问题解决 |
| latent KL 软约束（2.5e-6） | ✓ 训练项 | 平坦地面无显著影响 |
| latent 探索奖励衰减 | ✓ 训练项 | 前期有助探索，后期未阻止退化 |
| 2Hz 步态门控 + 0.5s 保持 + 反馈观测 | ✓ env 实现 | naive 自由运行 tick 被策略当钟表，反而有害（E2 vs E6） |
| 扰动课程 | ✓ 修正为 MuJoCo C2 语义 | aux 未因此获益（E7b 崩坏） |
| aux 正则（L2/rate）+ 紧偏航 | ✓ 新增 | 未消除超速/漂移（E10/E7b） |
| latent action（相位选择） | ✓ phase-mode | 离散 bin 无时序结构，MLP 学不出振荡器（E3 系列）；warm-start 可注入但 RL 微调退化（E8） |
| TVAE 潜空间 | ✗（SONIC token 替代） | 缺失平滑流形是 phase 线失败的根本原因 |
| vanilla RL 对比 | ✓ 自建 | 同等预算完全失败，先验必要 |

### 最终结论（Isaac 平坦地面）

1. **蒸馏先验是必要的**：冻结相位路由器（aux=0）全项最优（A 47m、B/C/D 全过）；
   同等预算 vanilla RL 无法行走（E9/E11 立即摔倒）。
2. **aux 在平坦地面无正向收益**：全部 learned-aux 变体存活率达标但超速+偏航漂移
   （A 净位移 3–32m vs 47m），部分变体评测时全倒（E7b）。
   与论文逻辑一致：aux 的价值在“先验覆盖不到的工况”（地形/强扰动），平坦地面 aux=0 最优。
3. **phase/latent 线需要结构化潜空间**：离散 SONIC token 相位选择不具备论文 TVAE 的
   时序平滑性，MLP PPO 无法从零发现步态振荡器；warm-start 可注入但 RL 微调不稳定。
4. **实现偏差记录**：naive 2Hz tick（自由运行）与论文的“步态决策绑定信号”不等价；
   后者需要步态决策状态而非纯时钟。
5. 工程：Isaac Lab 2.1 + Sim 4.5 训练栈已打通（安装踩坑、torch SONIC 解码器、
   批量相位路由器、A/B/C/D 评测），后续地形/真机实验可直接复用。

### E8 全量结果（phase warm-start it_300，已完成）

| 测试 | E8 phaseaux | noaux |
|---|---|---|
| A 60s | 3/3，vx 0.99–1.01（超速），位移 7–19m | 3/3，vx 0.79，47m |
| B 500N | 12/12（h_min 0.702–0.727） | 12/12 |
| C 68s | 3/3 | 3/3 |
| D 跳跃 | 3/3，vx 0.76–0.78（快于 noaux 0.68） | 3/3 |

- 结论：warm-start 相位策略全项存活；与 E6 相同的模式（A 超速漂移、D 跳跃更快）。
- 结果文件：`outputs/isaac_eval_e8_bcd.json`。

## 实验矩阵已完整（11 组训练 + 8 组评测，全部留档）

### E12：aux scale 0.05（600 iters，已完成）

- 训练：收益 1.86 稳定（比 E2/E6 后期更健康），无摔倒。
- A：aux 3/3，vx 0.56–0.59（低于指令 0.8，小尺度限幅），位移 0.6–4.5m；noaux 47m。
- 结论：**漂移与 aux 幅度关系不大**——即使 4 倍缩小 aux 权限，策略仍用持续的小幅
  关节修正造成慢速偏航累积。漂移是该动作空间/奖励下 aux 的稳健行为。
- 结果文件：`outputs/isaac_e12_train_log.json`（+ A 评测见服务器 /tmp/isaac_e12_A500.json）。

### E13：修正版 2Hz 门控（aux，600 iters，已完成）

- 实现：门控布尔信号只在“路由器组切换”时置 1（决策绑定），组选择在 2Hz 边界评估并保持 0.5s；
  取代 E2 的自由运行 tick。
- 训练：收益 1.74–1.83 **全程稳定（无 E2/E6 后期退化）**。
- A 60s 行走（it_500）：aux 3/3，vx 0.66，位移 **34–37m**；noaux 47m。
  - 对比：E2 naive-gate 3–11m，E6 gate-off 16–32m。
- **机制结论（正向结果）**：论文的 2Hz 门控按“决策绑定信号”语义实现后，
  显著减少 aux 的偏航漂移（3–11m → 34–37m），且训练更稳定。
  之前 E2 的漂移主要来自自由运行 tick 被策略当作钟表，属于实现偏差而非机制无效。
- B/C/D 评测进行中。

### E13 全量结果（修正门控 aux，it_500，已完成）

| 测试 | E13 aux | noaux |
|---|---|---|
| A 60s | 3/3，vx 0.66，位移 **34–37m** | 3/3，vx 0.79，47m |
| B 500N | 12/12（h_min 0.727–0.734） | 12/12 |
| C 68s | 3/3 | 3/3 |
| D 跳跃 | 3/3（vx 0.43–0.45，h_min 0.61） | 3/3（vx 0.68） |

- **最终 aux 对照结论**：修正门控把 aux 的 A 项位移从 3–11m（E2）提升到 34–37m
  （基线的 ~78%），但 D 跳跃速度下降（0.44 vs 0.68）。
- 综合所有变体：aux=0 仍是平坦地面最优；修正门控 aux 是最优 aux 变体。
- 结果文件：`outputs/isaac_eval_e13_bcd.json`、`outputs/isaac_e13_train_log.json`。

### 完整 Isaac 矩阵最终版（E1–E13）

| # | 变体 | A 位移 | B | C | D |
|---|---|---|---|---|---|
| E1 基线 aux=0 | 47m | 12/12 | 3/3 | 3/3 |
| E2 aux gate-naive | 3–11m | 12/12 | 3/3 | 3/3 |
| E6 aux gate-off | 16–32m | 12/12 | 3/3 | 3/3（vx 0.76） |
| E13 aux gate-fixed | **34–37m** | 12/12 | 3/3 | 3/3（vx 0.44） |
| E7b aux+dist+reg | 0/3 全倒 | - | - | - |
| E8 phase warm-start | 7–19m | 12/12 | 3/3 | 3/3（vx 0.76） |
| E9/E11 vanilla | 0/3 立即倒 | - | - | - |

### E14：修正门控 + 扰动课程（700 iters，已完成）

- 训练：收益 ~1.5（扰动成本），训练期零摔倒（对比 E7b 有摔倒）。
- A（it_500）：aux 3/3，vx 0.51–0.52，位移 5.7–12.6m；noaux 47m。
- 对比：E13（门控修正、无扰动）34–37m；E7b（naive 门控+扰动+正则）0/3 全倒。
- 结论：**修正门控显著改善扰动训练的稳定性（不崩），但扰动训练本身仍降低直线跟踪质量**；
  平坦地面 aux 的“扰动恢复”仍无法转化为正向收益。
- 结果文件：`outputs/isaac_e14_train_log.json`。

## 实验矩阵最终版（E1–E14）

| # | 变体 | A 位移 | B | C | D |
|---|---|---|---|---|---|
| E1 基线 aux=0 | **47m** | 12/12 | 3/3 | 3/3 |
| E2 aux gate-naive | 3–11m | 12/12 | 3/3 | 3/3 |
| E6 aux gate-off | 16–32m | 12/12 | 3/3 | 3/3（vx 0.76） |
| E13 aux gate-fixed | 34–37m | 12/12 | 3/3 | 3/3（vx 0.44） |
| E14 aux gate-fixed+dist | 5.7–12.6m | - | - | - |
| E7b aux dist+reg | 0/3 全倒 | - | - | - |
| E8 phase warm-start | 7–19m | 12/12 | 3/3 | 3/3（vx 0.76） |
| E9/E11 vanilla | 0/3 立即倒 | - | - | - |

## 地形实验（2026-08-12，进行中）

### 方法学修复：地形生成固定 seed

此前 `terrain_cfg.py` 未给 `TerrainGeneratorCfg` 设 seed，每次运行生成不同的随机
高度场——0.08 比 0.10 更差等非单调结果无法归因，aux/noaux 也无法在同一地形上
公平对比。已修复：`make_terrain_importer_cfg(terrain_type, noise, seed=0)`，
`train_apt_isaac.py` / `eval_fast.py` 新增 `--terrain-seed`（默认 0），
`eval_fast.py` 新增 `--keys aux,noaux` 以便只跑 noaux 基线。

### 先验（aux=0）盲走鲁棒性曲线（固定 terrain-seed 0，3 seeds，60s，vx=0.8）

| noise | 完成 | 说明 |
|---|---|---|
| 0.04 | **3/3** | vx 0.76–0.78，位移 44.6–47.0m |
| 0.06 | **3/3** | vx 0.73–0.76，位移 40.6–44.2m |
| 0.08 | **1/3** | seed0 24.3m 存活；seed1/2 约 29–41s 跌倒 |
| 0.10 | **0/3** | 6.5–29s 跌倒 |

- 结果文件：`apt_g1/outputs/terr_sweep_n{0.04,0.06,0.08,0.10}_s0.json`。
- 结论：冻结路由器（平坦数据蒸馏）的盲走能力悬崖在 noise 0.06→0.08 之间；
  这正是 E15 aux 应该有正向价值的难度区间（先验失败但 RL 仍能引导）。

### 修复效果确认

- reset root z 改为 `env_origins + 0.76 + jitter`（修复脚插地）；
- `reset_grace_steps=25`（修复 reset 碰撞早期误判终止）；
- 0.10 修复后不再"瞬倒"（最差 seed 也坚持到 ~6.5s），但离稳定行走仍有距离。

### 地形可复现性二次修复（重要）

`TerrainGeneratorCfg.seed` 并不能固定 `HfRandomUniformTerrainCfg` 的高度场：
`random_uniform_terrain` 内部使用**全局 `np.random.choice`**，与生成器的局部
rng 无关。因此即使传 `--terrain-seed`，不同进程仍生成不同地形（0.06 在扫描与
E15 评测间不一致的根因）。

修复：`eval_fast.py` / `train_apt_isaac.py` 在创建 env 前显式
`np.random.seed(terrain_seed)`（全局 numpy RNG 被固定 → 高度场可复现）。
修复后 0.06 s0 两次 noaux 均为 3/3（此前一次 3/3 一次 2/3）。

### E15 最终结果（gate-fixed aux，0.04→0.06→0.08 课程，固定 seed 0）

| noise | aux | noaux |
|---|---|---|
| 0.04 | 3/3（vx 0.40–0.43，3.7–11m） | 3/3（vx 0.76–0.78，42–46m） |
| 0.06 | 2/3（vx 0.43–0.48，1.7–9.7m） | 3/3（vx 0.71–0.78，36–46m） |
| 0.08 | 0/3（2.3–5.3s 倒） | 0–1/3（19–42s；seed2 一次 42.5m） |
| 0.10 | 0/3（10–25s；个别 seed 站住 vx≈0.04） | 0/3（6.5–29s） |

结果文件：`outputs/terr_fix_*.json`、`terr_e15_*.json`、`terr_sweep_*.json`。

**E15 结论**：纯 proprio 的 aux 在粗糙地形上无正向价值——平坦/轻度噪声下它把
速度砍半（0.4 vs 0.75），悬崖噪声下 0/9 完成行走（仅个别 seed 靠"站住"多撑
一会）。论文中 aux 的正向案例（越障、断腿补偿）依赖 elevation map 感知，
本次实验确认了该依赖：**无感知时 aux 只是反应式拐杖，不能提前修正落脚**。
这与用户记忆一致：论文先给特权地图，后蒸馏感知。

### E16：aux + 特权 elevation map（2026-08-12，进行中）

按论文教师式设计实现：`isaac/elevation_map.py` 包裹高度场生成函数记录网格
（与物理地形严格一致），`_get_observations` 采样机器人前方 9×9 @0.15m 局部
地形块（随 yaw 旋转，相对根高度）加入观测。训练/评测新增 `--use-elevation`。

### E16 结果（elevation 版 aux，0.04→0.06→0.08 课程，固定 seed 0）

| noise | E16 aux（+elevation） | E15 aux（盲） | noaux |
|---|---|---|---|
| 0.06 | **0/3**（2–4s 倒，vx≈0） | 2/3（1.7–9.7m） | 3/3（42–44m） |
| 0.08 | **0/3**（2.5–4.6s 倒） | 0/3（2.3–5.3s 倒） | 0–1/3 |

- 结果文件：`outputs/terr_e16_n{0.06,0.08}_s0.json`。
- **结论（负面但精确）**：把特权地图接进策略后 aux 没有变好，反而在 0.06 变差
  （0/3 vs 盲 aux 2/3）。原因不是"地图没用"，而是**我们只有 aux 动作通道，
  地图没有可表达的载体**——论文中 elevation 的价值通过 latent 动作 + gait
  logit（改变整身计划/步态选择）传导，而我们的相位路由器和相位头是冻结/哑的。
  这精确定位了架构缺口：要检验论文"先给地图"路径，需要让策略能用地形信息
  调节 latent/步态，而不是只调 12 维关节偏移。

### E15/E16 对论文对照的最终意义

- 论文 aux 的正向案例（越障、断腿、地形适应）依赖：elevation 感知 + latent/gait
  选择通道 + 力矩级解码器。三者我们目前都缺（感知刚接入但无表达通道；latent
  是离散相位；解码器是 SONIC 位置目标）。
- 因此平坦地面 aux=0 最优、粗糙地面 aux 无益，不是论文机制"无效"，而是我们的
  替代实现缺少其生效条件。完整复现论文需要（按优先级）：
  1. 让策略输出步态/组选择（gate logit）并用地形信息调制（论文 gait logit 类比）；
  2. 结构化连续潜空间（TVAE 或 Gaitor 式）替换离散相位；
  3. 力矩级数据/解码器（论文原方案，SONIC 数据无力矩标注）。

### E17：策略学习的步态/组选择（gate logit 类比）+ elevation（2026-08-12）

实现：策略输出 12 维 aux + 3 选 1 组选择（idle/slow_walk/walk_fwd），2Hz 决策
绑定（与 E13 门控同语义），命令/奖励仍跟踪 vx=0.8。训练同 E16 课程 + elevation。

结果（固定 seed 0）：

| noise | E17 aux（gate+aux） | E17 noaux（gate，aux=0） |
|---|---|---|
| 0.06 | 0/3（2.3–5.6s 倒，vx≈−0.5） | **3/3 但 vx≈0（站住）** |
| 0.08 | 0/3（2.5–4s 倒） | **3/3 但 vx≈0（站住）** |

- 结果文件：`outputs/terr_e17_n{0.06,0.08}_s0.json`。
- **结论（负面但揭示机制）**：给策略组选择通道后，最小化奖励下它学会"站住"来
  规避摔倒（noaux 3/3 但零位移；track_xy 在 vx=0 仍有 0.077 残值 + 终止惩罚
  −10 的权衡）。aux 版本则反向移动后跌倒。**论文的步态选择机制依赖"必须前进"
  的任务/奖励设计**（速度跟踪 + 风格奖励 + 真实部署），我们的最小奖励允许
  idle 坍缩。这正是论文奖励设计中"task + regularization + style"缺一不可的
  证据。

### E17b：gate + elevation + 前进进度奖励（2026-08-12，进行中）

在 E17 基础上加 `progress_scale=0.3`（vx 正向进度奖励），验证任务压力能否
阻止 idle 坍缩、逼出真正的地形自适应步态选择。

### E17b 结果（gate+aux+elevation+progress，固定 seed 0）

| noise | E17b aux | E17b noaux（gate，aux=0） |
|---|---|---|
| 0.06 | 0/3（3.2–3.8s 倒，vx 0.77–1.16） | 3/3 但 vx≈0（站住） |
| 0.08 | 0/3（3.2–11s 倒） | 3/3 但 vx≈0（站住） |

- 结果文件：`outputs/terr_e17b_n{0.06,0.08}_s0.json`。
- 训练期 vx 恢复到 0.67–0.82（进度奖励生效），但闭环 0.08 仍 0/3；noaux 行
  aux 置零导致观测分布偏移，gate 依旧选 idle。
- **结论**：任务压力能防止"站住"，但无法让 aux/gate/elevation 组合在粗糙地形
  稳定行走。与 E15/E16/E17 一起构成完整负面证据链：在我们的替代管道
  （SONIC 位置解码器 + 最小奖励 + 64 envs）下，论文的 aux、elevation、步态
  选择三个机制都不产生地形价值；根因指向缺失的力矩级结构化解码器与大规模并行。

## E15–E17 综合结论（地形与感知方向）

1. 冻结相位路由器先验在平坦地面全项通过（A 47m / B 12-12 / C 3-3 / D 3-3）；
   粗糙地形盲走能力悬崖在噪声 0.06→0.08。
2. 盲 aux（E15）、aux+elevation（E16）、aux+gate+elevation（E17/E17b）在
   0.06–0.08 全部无法稳定行走（0/3 或站住），部分配置还比 noaux 更差。
3. 论文机制的生效条件（per 我们的替代实现）：力矩级解码器 + 结构化连续潜空间
   + 大规模并行 + 强制任务奖励。缺任一环节，aux/elevation/gate 都无正向价值。

### E18：phase 直控（策略输出相位）+ aux + elevation + 进度奖励

- 配置：`--phase-mode --phase-warmstart-iters 150 --coef 10` + elevation +
  progress 0.3，0.04→0.08 课程。训练期 vx 0.5–1.1（能走），但闭环粗糙地形
  0/3：0.06 在 100–125 步倒、0.08 在 91–114 步倒。
- 结果文件：`outputs/terr_e18_n{0.06,0.08}_s0.json`。
- 结论：warm-start 衰减后策略相位偏离路由器精确相位，elevation 与进度奖励
  都无法阻止；直接相位控制比冻结路由器（noaux 0.06 3/3）显著更差。

### E19：全程相位锚定（进行中）

- 同 E18，但 `--phase-warmstart-iters 1000 --coef 2.0`——相位 MSE 监督全程
  保留（线性衰减但不归零），让策略只在路由器相位流形附近微调 + aux + elevation
  适应地形。这是论文「latent 动作在先验流形内自适应」的安全版测试。

### E19 结果（phase 锚定 + aux + elevation）

| noise | phase+aux | phase-only（aux=0） |
|---|---|---|
| 0.06 | **3/3 存活**（vx −0.06..−0.07，0.9–6.2m 漂移） | 3/3 存活（vx≈0.005 站住） |
| 0.08 | **3/3 存活**（vx −0.05..−0.11，2.1–5.6m） | 3/3 存活（vx≈0.004 站住） |

- 结果文件：`outputs/terr_e19_n{0.06,0.08}_s0.json`、`terr_e19b_*.json`。
- **关键发现**：相位锚定（策略相位被路由器监督拉回先验流形）后，学习策略首次在
  0.06/0.08 达到 3/3 存活——抗倒性超过冻结先验（0.08 先验仅 0–1/3）。但代价是
  丢失前进运动：aux 版缓慢倒走（vx≈−0.06），aux=0 版站住。**"存活 vs 任务"权衡**
  再次出现：最小奖励下策略选择不倒（倒走/站住）而非前进；论文的完整奖励（速度
  跟踪权重 + 风格项）会惩罚这种行为。
- 对论文对照的意义：**先验流形约束（相位锚定）是 learned 组件能存活的关键**，
  这佐证了论文"latent 动作 + 冻结 decoder"设计中先验约束的必要性；我们的实现
  缺的是迫使"前进"的任务奖励与更大的动作自由度（力矩）。

### P1：感知蒸馏机制演示（2026-08-12，进行中）

对应论文第 4 阶段与用户记忆"先给地图，后训练感知复原地图"：学生用
（粗 3×3 + 噪声的"感知"代理 + 基座状态）MSE 回归特权 9×9 elevation patch。
脚本 `train_perception_distill.py`。

### P1 结果

- 采集 3,000 组（privileged 9×9 patch, 粗 3×3+σ=0.04 噪声, 基座 9 维）样本
  （rough 0.08，seed 0）。学生 MLP（18→256→256→81）MSE 训练 60 epochs。
- **val MSE 0.00012、MAE 0.0085m（patch std 0.0357m 的 ~24%）、corr 0.954**。
- 结果文件：`outputs/distill_percept/{student.pt, percept_meta.json}`。
- 结论：**机制验证成立**——学生能从粗/带噪的"感知"输入复原特权地图，
  与论文 stage-4（depth+LIDAR → 学生回归教师 exteroceptive latent）同构；
  本演示用粗 patch 代理真实传感器（Isaac 无显示服务器无法渲染深度图），
  真实传感器版留待有渲染环境后替换输入即可。

### E19c：相位锚定 + aux 正则（进行中）

同 E19 但 `--aux-l2 0.01 --aux-rate 0.005`——迫使 aux 只做小修正，目标：
在保留 E19 存活率的同时不丢失前进运动（检验"存活 vs 任务"权衡能否被
aux 正则缓解）。

### E19c 结果

| 场景 | phase+aux（reg） | phase-only |
|---|---|---|
| rough 0.06 | 1/3（seed0 存活 vx 0.018，其余 144–146 步倒） | 0/3 |
| rough 0.08 | 0/3（107–1098 步倒） | 0/3 |
| 平坦 | 3/3（vx 0.27，1.8–2.5m） | 3/3（vx 0.28，2.2–2.8m，约 8s 前进后站住） |

- 结果文件：`outputs/terr_e19c_n{0.06,0.08}_s0.json`、`terr_e19c_flat.json`。
- **结论**：aux 正则（0.01 L2）把 E19 的 3/3 存活率打回 0–1/3——E19 的大幅
  aux 确实在承担地形稳定（代价是倒走），限制它反而破坏存活。平坦地面 phase
  策略只能前进约 8s（2.2m）后站住，任务表现远差于冻结先验（47m）。

## E15–E19c 最终一览（地形/感知/动作通道）

| 变体 | rough 0.06 存活 | rough 0.08 存活 | 前进质量 |
|---|---|---|---|
| 冻结先验 noaux | 3/3 | 0–1/3 | 42–46m |
| E15 盲 aux | 2/3 | 0/3 | 1.7–9.7m |
| E16 aux+elevation | 0/3 | 0/3 | - |
| E17 gate+aux+elevation | 0/3（或站住） | 0/3 | - |
| E17b +progress | 0/3 | 0/3 | 能移动 |
| E18 phase 直控 | 0/3 | 0/3 | 能移动 |
| E19 phase 锚定 | **3/3** | **3/3** | 倒走/站住 |
| E19c +aux 正则 | 1/3 | 0/3 | 平坦前进 8s |

**最终结论**：相位锚定（留在路由器先验流形）是学习组件存活的关键；但没有任何
learned 变体在粗糙地形上达到"冻结先验的前进质量 + 地形存活"兼得。这与
unitree_rl_mjlab 官方配方（地形高度扫描 + 丰富奖励 + 4096 envs）的差距一致：
我们缺的是任务奖励压力、大规模并行与力矩级动作自由度。**在现有替代管道内，
冻结先验 + 相位锚定学习是最优解；继续提升需要换硬件/数据/解码器条件。**

### E20：相位锚定 + anti-stop 方向压力（2026-08-12）

奖励新增 `anti_stop`（vx<0.3 时惩罚，scale 1.0）：把 E19 的"倒走存活"逼成
"正向移动"。

| 场景 | phase+aux | phase-only |
|---|---|---|
| 平坦 | 3/3（vx 0.86–0.88 **快**，但位移仅 1.8–3.5m → 画圈漂移） | 3/3（vx 0.28–0.30） |
| rough 0.04 | 0/3（175–294 步倒，vx 0.93–1.26 超速） | 1/3（22.7m） |
| rough 0.06 | 0/3（124–316 步倒） | 0/3 |
| rough 0.08 | 2/3（vx≈0.05 爬行） | 1/3 |

- 结果文件：`outputs/terr_e20_n{0.04,0.06,0.08}_s0.json`、`terr_e20_flat.json`。
- 结论：anti-stop 把速度压力加上了，但制造了**速度/存活冲突**——平坦/轻度噪声
  下超速画圈或跌倒，硬噪声下退回爬行。没有任何 anti-stop 系数同时拿到
  "直线快速 + 地形存活"。

### E20c：gate + anti-stop（阈值 0.1，进行中）

给 gate 变体加 anti-stop（vx<0.1 才惩罚，允许慢走前进），让策略有动机选
"慢走前进"组而非 idle——检验是否存在"慢速前进 + 地形存活"的甜点。

### E20c 结果（gate + anti-stop，阈值 0.1）

| 场景 | gate+aux | gate-only（aux=0） |
|---|---|---|
| rough 0.06 | 0/3（84–154 步倒，vx −0.3..−0.7 倒走） | **3/3，vx 0.71–0.77，38–43m** |
| rough 0.08 | 0/3（77–142 步倒） | 0/3（seed2 活到 2953 步 ≈ 59s，vx 0.71） |

**E20c gate-only 全量评测（B/C/D）**：

| 环境 | B 500N 扰动 | C 68s 切换 | D 跳跃 |
|---|---|---|---|
| 平坦 | **12/12**（vx 0.70–0.79） | **3/3**（vx 0.68–0.72） | **3/3**（vx 0.69–0.73） |
| rough 0.06 | **10/12** | **3/3** | **3/3** |

- 结果文件：`outputs/e20c_gateonly_{flat,rough06}_BCD.json`。
- **结论升级**：anti-stop 训练的 gate 头（aux=0）在平坦地面与冻结先验**全项持平**
  （B 12/12、C 3/3、D 3/3、A 3/3 前进），rough 0.06 除 B 略降（10/12，地形使
  扰动恢复更难）外全过。**可学习的步态选择机制在正确奖励压力下是"无损"的**——
  它收敛到选择先验自己的最优组；aux 通道则是唯一的破坏源。

**E20c gate-only 跨地形扩展（A 60s，3 个 terrain seed）**：

| noise | s0 | s1 | s2 | 合计 |
|---|---|---|---|---|
| 0.06 | 3/3（38–43m） | 3/3（40–45m） | 3/3（38–43m） | **9/9 前进** |
| 0.08 | 0/3（seed2 59s） | 0/3（21–42s） | 0/3（22–40s） | 0/9 完成，存活 21–59s |

- 结果文件：`outputs/e20c_gateonly_n{0.06,0.08}_s{1,2}.json`。
- 0.08 的"存活 20–60s 后倒"与冻结先验的边缘区一致（先验 0.08 为 0–1/3，
  19–42s）——gate-only 的行为即先验组行为，无增益也无损失。

- 结果文件：`outputs/terr_e20c_n{0.06,0.08}_s0.json`。
- **关键分解结论**：anti-stop 压力让 gate 头学会了**正确选择先验自己的 walk_fwd
  组**——aux=0 时行为与冻结先验一致（0.06 3/3 前进、0.08 接近 60s）；而带 aux
  的行依旧倒走跌倒。这证明：
  1. 可学习的**步态选择**（gate）在正确奖励下能收敛到"选先验最优组"，不破坏
     先验；
  2. **破坏性始终来自 aux 通道**——从 E2/E6/E10/E12/E15 到 E20c，任何学习到的
     aux 修正都让先验退化；
  3. 论文中 aux 的正向价值需要力矩级解码器（PD 稳定下的力矩修正），SONIC 位置
     目标 + 关节偏移的替代无法复现。

### E21a：gate + anti-stop + 特权地图（2026-08-12 夜）

把特权 elevation map 从 E16 的 aux 通道移到策略观测（含 gate 通道），训练
`isaac_e21a_gate_map`（rough 0.06、500 iters、64 envs、anti-stop 0.5@0.1m/s、
gate-sel、2Hz 门控）。命令与产出：

- 训练：`--terrain rough --terrain-noise 0.06 --use-elevation 1 --gate-sel 1
  --anti-stop 0.5 --anti-stop-thresh 0.1`
- 结果：`outputs/terr_e21a_n{0.06,0.08}_s0_it400.json`

| 场景 | gate+aux（E21a） | gate-only（E21a） | 对照 E20c gate-only |
|---|---|---|---|
| rough 0.06 A60s | 2/3 完成，vx 0.13–0.14（慢走 5–7m） | **3/3，37–43m，vx 0.71–0.74** | 3/3，38–43m |
| rough 0.08 A60s | 0/3（vx 0.13–0.14） | 0/3（vx 0.64–0.77） | 0/9 完成（21–59s） |

**结论**：
1. 地图+gate 让 gate+aux 从 E20c 的 0/3 倒走变成 2/3 慢速前进（map 缓解了
   aux 的破坏，但没有产生正向价值，仍远差于先验）。
2. gate-only（aux=0）行为 = 先验行为，0.08 上依旧 0/3——**特权地图不能突破
   token/解码器流形上限**：策略没有"地形对应的新行为"可学（流形里只有
   idle/slow/walk 原型），地图信息无处传导。与 unitree_rl_mjlab 配方对照：
   它的地形价值来自"策略直接输出新步态 + 千级并行 + 丰富任务奖励"，而不是
   仅仅把地图加进观测。
3. 这进一步收敛了方向：感知（地图/深度）不是当前管道的瓶颈；**动作流形
   （力矩/新步态）才是**。

### E21b：先验在离散地形（台阶/垫脚石/障碍）的直接评测（2026-08-12 夜）

`terrain_cfg.py` 新增 stairs / stairs_hi / stones / discrete（Isaac 内置
HfPyramidStairsTerrainCfg / HfSteppingStonesTerrainCfg /
HfDiscreteObstaclesTerrainCfg）。用冻结先验（无 RL）直接测：

| 地形 | A60s 结果 |
|---|---|
| stairs（0.04–0.08m 阶高，0.35m 阶宽） | **3/3 完成，vx 0.30–0.36，18–22m** |
| stairs_hi（0.08–0.14m） | **3/3 完成，vx 0.23–0.33，14–20m** |
| stones（0.25–0.4m 石宽，0.3–0.5m 间距，h≤0.06，洞深 0.5m） | 0/3（~2 步内倒） |
| discrete（0.05–0.10m 障碍 ×10） | **3/3 完成，vx 0.80，48m** |

E20c gate-only（it_450，带地图观测）复测：stairs_hi **3/3**（9–17m，
vx 0.15–0.29，略慢于先验）；stones **0/3**——学习过的 gate 在两类地形上都
保持先验行为。

E20c gate+aux（it_450）复测 stairs_hi：**3/3 "完成"但 vx 0.03–0.04（站住
不爬，位移 1.8–2.3m）**——aux 在台阶上也复现"退化站住"模式，再次确认
aux 是唯一破坏通道。

E21a gate-only（it_400，带地图在 rough 0.06 训练）复测 stairs_hi：
2/3 完成但 vx 0.063（慢爬 3.6–3.7m）+ 1/3 倒——**比 E20c gate-only
（3/3，9–17m）和冻结先验（3/3，14–20m）都差**。带地图在单一 rough
分布训练的 gate 对未见台阶的迁移更差；再次说明 gate 的价值上限 = 先验
流形 + 训练分布覆盖，地图本身没有带来跨地形增益。

### E22a/E22b：优先级 3 判据实验——aux 闭环修正是否"不劣于先验且提升鲁棒性"

按用户判据执行：A 60s 位移 ≥ noaux 的 90%（≈42.9m），且 B/C/D 不退化。

**E22a**（E13 风格：修正决策绑定门控 + aux，600 iters）：
`--use-2hz-gate 1 --latent-kl 2.5e-6 --latent-expl 0.01 --aux-scale 0.2`

| 测试 | aux | noaux | 判据 |
|---|---|---|---|
| A 60s | 3/3，vx 0.66，**34.0–35.9m** | 3/3，vx 0.79，47.3–47.7m | **不达标**（<42.9m） |
| B 500N | 12/12（h 0.720–0.733） | 12/12（h 0.720–0.747） | 持平 |
| C 68s 切换 | 3/3（disp 1.8–3.0m） | 3/3（disp 7.4–8.2m） | aux 更差 |
| D 跳跃 | 3/3（vx 0.42–0.45） | 3/3（vx 0.68） | aux 更差 |

**E22b**（E22a + aux L2/rate 正则 + 紧 yaw）：
`--aux-l2 0.01 --aux-rate 0.005 --yaw-sigma2 0.1`

| 测试 | aux | noaux | 判据 |
|---|---|---|---|
| A 60s | 3/3，vx 0.51，**0.9–4.3m** | 3/3，47.2–47.4m | **不达标（更差）** |
| B 500N | 12/12（disp 1.6–6.1m，原地转） | 12/12（35.3–35.8m） | aux 质量更差 |
| C 68s | 3/3（disp 0.3–5.3m） | 3/3（7.2–7.8m） | aux 更差 |
| D 跳跃 | 3/3（disp 0.6–1.3m，原地振荡） | 3/3（14.1m） | aux 更差 |

**结论（优先级 3 判据未达成，定论）**：在 SONIC 位置 token 先验 + aux 关节
偏移的替代管道下，PPO 学到的 aux 无法满足"不劣于先验 + 扰动/切换提升"：
不加正则（E22a）稳定但 A 只有基线的 ~73%；加正则（E22b）直接坍缩为原地
振荡。这与 E2/E4/E6/E7b/E10/E12/E13/E20c/E21a 全部证据一致——**aux 通道
在该管道下无正向价值；复现论文 aux 正向价值的前提仍是力矩级解码器 +
真实 TO/逆动力学力矩数据（或换 unitree_rl_mjlab 官方 4096-env 配方）**。
结果文件：`outputs/isaac_eval_e22a.json`、`outputs/isaac_eval_e22b.json`。

**意外正向**：小台阶（4–14cm）和离散小障碍都在先验能力内——walk 原型本身
带小障碍跨越能力；垫脚石（需精确落脚）是明确盲区。这修正了"先验完全不含
地形行为"的说法：连续粗糙 >0.06 和垫脚石不行，但离散台阶可以。

### E23：连续潜空间（相位插值）+ RL 相位调制（2026-08-13，方向 B）

E23（64 envs × 800 iters，平坦地面，warm-start 相位 200 iters coef 10.0，
latent-kl 2.5e-6，expl 0.01，entropy 0.001）：
`--use-2hz-gate 1 --phase-mode --phase-warmstart-iters 200
--phase-warmstart-coef 10.0 --latent-kl 2.5e-6 --latent-expl 0.01
--entropy 0.001`

| 测试 | phaseaux（E23） | 对照 noaux（E22a 冻结先验） | 判据 |
|---|---|---|---|
| A 60s | 3/3 done，vx 0.52–0.54，**disp 0.8–2.0m** | 3/3，47.3–47.7m | **不达标**（<42.9m） |
| B 500N | 11/12（left_s0 976 步倒，h_min 0.214） | 12/12 | 略差 |
| C 68s 切换 | **0/3**（659–1011 步倒 ≈ 13–20s） | 3/3 | **不达标** |
| D 跳跃 | 3/3 done，vx 0.05–0.15（无前进） | 3/3，vx 0.68 | 退化 |

**结论**：连续插值读取机制本身无损（MuJoCo 闭环 3/3，见
DATA_GENERALIZATION_LOG 14.1），但 RL 相位调制在无"前进压力"奖励下收敛为
**存活但不前进**（v_speed 0.56–0.58，净位移 <3m 的原地振荡/画圈），命令
切换鲁棒性远差于冻结先验。与 E17/E19/E22 结论一致：**连续潜空间不改变
流形上限，奖励压力才是关键**。结果文件：`outputs/isaac_eval_e23.json`、
`outputs/isaac_e23_phase_interp/`。

### C 方向：Isaac 并行扩展压力测试（2026-08-13）

128 envs × 300 iters（其余同 E23）：
- env 数生效（日志头部 Number of environments: 128）；
- dt/iter 均值 0.897s vs 64 envs 0.721s；样本吞吐 2,130 → 3,425/s（+61%）；
- 显存 2.8GB、GPU 利用率 62%（64 envs 时 2.8GB / 60%）——显存和算力都未
  用满，瓶颈在每迭代 rollout/update 固定开销。
- **结论**：当前 Isaac 管道在 3060 上并行收益有限；千级并行需换 mjlab
  （MuJoCo-Warp）官方配方（默认 4096 envs），那是从零学速度跟踪的独立
  轨道，与 SONIC 蒸馏管道不共享代码。服务器已具备条件（外网通、warp-lang
  1.16 已装、mjlab 可 pip 安装），12GB 需降到 1024–2048 envs 试跑。

### E24：相位 + anti-stop + 进度奖励（2026-08-13，方向 B 续）

E23 的"存活不前进"说明自由相位缺前进压力。E24 在 E23 基础上加
`--anti-stop 1.0 --anti-stop-thresh 0.1 --progress-scale 0.3`（64 envs ×
600 iters，平坦）。评测（phase-mode，A/B/C/D × 3 seeds）：

| 测试 | phaseaux（E24） | 对照 noaux（冻结先验） | 判据 |
|---|---|---|---|
| A 60s | 3/3 done，vx 0.28–0.31，**disp 9.2–12.1m** | 3/3，47.3–47.7m | **不达标**（<42.9m） |
| B 500N | **12/12 done**（h_min 0.699–0.724） | 12/12 | 持平 |
| C 68s 切换 | **3/3 done**（h_min 0.718–0.724） | 3/3 | 持平（E23 是 0/3） |
| D 跳跃 | 3/3 done，vx 0.49–0.53 | 3/3，vx 0.68 | 略降 |

**结论**：anti-stop+progress 修复了 E23 的坍缩——鲁棒性全部恢复（B/C/D
全过，C 从 0/3 变 3/3），但 A 位移只有 ~10m（vx 0.3 vs 命令 0.8）。原因：
自由相位下策略没有重建路由器那种按 proprio 历史自然旋转的相位振荡器，
EMA 平滑后相位推进太慢 → 慢动作步态。这指向下一步（E25）：相位锚定到
冻结 PhaseNet 的步态时钟，策略只学有界偏移。结果文件：
`outputs/isaac_eval_e24.json`、`outputs/isaac_e24_phase_antistop/`。

### C 方向：mjlab（MuJoCo-Warp）官方配方冒烟测试（2026-08-13）

服务器搭建完成（独立 venv `.venv_mjlab`）：
- 依赖踩坑：mujoco-warp 3.5.0 需配对 mujoco 3.5.0（3.11.0 移除
  mjENBL_MULTICCD）；mjlab 1.2.0 需 warp-lang 1.12.0（1.16.0 移除
  wp.context）；默认 logger=wandb 需 `--agent.logger tensorboard`；
  venv 用 `--without-pip` + get-pip.py 引导（无 sudo）。
- 冒烟：`scripts/train.py Unitree-G1-Flat --env.scene.num-envs=1024`，
  跑在 **cuda:0（RTX 3060）**，60 iters，**Iteration time ≈1.03s/iter**
  （1024 envs × 24 步 ≈ 2.4 万 env-steps/s），中途显存 0.84GB / GPU 69%。
- **结论**：3060 12GB 上官方配方 1024 envs 完全可行（显存仅 0.84GB，理论
  上 2048 也可试）；官方 4096 默认配置是给 24GB 级 GPU 的，但 1024 已满足
  "千级并行"的验证目标。该轨道与 SONIC 蒸馏管道独立，是从零学速度跟踪的
  对照基线。产出：`unitree_rl_mjlab/`、`.venv_mjlab/`、`mjlab_smoke1024.log`。

### E25：相位锚定（路由器时钟 + 有界偏移）（2026-08-13，方向 B 关键转折）

针对 E24 根因（自由相位学不会步态时钟）：phase = normalize(冻结 PhaseNet
时钟 + 0.15×策略偏移)，时钟按 proprio 历史推进，策略只做有界调制。
训练：E24 奖励（anti-stop 1.0@0.1 + progress 0.3），600 iters。

| 评测（E25 policy_final，EMA 时钟开/关 × aux 开/关） | A 60s 位移 | B/C/D |
|---|---|---|
| raw 时钟 + aux + offset | 20.2–28.3m（vx 0.48–0.55） | B 12/12、C 3/3、D 3/3 |
| EMA 时钟 + aux + offset | 21.3–34.2m（vx 0.60–0.65） | B 12/12、C 3/3、D 2/3（jump s1 倒） |
| **EMA 时钟 + aux 归零 + offset** | **49.2–49.5m（vx 0.82）** | **B 12/12、C 3/3、D 3/3（vx 0.59–0.60）** |

**结论（方向 B 的定论）**：
1. **相位锚定 + 连续插值读取 + 前进压力 = 首个"学习型策略全项不劣于冻结
   先验"的结果**：aux 归零时 A 49m（冻结先验 47.3–47.7m）、B/C/D 全过、
   D 满速。策略偏移通道本身无损（甚至略正）。
2. **速度缺口 100% 来自 aux 通道**：同一检查点 aux 开只有 21–34m；EMA
   时钟另外贡献 ~5m（raw 时钟相位推进仍偏慢）。
3. 与 E2–E24 全部证据闭环：**aux 关节偏移通道在该管道下始终是破坏源；
   连续潜空间只有在"锚定到先验时钟 + 关掉 aux"时才展现无损价值**。论文的
   aux 正向价值依旧依赖力矩级解码器/TO 数据，未被本管道复现。

产出：`outputs/isaac_e25_phase_anchor/`、`isaac_eval_e25*.json`。

### E26：纯相位偏移 RL（aux_scale=0，训练中）

把 E25 的 aux 通道在训练时就置零（策略只学相位偏移），干净归因"相位调制
是否独立具备价值"。判据：A ≥ 47m 且 B/C/D 全过（与 E25+aux归零 持平或更好）。

### E26 结果（完成）

| 测试 | phaseaux（E26，aux_scale=0） | 对照 noaux（冻结先验） | 判据 |
|---|---|---|---|
| A 60s | 3/3，vx 0.78，**45.7–46.1m** | 3/3，47.3–47.7m | **达标**（≈97%） |
| B 500N | **12/12**（h_min 0.727–0.746） | 12/12 | 持平 |
| C 68s 切换 | **3/3**（h_min 0.739–0.745） | 3/3 | 持平 |
| D 跳跃 | **3/3**，vx 0.62–0.66 | 3/3，vx 0.68 | 持平 |

**方向 B 最终定论**：
1. 连续潜空间（相位锚定 + 插值读取）+ 前进压力 + **aux 全程关闭**时，
   学习型策略全项不劣于冻结先验（A 46m/97%，B/C/D 全过）——这是
   E2–E24 中第一个达标的 learned 策略。
2. 但相位偏移通道本身是"机制中性"的（E26 A 46m ≈ E25+aux归零 49m ≈
   冻结先验 47m）：**没有产生正向增益，只是不再破坏**。连续潜空间解决
   了"平滑性/梯度"问题（E3 离散选择的死因），但没解决"流形上限"问题。
3. aux 关节偏移通道依旧是唯一破坏源（E25 aux 开 → 21–34m）；论文 aux
   正向价值仍依赖力矩级解码器 + TO/逆动力学数据（方向 A 已定论）。
4. 对用户问题"蒸馏是否必要"的实证答案：我们的蒸馏先验 + 显式回退表
   （24/24 平坦命令）已经是可用下限；连续潜空间可以无损接入；官方 mjlab
   从零配方是独立对照轨道，可回答"不蒸馏能否更强"，但它与当前管道不共享
   代码，需要单独立项。

产出：`outputs/isaac_e26_phase_only/`、`outputs/isaac_eval_e26.json`。

### E26 地形扩展：相位偏移在粗糙地形上是负资产（2026-08-13）

把 E26（相位锚定纯偏移，aux_scale=0）放到 rough 0.06/0.08 评测（A 60s，
terrain-seed 0，同一评测 harness）：

| 变体 | rough 0.06 | rough 0.08 |
|---|---|---|
| 冻结先验 noaux（对照） | 3/3，38–43m | 0/9 完成（存活 21–59s） |
| **E26 带偏移** | **3/3，22.7/42.5/38.0m** | **0/6 立即倒**（s0 3/3、s1 3/3 全倒） |
| **E26 相位归零（纯时钟）** | - | **2/6 完成**（s0 2/3：30.1/33.3m；s1 0/3） |

**结论**：
1. rough 0.06 上 E26 基本持平基线（3/3，2/3 位移达 38–42.5m），一个 seed
   位移降为 22.7m。
2. rough 0.08 整体在能力边界之上：带偏移 0/6、纯时钟 2/6 立即倒。
   两个 seed 的对照（s0：0/3 vs 2/3；s1：0/3 vs 0/3）与"偏移在地形上
   有害"一致但不充分——地形 seed 方差大，纯时钟自身在 s1 也全倒。
3. 0.08 的结论修正为：**偏移通道在平坦无损、rough 0.06 基本持平、
   rough 0.08 无增益甚至可能有害**；地形鲁棒性仍由路由器时钟/先验决定，
   0.08 是两类路径的共同能力边界（对照先验 0/9 完成）。连续插值在 0.06
   与离散路径持平，未见明显优势。

产出：`terr_e26_n0.06_s0.json`、`terr_e26_n0.08_s0.json`、
`terr_e26_n0.08_s0_zerophase.json`。

### E27：latent→VAE→冻结 SONIC 解码器（无行为先验，2026-08-13）

用户指出的缺失实验：**自主学习（不用相位路由器/Sonic 行为先验），但运动
仍经过冻结 SONIC 解码器**。复用论文 TVAE 结构与既往经验实现：

- **相位条件化 token VAE**：因果窗口（t-9..t，10×64）编码为 z∈ℝ16；解码器
  D(z, sinφ, cosφ)→token。φ 来自 walk（mode 2）token 的 2-PC PCA 圆相位，
  步态时钟速率 0.121 rad/步（数据实测）。val recon MAE 0.079。
- **Isaac env latent 模式**：策略每步输出 z；env 用固定时钟推进 φ；
  D(z,φ)→token→冻结 SONIC 解码器→关节目标。无路由器、无 aux 通道。
- **latent warm-start**：前 200 iters 监督策略 z→z_walk（walk 窗口的平均
  编码），之后自由。奖励 = 速度跟踪 + anti-stop + progress（E24 验证组合）。
- 训练 64 envs × 800 iters，无跌倒，vx 0.35–0.42（随机命令均值 ≈0.4）。

| 测试 | E27（latent，无行为先验） | E1 冻结先验 | E9/E11 vanilla（无 Sonic） |
|---|---|---|---|
| A 60s | 3/3，vx 0.32，**19.1m** | 3/3，47m | **0/3 立即倒** |
| B 500N | **12/12**（h_min 0.74–0.76） | 12/12 | - |
| C 68s 切换 | **3/3**（disp 19.7–19.8m） | 3/3 | - |
| D 跳跃 | 3/3（vx 0.34，不跳但存活） | 3/3（vx 0.68） | - |

**结论（回答用户的架构问题）**：
1. **"不通过 Sonic 学习、但让 Sonic 动"可行**：仅靠 token 流形（VAE）+ 相位
   时钟 + z_walk 初始化，64 envs 的从零 RL 就能学会稳定行走——对比 E9/E11
   直出关节的 vanilla RL 同等算力 0/3 立即倒。**Sonic 的 token 流形本身就是
   关键先验**，即使不蒸馏它的行为策略。
2. 速度只有 0.32 m/s（先验 0.8）：策略的 z 收敛到慢速技能，没找到/压满
   walk 技能；这是"无行为先验"的代价，不是流形失效。B/C 全过说明鲁棒性
   结构完整。
3. 与 E25/E26 对照：路由器先验 = 满速（47m）；纯流形 + 时钟 = 半速但
   可行（19m）；无 Sonic 直出 = 完全失败。三者正好排成"先验强度 → 样本
   效率"的阶梯。
4. 下一步候选：latent 速度奖励加权（vel_sigma2 调小）+ 128 envs + 更长
   训练，或给 z 加"向 walk 技能推"的课程。

产出：`outputs/token_vae_e27/`（vae.pt/pca.npz/z_walk.npy/meta.json）、
`outputs/isaac_e27_latent/`、`outputs/isaac_eval_e27.json`、
`train_token_vae_e27.py`、`isaac/token_window_vae.py`。

### E28：E27 速度天花板的归因消融（2026-08-13）

**动机**：E27 得 0.32 m/s（先验满速 0.8），结论文写"半速是无行为先验的
样本效率代价"。但代码审查发现两个更具体的可疑瓶颈，需用消融定责：

1. **冻结相位时钟**（`apt_flat_env.py` latent 分支）：φ 以**标量固定速率**
   0.121 rad/步推进，全 episode/全 env 不变 → 策略无法改变步频，命令
   vx∈[0,0.8] 跟不上高命令。
2. **奖励地形**：`anti_stop_thresh=0.1`（>0.1 无前进激励）+ 硬编码
   `stillness=-0.05·vx²`（二次罚前进速度）→ 跟不上时停在 ~0.35 是舒适最优。

（旁证：E27 训练曲线 iter 700 后 rew/vx 平台、expl 退火到 0；latent KL=56
而 coef 仅 2.5e-6——流形约束形同虚设，但非速度主因。）

**改动（全部 config 开关，默认值保持 E27，可 bit-for-bit 复现）**：
`apt_flat_env.py` 加 `latent_cmd_phase_rate`（命令条件化步频：rate =
base·clamp(cmd_vx/0.6, 0, 2.0)）/`latent_phase_rate_ref`/`_max` +
`stillness_vx_scale`；`train/eval_apt_isaac.py` 加对应 4 个 CLI flag。

**消融（64 envs × 800 iters，A_walk60 = 60s 前进，6 seed）**：

| Run | 改动 | mean vx | mean disp | fall | 解读 |
|---|---|---|---|---|---|
| E27（基线） | 冻结时钟 + 基线奖励 | 0.317 | **19.1m** | 0/3 | — |
| E28a | + 解冻步频 | 0.342 | 13.1m | 0/6 | vx 微涨(+8%)但 disp 暴跌(-31%)→**方向漂移** |
| E28b | + 解冻步频 + 奖励重调* | **0.253** | 14.9m | 0/6 | **更慢**(-20%) |

\* E28b 奖励重调：`anti_stop_thresh 0.1→0.6`、`progress 0.3→0.5`、
`stillness_vx_scale 0.05→0`。

**结论（细化 E27 结论 #2，且推翻了上述两个假设）**：

1. **解冻步频不够**（E28a）：策略有自由调速却不快，且更快播放冻结 walk
   token 会**破坏直行**（disp 13<0.342×60≈20.5，即大量偏航）。说明
   SONIC walk token 序列在非原始步频下不保直。
2. **奖励重调反而更慢**（E28b）：若速度是"激励不足"，加权应变快；它变慢
   → 速度**不是奖励激励问题**。策略被推向达不到的 0.6（anti_stop_thresh
   抬高）反而失稳降速。
3. **三者一致收敛到 ~0.25–0.34 m/s**，与样本效率/时钟/奖励都无关 →
   **~0.3 m/s 是冻结 SONIC walk 解码器流形的固有前进速度上限**（先验本身的
   速度）。walk 数据虽录于 0.6 m/s，但 RL 找到的 z + 冻结 D(z,φ) 复现出的
   步态动力学上只到 ~0.3（足部滑动/顺应/解码重建误差），这正是"冻结先验是
   跟踪/稳定先验、非速度优化先验"的又一佐证。
4. E27 结论 #2 原说"半速是无行为先验的样本效率代价"——**E28 修正为**：
   半速是解码器流形的固有速度天花板，不是样本效率（加探索/加 iters 也无效，
   E28a/b 用了全新探索退火仍封顶）。要突破需：更忠实的 walk 解码器、或
   aux/残差通道（即项目一贯的 aux 破坏源）、或更高速度的 walk 数据。

E28c（仅奖励重调、不解冻步频）按计划为条件项；a/b 已不歧义，**跳过**。

产出：`outputs/isaac_e28a_cadence/`、`outputs/isaac_e28b_cadence_rew/`、
`outputs/isaac_eval_e28a.json`（A 段 6 seed）/`isaac_eval_e28b.json`、
`e28{a,b}_train.log`、`e28a_eval.log`（B/C/D 段因结论已定提前终止）。
代码改动见 commit（`apt_flat_env.py` + `train/eval_apt_isaac.py` 的 E28
config 开关）。

### E29：latent KL 先验对齐 walk 流形（首个正向杠杆，2026-08-13）

**动机**：E28 排除了时钟/奖励，但留了一个未测且更可疑的轴——**KL 先验设错**。
E27 的 latent KL 先验是 N(0,I)，但 walk 流形中心是 z_walk（≠0，||z_walk||²≈100）。
所以 KL 把 z 往原点拉 = **往 walk 流形外拉**；E27 用极小 coef 2.5e-6 致 KL=56（z
严重漂离流形）。论文 APT-RL 的 KL=0.1 是配它自洽的潜空间先验。这里先验与流形
不一致。

**改动（config 开关，默认 = E27）**：`ppo_core.py` 加 `kl_normal(mean,log_std,
prior_mean)` 与 PPOTrainer 的 `latent_prior_mean` 缓冲；`train_apt_isaac.py` 加
`--latent-kl-prior {zero,walk}`（walk 时加载 z_walk.npy 作先验中心）。

**E29 = E27 + `--latent-kl-prior walk --latent-kl 1e-2`**（其余不变，孤立先验变量）。
A_walk60（6 seed，latent 模式 aux/noaux 同值）：

| Run | KL 先验 | KL coef | mean vx | mean disp | 结论 |
|---|---|---|---|---|---|
| E27 | N(0,I) | 2.5e-6 | 0.317 | 19.1m | 基线 |
| E28a | N(0,I) | 2.5e-6 | 0.342 | 13.1m | 解冻步频→漂移 |
| E28b | N(0,I) | 2.5e-6 | 0.253 | 14.9m | 奖励重调→更慢 |
| **E29** | **N(z_walk,I)** | **1e-2** | **0.348** | **18.3m** | **最佳 vx，不漂移** |

**观察**：训练中 KL 仍 ~55–58（ coef 1e-2 不足以把 z 钉死在 z_walk——奖励仍主导），
但 coef 比 E27 大 4000×，把 z 拉得**更靠近 walk 流形** → vx +10%（0.317→0.348）
且位移稳定不漂移（18.3m，远好于 E28a 的 13m）。这是 E28 之后的**首个正向杠杆**。

**结论**：
1. "z 靠近 walk 流形 → 更快更稳"方向**成立**（E29 是 E27/E28a/E28b 中最快且无回退的）。
2. 但天花板仍 ~0.35（数据 walk 速度 0.6 仍不可达）——解码器保真度硬上限依旧；
   流形对齐只能逼近它，不能突破。
3. KL 没钉死 z 说明 1e-2 还偏小；要进一步逼近流形需更大 coef 或**按相位**钉 z
   （z_walk 是窗口均值，丢失了逐相位细节）。

**下一步候选（E30）**：把 KL coef 加大到 1e-1/1.0 看能否把 vx 再推高一档，或改用
逐相位 walk 后验作先验（而非均值）。

### E30：流形 KL coef 扫描的第二点（coef 1e-1，2026-08-13）

E29 的 coef 1e-2 把 KL 压到 ~55（z 更靠流形，vx 0.348）。E30 把 coef 再 ×10 到
1e-1（其余同 E29），测"更硬钉向流形"是否再提速。A_walk60 6 seed：

- **E30（coef 1e-1）**：vx **0.336**、disp 16.0m、实测 KL **46.7**（比 E29 的 55 更低，
  z 更靠 z_walk）。
- 对比 E29（coef 1e-2）：vx 0.348、disp 18.3m、KL 55。

**KL 扫描非单调，峰值在 E29**：

| KL coef | 实测 KL | mean vx | mean disp |
|---|---|---|---|
| 2.5e-6（E27） | ~56 | 0.317 | 19.1m |
| **1e-2（E29）** | ~55 | **0.348** | **18.3m** |
| 1e-1（E30） | ~47 | 0.336 | 16.0m |

**结论**：过度把 z 钉向 z_walk（E30，KL 47）反而比温和钉（E29，KL 55）略差——因为
z_walk 是 walk 窗口的**均值**，丢失逐相位细节；过度约束让策略无法利用相位结构。
**流形对齐 KL 的峰值在 E29（coef 1e-2，vx 0.348）**，是本系列最快配置。

综合 E28–E30：时钟/奖励/流形-KL 三轴均无法把 vx 推过 ~0.35（数据 walk 0.6 仍不可达），
**解码器保真度硬上限稳固**。要真正突破需动作表征层的改动——**速度条件化 VAE 解码器**
D(z, φ, v_cmd)→token（让流形本身编码速度维度），是下一阶段候选。

产出：`outputs/isaac_e30_klwalk_h1/`、`outputs/isaac_eval_e30.json`、`e30_train.log`、
`e30_eval.log`、`apt_g1/plot_latent_cmp.py`（E27–E30 对比图脚本，产出
`outputs/latent_cmp.png`）。

**渲染缺口**：`render_walk.py` 已加 latent 支持（cfg/policy/action 循环），但 Isaac
Sim 在本服务器开相机（`enable_cameras=True`）时于 stage 创建阶段段错误
（viewport hydra 引擎 `__enable_hydra_engine`，崩溃在 AppLauncher 初始化、
业务代码之前；重试复现；服务器历史上从无 Isaac 渲染产物，仅 MuJoCo router 视频）。
故本阶段未产出 E29 行走视频，待渲染环境修复。

产出：`outputs/isaac_e29_klwalk/`、`outputs/isaac_eval_e29.json`（A 段 6 seed）、
`e29_train.log`、`e29_eval.log`。代码：`ppo_core.py`（`kl_normal` +
`latent_prior_mean`）、`train_apt_isaac.py`（`--latent-kl-prior`）、
`render_walk.py`（latent 支持）。

### E31：速度条件化 VAE 解码器（D(z, φ, v_bin)→token，首个破速度上限，2026-08-13）

**动机**：E28-E30 证明冻结 walk 解码器流形有 ~0.35 m/s 保真度上限（时钟/奖励/KL
三轴都无法突破）。要真正破上限需**动作表征层**改动：让流形本身编码速度。

**改动**：
- `train_token_vae_e31.py`：`SpeedPhaseTokenVAE`——decoder 输入加 v_bin embedding
  （3 档，由 walk 数据实测相位速率三分位分 bin：慢<0.08/中/快>0.14 rad/步）；
  encoder 不变（窗口→z∈ℝ16）。**val recon MAE 0.0753**（≈E27 的 0.079，速度条件化
  未损重建）。
- env：`latent_speed_bins` cfg——命令 vx 经 `torch.bucketize` 映射到 v_bin 喂 decoder。
- `token_window_vae.py` 加 `SpeedPhaseTokenVAE`；train/eval 加 `--latent-speed-bins`。

**E31 = E29 配方 + speed-bins**（64 envs × 800 iters）。A_walk60 6 seed：

| Run | 表征 | mean vx | mean disp | vx/disp 比 |
|---|---|---|---|---|
| E29 | E27 流形 | 0.348 | 18.3m | 0.88（直） |
| **E31** | **速度条件化流形** | **0.535** | ~10m | **0.31（漂移）** |

**突破**：vx **0.535（+54% vs E29）**——**速度条件化流形确实打破了 ~0.35 上限**！
这是 E27 以来的首个架构级正向突破。但 disp 仅 ~10m（若直应 ~32m）→ 严重偏航。

**漂移分析**（rollout base_xyz/quat）：**持续左转，yaw 从 0 线性漂到 -27°（~4°/s）**
——是**系统性转向偏置**（速度条件化的快速技能自带固定偏航，像未校准步态），
非打转、非命令问题（A 段命令固定 (0.8,0,0)）。

### E32：yaw×2 + heading 奖励修复漂移（失败，2026-08-13）

**动机**：E31 快但偏。尝试用奖励修方向：`yaw_scale`（track_yaw 权重 0.5→2.0）+
`heading_scale`（新增速度方向 vs 命令朝向的 cos 对齐奖励 0.8）。

**E32 = E31 + 方向奖励**（其余同）。A_walk60：**vx 0.354、disp ~10.3m**——
heading 强化把速度**压回 0.35**（训练 vx 0.26-0.35，远低于 E31 的 0.45），但
**漂移没修好**（disp/vx 比仍 0.5）。

**结论**：系统性转向偏置**不是奖励可修的**——policy 已把转向当"技能的一部分"，
奖励只能逼它慢下来换方向，不能消除偏置。E32 负结果（heading 奖励无效 + 速度被压）。

**下一步（E33）**：系统性偏置适合**外部开环补偿**——在 eval/部署层把命令 yaw
设为 +偏航率（≈+4°/s）抵消 E31 的 -4°/s 左转。若成立，E31+补偿 = 快且直。

产出：`outputs/token_vae_e31/`（vae.pt/pca.npz/z_walk.npy/vbin_meta.json）、
`outputs/isaac_e31_speedvae/`、`outputs/isaac_eval_e31.json`、
`outputs/isaac_e32_speedvae_heading/`、`outputs/isaac_eval_e32.json`、
`e31{eval,rollout,train}.log`、`e32_{train,eval}.log`。
代码：`train_token_vae_e31.py`、`token_window_vae.py`（SpeedPhaseTokenVAE）、
`apt_flat_env.py`（latent_speed_bins + yaw_scale/heading_scale）、
`train/eval_apt_isaac.py`（--latent-speed-bins/--yaw-scale/--heading-scale）。
