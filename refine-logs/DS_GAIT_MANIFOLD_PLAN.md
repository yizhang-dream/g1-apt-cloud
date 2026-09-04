# DS 步态流形计划（Phase 0–5）：像论文一样在多步态间切换，全地形 + 全速度段

> 【层位 L2 侧轴｜执行计划（2026-09-04 定稿；owner 指令「写计划，另开会话执行」；
> 前序 D029–D033 五轮实验全部支撑本文，事实源 = `tracker/D.md`）】↑
> `refine-logs/README.md`（扇出树根）｜上游：`DS_RECOLLECT_PLAN.md`（采集线设计与
> benchmark 选型）、`TO42_PLAN.md`（选择器配方，Phase 5 平移其预注册框架）｜
> 执行事实源：`tracker/D.md`（Run 行只进那里）｜状态：**待执行**（本文为唯一
> 执行依据，另开会话按 Phase 0→5 顺序执行，每 Phase 完成后落盘 Run 行）

---

> ## ⚠️ 执行会话第一件事（owner 09-04 指定）
>
> **Phase 0「Isaac 执行保真度校准」必须最先执行（§2）——未完成或未明确
> 判定（PASS / 降级 G3）之前，禁止开始 Phase 1 采集及之后任何步骤。**
> 原因：校准结果决定 G3 全速度段目标是否成立、Phase 4 的 cmd 采样上限、
> RUN 族在流形中的档位；先采集再校准会在错误的执行假设上烧掉 2.5h 机时。
> 若 Phase 0 无法执行（如 Isaac 环境故障），停下向 owner 汇报，不得跳过。
>
> **✅ 判定已完成（2026-09-04，D034）：PASS（第 1 轮，零对齐迭代）——
> Isaac oracle 回放实现率 1.61（1.6657 / 1.033，3 seed 零摔）≥ 0.9；
> Phase 1 解禁，G3 维持，Phase 4 U(0,1.5) 第二臂解锁。详见 §2 末修订
> 记录、`tracker/D.md` D034、`apt_g1/outputs/sync/ds_phase0_calibration.md`。**

## 0. 北极星与目标分解

**北极星**：在冻结 SONIC 解码器底板上，用官方回路采集的最小步态族数据训练一个
**动作流形**（token VAE），并像论文一样让策略**学习在多个步态间切换**，使 G1
以最少的动作集（4 族）覆盖论文七地形中档可通行域 + 全速度段 0.2–3.0 m/s。

| # | 目标 | 可证伪判据 |
|---|---|---|
| G1 | 4 族步态流形（VAE） | val MAE ≤ 0.08（E39 0.077 同级）；held-out 族近邻重建误差 ≤ 训练族 1.5× |
| G2 | 全地形通行 | mjlab 论文地形中档（rough/stairs/stones/discrete/gap）策略通过率 ≥ 各地形最优单族 |
| G3 | 全速度段 | 0.2–0.8（SLOW 可调）+ ~0.45（HAPPY）+ 1.5–3.0（RUN 可调）命令谱全部 cmd 可响应（推翻 TO41 §3d「realized 不随 cmd」） |
| G4 | 学习型切换 | TO42 配方（2Hz logit + 0.5s 锁）4 族版；切换策略 ≥ best-fixed 单族（全地形矩阵） |
| G5 | （远期，不在本轮）感知条件化 | 高度图→相位对齐把单栏 0.1m 通过率 38%→~100%（D031 靶） |

**两个已裁定默认**（owner 可在执行前推翻，推翻只需改 Phase 1 网格）：
① WALK(2) 不进首版训练集，作 held-out 验证族（stones 0/3 短板 + 23 号全覆盖其速度档；
exp_all 11k 步 mode2 数据保留作流形内插探针）；② JUMP(17) 进首版（负障碍专长 +
感知越障线的未来载体，接受其平地慢 adv~1m 的数据稀释）。

## 1. 已冻结的技术决策（D029–D033 沉淀，执行时不得重新发明）

1. **采集回路 = 官方 deploy 回路**（deploy C++ + run_sim_loop WBC + MuJoCo，
   `.venv_sim`，pty 键盘驱动；setup 模板 = `/tmp/setup_ds_smoke.sh`（lab-ts），
   驱动模板 = `apt_g1/drive_ds_smoke.py`）。**禁止从 planner ONNX 开环提取数据**
   （context 自回归漂移 + 无物理检验 + 不可执行轨迹入集；官方回路的物理检验正是
   把 planner 2.12 m/s 压到可执行 1.0 m/s 的闸，D033）。
2. **速度标尺（三层，校准靶）**：planner 裸输出 RUN 2.12 / 官方 WBC 回路 ~1.0
   （48%）/ 我方 MuJoCo harness 0.37（17%）；WALK 0.91/0.51、SLOW 0.54/0.15。
   我方 harness 的地形测试通过率对快步态系统性偏悲观（已知偏差，只用于相对排序）。
3. **4 族主案 {1 SLOW, 23 HAPPY, 3 RUN, 17 JUMP}**（D030 全矩阵 set-cover 解）：
   23 号 = 全地形全存活中速主力（rough 9.1m/20s）；1 号 = 低速段 0.2–0.8 连续可调
   + stones 稳；3 号 = 高速段 1.5–3.0（planner 材料真，执行层衰减由 Phase 0 处置）；
   17 号 = stones/rough 独特存活（负障碍）。held-out = {18 STEALTH, 19 INJURED, 2 WALK}
   （流形连续性验证族）；crawl(8)/ledge(20) 留 harder 档备用不进首版。
4. **过渡段是全局难点，必采**：stones 死点 = 平台↔石头阵过渡带（x≈±0.5±0.6，
   D032）；单栏 = 相位彩票（38%，D031）；步态切换 transition 脆弱（MQ12：切换后
   1/3 vs 从头 3/3）。流形训练数据必须含族间过渡段。
5. **接口 gotcha**：deploy 键盘 '9'/'0' 调速只对 SLOW_WALK/RUN 生效（WALK 恒 -1）；
   planner ONNX 的 target_vel 非单调（-1≈2.5>1.5），**一律用 -1 哨兵**；standing
   motion set 键位 "1"-"6" = SLOW/WALK/RUN/FORWARD_JUMP/STEALTH/INJURED，styled
   set（'n' 切换）第一位 = LEDGE；HAPPY_DANCE(23) 在 styled set 第 4 位。
6. **地形基准 = mjlab 官方生成器**（论文原生不可得；`ds_mode_terrain.py` 已实现
   内存组装：rough 对称化/stairs/stones/discrete 原生 + hurdle/gap 自建标注；
   gap 曾有几何 bug 已修——两板 center±(half+gap/2) 才是真缝）。
7. **静态 6 mode 排除**（IDLE/SQUAT/KNEEL×2/LYING/BOXING，无前进，通过率无意义）。

## 2. Phase 0：Isaac 执行保真度校准（前置门，不做则 G3 无意义）

**动机**：RL 在 Isaac 训练；若 Isaac 执行上限也是 ~0.4 m/s，快步态学不出来
（D033 升格任务）。

- **步骤**：①官方回路录 RUN 段 60s（`drive_run_probe.py` 模式 + `--record-*`
  csv 三件套），从 token 列提取 token 序列；②Isaac `apt_flat_env` 写 oracle
  token 回放入口（D002 流程 Isaac 版：token 直接进 decoder，policy 旁路）；③
  同序列回放 60s × 3 seed，记 realized vx。
- **判据**：Isaac 实现率（realized / 官方回路 realized 1.0）≥ 0.9 → PASS；
  < 0.9 → 对齐 `base_sim.py` 执行参数（PD kp/kd、控制频率、decoder 输出到
  q_des 的消费路径）后复测，迭代 ≤ 3 轮；仍 < 0.9 → 降级 G3 速度目标到
  Isaac 实测上限并记录（不阻塞后续 Phase）。
- **产物**：校准报告（sync 目录）+（若改）`apt_flat_env.py` 参数 diff +
  tracker D 行。**预算 ~0.5 天**。

> **修订 09-04（D034，门判定回写）**：**PASS（第 1 轮，零对齐迭代）**——
> D033 drive_run_probe 录音复用（零采集机时），token 窗 [1048,4048) 3000 行
> lattice 违例 0；AptFlatG1Env 子类 oracle 回放（canonical env 零改动）
> 3 seed 零摔，realized vx 均值 1.6657 m/s ÷ 官方 1.033 = **实现率 1.61**。
> 执行层衰减排序修正（对 planner 参考 2.086 m/s）：**Isaac 79.8% > 官方
> WBC 48.7% > 我方 harness 17.5%**——harness 0.37 的归因收窄到 harness 自身
> 执行配置，与 token/decoder 无关。**caveat（gate≠机制）：Isaac 比官方快
> 61%，两套 realized 互不外推**；足底接触真实性列 P2 抽检不阻塞。
> G3 维持；Phase 4 U(0,1.5) 第二臂解锁；Phase 1 解禁。
> 报告 = `apt_g1/outputs/sync/ds_phase0_calibration.md`。

## 3. Phase 1：采集（官方回路；稳态 + 速度梯度 + 过渡段）

**网格**（全部官方 deploy 回路，段间 idle 10s + fall 计数，events.json 留档）：

| 族 | 键位 | 速度档 | 方向 | 时长 | 段数 |
|---|---|---|---|---|---|
| SLOW(1) | "1" | '0' 键调 {0.2,0.4,0.6,0.8} | 前/后/左右斜（4 主向） | 60s×2 | 32 |
| HAPPY(23) | 'n'→"4" | 默认 | 8 方向 bin | 60s×2 | 16 |
| RUN(3) | "3" | {1.5,2.0,2.5,3.0}（'0' 调） | 前 + 4 主向（仅 1.5/2.5 档） | 60s×2 | 24 |
| JUMP(17) | "4" | 默认 | 前/左/右 | 60s×2 | 6 |
| 过渡段 | 序列切键 | — | 前 | 20s×3 相位 | 12 对×3=36 |
| （可选 C）stones 过渡 | 见风险 5 | — | 前 | 30s×6 | 6 |

- 过渡段采法：驱动脚本按 `A 键 10s → idle 2s → B 键 10s` 序列（12 对 = 4 族
  两两双向），3 个起切相位（idle 后第 0/2/4 秒切）。
- **预算**：~2–2.5h 机时（实时比 1:1）+ 驱动脚本改造 ~0.5 天。
- **产物**：`apt_g1/data/ds_manifold/`（csv 三件套 + events + deploy.log 分段）。

## 4. Phase 2：数据集构建 + 质量门

- 新脚本 `build_ds_dataset.py`（`build_exp3_dataset.py` 模式；SCRIPT_MAP 登记）：
  合并 csv → npz：token/proprio/cmd/**mode/speed(命令标注)/angle_bin/regime(4+3 held-out one-hot)/transition_flag**。
- **质量门（不过不进 VAE）**：①D002 式 oracle 回放抽检（每族 2 段 × 500 步，
  存活 + realized 速度与采集记录一致）；②token lattice 合法性（D001：k/16 格点）；
  ③过渡段 token 连续性（切换点前后 token 距离 ≤ 族内稳态 P95）。
- **预算 ~0.5 天**。产物：`data/ds_manifold/ds_manifold.npz` + 质检报告。

## 5. Phase 3：流形 VAE 训练 + held-out 验证

- 新脚本 `train_token_vae_ds.py`（`train_token_vae_e39.py` 架构扩展，登记）：
  条件轴 = **vb 连续速度（命令标注，弃相位反推）+ regime one-hot（4）+ db 8bin**；
  z 16d；dir/speed 对抗解耦头沿用（E39 配方）；过渡段按 transition_flag 加权
  （×2 采样权重——过渡是稀缺样本）。
- 训练：lab-ts GPU（3060 可，~1h 级）；数据 60–80k 步。
- **验证三件**（G1 判据）：val MAE ≤0.08；held-out 族 {18,19,2} 最近邻重建误差
  ≤ 训练族 1.5×（流形连续性可证伪检验）；z 空间族间线性插值 token 的 lattice
  合法率 ≥90%。
- **预算 ~1 天**（含失败重训一次）。

## 6. Phase 4：RL 底板重训 + 判读门

- 配方：E46/E47 口径 `--latent-mode --latent-vae-path ds_manifold/vae.pt
  --latent-speed-bins(连续 vb) --latent-dir-bins --latent-kl-prior zero
  --progress-scale 1.0 --heading-scale 0.4`，128 envs × 2000 iters，seed 0，
  平地，cmd U(0, 0.8)（若 Phase 0 PASS 且 RUN 稳，扩 U(0,1.5) 第二臂）。
- **判读门**：A 60s ×6 seed 对照 E39 底板（vx 0.418/直行 0.86/h_min 0.75）——
  新底板慢档不劣于 E39 且存在 ≥1 个 cmd 档 realized ≥0.8 m/s（E40 纪录 0.456
  的突破）；KL 漂移监测（E45 教训：后期窗衰减是已知形态，选 ckpt 用 50-iter
  窗口机械规则）。
- **预算**：lab-ts ~4h（128 envs）或云 A10 1024 envs ~40min（若用云，沿用
  TO42 wave 骨架；余额管控见 [[limx-flux-platform-api]]）。

## 7. Phase 5：学习型切换（TO42 配方 4 族平移）

- 选择头：policy encoder 之上 4-way softmax（替换单 logit；TO42 §3 其余逐字：
  2Hz gate、0.5s 锁、gate 布尔+当前选择进 obs、两臂 obs 对齐）；τ 恒 OFF。
- 配对：{learned-selection, frozen-best-fixed} × {s0,s1} 同 seed；frozen 臂 =
  按 cmd 查 D030 通过率矩阵的冻结函数。
- **判读**：ds_mode_terrain 地形矩阵（5 地形 × 7 命令点 × 3 seed）两臂对比 +
  TO42 停止规则三支预注册（选择器塌缩 / 无改善 / seed 换向——负结果措辞沿用
  TO42_PLAN §2）。
- **预算**：~1 天（含训练 4 runs + 矩阵评测 ~2h）。

## 8. 总预算与里程碑

| Phase | 墙钟 | 机时 | 关键门 |
|---|---|---|---|
| 0 校准 | 0.5d | <1h | Isaac 实现率 ≥0.9（或降级 G3） |
| 1 采集 | 1d | 2.5h | fall 率 <5%/段、events 完整 |
| 2 数据集 | 0.5d | — | oracle 回放门 + lattice 门 |
| 3 VAE | 1d | 1h | G1 三件 |
| 4 RL 底板 | 0.5d+4h | 4h | 对照 E39 门 |
| 5 切换 | 1d | 4h | 两臂矩阵 + 停止规则 |
| 合计 | ~5 个工作日 | ~12h 机时 | |

## 9. 风险清单（按杀伤力排序）

1. **Isaac 校准不达 0.9**：G3 降级为「Isaac 实测上限」，RUN 族保留在流形中
   （材料有效性独立于执行），流形结论不受损，速度段声明打折。
2. **HAPPY(23) token 分布离群**（跳舞形态）：对抗头 + regime one-hot 让族间
   差异走显式轴；若 held-out 检验失败且归因 23 号，备选 = 23↔2 WALK 对调
   （重采 WALK 稳态 ~16 段，1h）。
3. **RUN 蒸馏开环漂移**（E45 同款 KL 后期衰减）：质检门 ② 在 Phase 2 就抽检
   RUN 段 oracle 回放；不过则 RUN 降档（速度上限 2.0）或加 KL prior walk。
4. **过渡段 token 不连续**（切换点断裂）：Phase 2 门 ③；失败则 VAE 加
   transition 段专用 reconstruction 权重 ×4 重训一次；再失败则接受族间
   硬切换（选择器语义退化为 TO42 原设）。
5. **stones 官方回路工程量**（可选 C 项）：run_sim_loop 场景由 wbc config
   决定，stones 需生成对应 scene xml + config（~0.5d 工程）；首版可跳过
   （过渡带难点已有 D031/D032 证据，Phase 5 判读用 harness 矩阵兜底）。
6. **target_vel/键位 gotcha 复发**：采集驱动一律先跑 5 分钟冒烟打印
  "Replanning with mode" 行核对 mode/speed 再放全程。

## 10. 纪律

- Run 行只进 `tracker/D.md`（D034 起）；新脚本全部登记 SCRIPT_MAP；本计划
  执行中的设计变更须回写本文（append 修订记录，不删原文）。
- 每完成一个 Phase：commit + tracker 行 + 若干判读摘要；全计划完成后回到
  owner 讨论感知线（G5，高度图→相位对齐，另立计划）。

## 11. 修订记录（append-only）

- **09-04 D034**：Phase 0 前置门判定 **PASS**（第 1 轮，实现率 1.61 ≥ 0.9，
  3 seed 零摔，零对齐迭代）。要点：D033 录音复用零采集；canonical env 零
  改动（oracle 回放走 SCRIPT_MAP §9 子类入口）；执行层衰减排序修正
  Isaac 79.8% > 官方 WBC 48.7% > harness 17.5%（两套 realized 互不外推，
  引用须注明执行栈）；G3 维持、Phase 4 第二臂解锁、Phase 1 解禁。
  详见 §2 修订块与 `sync/ds_phase0_calibration.md`。
