# DS 步态流形计划：A 线（官方回路 Phase 0–5）× B 线（官方数据 B1–B4）双源架构

> 【层位 L2 侧轴｜执行计划（2026-09-04 定稿，同日**双线架构重构**（owner 指令
> 「参考官方数据立项内容彻底修改计划与整体规划」）；前序 D029–D033 支撑 +
> D034/D035 执行中回写，事实源 = `tracker/D.md`）】↑
> `refine-logs/README.md`（扇出树根）｜上游：`DS_RECOLLECT_PLAN.md`（采集线设计
> 与 benchmark 选型）、`TO42_PLAN.md`（选择器配方，Phase 5 平移其预注册框架）、
> `DS_SONIC_OFFICIAL_DATA.md`（**B 线文件级规范**：官方数据资产清单/B1–B4
> 细则/坑清单）、`LITERATURE_SURVEY_DS_MANIFOLD.md`（SONIC 论文精读 = B 线出处）｜
> 执行事实源：`tracker/D.md`（Run 行只进那里）｜状态：**执行中**
> （Phase 0 已 PASS = D034 + D035；Phase 1 驱动标定完成待全网格；B1–B3 待
> 服务器会话启动，与 Phase 1 并行；B4 合并策略 = 本计划唯一剩余 owner 裁定点，
> Phase 3 前裁定）

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
| G1 | 步态流形（VAE，双层架构见 §0.5v2） | val MAE ≤ 0.08（E39 0.077 同级）；held-out 近邻重建误差 ≤ 训练族 1.5×；〔v2 升级〕可扩展留出（训练时留出整个 BONES-SEED 大类，测流形真泛化，VAE-L 轨执行） |
| G2 | 全地形通行 | mjlab 论文地形中档（rough/stairs/stones/discrete/gap）策略通过率 ≥ 各地形最优单族 |
| G3 | 全速度段 | 基线判据（VAE-S 轨）：{0.2–0.8, ~0.45, 1.5–3.0} 命令档位 cmd 可响应（推翻 TO41 §3d）；〔v2 升级判据，VAE-L 轨〕连续速度谱回归：cmd–realized 斜率在 0.2–3.0 全段显著非零（BONES-SEED 人跑 0–8 m/s 连续材料支撑） |
| G4 | 学习型切换 | TO42 配方（2Hz logit + 0.5s 锁）4 族版；切换策略 ≥ best-fixed 单族（全地形矩阵） |
| G5 | （远期，不在本轮）感知条件化 | 高度图→相位对齐把单栏 0.1m 通过率 38%→~100%（D031 靶） |

**两个已裁定默认**（owner 可在执行前推翻，推翻只需改 Phase 1 网格）：
① WALK(2) 不进首版训练集，作 held-out 验证族（stones 0/3 短板 + 23 号全覆盖其速度档；
exp_all 11k 步 mode2 数据保留作流形内插探针）；② JUMP(17) 进首版（负障碍专长 +
感知越障线的未来载体，接受其平地慢 adv~1m 的数据稀释）。

## 0.5 双源总体架构（09-04 重构：A/B 线一等公民化）

```
                    ┌── A 线（官方 deploy 回路，Phase 1–5）──────────────┐
                    │   稳态+速度梯度+方向网格+过渡段（唯一物理检验源）      │
  Phase 0 校准 PASS ─┤                                                    ├→ Phase 3 VAE
  （D034，1.61）     │   B1 下载 ─ B2 对齐冒烟 ─ B3 抽检门(≥95%) ─ B4 npz │   （数据组成 =
                    └── B 线（BONES-SEED × 官方 encoder 离线编码）────────┘    A 产物必选
                                                                      + B4 产物按
                        B4 合并策略三选一（①对照臂 ②合并加权 ③探针）          合并策略）
                        = 本计划唯一剩余 owner 裁定点，Phase 3 前裁定
```

**分工铁律（三条，违反即数据卫生学事故）**：
1. **A 线独占**：方向网格（8 bin）、族间过渡段、held-out 探针、命令标注速度梯度
   ——BONES-SEED 方向偏斜（直行/跑/舞为主，转向/横移稀少，`HUMAN_READABLE:859`
   早有记录）且无命令语义，这些材料只有官方回路能给；
2. **B 线定位三重**：①G3 速度段放大器（含跑/舞高速材料，绕开官方回路 RUN 48%
   执行衰减的数据侧瓶颈）②规模放大器（142k 条/288h 离线编码 vs 1:1 实时采集，
   量级差 2–3 个数量级）③A 线工程受阻时的备胎——**任何情况下不替代 A 线**；
3. **B 线准入双门**：D001 lattice 检验 + B3 oracle 回放抽检（类级存活 ≥95%，
   复用 D034 机制）——动捕重定向非物理解，不过门不入集。卫生学边界：D033 禁的
   是 planner 开环提取路径；B 线走「官方 encoder 编码 + Isaac 回放检验」，
   与 D002/D034 质检框架同源，不冲突。

**时序**：B1–B3 与 Phase 1 并行（B1 下载不占机时）；B4 在 Phase 2 完成后、
Phase 3 启动前完成，合并策略裁定点即 Phase 3 的数据组成决策。文件级规范、
资产清单（HF `nvidia/GEAR-SONIC` encoder 50.1MB/sample_data + `bones-studio/
seed` 142,220 条双格式）、license 门与六个已知坑 = `DS_SONIC_OFFICIAL_DATA.md`
（§7B 为本计划内的计划级摘要）。

### §0.5v2 三层架构（09-04 二次重构：BONES-SEED 量级重估后）

**量级重估（owner 研讨后）**：BONES-SEED = 288h/5180 万帧/33 大类/522 演员 =
现有数据的 **~760 倍**、SONIC 官方训练集（611h）的公开子集——移植研究的
数据基础从「22 分钟自采」跃至「与论文同量级」。三个后果：

1. **双层架构（材料多 × 接口少的和解）**：底层流形用全量材料训（A 锚定 +
   B 放大），上层策略接口保持 4 族选择器（owner 需求 2「动作少」由接口满足，
   需求 1「全覆盖」由厚底座支撑）——两需求不再互斥；
2. **双流形对照实验（把「用多少数据」变成实验）**：VAE-S（A only）vs
   VAE-L（A+B）同判据对比——L 显著优 ⇒ 数据规模是历史流形瓶颈（E43 伤流形
   的重新归因）；S≈L ⇒ 冻结 decoder 的 token 空间容量才是天花板。两个方向
   都是硬结论，预注册不预设赢家；
3. **两个升维判据**：G1 加可扩展留出（整大类留出测真泛化）；G3 加连续速度
   谱回归（人跑 0–8 m/s 连续材料）。

**不变量（重估后依然成立，防止期望过热）**：BONES-SEED 是平地动捕——G2
地形材料零贡献（A 线/harness 独占）；无命令语义——方向网格/命令标注速度仍
A 线独占；一切以 **B3 门通过率为第一前提**（即使 10% 通过率仍 ≥ A 线 30 倍，
量级优势不因通过率崩塌，但具体判据按实测校准）；**自然过渡段是意外红利**
（288h 连续动作流含海量自然步态转换，可切分则过渡材料从 36 段受控样本升为
海量，直接攻击全局最大难点 transition——B3 抽检时加切分试点验证）。

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
   〔09-04 冒烟+探针勘误与标定，驱动 `drive_ds_manifold.py` 已实现〕：
   ①'n' 不是二态切换，是 **4 组循环**（0 站立/1 蹲爬/2 拳击/3 styled——
   `localmotion_kplanner.hpp get_motion_set`；越界回落站立组），HAPPY = 3×'n'
   到 styled 组后按 "4"，下行用 'p'；②SLOW_WALK **基速 0.2**、RUN 基速 1.5、
   '0' = +0.1/次（'0'×1→0.3、×5→2.0 已核）；③**'a'/'d' = 每按一次 movement+
   facing 旋转 ±5.73°（累加制，非长按动量制——长按 300 次转出 139.5° 定值），
   'q'/'e' = 纯 facing ±30°（movement 不动），'w' 沿当前 movement 维持动量
   （不复位方向），'r' 清 movement 保 facing**；斜向 = 'a'/'d' 点按 N 次
   （45°≈8 次）后用 'w' 维持。
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

## 3. Phase 1（A 线主案）：采集（官方回路；稳态 + 速度梯度 + 过渡段）

**网格**（全部官方 deploy 回路，段间 idle 10s + fall 计数，events.json 留档；
驱动 = `drive_ds_manifold.py`（SCRIPT_MAP 已登记），段段自报末条
"Replanning with mode" 行做 mode/speed/方向自证）：

| 族 | 键位（含组切换） | 速度档（'0'=+0.1/次） | 方向（点按制） | 时长 | 段数 |
|---|---|---|---|---|---|
| SLOW(1) | 站立组 "1" | 基速 0.2：{0.2,0.4,0.6,0.8}='0'×{0,2,4,6} | fwd/back(diagL45/diagR45='a'/'d'×8+'w'维持) | 60s×2 | 32 |
| HAPPY(23) | 3×'n'→styled 组 "4" | 默认 | 8 bin：{0,45,90,135,180,225,270,315}°='w'/'a'×8/×16/×24/'s'/'d'×24/×16/×8 | 60s×2 | 16 |
| RUN(3) | 站立组 "3" | 基速 1.5：{1.5,2.0,2.5,3.0}='0'×{0,5,10,15} | fwd×4 档；{back,L45,R45,L90}×{1.5,2.5} | 60s×2 | 24 |
| JUMP(17) | 站立组 "4" | 默认 | fwd/L45/R45 | 60s×2 | 6 |
| 过渡段 | 组状态机序列切键 | — | 前 | 20s×3 相位 | 12 对×3=36 |
| （可选 C）stones 过渡 | 见风险 5 | — | 前 | 30s×6 | 6 |

- **方向语义（09-04 探针标定，§1.5 勘误③）**：'a'/'d' 每按一次 movement+facing
  旋转 ∓5.73°（累加），45°≈8 次；'w' 沿当前 movement 维持动量。斜向段 =
  先点按 'a'/'d' N 次、再长按 'w' 维持；纯后向直接 's'。
- 过渡段采法：驱动脚本按 `A 键 10s → idle 2s → B 键 10s` 序列（12 对 = 4 族
  两两双向），3 个起切相位（idle 后第 0/2/4 秒切）。
- **对齐资产**：deploy `--enable-csv-logs` 逐帧 csv 带 `time_realtime_ms`
  墙钟戳（logs/ 下 action/base/encoder_mode/motion_name 等），与 events.json
  墙钟直接对齐——Phase 2 切段不靠行数算术。
- **预算**：~1.7–2h 机时（实测段表）+ 驱动脚本改造 ~0.5 天（已完成，冒烟
  两轮标定 mode/speed/组切换全对齐）。
- **产物**：`apt_g1/data/ds_manifold/`（csv 三件套 + logs/ 逐帧 csv +
  events.json + manifest.json + deploy.log）。

## 4. Phase 2：数据集构建 + 质量门

- 新脚本 `build_ds_dataset.py`（`build_exp3_dataset.py` 模式；SCRIPT_MAP 登记）：
  合并 csv → npz：token/proprio/cmd/**mode/speed(命令标注)/angle_bin/regime(4+3 held-out one-hot)/transition_flag**。
- **质量门（不过不进 VAE）**：①D002 式 oracle 回放抽检（每族 2 段 × 500 步，
  存活 + realized 速度与采集记录一致）；②token lattice 合法性（D001：k/16 格点）；
  ③过渡段 token 连续性（切换点前后 token 距离 ≤ 族内稳态 P95）。
- **预算 ~0.5 天**。产物：`data/ds_manifold/ds_manifold.npz` + 质检报告。

## 5. Phase 3（v2 双轨）：流形 VAE 训练 + held-out 验证（A/B 双源汇合点）

- **双轨设计（§0.5v2；原 B4 三选一降级为 VAE-L 内部实现细节）**：
  - **VAE-S 轨（必做，金标准小流形）**：A 线 `ds_manifold.npz` only——原判据
    全部不变，与 E39 底板直接可比；
  - **VAE-L 轨（B3 通过量 ≥ 10 万帧即启动）**：A 锚定 + B 放大（过门大类全量
    + A 全量；首轮 A 线 ×10 上采样保命令语义占比，可调）。两轨同判据对比 =
    双流形对照实验（L 优 = 数据规模瓶颈 / S≈L = token 空间容量天花板，预注册
    不预设赢家，两向皆硬结论）。
- **裁定前置 = Phase 2 与 B3 双门全绿**；owner 只裁 VAE-L 启动与否与最终底板。
- 新脚本 `train_token_vae_ds.py`（`train_token_vae_e39.py` 架构扩展，登记）：
  条件轴 = **vb 连续速度（A 线命令标注优先；B 线段落用参考轨迹速度代理标注，
  来源列区分）+ regime one-hot（4 + bones 来源类）+ db 8bin**；z 16d；dir/speed
  对抗解耦头沿用（E39 配方）；过渡段按 transition_flag 加权（×2 采样权重——
  过渡是稀缺样本）。
- 训练：lab-ts GPU（3060 可，~1h 级）；数据 60–80k 步。
- **验证三件**（G1 判据）：val MAE ≤0.08；held-out 族 {18,19,2} 最近邻重建误差
  ≤ 训练族 1.5×（流形连续性可证伪检验）；z 空间族间线性插值 token 的 lattice
  合法率 ≥90%。
- **预算 ~1 天**（含失败重训一次）。

## 6. Phase 4：RL 底板重训 + 判读门

- **底板候选 = 双流形对照胜者**（四件判据：val MAE / held-out 重建 / 内插
  lattice 合法率 / G3 速度谱响应；平手则 VAE-S 上场保 E39 可比性，VAE-L 记
  descriptive）。
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

## 7B. B 线（与 Phase 1 并行）：BONES-SEED × 官方 encoder 离线编码

> 架构级定位与分工铁律见 **§0.5**；文件级规范与资产清单 =
> `DS_SONIC_OFFICIAL_DATA.md`（owner 09-04 提权立项）；本节只定计划级
> 位置、门与预算。**不替代 A 线 Phase 1**。

- **定位三重**：①G3 速度段放大器（BONES-SEED 含跑/舞高速材料，绕开官方回路
  RUN 48% 执行衰减的数据侧问题）；②规模放大器（离线编码 vs 1:1 实时采集，
  差 2–3 个数量级）；③Phase 1 工程受阻时的备胎。
- **步骤**：B1 下载盘点（`nvidia/GEAR-SONIC` 的 encoder ONNX 50.1MB +
  observation_config + sample_data <60MB；`bones-studio/seed` G1 格式
  locomotion/dance 大类抽样；**license 门先核对**）→ B2 格式对齐冒烟
  （observation_config ↔ `planner_sonic.py` encoder 输入逐字段对齐；
  sample_data encode→token→decoder 回环 sanity）→ **B3 抽检门（不设门不入集）**：
  每大类 ≥10 段 ×500 步 Isaac oracle 回放（复用 D034 机制），类级存活 ≥95%
  准入 → B4 全量编码 npz（`apt_g1/data/ds_bones/`，gitignored）。〔v2〕B3 同批
  加**自然过渡段切分试点**：抽 10 条跨动作 recording 按速度/姿态变化点切出
  walk→run / 舞→走 转换段做 oracle 回放——验证「288h 连续动作流 = 海量过渡
  材料」假设（§0.5v2 红利项）。
- **判据**：B3 类级存活 ≥95%（首轮抽检后可校准阈值）；token 先过 D001
  lattice 检验 + 与官方回路 token 分布对照（mode 匹配段）。
- **B4 合并策略三选一（Phase 3 VAE 训练前 owner 裁定，预注册）**：
  ①纯对照臂（bones-only 重训 VAE 看 held-out 差异）；②合并加权（bones 作
  速度段补充，官方回路作稳态/过渡/方向主料）；③探针（仅作流形内插验证材料）。
- **预算**：~1 个工作日（机时 <1h GPU）；与 Phase 1 并行（B1 下载不占机时）。
- **卫生学**：D033 禁的是 planner 开环提取路径；本轨走「官方 encoder 编码 +
  Isaac 回放检验」，与 D002/D034 质检框架同源，不冲突。

## 8. 总预算与里程碑

| Phase | 墙钟 | 机时 | 关键门 |
|---|---|---|---|
| 0 校准 | 0.5d | <1h | ~~Isaac 实现率 ≥0.9~~ **已完成 D034 PASS(1.61)+D035 打滑 HONEST** |
| 1 采集 | 1d | ~1.7–2h | fall 率 <5%/段、events 完整 |
| 2 数据集 | 0.5d | — | oracle 回放门 + lattice 门 |
| 3 VAE-S（必做） | 1d | 1h | G1 三件 |
| 3L VAE-L（B3≥10万帧启动） | +1d | +3h（量大走云 A10） | G1 三件 + 连续速度谱 + 大类留出 |
| 4 RL 底板 | 0.5d+4h | 4h | 对照 E39 门 |
| 5 切换 | 1d | 4h | 两臂矩阵 + 停止规则 |
| **B 线数据源（§0.5/§7B，与 Phase 1 并行）** | ~1d | <1h GPU | B2 回环 sanity；B3 类级存活 ≥95%；B4 合并策略 owner 裁定 |
| 合计 | ~6 个工作日 | ~12h 机时 | |

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
7. **BONES-SEED 方向偏斜**（§7B）：转向/横移材料天然稀少——不得替代 Phase 1
   方向网格，只作速度段/规模放大器；入集前必须过 B3 门（动态可行比例未知）。
8. **采集网格有效性依赖方向键语义**：'a'/'d' 为逐次累加转向（长按打转）——
   驱动已改点按制；若后续 deploy 版本换键位语义，冒烟自证行
   （movement/facing 向量）是唯一放行依据。

## 10. 纪律

- Run 行只进 `tracker/D.md`（D034/D035 已用，后续 D036 起）；新脚本全部登记
  SCRIPT_MAP；本计划
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
- **09-04 D035**：打滑核验 **HONEST**（3/3，中位接触足速 0.038 m/s、双足
  占空 ~0.8 = 20% 周期腾空、q 跟踪 MAE 0.171 rad 排除刚性拖拽）——Isaac
  1.6657 m/s 为物理诚实跑步，速度声明解除打滑保留。
- **09-04 全面修订（owner 指令「彻底修改计划方案和整体规划」）**：
  ①并入 **Phase B**（BONES-SEED × 官方 encoder，§7B，owner 同日提权立项，
  规范 = `DS_SONIC_OFFICIAL_DATA.md`）——数据双源策略入 §0，预算表加行，
  风险 7 新增，B4 合并策略留 Phase 3 前 owner 裁定；②Phase 1 网格按冒烟
  两轮 + 方向探针**标定重写**（§1.5 勘误：'n'=4 组循环、SLOW 基速 0.2、
  'a'/'d' 逐次累加 ±5.73°/q'e' 纯朝向 ±30°/'w' 维持；§3 网格表全部改为
  点按制方向预置 + 键位含组切换 + logs/ 墙钟戳对齐资产），驱动
  `drive_ds_manifold.py` 同步实现；③头部状态改「执行中」。

- **09-04 双线架构重构（owner 指令「参考官方数据立项内容彻底修改计划与整体规划」）**：
  §0.5 新增（A/B 双源总体架构图 + 分工三铁律 + 时序），B 线从「§7B 附录式侧轨」
  升格为一等公民；Phase 3 改写为「A/B 双源汇合点」（数据组成 = A 必选 + B4 按
  策略，裁定前置 = Phase 2 与 B3 双门全绿）；vb 标注来源列区分（A 命令标注 /
  B 参考轨迹代理）；Run 行编号修正 D036 起；上游文档链补
  `DS_SONIC_OFFICIAL_DATA.md`（B 线规范）与 `LITERATURE_SURVEY_DS_MANIFOLD.md`
  （出处）。A 线全部 Phase 内容与判据零改动（append-only 纪律维持）。

- **09-04 v2 三层架构二次重构（owner 研讨「官方数据能否大幅增长进度/能力/期望」；
  分两次落盘 = 7c23bd9（§0 判据升级 + §0.5v2）+ 本条（Phase 3 双轨 / Phase 4
  底板候选 / 预算 VAE-L 行 / B3 过渡切分试点））**：核心 = 双流形对照
  VAE-S/VAE-L + 双层架构（厚底座 × 4 族接口）+ 升维判据（G1 大类留出、G3 连续
  速度谱回归）+ B3 过渡试点；不变量 = 地形/命令语义归 A 线、B3 通过率第一前提
  （≥10 万帧即启 VAE-L）。
