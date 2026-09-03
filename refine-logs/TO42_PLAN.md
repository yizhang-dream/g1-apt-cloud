# TO42 设计定稿（protocol draft）：学习型 regime 选择（learned regime selection，Rung 1 selection-only）

> 【层位 L2 侧轴｜设计定稿：TO42（2026-09-03 文献调研改向后定稿；协议冻结候选，
> 开跑须 owner 另行授权）】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`tracker/TO.md` TO41 §四十三轮（TO41 终态 + TO42 初步方向 = A/B 固定臂
> τ 干预，未立项）、`TO41_RUNG1_CLOSURE.md`（Rung 1 关线：regime 支配行为 +
> τ_ff 效应 regime/seed 依赖）、`TO41_C_DIAGNOSIS.md`（bucketize 边界与 C1/C2
> 身份重构）、`TO40C_PLAN.md`（§3 E47 配方与对照臂重训先例 / §5 评测协议骨架，
> 本轮逐字平移）、`tracker/TO.md`（Run 行唯一事实源）｜状态：**活跃**
> （协议冻结候选 → **FROZEN**（2026-09-03 owner 开跑授权 + 数据保全裁定后
> 顺序冻结：Recovery Gate → lineage audit → scientific readout，判读前置于
> Gate PASS）；开跑 / compute 已获 owner 批准）

**这篇讲什么**：TO41 证明了两件事——**decode regime（vb0/vb1 locomotion
manifold）支配行为，强度 ≥ τ_ff**；以及移植度比对确认 APT-RL 移植里**唯一没
复现的论文成分是学习的 gait 选择器**（我们的对应槽位是冻结的
`bucketize(cmd)`）。本轮把这一个成分外科手术级补上：在冻结解码器底板上，把
regime 从"冻结混杂因子"升级为**学习型离散动作**（论文配方逐字移植），τ_ff 全程
OFF，只问一个问题——**让策略自己拥有 regime 接口，能不能在两流形夹住的中速带
里插值出速度**。负结果（插不出来）同样是合格产出。本文只立协议，不动代码。

---

## 0. 改向记录与授权链

- **四十三轮初步方向（未立项）**：TO42 = regime-controlled τ intervention——
  A 臂组固定 vb0、B 臂组固定 vb1，2×2 估计 Δτ(r), r∈{vb0,vb1}（见
  `tracker/TO.md` TO41 §四十三轮）。该方向继承 TO41 收束时的「唯一提前锁死
  原则」（先固定 regime 再测 τ_ff）。
- **2026-09-03 owner 授权**：跳出已入档的 A/B 固定臂候选，按文献调研重新选
  方向；调研完成后 owner 确认改向 = **learned regime selection**。
- **A/B 固定臂设计不废弃**：降级为 Rung 2 的 post-selection 子问题（§7）——
  在 policy 自己选进并锁定的 regime 稳态内测 Δτ，比训练期硬锁更干净地拿到
  「分布内 regime」。
- **与「唯一提前锁死原则」的关系**：Δτ(r) estimand 不作废、延后——Rung 1 τ 恒
  OFF 只做 selection；Rung 2 的 post-selection Δτ(r_sel) 即原锁死原则的实现
  形式。这不是 reopen TO41（九项冻结清单与 maintenance mode 均不受影响），是
  TO42 内部的 Rung 分层。本文正是 maintenance baseline 记录中「下次开发入口 =
  TO42 protocol draft only（非代码改动）」的执行物。
- **regime 锁定三候选（training-time / eval-time / stratified）处置**：被本次
  改向**取代**而非搁置——Rung 1 中 regime 由 policy 拥有，不存在实验者锁定；
  Rung 2 的锁定机制 = policy 自选 + lock 窗口（因果语义最接近候选 A
  training-time，因 policy 从始适应自己的选择）。

## 1. 调研依据（为什么改向是"更正确"）

1. **论文缺的成分，结构上与我们完全同构**。论文（Materials and Methods）的
   gait 选择器 = 单个 logit 过 sigmoid、阈值 0.5 在两个解码器（trot/bound）间
   二选一，2 Hz gate 触发、选中后锁定 0.5 s、gate 布尔与当前选择回馈 obs。
   我们 {vb0, vb1} 的对应槽位 = `apt_g1/isaac/decft_policy.py:150`
   `vb = torch.bucketize(cmd[:, 0], edges)`（`n_vbins=3`、`vx_max=0.8` → 边界
   [0.267, 0.533]）。本轮移植度比对（论文 M&M ↔ 本仓 runtime）确认这是
   **唯一未被复现的论文成分**，而它的移植是外科手术级的：动一个离散头 + 一条
   obs 回馈，解码器、Mode A、τ 材料全部不动。
2. **文献在"谁拥有 regime 接口"上口径高度一致**。[Walk These Ways（ICRA
   2023）](https://arxiv.org/abs/2212.03238)把步态参数作为命令条件化，是四足
   command-following 与泛化的标准解；[CALM（NeurIPS
   2023）](https://arxiv.org/abs/2305.02195)证明高层控制器应通过学习的潜空间/
   语义接口驱动低层；[DeepPhase（SIGGRAPH
   2022）](https://dl.acm.org/doi/10.1145/3528223.3530178)显示相位/步态流形
   嵌入作为条件化显著改进技能学习；[Learned Gait Transitions（CoRL
   2021）](https://proceedings.mlr.press/v164/yang22d/yang22d.pdf)与
   [Kim & Son 分层多步态框架](https://www.semanticscholar.org/paper/614999ad5f6f4927506f12bbe6e07ca49cc2a122)
   都观察到**速度依赖的离散步态选择自发涌现**。四条线同一句话：regime/步态
   接口应由学习器拥有，而不是环境冻结。
3. **人形侧这块是公开空白**。[APT-RL 原文](https://www.science.org/doi/10.1126/scirobotics.adz7397)
   的人形扩展只有 movie S4 一句 claim（[项目页](https://skillquadsr.github.io/)），
   截至本轮检索无公开跟进；最近参照 [Radosavovic 真机人形（Science Robotics
   2024）](https://arxiv.org/html/2303.03381v2)与 [UCL gait-conditioned 多相位
   框架（Humanoids 2025）](https://discovery.ucl.ac.uk/id/eprint/10219246/1/2025_Humanoid_Conference_Paper-13.pdf)
   都印证"条件化 + 步态间转移"是人形主线，但没人做过"冻结第三方解码器 +
   学习型 regime 选择"的识别级审计。补此成分 = 论文 port 的最后一块 + 对
   robot-agnostic claim 的人形版检验。
4. **TO41 自己的数据给出了 selection 的可测性能前沿**：C1/C2 realized speed
   （~0.13 / ~0.61 m/s，收束文 §3d）**夹住**整个 cmd 网格（0.200–0.325）；
   mid-band 四点 best-fixed OFF err = {0.104, 0.109, 0.138, 0.198}（收束文
   §3c）。冻结底板上，regime 的时间复用（duty-cycle 切换）是**族内唯一的速度
   插值机制**——论文正是以 2 Hz logit 服务 −2~7 m/s 全速度段的方式。

## 2. estimand 与主假设（开跑前预注册）

**estimand（中性命名）**：**selection-interface contrast**
= err60s(learned-selection) − err60s(frozen-bucketize)，按 v 分段报告，主对照
段 = mid-band {0.275, 0.277, 0.300, 0.325}（TO41 mapping v2 的 mid 语义，保
TO41 可比）。两重基线，角色分明：

- **配对基线** = frozen-bucketize 臂（同 commit / 同 seed 重训，obs 结构对齐）
  ——回答「接口由学习还是冻结拥有」的 contrast；
- **绝对前沿** = TO41 effect_table OFF 臂 per-v best-fixed-regime err（两点取
  较优的 oracle 参照：mid-band 四点即 C1 OFF 的 {0.104, 0.109, 0.138, 0.198}）
  ——回答「切换能否超越任何固定 regime」。

**主假设 H1（可证伪）**：learned time-multiplexed selection 使 mid-band
err60s 相对 best-fixed-regime 前沿降低 ≥ 0.02（仓库决策边界，继承 TO40C §9.1
practical-equivalence 口径，TO41 沿用）。

**H1 的机制内容（预注册，防误读）**：在 per-step |vx−cmd| 指标 + 两 regime 固有
速度冻结（~0.13 / ~0.61 m/s）的前提下，duty-d 切换的期望误差
= (1−d)·|v0−cmd| + d·|v1−cmd| ≥ min(|v0−cmd|, |v1−cmd|)——**纯混合在算术上
不可能超过 best-fixed**。故 H1 若成立，唯一通路是**训练期可塑性**：学习型选择
接口使各 regime 的 realized speed 本身变得 cmd 可响应（即推翻/修正 TO41 §3d
「realized speed 由 regime 决定、几乎不由 cmd 决定」），或产生真正的新中间
速度行为。预注册描述性检验：learned 臂 per-regime realized-speed vs cmd 斜率
（对照 TO41 ≈ 0）；若 err 改善而速度斜率仍 ≈ 0，疑转移段扫速对指标的贡献，
标记为 metric artifact 线索，不升格为结论。

**停止规则（预注册三支，全部为合格产出）**：

| 支 | 触发 | 结论（负结果措辞预注册） |
|---|---|---|
| (a) | 选择器塌缩：训练后期 ≥90% 单一 regime 且不随 cmd 变化 | 「冻结底板接口密度不足以支撑学习型选择」 |
| (b) | mid-band 无 ≥0.02 改善（vs best-fixed 前沿） | 「冻结底板上 regime 切换不能插值速度（混合算术预期成立）；cmd→realized-speed 通路缺失持续」——直接延伸 TO41 §3d |
| (c) | mid-band 改善但随训练 seed 换向 | 沿用 TO41 regime/seed-dependent 口径，不宣称稳定主效应 |

## 3. 机制规格（论文配方逐字移植；implementation 阶段才动代码）

- **selection head**：policy encoder 之上加单 logit s；2 Hz gate tick（0.5 s
  周期，与 env 控制频率的整除对齐由 implementation 章程定）；tick 时
  σ(s) > 0.5 → 目标 regime = vb1，否则 vb0；与当前不同才切换；**切换后锁定
  0.5 s**（锁定期内 gate 不生效，decode 输入恒定）。
- **selector 值域 = {vb0, vb1}**（vb2 不进 Rung 1，保 TO41 可比）。
- **obs 回馈**：当前选择（标量或 2 维 one-hot）+ gate 布尔进 obs；**两臂 obs
  维度与语义槽位完全一致**——差异唯一归因于「选择由策略学出，还是由 cmd 的
  冻结函数产生」。
- **baseline 臂对齐**：frozen-bucketize 臂把 clamp 后的冻结 bucketize 结果写进
  同一 selection 槽位、gate 信号恒 0。在 eval 网格（全部 v ≤ 0.325 < 0.533）上
  与 TO41 自然分配**逐位一致**；仅训练分布 cmd > 0.533 段与 TO41 ctrl 不同
  （该段本就在 TO41 eval 支撑之外）。
- **冻结面（全不动）**：SONIC decoder（frozen）、e39 VAE（frozen）、Mode A
  runtime、τ 材料（G↓ LUT）、eval protocol / jitter、speed grid 7 点。
- **实现落点（未来 implementation）**：`decft_policy.action_mean` 的 vb 计算
  与 env obs 拼接处；receipt schema 增 selection 时间线字段（封闭 schema 纪律
  照 rung1 先例）；SCRIPT_MAP 届时登记（runtime / checker 分列）；服务器目录
  `TO42/{protocol, train, eval, analysis}`（集合预冻结，不污染 TO41）。

## 4. 实验设计（2×2 同 seed 配对）

- 臂：{learned-selection（`to42-lsel`）, frozen-bucketize（`to42-fbkt`）} ×
  {s0, s1}。
- 配方：E47 精确配方逐字 + ctrl 臂旗标（`TO40C_PLAN` §3：`--latent-mode
  --latent-vae-path token_vae_e39/vae.pt --latent-speed-bins --latent-dir-bins
  --latent-kl-prior zero --progress-scale 1.0 --heading-scale 0.4 --to-ref
  --to-ref-npz to38_ref.npz --to-ref-obs-zero --to-ref-w 0`），128 envs ×
  2000 it，lab-ts 串行；**τ 恒 OFF（无 --to-tau，selection-only 干净
  estimand）**。
- **重训 baseline 而非复用 TO41 ctrl ckpt**：配对纪律（同 commit / 同 seed /
  同机，TO40C §3 先例）+ obs 结构对齐；TO41 ctrl ckpt 可作 **legacy 锚点**
  免费 eval（跨轮旁证，不进主对照）。
- eval：28 receipts（2 臂 × 2 train seed × 7 v）× 3 eval seeds = 84 episodes；
  jitter rng(1000+seed) 唯一随机入口、policy deterministic、disturbance_prob=0
  （TO41 口径平移）。
- ckpt 规则：50-iter 窗口 argmax（对称非手挑），manifest 先冻结再启动任何正式
  eval（TO41 四硬点平移）。
- 预算：训练 4 runs ≈ 2.5 h + eval 28 receipts ≈ 1.6 h，全 lab-ts 串行；零云
  算力、零新 TO 解、零解码器改动。

## 5. 门与判据（开跑前预注册）

- **floor**（每臂各自，失败即报失败）：60 s completed、h_min ≥ 0.6、
  disp > 0.5 m（TO36-C 门口径）。
- **G0 wiring（开跑前，negative cases 先行）**：fbkt 臂在 eval 网格上 selection
  时间线 ≡ 冻结 bucketize 时间线（逐位），gate 恒 0；lsel 臂 integrity：选择
  只在 tick 边界变化、lock 窗口内 decode 输入恒定、selection 头无未来信息泄漏。
- **G1 execution**：84/84 episodes completed；selection 时间线（选择序列 +
  gate 触发时刻）入 receipt。
- **G2 behavioral**：selection 随 cmd 变化（低带 vb0 主导 / 高带 vb1 主导的
  涌现本身即通过证据；全程单 regime → 走停止规则 (a)）。
- **判读**：效应曲线 7 v 全报（主对照 mid-band）+ §2 停止规则；先审计后分析
  （checker 全 PASS 前不读行为指标，TO41 四硬点平移）；论文聚合指标
  best-performance rate / regret 仅作描述性附件在 receipts 上复算（论文四足
  报告值 44.44% / 4.99% 来自其 gait grid 协议，数值不可直接比，只作结构参照）。

## 6. 本轮不主张什么（边界）

- **不测 τ_ff**（恒 OFF）——本轮任何结果不得读作「τ_ff 有效 / 无效」；TO41
  收束文 §6 canonical 英文结论段继续是 τ 侧唯一引用口径（Q2 / Q3 NOT
  IDENTIFIED 不变）。
- selection 涌现 ≠ 机制被识别：gate≠机制、PASS-AS-CHANNEL 纪律沿用；不解释
  解码器内部机制。
- 不扩 speed grid、不加第三训练 seed（如需 robustness 另立独立实验）、不改
  decoder binning、不改 Mode A。
- TO41 九项冻结清单（收束文 / effect table / audit-receipt-manifest / G1–G10
  结果 / Δ_ff 解释 / seed 配置 / Mode A / binning / τ material identity）一字
  不动；TO42 全部产物走新目录。

## 7. Rung 2 前瞻（不预授权，selection 建立后再议）

- **post-selection Δτ(r_sel)**：在 policy 选定并锁定的 regime 稳态段内做
  τ ON/OFF——回收四十三轮 A/B 固定臂设计（owner 设计不废弃，换到更干净的
  测量位置：分布内 regime，policy 自适应后的稳态）。
- vb2 / speed grid 扩展评估（连续速度段下选择器行为与 trot/bound 型多 regime
  扩展）。

## 8. 执行序与产物清单

1. 本 protocol 冻结候选 → owner 批准（开跑授权，含 compute）。
2. implementation：selection head + obs 回馈 + fbkt 对齐槽位 + receipt 扩展
   字段；G0 selftest（negative cases 先行）；SCRIPT_MAP 登记。
3. 训练 wave（4 runs 串行）→ ckpt manifest 冻结。
4. formal eval（28 receipts）+ checker（G1/G2）→ 判读（效应曲线 + 停止规则 +
   速度斜率描述性检验）。
5. Run 行入 `tracker/TO.md` TO42 节；收束时另立 `TO42_RUNG1_CLOSURE.md`
   （命名照 TO41 例），停止规则命中支 = 论文结果段骨架的身份。

## 9. 开跑留痕（2026-09-03 owner 授权后补记）

- **授权**：owner 指令「请使用云算力，开始实验」（2026-09-03）= §8 步 1 的
  开跑授权（含 compute）；venue 由「lab-ts 串行、零云算力」改为 **LimX 云
  平台单 A10 pod**（机型 ESKU000004，¥4.01/时；镜像 BJX00000002/V000125 =
  IsaacSim 4.5 / IsaacLab 2.0.0 / torch 2.5.1，与 lab-ts `.venv_isaac` 同版；
  TO38/TO39 已在同一镜像验证 git-mode 全链）。**同机配对纪律保持**：4 runs
  同 pod 串行；「确定性复现/哈希对照留原环境」策略例外不适用——TO42 定义
  自己的 env 身份，全链哈希入 receipt。预算估算 4–6 h ≈ ¥17–25（赠款额度内；
  平台无余额 API，实额以 Web 台账为准）。
- **执行身份**：冻结资产全部 git 白名单在仓（pod clone 即得）——e39
  `vae.pt`（+pca.npz/z_walk.npy）、`sync/to38_ref.npz`（**中性脚手架**：obs
  置零 + w=0 + to_tau 关，内容不进动力学，文件 sha 入 receipt）、
  `distill_final`、SONIC decoder onnx。
- **实现面**（SCRIPT_MAP §8d）：`to42_gate`（纯 torch 状态机，env 与自检同一
  份代码）/ `to42_selftest`（G0 负例先行，本机全绿）/ `to42_eval` /
  `to42_select` / `to42_checker`（先审计后分析，不读行为指标）/
  `to42_cloud_wave`（gm-run 入口，`--stages` 可重入）+ canonical 两文件
  cfg 门控增量（默认 off = TO41 逐位，TO41 九项冻结清单不受影响）。
- **eval harness 逐字继承 TO41**：jitter rng(1000+seed)（root z ±5mm / 29
  joint pos ±0.01rad / joint vel ±0.02rad/s）、恒定 cmd 每步重申、确定性策略、
  episode_length_s=120（60s eval 内无 auto-reset）、eval seeds {0,1,2}、
  ckpt 50-iter 窗口 argmax 规则逐字。
- **与协议正文的偏差（两处，owner 授权链内留痕）**：① venue 云化（本节）；
  ② legacy 锚点（TO41 ctrl ckpt 免费旁证 eval）**不执行**——TO42 obs +2 维
  使 TO41 ckpt 不可直接加载，如需旁证须另加 padding 兼容加载器，留待按需。
- **执行身份修订 v2（2026-09-03 深夜 owner 授权「worker-pool 最省时间」）**：
  venue 不变（A10 pod），科学配置零变化（冻结矩阵/配方/判据逐字），编排层
  改并发——① **训练 2 并发**：同 seed (lsel, fbkt) 成对并发（对内共享 GPU
  争用状态，配对对称性按对保持；s0 对先于 s1 对），单臂浮点路径受争用影响
  如实声明：单臂不再可 solo 逐位复现（重跑语义 = 同配置重训，与既有跨机
  不可逐位一致口径一致）；② **评测 3 并发**：每 cell 单 env 确定性策略 +
  冻结 jitter，并发零科学足迹；**mid-band 16 cells 优先出队**（中途死亡的
  salvage 价值最大化）。并发位次入 receipt（execution.worker_tag）与
  bundle meta.concurrency；bundle 写全部收敛在编排主线程（无并发写）+
  原子写不变。依据：串行惯例的起源是 3060 12G 装不下两份 128 envs 的物理
  约束 + 「同机同条件」冻结纪律——并发位设计保住后者。预期全程
  4.4h → ~2.8–3.2h（省 ~1.2h ≈ ¥5），同时压缩中途被平台终止的风险窗口。
  实现 = to42_cloud_wave.py worker-pool（SCRIPT_MAP §8d）。
