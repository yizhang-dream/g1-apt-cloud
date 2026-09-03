# Experiment Tracker —— 总索引

> 【层位 L3｜Run 台账总索引】↑ `HANDOFF/02_EXPERIMENT_HISTORY.md`（L2 阶段
> 索引 → stages/ 叶子）｜↓ `refine-logs/tracker/` **五个系列文件（数据唯一
> 事实源）** → `HANDOFF/03_OUTPUTS_INDEX.md` → 服务器 `outputs/`（L4）｜
> ≈侧轴：`apt_g1/SCRIPT_MAP.md`（代码轴）｜树根：`refine-logs/README.md`
> （实验记录扇出树，`python refine-logs/tools/tree_check.py` 闸门）。

**2026-08-28 起口径变更**：原 98KB 单表已按系列整节拆入 `tracker/`，五个系列
文件**共同构成数据唯一事实源**；本文件只做索引与统计，不再直接存 Run 行。
**新实验（含负结果）一律追加到对应系列文件**，并同步本表行数。

| 系列 | 事实源文件 | 覆盖 | 表行数 |
|---|---|---|---|
| R | [tracker/R.md](../refine-logs/tracker/R.md) | R001–R020 MuJoCo RL 线（token/VAE/skill/aux 变体，全劣于 aux=0） | 21 |
| D | [tracker/D.md](../refine-logs/tracker/D.md) | 蒸馏 Distillation 线（Distillation Experiment/Phase2/Phase3 + Stress Test） | 25（另有 D021–D028 共 4 行混排在 E.md『地形/数据泛化/感知』节，该节以 E 线为主未拆） |
| E | [tracker/E.md](../refine-logs/tracker/E.md) | Isaac APT 主线 E01–E48（含 FB/I21/T1-T2 等辅助行与上述混合节） | 87 |
| MQ | [tracker/MQ.md](../refine-logs/tracker/MQ.md) | 官方规划器复刻 MQ07–MQ12 + Gate 0 论文形状地形评测（G0a/G0b） | 22 |
| TO | [tracker/TO.md](../refine-logs/tracker/TO.md) | TO 数据管线 + TO18–TO35 力矩/WBC 线 + TO36 腿级 TO（A 门膝可行/B 门双验证/C 门负结果收束）+ TO37 解族三点与跨速度泛化 FAIL + TO38 RL 叠加 TO 参考（收束：分支一可消化，低速带 50×）+ TO39 低速消融 + TO40a 条件化解码器插值 FAIL + TO40C 力矩前馈通道门控（Rung 0：分支①通道可用+低速带正向，非 E48 破坏）+ TO41 G↓ 低速 downward-continuation 材料 sub-campaign（4/4 k=0 首过 + 7/7 availability map，owner 裁定 CLOSED/PASS；D1/D2/D3 授权；09-02 三十四轮评审 9.8/10：conditioning runtime implementation 授权（仅 Mode A），D dry-run 协议落盘，compute 仍 BLOCKED；同日三十五轮评审 9.9/10：D 协议定稿 FROZEN + implementation 开工；同日三十六轮 owner 批准 implementation 开工 → Mode A runtime + independent checker 落码（negative tests 本机/服务器 9/9）→ 28-cell D dry-run 执行，independent checker 全 PASS（D2 fingerprint 7 v same-τ 全中；report = sync/to41_d/D_report）——execution freeze 待 owner，compute 仍 BLOCKED；同日三十七轮 owner 改判：暂不 freeze——D decode-only PASS≠完整训练环境 plumbing，先过 L1–L4 真实 env 接线 launch sanity（判据=TO41_LAUNCH_SANITY.md）；09-03 执行完毕：gate 首格发现并修复 L0 材料格式断链（7 冻结材料 to36 dump 格式 vs env 所需 TO38 LUT——既有链确定性导出，F11b 交叉验证逐位 MATCH，材料身份不动）+ 28-cell 真实 env 接线验证 independent checker 全 PASS（L1 三层 Mode A fingerprint 7/7、L2 interventional 逐 call 300/300、L3 两臂 cfg diff=={to_tau}、L4 env 源哈希锚；report = sync/to41_sanity/L_report）——execution freeze 归还 owner；同日三十八轮 owner 裁定 GO：execution freeze 已执行（freeze commit 76954a0，仅状态板字段零内容变化）→ Rung 1 compute UNBLOCKED，设计审查结束，直接进入 Rung 1 训练/评估（lab-ts frozen env；两层身份纪律：canonical τ(v) → derived LUT → env，material hash ≠ runtime-consumed LUT hash）；同日 Rung 1 训练 wave launch @freeze HEAD 8f6ba1e：{ctrl,t10}×{s0,s1} 4 runs 串行（TO41-R1-*，TO40C §3 配方逐字，保守取交集裁定留痕）；三十九轮：eval 栈四件落码（SCRIPT_MAP §8c，selftest 9/9、static 56/56，负例先行）；四十轮三连收官（09-03）：训练 4/4 rc=0（窗口最优 ctrl it350/it1250、t10 it200/it50 早期窗机械接受）→ ckpt manifest 冻结（8fab587f）→ 56-cell formal eval + eval_checker G1–G10 全 PASS（168/168 completed 零倒地）→ 第一轮判读：Δ_ff 训练 seed 符号翻转（C1 臂 s0 正/s1 负、C2 臂两 seed 同正）、Δ_cond 大且结构化但方向 seed 依赖、conditioning override 强效应（两 seed 一致）；四十一轮 owner 裁定 (c)（09-03）：现有产物三块诊断（零新增执行；详文 TO41_C_DIAGNOSIS.md）——V 方差：eval-seed sd 中位 0.0018 vs train-seed |Δdff| 中位 0.052（F≈133，train 主导）；S：eval seed 确进执行路径（0/56 逐位全同），可忽略=强镇定非未生效；N：bucketize 边界 [0.267,0.533] → 每 (v,C) 恰一臂 natural 一臂跨 bin 强制，Δ_cond 双段拼接非 homogeneous estimand，主结构=decode-regime(vb0/vb1)×seed 交互（regime 差 ±0.064–0.068 ≫ matched-ness 差 ≤0.017）；D：OFF 臂 C1/C2 全 7 v 区间分离（最低间隙 0.040>0.02 边界）不可比 → 落点分支 (b) condition-specific contrast；第三 seed 仍 NOT AUTHORIZED，待 owner 裁定） | 82 |

表行合计 235（其中 R/D/E/MQ/TO 主线编号行 169，其余为蒸馏 gait 行、FB/I/T/P/G
辅助行等，均随原节归属）。拆分为整节逐字搬运，原表可由系列文件按
R(7–30) / D(31–65,91–100) / E(66–90,101–257,346–368,645–693) /
MQ(258–345,611–644) / TO(369–610,694–830) 行区间逐字还原。

各系列的叙事版结论见 `HANDOFF/02_EXPERIMENT_HISTORY.md` 阶段索引与
`refine-logs/stages/` 叶子；专题细节见 FINAL_REPORT / ISAAC_APT_LOG /
DATA_GENERALIZATION_LOG / TO_TORQUE_LINE_REPORT / WBC_BRINGUP_REPORT。
