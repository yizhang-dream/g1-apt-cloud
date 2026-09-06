# E49-C KL 信任域守卫——退化调研结论与执行计划

> 【层位：专题日志域·执行计划｜↑ `refine-logs/README.md`（扇出树根地图）·
> `E49_STATUS_2026-09-05.md`（上一时点快照，§3c 诊断四步）｜↓
> `refine-logs/tracker/E.md`（Run 行唯一事实源）·`apt_g1/isaac/ppo_core.py`
> （实现本体）·`apt_g1/isaac/e49_kl_guard_test.py`（单测）】
>
> **状态：活跃**——E49-C 唯一执行依据。§A 调研结论冻结；§B 协议为预注册稿，
> 发射 E49-C2 前冻结，此后改动须进 §G 修订记录。
> 2026-09-06 深夜审计修正：§A.1/A.2/A.3-2 措辞修正、守卫 v1 三缺陷、
> thr=45.1597 作废——详见 **§A′/§B.3′/§C.5**，与 §A/§B/§C.1 冲突处以其为准。
> 2026-09-06 定稿。裁决：owner（下一干预 = KL 信任域更新，不改奖励；固定
> log_std 初值/熵系数/训练器其余配置；不能只做 KL early-stop，必须含回滚；
> 先短探针定阈值，再单 seed）。

---

## §0 一页执行卡（发射从这里开始）

> 09-06 深夜审计后流程改版：v1 守卫三缺陷（§A′-4）→ C1/C1.5/C2 旧流程
> 与 thr=45.1597 作废；v2 守卫已实现并单测（§C.5），新流程 = **B.3′ 探针
> → C2′**。C2（v1 守卫，thr=45.1597）按协议跑完取数作对照保留。

| 步 | 做什么 | 判什么 | 预算 |
|---|---|---|---|
| **P 探针×3** | fix2 逐字配方 + `--iters 1000 --probe-iters 300 --seed 0` + `--kl-guard {0.01,0.03,0.1}`（v2 守卫，schedule-fair） | 只收数据：rew 是否离开地板上行、回滚率、kl_mb 分布（§B.3′-3 标准） | 3×~8 min |
| **定阈** | 按 §B.3′-3 选 thr（最紧且可学习者） | 候选全灭（全冻死或全不约束）→ 停，回 owner | 1 min |
| **C2′ 单 seed** | fix2 逐字配方 + `--kl-guard <thr>` 跑 1000 it + 双评 | 预注册 J1/J2/J3（§B.4）与分支判读（§B.6） | ~20 min + eval |
| **中止规则** | C2′ 发射后 60 it 内 rew<1.0 或 NaN 连发 → 杀；回滚失控形态（rew 钉地板、roll>50%）→ 回 §B.6 分支④放宽候选重探一次 | — | — |

对照基准 = `E49-A-fix2-s0`（同配方同 seed 无守卫，已有，不重跑）+
`E49-C2-guard-s0`（v1 守卫，判读=不充分对照）。发射命令与监控判读的
逐字命令见 **§D**；预注册判据全文见 **§B**。

---

## §A 退化调研结论（2026-09-06，冻结）

数据源：四条 run 的 `train_log.json` 逐迭代提取（fix2 / lr1e4 / E49-B /
VAE 参照，服务器 `outputs/isaac_e49*`；本地提取件
`tmp/e49_investigation/`，gitignored）；代码核验在 HEAD c19bfe6 工作树。
外部锚点见 §F。

### A.1 核心改写：「KL/裁剪失控打崩策略」被日志时序否决

> 【09-06 深夜审计修正】本节时序证据只否决「KL 尖峰**即时**打崩」叙事，
> **不能排除「持续累积偏离」机制**（性能损失可滞后；且 v1 守卫的 KL 观测
> std 盲，既有 KL 数据不完整）。「更新问题已排除」不成立。见 §A′-1。

三次退化的 rew 谷底**无一**出现在 KL 高点，全部出现在 **approx_kl 低位 +
速度失控（超速/侧漂/后漂）**处；最大的 KL spike 反而与恢复同相：

| 退化事件 | 谷底时 akl（k3，minibatch 均值口径） | 同时发生 |
|---|---|---|
| fix2 it656 rew=0.194 | 65.9（健康段量级 20–160） | vx=1.81 超速（cmd≈0.4） |
| E49-B it575 rew=0.405 | 33.4 | vx 0.53→1.75、drift 0.41→3.94，随后 fall↑ |
| VAE it467–470 rew≈0.72 | 先塌 158.9→5.6→3.2 | vx_fwd 转负 −0.54 后漂 |

反例（KL 大 ≠ 崩）：fix2 it680 `akl=5098 / pkl=16496` 之后 it750 rew 回
1.55；VAE/E49-B 每次恢复也伴随 akl 回升到几十。**形态结论：大 KL = 大
修正步；退化 = 低速漂移（慢漂移）+ 随机游走出行走盆地，不是灾难性大步。**

### A.2 结构实锤：clip 恒 0.833 = 5/6，裁剪完全失效

> 【09-06 深夜审计修正】「完全失效/后五批没有有效梯度」措辞过头——PPO
> 是否截断梯度还取决于优势正负。准确表述 = 更新长期严重超出裁剪区间
> （三 run 实测区间均值 82.9%–83.3%）、**clip 未提供足够约束**；结构数字
> （首 minibatch ratio≡1、5/6 饱和）不变，核心结论（无信任域使能）不变。
> 见 §A′-2。

`--ppo-epochs 1` 下每 iter 恰 6 个 minibatch（128×24/512）；`logp_old`
存自 rollout 策略（`ppo_core.py:294`），故**首个 minibatch 的 ratio 恒 ≡1
——裁剪对它零约束，等于一次纯策略梯度步**；其后 5 个 minibatch 的联合
概率比（logp 对 64 维求和，:332；σ≈0.018/dim）几乎全部出 ±0.2 窗 →
clip_frac≈1。六个 minibatch 平均 = (0+5×1)/6 = **0.8333**，与三条
lr 3e-4 run 的常数逐位吻合；lr1e4（步长小 10×）落中段 0.42–0.55。
**PPO 目标实际 = 每 iter 一次无约束步 + 五个近全裁 minibatch**；文献
健康带 0.1–0.2（§F）。这从代码上证实 owner 裁决「第一步更新本身就可能
过大，必须包含回滚」——epoch 级 early-stop 救不了首步。

### A.3 两个跨 run 稳定规律与一个指标澄清

1. **act_std 全程单调上升**（`exp(mean(64 log_std))`）：fix2 0.0183→0.0262、
   E49-B →0.0299（四 run 最高）、VAE →0.0258；唯一不学的 lr1e4 仅
   →0.0197。代码里**三项 loss 都在推 log_std 上行**：熵系数 0.001×64 维
   （`ppo_core.py:357`）、expl_coef×pd 熵（:368，0.01 线性衰减）、
   kl_prior 项在 logσ<0 处梯度为负即推向 σ=1（:365-369）。与「超速/侧漂
   乱动」形态吻合；但各 run 退化起始 σ 水平不同（0.019–0.024）→ 慢使能
   项非触发器。**嫌疑队列第 1 位**（本轮冻结不动）。
2. **价值函数健康度不可观测**：日志 `expl` 键 = 探索系数 schedule
   （`0.01·(1−it/max_iters)`，ppo_core.py:305,417），**不是** explained
   variance；vloss 计算了但不落 hist（train:585-600）。**嫌疑队列第 2 位**
   ——本轮以纯日志方式补观测（§C.3），不干预。
   【09-06 深夜审计修正：ev 已可观测且结论反转——探针全程均值仅 0.409
   （尾 50 it 0.964）、C2 前 300 it ≈0 → §G 上条「探针 ev≈0.97 价值函数
   健康」**撤回**；正确口径 = critic 早期不健康/晚熟（C2：0–299 it
   ≈−0.006，350–399 it 0.902，it800+ ≈0.99），嫌疑维持且获新证据，
   但不能单独定因退化。见 §A′-3。】
3. lr1e4 臂负结果维持：KL/clip 全程低位也无退化也无学起——低 lr 把策略
   冻在初始站立盆地，「降温」救不了。

### A.4 E49-B 与 VAE 参照（Run 行已落账 tracker/E.md）

- **E49-B（相位归因臂）**：it50 6/6 真直行（vx 0.450–0.453、disp
  23.7–24.3 m）= 把 A-fix2 早期「有速度不直行」改善为真直行；final 6/6
  立即塌（disp≈0.001 m、h_min 0.201–0.229）；训练内两次形态不同的退化
  （it170–260 KL 抬升期、it575 低 KL 崩塌）。**裁决：相位 = 有效表征
  变量，非优化稳定器。**
- **VAE 参照（owner 中止于 it950，不作完成态）**：it301 峰 rew 2.0781 正
  常行走；it460–470 首退化（akl 先塌 5.6、vx_fwd −0.54）；it690–760 二次
  退化（akl 低至 1.09）；it880–949 恢复至 1.684；it949/950 = 1.832/1.861
  零摔。**VAE 能缓冲直出 token 的退化，但不证长期稳定**（未跑满、无终点
  评测，ckpt/日志保留）。

### A.5 原因裁决总表

| 假设 | 裁决 | 依据 |
|---|---|---|
| 裁剪失效 = 无信任域（结构使能项） | **成立** | A.2 实锤；官方配方全带 KL 反馈而我方没有（§F） |
| 「KL 失控打崩策略」 | **否决**（作为动机叙事） | A.1 时序 |
| act_std 单调增长 | **强嫌疑**（慢使能项） | A.3-1；三项 loss 推 σ↑ |
| 价值函数后期失效 | **待观测** | expl_var 此前不可观测；本轮补日志 |
| 奖励地板/走站差弱 | 维持主嫌之一（重标冻结） | 地板 1.641 解析吻合；但尾段 stand_frac 仅 0.30–0.36 且 2/3 时间 \|vx\|>0.05 = 乱动非纯站，「站立吸引子」叙事弱化 |
| 环境契约/探索扰动/trunc 自举 | 已排除 | diag-matrix；fix2 带 timeout 自举仍退化 |
| 相位/表征 | 表征变量非稳定器 | A.4 |

**对 E49-C 的含义（预期管理，进预注册）**：KL 信任域的机制目标是压制
灾难步与振荡幅度（it680 型 spike、退化-恢复循环深度），**不保证阻止慢漂移
退化**——证据显示谷底发生在任何合理阈值都拦不住的低 KL 处。判据按此
诚实预注册（§B.4），负空间分支预定（§B.6）。

---

## §A′ 2026-09-06 深夜审计修正（权威度高于 §A 对应条目）

来源：owner 侧独立审计（服务器原始日志 + 实际执行代码核验 + 不运行仿真的
最小复现）；**全部关键数字与代码论断已在本仓二次复核通过**（本地提取件
`tmp/e49_curve_audit/{fix2,guard,probe,phase}.json`，gitignored）。

1. **§A.1 时序否决降级**：谷底不出现在 KL 高点，只能否决「KL 尖峰即时
   打崩策略」的动机叙事；**不能排除「PPO 更新长期偏离采样策略、逐渐丢失
   行走能力」的累积机制**——性能损失可以滞后，也可以来自持续累积的变化，
   而 v1 守卫的 KL 观测本身 std 盲（见下），既有 KL 数据不完整。
2. **§A.2 措辞修正**：实测三 run 区间均值 clip = 82.9%–83.3%（A-fix2
   80–119 = 82.9%、800–999 = 83.3%；E49-B 800–999 = 83.3%；C2 守卫
   350–399 = 83.2%）——「更新严重超出裁剪区间、clip 未提供足够约束」
   成立；「裁剪完全失效、后五批没有有效梯度」说过头（截断与否取决于优势
   正负）。PPO 官方说明亦明确：裁剪不保证新策略始终接近旧策略（§F
   Engstrom 条目的同一结论）。
3. **价值函数结论反转**（撤回 §G 上条「探针 ev≈0.97 健康」）：探针全程
   expl_var 均值 0.409、尾 50 it 0.964；C2（guard_s0）0–299 it ≈−0.006、
   300–349 = 0.124、350–399 = 0.902、it800+ ≈0.99——critic **早期不
   健康/晚熟**。expl_var 不足不能单独定因退化，但「critic 全程健康」
   不成立。
4. **守卫 v1 三缺陷（代码实锤，本仓逐一核实）**：
   - **参照错**：KL 的「旧」侧 = 本 minibatch loss 计算时的 forward 结果
     （`ppo_core.py:450-466`），即「本次梯度步前后」对比，不是与产生
     rollout 的策略比——连续小步可累计走远而每步过阈；
   - **std 盲**：`forward_actor` 返回的 `phase_log_std` 是参数张量
     `expand_as` 视图（`ppo_core.py:82`；`aux_log_std` :78 同），detach
     不解除别名，`optimizer.step()` 原地更新后「旧」值已变新值——仅方差
     变化时守卫恒报 0（服务器复现：真 KL 0.685，v1 报 0）；
   - **阈值来源不成立**：45.16 = 失稳分布 P95（只描述历史、不构成安全
     阈；均匀位移假设下允许每维均值移动约 1.19σ），且探针 `--iters 300`
     把 expl 衰减调度长度一并压缩（`train_apt_isaac.py:375`
     `max_iters=cli.iters`）= 与 1000 it 正式配方不同配置。
   - 附带：§C.1「rsl_rl 口径」表述不准——rsl_rl v1.0.2 实际用 k1 估计
     E_old[logπ_new−logπ_old]（≈−KL(old‖new) 取幅度），我方是解析
     KL(new‖old)，方向不同；量级对标（desired_kl=0.01 体系）仍成立。
5. **C2（guard_s0）中途读数**（审计时点 it≈400：350–399 rew 0.899/clip
   83.2%；it820 复查：rew 1.15–1.22、d_xy≈0.51、ev 0.99、kl_g 16–66、
   roll 1–3/iter）：守卫在动作但未阻止退化，J1 轨迹注定 FAIL。预注册
   判读：该结果**只能判「v1 守卫不充分」，不能否定正确实现的 KL 约束**。
6. **曲线事实**（区间均值，二次复核与审计逐位一致）：fix2 80–119
   rew 1.586 / d_track_xy 0.736 → 800–999 rew 1.082 / 0.472；E49-B
   800–999 rew 0.941；C2 350–399 rew 0.899——下降伴随实际速度跟踪恶化，
   非总奖励的统计波动。act_std：fix2 0.0184→0.0262（方差近翻倍），
   guard 0.0184→0.0244（it400）。
7. **嫌疑队列更新（本轮冻结口径）**：①PPO 更新长期偏离采样策略（累计
   机制，修正守卫+固定方差对照裁决）；②探索方差单调增长（熵 0.001 +
   expl 0.01 初始合计 0.011 同推一个动作头 + kl_prior 小系数助推）；
   ③critic 早期 expl_var≈0；④奖励设计（冻结待①②裁决后再对照）。
   「不能据此断言 SONIC 能力不足，也不能定成单纯学习率过大」（审计
   原话，与 §A.5 口径合并）。

---

## §B E49-C 预注册协议

### B.1 假设与干预

- **H-C**：给每 minibatch 更新加 KL 信任域（解析高斯 KL 超阈 → 回滚该步
  + 缩 lr）能抑制 fix2 的更新失控与退化-恢复振荡，使行走能力在训练尾段
  得到保持（程度见 J1 分档）。
- 干预 = **仅**新增 `--kl-guard` 及其三参数（§C.1）；对照 = fix2 同配方
  同 seed（已有 run，不重跑）。

### B.2 不变量（全部冻结，违者 run 作废）

奖励函数与权重、log_std 初值 −4.0、熵系数 0.001、expl_coef 0.01 衰减、
latent_kl_coef 2.5e-6、lr 3e-4、epochs 1、minibatch 512、clip 0.2、
γ/λ 0.99/0.95、grad clip 0.5、num_envs 128、seed 0、token-stats npz、
训练器其余全部逻辑。**唯一新自由度 = kl_guard 四参数**（取值见 B.3）。

### B.3 阈值决定程序（C1→C1.5，预注册）

> 【09-06 深夜作废】本程序产出的 thr=45.1597 作废——失稳分布 P95 不构成
> 安全阈 + 探针 `--iters 300` 压缩调度 = 配方不公平（§A′-4）。重定程序
> 见 **§B.3′**；原文保留作历史。

1. C1 探针：fix2 逐字配方 + `--kl-guard 1e9 --diag-log`，300 it，seed 0，
   输出 `outputs/isaac_e49c_probe_s0/`。
2. C1.5 用 §D.2 脚本对 `train_log.json` 的 `kl_mb_all`（逐 minibatch 解析
   KL，KL(new‖old)、64 维求和、batch 均值 = rsl_rl 口径）计算：
   **thr = P95（it≥10 的全体 minibatch 值）**。
3. 合理带检查：文献锚 desired_kl=0.01 / APT 0.008 是他们自己配方下的量
   级，**不构成我方数值约束**；我方 lr/batch/维度不同，thr 落在任何量级
   都可接受。但若 thr < 1e-4（疑似过紧，会扼杀更新）或探针期间 kl_mb 分布
   无离散度（P95≈P50），**停下回 owner**，不自动发射 C2。
4. thr 与 P50/P90/P95/P99/max 一起写入本节下方空栏后**冻结**：
   - thr = `45.1597`（P95）；P50=`13.2371` P90=`38.0389` P99=`55.5393`
     max=`91.9314`，n=1740（C1.5 已回填冻结，2026-09-06；合理带检查通过：
     thr≫1e-4 下界、P95/P50=3.41 离散度充分、C1 全程 roll=0 与 1e9
     探针语义自洽；C1 产物 `outputs/isaac_e49c_probe_s0/`）

### B.3′ 阈值重定程序（v2，2026-09-06 深夜审计后；取代 §B.3）

前置：守卫 v2（§C.5）实现落地并单测通过（含旧代码 FAIL 证据链）。

1. 候选 = 文献量级 **{0.01, 0.03, 0.1}**——新度量 KL(θ_step后‖θ_rollout)
   与 rsl_rl desired_kl（0.01）/APT kl threshold（0.008）同定义体系，
   文献值首次成为可比候选；探针只作候选筛选，**不保证任何常数正确**。
2. 每候选一个 **schedule-fair 短探针**：`--iters 1000 --probe-iters 300`
   （调度长度 = 正式配方，仅提前结束采样），fix2 其余逐字配方 + seed 0。
3. 选阈标准（预注册）：有可测学习进展（rew 轨迹离开站立地板 1.64 上行）
   且回滚未失控（参照 J3(b) ≤30% 量级）；**不接受以「完全不更新」换
   稳定**（lr1e4 教训）。
4. 选定后单 seed 1000 it（C2′）+ 双契约评测，判据沿用 J1/J2/J3（对照仍
   = fix2 同窗）；探针 run 落 tracker 只记数据不定 PASS/FAIL。

### B.4 判据（C2，预注册；对照 = fix2 同窗）

- **J1 尾段行走保持**（主判据，it800–999 窗口均值，diag-log 同窗量）：
  - PASS：rew ≥ 1.70 **且** d_track_xy ≥ 0.75 **且** fall_rate ≤ 0.01
    （参照：fix2 it100 行走峰 rew 1.731/d_xy 0.824；fix2 同窗 1.04–1.20；
    站立地板 1.641/d_xy 0.541）；
  - PARTIAL：1.60 ≤ rew < 1.70 且 d_track_xy ≥ 0.65（退化显著缓解但
    未守住行走）；
  - FAIL：rew < 1.60 或 d_track_xy < 0.65。
- **J2 终点评测**（final = it1000 ckpt；另按预定规则评 best-tail ckpt =
  it800–999 中 rew 最高 50-it 窗中点的 `policy_it_*.pt`，只记录不判）：
  - 行走格定义：`--contract train`（20s）disp ≥ 5 m 且零摔；60s 标准
    eval disp ≥ 15 m 且零摔。
  - PASS：train 契约 det ≥ 4/6 格行走 **且** 60s det ≥ 2/3 行走
    （`--num-rollouts 6` / `3`，seed 0；det 与 sample 各跑一组，sample
    组只记录）。
  - 参照：fix2 final = 纯站立 0.19–0.47 m；E49-B it50 = 23.7–24.3 m。
- **J3 机制判据**（守卫是否按设计生效）：
  - (a) 上界生效：it300–999 内所有 kl_mb > thr 的 minibatch 均被回滚
    （由 kl_rolls>0 与 kl_mb_all>thr 计数对账）；
  - (b) 未扼杀训练：it300–999 回滚步占比 ≤ 30%（>30% → 判「阈值过紧」，
    走 §B.6 分支④；参照 lr1e4 教训=更新被冻死也不行）；
  - (c) 灾难步减少：任意相邻 10-it 窗 rew 降幅 >0.5 的次数 ≤ fix2 同窗
    （fix2 计数由分析脚本给出，写入本节回填栏：fix2 = `7` 次 /
    C2 = `19` 次——反增 2.7×，FAIL【v1 守卫口径下的读数；口径 = i 从
    10 起扫，max(rew[i-10:i]) − rew[i] > 0.5 记 1 次并跳 i+=10 不重叠；
    前两事件 it51/it80 两 run 逐位同（it161 首次回滚前一致），分化始于
    守卫首次干预后；v2 重跑时按同口径重算】）；
  - (d) 观测项（只记录不判）：act_std 轨迹、vloss/expl_var 轨迹、lr_now
    轨迹——A.3 两个嫌疑的直接证据，供下一轮裁决。

### B.5 评测契约

训练日志窗口口径差异照旧（rew 全窗 vs fwd 瞬时，E49_STATUS §3c-③ 收紧
点），判读以 rew + d_* 同窗量为准；eval 全部走 `--seed`（c19bfe6 后语义：
同控初态+RNG+相位）。

### B.6 分支判读（预注册决策树）

1. **J1 PASS + J2 PASS** → H-C 成立，KL 守卫进配方候选；下一轮 = seed 1
   复现 + 同守卫下 E49-B/VAE 参照重跑（归因完整化），归 owner 点火。
2. **J1 PARTIAL** → 守卫有效但不充分；按 PARTIAL 形态分流：仍见慢漂移
   （低 KL 超速）→ 嫌疑队列第 1 位 act_std/熵项升主嫌；KL 已被压但 rew
   停滞 → 先查 J3(b) 是否实际过紧。
3. **J1 FAIL + 慢漂移形态** → H-C 否决为「不充分」；act_std 单变量
   （熵系数/log_std 处置，需 owner 解冻裁决）升首选。本 run 数据仍用于
   下一轮判别，不算白跑。
4. **阈值过紧形态**（rew 全程 ≈地板、kl_mb 钉低位、回滚率 >50%、lr 持续
   缩）→ 判 C1.5 失败，thr 放宽 2× 重跑一次 C2（仅此一次，再紧即停回
   owner）。
5. 任何分支的最终裁决与下一轮点火归 owner。

---

## §C KL 守卫实现规格（E49-C0）

### C.1 语义（冻结）

> 【09-06 深夜】v1 实现三缺陷（步内参照 / std 盲 / Adam 不回滚，§A′-4）
> → v2 规格见 **§C.5**，实现语义以其为准；「rsl_rl 口径」表述修正见
> §A′-4 附带条。

- 判据量 = **解析对角高斯 KL，KL(new‖old)（rsl_rl 口径）**：
  `Σ_d[(ls_old − ls_new) + (var_new + (μ_new−μ_old)²)/(2·var_old) − 0.5]`，
  先对执行动作维求和（token 模式 = 64 维 phase 头；aux 头仅在 aux_scored
  时并入；gate 跳过）、再对当前 minibatch 样本取均值。**不用**日志既有
  akl/pkl（联合 log-ratio 的 k3，量级被 64 维小 σ 放大、首 minibatch 恒
  ≈0，无文献对标值）；akl/pkl 照旧记录保持可比。
- 每 minibatch `optimizer.step()` 前 snapshot 全模型 state_dict；step 后
  测 KL。**KL > thr → load_state_dict 回滚整步 + lr ×0.8**（下限 1e-6）；
  KL < thr/2 → lr ×1.2 回升（上限 = 初始 lr；`--kl-guard-grow 1.0` 关闭）；
  同一 update 内**连续 3 次回滚 → 提前结束本轮更新**。Adam 动量不回滚
  （常规做法）。
- 已知近似（不影响 E49）：decoder_ft 模式 old 分布取 forward 头近似
  （E49 不用该模式）；nan_skip 的 minibatch 不测不拦（分布未变）。
- CLI：`--kl-guard`（默认 None=完全关闭）/ `--kl-guard-shrink 0.8` /
  `--kl-guard-grow 1.2` / `--kl-guard-max-rolls 3`。
- 新日志键：`kl_mb`（iter 内 minibatch 均值）、`kl_mb_all`（逐 minibatch
  列表）、`kl_rolls`（回滚计数）、`lr_now`（仅 guard 开启时）；**无条件**
  新增 `vloss`、`expl_var`（= 1 − Var(ret−V)/Var(ret)，纯日志）。
- **冻结版不变性声明**：`--kl-guard` 未给时零快照零额外前向（除 expl_var
  一处 no_grad 方差计算）、RNG 路径零变化——同 seed 同配方与 fix2 逐位
  同动力学。

### C.2 与官方配方的对应

本守卫 = rsl_rl `schedule="adaptive"`（desired_kl=0.01，per-minibatch
KL→lr 负反馈）**加强版**：在其「超阈降 lr」之上加「回滚当步」（TRPO
回溯线搜索的接受条件语义；TRPO 原文：无回溯线搜索时算法偶尔算出灾难性
大步——正是 it680 型 spike 的对策）。APT 原文 Table S4 亦有 kl threshold
0.008（§F），即被复现管线本身带 KL 稳定器，我方此前为唯一缺失项。

### C.3 新增观测（纯日志，默认开启）

vloss 落 hist、expl_var 落 hist/控制台——补 A.3-2 的「价值健康度不可
观测」缺口；act_std 原有。

### C.4 实现与验证状态

- [x] ppo_core.py 守卫逻辑 + expl_var（commit d9ce851；diff 零删除行）
- [x] train_apt_isaac.py CLI/hist/回显（同上）
- [x] e49_kl_guard_test.py 单测——服务器 **6/6 PASS**（2026-09-06；
      另旧 e49_gae_test 复跑 **10/10 零回归**；部署 = sync 克隆
      0d3500f→执行根，sha256 三对逐位一致：ppo_core 552849c1…、train
      d6682991…、test 529972ff…→case6 判据加固 594c11d 后复跑 6/6）。
      case6(b) 首版判据把「期望 ≤0」当「单次样本 ≤0」，+0.0049 落
      ±1/√N 噪声带 FAIL——实现无嫌疑，判据改大方差独立 value
      （断言 < −1.0，实测 −23.79）。
- [x] SCRIPT_MAP 登记（DEV 角色，实验号 E49-C）
- 发射门已开：C1 探针可直接按 §D.1 发射。

### C.5 守卫 v2 规格（2026-09-06 深夜审计后；已实现并单测通过）

修正 §C.1 v1 三缺陷，语义变更点：

- **旧分布 = rollout 采样时存储**的 phase/aux 均值与 log_std（存储侧
  `detach().clone()`，杜绝 `expand_as` 别名）；update 内固定作参照，每个
  候选步测 KL(θ_step后 ‖ θ_rollout)——**天然含累计效应**，并覆盖首
  minibatch ratio≡1 无裁剪的洞（A.2）。
- **拒绝步恢复 policy 参数与 Adam 状态**（state_dict 双快照，替换 v1
  「动量不回滚」口径）。
- 不变：`kl_diag_gaussian` 解析式（KL(new‖old)、维求和、batch 均值）、
  lr shrink/grow、max_rolls 提前结束、nan_skip 不测不拦、日志键
  （kl_mb/kl_mb_all/kl_rolls/lr_now）、`--kl-guard` 未给时零快照零额外
  前向零 RNG 变化。
- 新增 CLI `--probe-iters N`：训练提前结束但调度 max_iters 仍按
  `--iters`（探针配方公平性，§B.3′-2）。
- 单测（e49_kl_guard_test.py）：旧 6 case 适配新参照口径（逐 case 注明
  改动理由）+ 新 3 case——**case7 方差盲复现**（仅 log_std 变化，v1 必须
  FAIL / v2 PASS）、**case8 累计超阈**（单步小、相对 rollout 累计超阈，
  第 1 步过后续步拒）、**case9 拒绝步全恢复**（参数 + Adam 状态逐张量
  相等）；另 e49_gae_test 10/10 零回归。服务器隔离目录
  （`~/ros2_data/tmp_guardtest`）验证，**不碰执行根**。
- **验证结果（2026-09-06 深夜）**：隔离目录两轮——旧代码 6/9（case7
  kl_mb=0.0000 vs 解析 0.6275 = 方差盲直接复现 / case8 kl 全 0 / case9
  Adam 动量残留，三 FAIL）→ 新代码 **9/9 PASS**（case7 偏差 <5%、case8
  kl 序列 0.043→0.184×5 单调、case9 48 张量逐位相等）+ **gae 10/10 零
  回归**；case4 kl_mb_all 由旧口径递减 [1.47,0.82,0.48,0.21] 变递增
  [1.47,2.57,4.98,5.19] = 参照切换直接证据。部署走 git+sync（待 C2
  评测跑完后执行根换装 + sha256 三对账）。

---

## §D 执行手册（逐字命令）

> 路径约定（已按服务器实测核对，2026-09-06）：包装脚本 `/tmp/run_apt_isaac.sh`
> = source `.venv_isaac` + `PYTHONPATH=~/ros2_data:~/ros2_data/apt_g1:~/ros2_data/
> GR00T-WholeBodyControl` + **cwd 切到 `~/ros2_data/GR00T-WholeBodyControl`** 后
> `exec python "$@"` → **脚本路径必须用绝对路径**（`~/ros2_data/apt_g1/isaac/
> ...`；仓库 apt_g1 不在 cwd 之下）；`--out` 相对路径落在
> `GR00T-WholeBodyControl/outputs/`（= 历史训练产物位置）；eval JSON 历史上落
> `~/ros2_data/apt_g1/outputs/e49/`；token-stats =
> `~/ros2_data/apt_g1/outputs/e49/token_stats_e49.npz`。
> 发射前核对：历史日志不回显 argv（多路 grep 证实），本轮起新增 `[CFG]`
> 配置回显行——发射后 `head` 日志核对该行与本节命令一致即可。

### D.1 C1 探针（~6 min）

```bash
ssh lab-ts
cd ~/ros2_data/GR00T-WholeBodyControl
nohup bash /tmp/run_apt_isaac.sh ~/ros2_data/apt_g1/isaac/train_apt_isaac.py \
  --num-envs 128 --iters 300 --env apt --token-mode \
  --token-stats ~/ros2_data/apt_g1/outputs/e49/token_stats_e49.npz \
  --lr 3e-4 --ppo-epochs 1 --seed 0 --diag-log --kl-guard 1e9 \
  --out outputs/isaac_e49c_probe_s0 \
  > outputs/isaac_e49c_probe_s0_train.log 2>&1 < /dev/null & disown
# ssh 可能等 20–30s 超时，进程其实已起；第二条 ssh 验证：
# tail -3 outputs/isaac_e49c_probe_s0_train.log
```

### D.2 C1.5 定阈脚本（探针完成后）

```bash
cd ~/ros2_data/GR00T-WholeBodyControl
python - <<'EOF'
import json, numpy as np
h = json.load(open("outputs/isaac_e49c_probe_s0/train_log.json"))
flat, its = [], []
for i, xs in enumerate(h["kl_mb_all"]):
    flat += list(xs); its += [i] * len(xs)
flat, its = np.array(flat), np.array(its)
sel = flat[its >= 10]
for name, v in [("P50",50),("P90",90),("P95",95),("P99",99)]:
    print(name, f"{np.percentile(sel, v):.6g}")
print("max", f"{sel.max():.6g}", " n=", len(sel))
# thr = P95；连同分位数回填 §B.3 第 4 步并冻结
EOF
```

合理带检查（§B.3-3）通过后，回填 §B.3 并进入 C2。

### D.3 C2 单 seed（~20 min + eval）

```bash
# 训练（<thr> 换成 §B.3 冻结值）
nohup bash /tmp/run_apt_isaac.sh ~/ros2_data/apt_g1/isaac/train_apt_isaac.py \
  --num-envs 128 --iters 1000 --env apt --token-mode \
  --token-stats ~/ros2_data/apt_g1/outputs/e49/token_stats_e49.npz \
  --lr 3e-4 --ppo-epochs 1 --seed 0 --diag-log --kl-guard <thr> \
  --out outputs/isaac_e49c_guard_s0 \
  > outputs/isaac_e49c_guard_s0_train.log 2>&1 < /dev/null & disown
```

监控（60 it 内中止规则 + 全程）：`tail` 训练日志看
`rew/akl/pkl/clip/std/vloss/ev/kl_g/roll/lr` 新列；退化段（it300+）重点看
kl_g 是否贴 thr 上界、roll 是否集中在早期。

### D.4 C2 评测（训练完成后）

```bash
# final（it1000）双契约四组（命令形态 = 服务器 e49_run_a.sh eval 逐字模板
# + 本轮新旗标；ckpt 在 GR00T-WholeBodyControl/outputs/，JSON 落 apt_g1/outputs/e49/）：
bash /tmp/run_apt_isaac.sh ~/ros2_data/apt_g1/isaac/eval_apt_isaac.py \
  --tests A --token-mode \
  --token-stats ~/ros2_data/apt_g1/outputs/e49/token_stats_e49.npz \
  --checkpoint ~/ros2_data/GR00T-WholeBodyControl/outputs/isaac_e49c_guard_s0/policy_it_1000.pt \
  --contract train --num-rollouts 6 --seed 0 \
  --out ~/ros2_data/apt_g1/outputs/e49/eval_e49c_guard_s0_final_train_det.json
# 再分别跑：--sample 组 / 去掉 --contract train 且 --num-rollouts 3 的 60s det 组 / 60s sample 组
# best-tail ckpt（按 §B.4 预定规则从 train_log.json 选 policy_it_*.pt）同四组，只记录
```

（e49b eval JSON 内仅记录 `mode/contract/seed` 等参数无完整 argv；上方命令以
e49_run_a.sh 的 eval 逐字模板为基、旗标语义均经代码核验——若与实跑行为有出入，
以实际 JSON/log 回显为准并回写本节。）

### D.5 判读与落账

按 §B.4 计 J1/J2/J3 → §B.6 分支 → Run 行落 `tracker/E.md`（E49-C1-probe /
E49-C2-guard-s0）+ 结论 8 + `EXPERIMENT_TRACKER.md` 行数同步 → owner 裁决
分支走向。

### D.6 v2 流程逐字命令（09-06 深夜审计后；前置 = v2 三件套已部署执行根 + sha256 对账）

```bash
# P 探针 ×3（thr ∈ 0.01 / 0.03 / 0.1；dir 名 t001/t003/t010 相应替换）
ssh lab-ts
cd ~/ros2_data/GR00T-WholeBodyControl
nohup bash /tmp/run_apt_isaac.sh ~/ros2_data/apt_g1/isaac/train_apt_isaac.py \
  --num-envs 128 --iters 1000 --probe-iters 300 --env apt --token-mode \
  --token-stats ~/ros2_data/apt_g1/outputs/e49/token_stats_e49.npz \
  --lr 3e-4 --ppo-epochs 1 --seed 0 --diag-log --kl-guard 0.01 \
  --out outputs/isaac_e49c_p2_t001_s0 \
  > outputs/isaac_e49c_p2_t001_s0_train.log 2>&1 < /dev/null & disown
# 三个探针串行（3060 12G 不并行多 Isaac 实例）；每个 ~8 min
# 发射后核对日志头 [CFG] 行：iters=1000 probe_iters=300 kl_guard=<thr>

# 定阈判读（三探针齐后）：rew 是否离开站立地板 1.64 上行（对照 fix2
# 80–119 均值 1.586 / 峰 1.731@it100）+ 回滚率（kl_rolls 累计 / minibatch
# 总数）+ kl_mb 分布；按 §B.3′-3 选最紧且可学习者

# C2′（<thr> 换选定值；与 D.3 同形，无 --probe-iters）
nohup bash /tmp/run_apt_isaac.sh ~/ros2_data/apt_g1/isaac/train_apt_isaac.py \
  --num-envs 128 --iters 1000 --env apt --token-mode \
  --token-stats ~/ros2_data/apt_g1/outputs/e49/token_stats_e49.npz \
  --lr 3e-4 --ppo-epochs 1 --seed 0 --diag-log --kl-guard <thr> \
  --out outputs/isaac_e49c2p_guard_s0 \
  > outputs/isaac_e49c2p_guard_s0_train.log 2>&1 < /dev/null & disown
# 评测 = D.4 命令把 isaac_e49c_guard_s0 换成 isaac_e49c2p_guard_s0
```

---

## §E 数据与落账规矩

- 产物：`outputs/isaac_e49c_probe_s0/`、`outputs/isaac_e49c_guard_s0/`
  （ckpt 每 50 it + train_log.json）、eval JSON 入 `outputs/e49/`。
- Run 行：E49-C1（探针，Status=DONE，Result 记阈值分位数表）、E49-C2
  （判据 J1/J2/J3 读数 + 分支结论）。VAE 参照维持「ABORTED@it950 非完成
  态」表述（owner 09-06 钉死）。
- 本文档改动（协议冻结后）一律进 §G 修订记录。

## §F 外部锚点（2026-09-06 核至源码/论文原文）

| 锚点 | 内容 | 用途 |
|---|---|---|
| rsl_rl PPO（master/v1.0.2） | per-minibatch KL→lr：>2×desired_kl → lr/1.5、<÷2 → lr×1.5，界 [1e-5,1e-2]；解析对角高斯 KL、维求和 batch 均值；desired_kl=0.01 | 守卫口径与量级的官方参照 |
| Isaac Lab velocity 任务（a1 等全族） | `schedule="adaptive", desired_kl=0.01, clip 0.2, lr 1e-3` 显式开启 | 「官方配方自带 KL 稳定器」证据 |
| APT（Science Robotics adz7397 / arXiv 2607.13579，KAIST Hae-Won Park 组，无开源） | Table S4：**kl threshold 0.008**、lr 3e-4、clip 0.2、熵 0.001、latent KL 2.5e-6、horizon 50、batch 204800/40960、~50k iter、动作 29 维（12 aux+16 latent+gait logit） | 被复现管线本身带 KL 稳定器 |
| Engstrom et al. 2020（arXiv 2005.12729） | 裁剪不构成信任域："can end up moving arbitrarily far from the trust region"；PPO-NoClip 不掉点 | A.2 机理背书 |
| TRPO（arXiv 1502.05477）+ Spinning Up | 回溯线搜索：KL 约束不满足即拒绝、β 指数收缩（典型 0.8×10 次）；原文「无回溯则偶发灾难性大步」 | 回滚语义合法出处 |
| PPO 调参实践（37 details 等） | clip_frac 健康带 ~0.1–0.2 | 我方 0.833/0.42–0.55 显著饱和的参照 |
| Tang & Agrawal AAAI 2020 等 | 连续动作高斯 PPO 本身不稳，维度越高越甚 | 64 维直出 token 的背景 |

## §G 修订记录

- 2026-09-06：定稿（§A 调研 + §B 预注册 + §C 规格 + §D 手册）；C0 实现
  完成、单测服务器 **6/6 PASS**（旧 e49_gae_test 10/10 零回归）、部署
  sha256 双端一致；§B.3 阈值栏待 C1.5 回填，属正常状态。
- 2026-09-06（补）：case6(b) 判据加固（594c11d，详见 §C.4）；§D 命令路径
  按服务器实测核对修正（脚本须绝对路径，eval JSON 落 apt_g1/outputs/e49/）。
- 2026-09-06（C1/C1.5 执行）：C1 探针 300 it 完成（[CFG] 回显核对一致，
  全程 roll=0 符合 1e9 只测不拦语义）；C1.5 定阈 thr=P95=45.1597 回填 §B.3
  并冻结，合理带检查通过（P95/P50=3.41）；C2 以 `--kl-guard 45.1597`
  发射（outputs/isaac_e49c_guard_s0）。附带观测：探针全程 ev(expl_var)
  ≈0.97——价值函数在 300 it 内健康，A.3-2 嫌疑的首次实测证据（只记录
  不判，待 C2 全程轨迹）。【本条 ev 结论已被 09-06 深夜审计撤回，见
  §A′-3。】
- 2026-09-06（深夜·审计修正轮）：owner 侧独立审计落地、本仓二次复核全
  通过——§A.1 时序否决降级（不能排除持续累积偏离机制）、§A.2 措辞修正
  （clip 未提供足够约束 ≠ 完全失效）、§A.3-2/上条「ev≈0.97 critic
  健康」撤回（探针均值 0.409、C2 前 300 it ≈0 = 早期不健康/晚熟）、
  守卫 v1 三缺陷（步内参照 `ppo_core.py:450-466` / log_std `expand_as`
  别义 std 盲 `:82`（仅方差变时真 KL 0.685 报 0）/ thr 来源不成立）、
  探针配方不公平实锤（`train_apt_isaac.py:375` max_iters=cli.iters）→
  **thr=45.1597 作废**；新增 §A′（审计修正全文）/§B.3′（v2 重定阈：
  文献量级 {0.01,0.03,0.1} schedule-fair 探针）/§C.5（守卫 v2 规格，
  rollout 参照 + 拒绝步恢复参数与 Adam + `--probe-iters`）。C2（guard_s0）
  跑至 it820 rew≈1.15、d_xy≈0.51——J1 注定 FAIL，按协议跑完取数（判读
  = 「v1 守卫不充分」，非「KL 约束无用」）。嫌疑队列重排：累计更新
  偏离 > 探索方差单调涨 > critic 早期 ev≈0 > 奖励设计（冻结待裁决）。
- 2026-09-06（C2 完成收账）：C2 训练+8 组评测全部完成（final 四组
  24/24 rollout 100% 摔、disp 0.001 原地踏频；btail=it800 四组 36
  rollout 零摔但纯站漂移 60s disp 0.36–1.3 m）；J1 双 FAIL（0.9798
  vs fix2 1.0818）、J3(a) 313=313（按 v1 判据量）、J3(b) 7.45%、
  J3(c) 19 vs 7 反增——判读与 Run 行落 tracker/E.md（E49-C1-probe /
  E49-C2-guard-s0 两行 + 结论 8，EXPERIMENT_TRACKER 104→106/247→249
  同步）。§B.4(c) 回填 fix2=7 / C2=19（v1 口径）。§D.6 前置的「C2
  评测跑完」条件已满足 → 下一步 = v2 三件套部署执行根（sha256 对账）
  → P 探针 ×3 发射。
