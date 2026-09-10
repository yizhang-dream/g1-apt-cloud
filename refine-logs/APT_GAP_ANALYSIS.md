# APT 论文差距分析 + decoder 性能画像卡（持续维护）

【层位】L1 路线图/机制对照 | 上级：[refine-logs 扇出树](README.md) · 相关：[DS_CONTINUOUS_EXECUTION_PLAN](DS_CONTINUOUS_EXECUTION_PLAN.md)（D048 系列 §5h） · 结论口径：[HANDOFF/README](../HANDOFF/README.md) §3

> 状态：ACTIVE（2026-09-10 owner 授权「自行决定下一步，逐渐补齐与 APT 论文之间的
> 差距，并完全吃透 decoder 性能」后建立）。本文档两个作用：①论文机制 vs 我方实现
> 的**结构差距台账**（补齐进度追踪）；②冻结 SONIC 解码器 + token VAE 的**性能画像卡**
> （证据索引 + 待补空白）。
>
> **口径纪律**：本文档只做机制/结构对照，不做结果可比性论断（论文=四足力矩解码器
> +TO 数据；我方=人形关节目标解码器+SONIC token——两栈结构性不可比，混线综合
> 对照框架 2026-09-03 已被 owner 降级勿复用）。论文全文
> `tmp/pdfs/paper.txt`（Methods 引行号为该文件行号）。

## 1. 结构差距台账

| # | 论文机制（Methods 锚点） | 我方现状 | 差距性质 | 状态 |
|---|---|---|---|---|
| G-1 | **gait 选择权归策略**：actor 输出 logit_gait∈ℝ¹（动作空间 ℝ²⁹=latent16+aux12+logit1），2Hz 决策 + 0.5s 锁定，σ(logit)<0.5→trot 解码器否则 bound（paper.txt:1171-1188, 1248） | vb 三档由**环境**从命令等宽分桶（apt_flat_env.py `_compute_q_des` bucketize），策略无选择权 | **可补、当前最重**：D048l 已证命令→速度效应完全由 vbin 中介且命令→档位映射错码（b1→0.61/b2→0.45 反向）——策略自主选档可绕开该映射环节，且与论文结构对齐 | D049 草案见 §3 |
| G-2 | 命令空间：v_x∈[−1,7] m/s、v_y∈[−1,1]、heading±1.4rad（yaw rate 跟踪）；>2m/s 时 v_y=0、heading±0.05（paper.txt:1240-1246 附近） | v_x∈[0,0.8]（apt_flat_env.py:284 vx_max）、无 v_y、heading 项（--heading-scale 默认 0，D048b 双旋转 bug 修复后默认关） | 部分可补：量程受语料上限约束（v2.1 材料 1.694 m/s、p90 1.30）；v_y/宽 heading 属阶段2+ | OPEN |
| G-3 | 速度控制经连续 latent（ℝ16）+ 离散 gait 双层 | vb 离散三档（nn.Embedding 查表）吸收全部命令速度效应（D048l：固定档内换命令 vx 差≤0.02） | 待测：vb 软插值（嵌入线性混合）闭环行为=连续可调 or 锁死吸引子 → D048q-Q1 | 本轮补测 |
| G-4 | 混合控制 τ_input=τ_dec+kp(q_default−q)+kd(a_scale·a_aux−q̇)，kp=80/kd=2/a_scale=0.2（aux=**速度目标**、位置反馈锚 default；paper.txt:1189-1200） | q_des=q_dec+res_scale·a_aux→PD（aux=**位置残差**，res_scale 0.15；isaac PD 内置） | **结构性不可比**：我方解码器输出关节目标（29 维 q_des）非力矩，τ_ff 形式无逐字移植路径；登记为设计差异，不作为可补差距 | CLOSED(as-design) |
| G-5 | latent KL 正则系数 2.5e-6（paper.txt:1228-1240） | ppo_core.py:228 `latent_kl_coef=2.5e-6` **逐字一致**；另有 E49/D048i 步级 KL 看门（0.05 夹紧）为我方增量机制，论文无对应 | 已对齐 ✓ | CLOSED |
| G-6 | 感知蒸馏：特权 elevation map teacher → depth+LIDAR student（paper.txt:1249+） | 无（平地命令跟踪主线；地形线为第二阶段范围） | 阶段边界，暂不补 | DEFERRED |
| G-7 | 2Hz 门控（latent/aux 100Hz、gait 2Hz） | eval_apt_isaac.py `--use-2hz-gate` 默认 1（开） | 已对齐 ✓ | CLOSED |
| G-8 | 双 gait 解码器（trot/bound 各自训练，共享状态编码器） | 单解码器 + vb 三档条件（嵌入通道） | 结构不同但「离散步态槽位」同构（历史判读口径）；vb 槽位≈gait 槽位。换双解码器=动冻结解码器，超表示层对照范围 | CLOSED(as-design) |

**补齐路线判断**（2026-09-10）：G-1 是当前唯一「既有直接实验证据支撑可补且补了
就动主矛盾」的差距（D048l/D048m/D048n 证据链：vb 轴语义错码=表示层本征 → 速度
锚定标签可恢复单调但量程失手）。G-3 是 G-1 的前置测量（若 vb 连续可调，命令跟踪
可走软条件路线绕开离散选档；若锁死吸引子，G-1 的策略选档路径权重上升）。G-2 量程
部分依赖 D048p 反标定结果。

## 2. decoder 性能画像卡（冻结 SONIC 解码器 + token VAE 接口层）

> 「decoder」= 冻结 SONIC 解码器（64-d FSQ token→29 维关节目标）+ 其上游可训
> 接口（token VAE：z16+phase2+speed_embed3×8+dir_embed8×8）。合作者实测无 VAE
> 直接 RL 可达几 m/s=解码器本体非瓶颈（2026-09-08 owner 结论）；我方瓶颈画像
> 集中在**条件接口层语义**。

| 维度 | 证据 | 结论 | 状态 |
|---|---|---|---|
| 速度-三档吸引子（canonical e39） | D048l force-vbin 电池 + D048m b0 补测 + init 本征图 | {b0 0.16, b1 0.57-0.61, b2 0.43-0.45} 非单调；命令近零效应（固定档内换命令 vx 差≤0.02）；init 与训练图同形=VAE+解码器本征 | 已闭合 |
| 速度-标签语义修复 | D048n G2（speedA 速度锚定标签重训 VAE，架构冻结） | init 档位图 {0.29,0.47,1.31} 严格单调、命令无关、种子稳 → 速度标签→冻结解码器因果链闭合；语料效应并行发现（B4-lite ctrlB 弱单调） | 已闭合 |
| 速度-量程 | D048n G2 | speedA b2=1.31>0.8（edges=语料 WALK 三分位 [0.42,0.98] 非命令带反标定）=D048n 唯一失手项 | **D048p 补** |
| 速度-连续可调性 | 无（未测） | vb=nn.Embedding 查表，软混合（α·E[ba]+(1−α)·E[bb]）接口可行但闭环行为未知 | **D048q-Q1 补** |
| 方向轴 | D046b（token 层 vb/db 读出不可判读）+ D048m（方向=z 彩票：init 种子不同 drift 0.9/6.3m）；db 八档闭环语义**从未测** | 方向不可控证据充分，但 db 条件通道闭环是否有方位角语义=空白 | **D048q-Q2 补** |
| 速度-方向耦合 | D048l ③ + D048m | b2 全模型横漂+航向败、b1 部分过门=速度-方向反相耦合系表示层本征（init 同构） | 已闭合（机制归因开放） |
| b1/b2 token 退化 | D048m 探针（rate(b1)=rate(b2)，vb2-vs-vb1 +0.1%） | canonical 标签下 b1/b2 节拍维退化；speedA 使条件轴变强（token 位移 2.26×） | 已闭合 |
| 解码器本体上限 | 合作者无 VAE 直接 RL 几 m/s（09-08）；v2.1 语料速度上限 1.694（p90 1.30） | 解码器本体非瓶颈、VAE 接口是；语料侧速度上限 ~1.7 | 已闭合（外部证据） |

## 3. D049 草案（G-1 闭合路径：vb 选择权归策略，本轮不执行）

G-1 的闭合实验（策略自主 gait 选择，论文同构 2Hz+0.5s 锁）：
- 臂 a（表示层单变量）：D048p 量程校正 VAE + 环境命令自然档——检验「标签+量程
  修复是否恢复命令跟踪」（=D048n G3 原设计，判读三分支沿用 §5g）。
- 臂 b（选择权单变量）：同 VAE + 策略输出 vb logits（3 档 softmax 或 argmax），
  2Hz 更新 0.5s 锁存（对齐论文 gait 选择器节律），奖励不直接惩罚档位、只看速度
  跟踪+存活+航向。b−a 差=选择机制单变量。
- 判读：联合门（vx_rmse/lat/yaw/task_success）+ 档位使用分析（策略是否收敛到
  单调速度映射）。若 b 过门而 a 不过 → G-1 差距闭合且「映射错码由选择权归属修复」
  获证；均不过 → 表示层候选收窄至连续条件/语料量程。
- 预算：2×200it（D048i 配方逐字）。正式预注册待 D048p/D048q 结果后落
  DS_CONTINUOUS_EXECUTION_PLAN 新节（§5i），点火时点按本轮 owner 授权裁量。

## 4. 更新日志

- 2026-09-10：建卡。差距台账 8 项（CLOSED 4 / OPEN 2 / 本轮补测 2）；画像卡 8 维
  （已闭合 6 / 本轮补 3——量程、连续性、db 方向轴）。依据：论文 Methods 精读 +
  D046b/D048l/D048m/D048n 证据链 + ppo_core/eval 接口核验（KL 系数、2Hz 门控
  对齐项确认）。
