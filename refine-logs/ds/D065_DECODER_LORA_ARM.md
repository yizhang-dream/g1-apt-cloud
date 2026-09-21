# D065 预注册：冻结 SONIC 解码器上的 rank-16 零初始化 LoRA 对照臂——「接口代价能否被低秩适配消掉」

> 【层位 L3 执行计划/预注册（2026-09-21 立项；owner 09-21 拍板执行 ORCS/ViBe 调研报告建议 A）】↑
> 上级纲领：[DS_TERRAIN_ADAPTER_CHARTER](DS_TERRAIN_ADAPTER_CHARTER.md)（§6 冻结/解冻裁决规则与四分支；冲突条款以纲领为准）｜
> 同线计划：[DS_TERRAIN_AUTHOR_PLAN](DS_TERRAIN_AUTHOR_PLAN.md)（作者阶梯 D061/D062，两线并行互不冲突）｜
> 调研依据：[ORCS_SURVEY](../surveys/ORCS_SURVEY.md)（§4 建议 A 与全部技术细节）｜
> 前置实验：[tracker/D.md](../tracker/D.md) D064 行（接口代价的受控证据）｜Run 事实源：[tracker/D.md](../tracker/D.md)｜
> 判读口径：[HANDOFF/README](../../HANDOFF/README.md) §3｜状态：**活跃（预注册）**

## 0. 三十秒版

- D064 已判定「冻结 decoder 接口 = 机制性瓶颈」（rough_paper 0.08 存活 A direct 直出 0.75/0.79 vs B token→冻结 decoder 0.167/0.083，差 4.5–9.5×）。
- 本实验问：这个接口代价**能不能用低秩适配消掉**——在冻结 decoder 的全部 7 个 Linear 上挂 rank-16 零初始化 LoRA、只训 adapter，与 D064 的 A/B 两臂同台构成三臂单变量对照（**09-21 发射前结构修正：C 臂改 in-graph 形态，对照结构见 §3a**）。
- 先过 G0 对齐冒烟门（**零训练**）：ORCS Dodge ckpt 的 `decoder.base.*` 与我们的 decoder 逐层 maxdiff==0（关节序置换复原后）、坐实 adapter rank/alpha、21 维 augmentation 流前向跑通输出 29 维有限值；任一失败不上 G1，D065 转记录性收束。
- 主判据：C 臂 rough_paper 0.08 存活 ≥ A−25%（≈把 A vs B 的 4.5–9.5× 差距收回 ≤1.33×）；辅判据：det 推理 actual_vx 非近静止、flat vx 跟踪有效（评测命令通道先修复）。
- 结果**不作主假设证据**，按纲领 §6「受限低层适配」对照路线与分支 2 裁决；D062 作者 v1 300M 照走不误，两线互为对照。

## 1. 定位与授权链

1. **科学问题**：D064 用同资产/同官方配方/同 envs 同 iters 的单变量对照，把「冻结 decoder 接口」的代价钉在 rough 0.08 存活率上（A 0.75/0.79 vs B 0.167/0.083，差 4.5–9.5×；行内限定语四条见 §7）。本实验 = D064「接口改造」候选的具体化：**接口代价是「离散 token 是唯一入口」造成的，还是「基座权重不能动」造成的？** LoRA 回答后者——它把连续修正注入 decoder 内部、同时保持基座冻结。
2. **纲领内定位**：这正是 [DS_TERRAIN_ADAPTER_CHARTER](DS_TERRAIN_ADAPTER_CHARTER.md) §6 三条对照路线中的「**相同上层 + 受限低层适配**」（检验修改少量低层参数是否带来关键收益，同时测原技能退化），也是「冻结/解冻裁决」四分支的判据来源。ViBe/ORCS 是这条路线在**与我们字节级同一份 SONIC decoder** 上的外部同构实例（[ORCS_SURVEY](../surveys/ORCS_SURVEY.md) §1.2/§2）。
3. **文献先例**：我方文献综述早已把「潜空间瓶颈 / LoRA / SPAR 整流受控改」列为解法③（[LITERATURE_SURVEY_FROZEN_DECODER](../surveys/LITERATURE_SURVEY_FROZEN_DECODER.md) :21/:49–51），本期由 ORCS/ViBe 补上外部存在性证明；而 E44 的全权重微调负结果（任何程度微调都把直行破坏成打转，见 [tracker/E.md](../tracker/E.md) E44 行）与本臂「低秩 + 零初始化 + 只训 adapter」构成「改多少算多」的受控对照。
4. **证据资格（口径纪律）**：本臂**违反主假设「冻结 decoder 一权不动」**，结果**不得作为主假设证据**；无论正负一律按纲领 §6 落账，正结果走「允许转向适配」的授权路径（不得写成「冻结路线被推翻」）。

## 2. G0 对齐冒烟门（零训练，lab-ts）

目的：把 ORCS_SURVEY §2 的「权重对齐成立」从文档结论变成端到端实测，并坐实发布 ckpt 的实际超参。**三小项全过才允许发射 G1；任一失败 → G1 不发射，D065 转记录性收束（结论限定为「公开件在本仓环境下无法对齐」，不延伸为机制结论）。**

- **(a) 权重逐层比对**：取 ORCS Dodge 发布 ckpt（HF `lkrajan/orcs` `Orcs-Dodge-AdaptSonic`，revision `v0.1.0`）的 `decoder.base.*`，与我们 `apt_g1/isaac/sonic_decoder_torch.py` 的 7 个 Linear 逐层比对：**中间 5 层（层 1–5）直等；首层经 IsaacLab↔MuJoCo 关节序列置换后相等；末层经行置换后相等；全部 maxdiff==0**；**token 列（输入前 64 维）在置换下恒等**（置换只作用于 930 维本体历史段）。置换自身须被打印落盘，作为后续接线（若走「保留我方 decoder + 按同一置换喂/逆置换解释」方案）的唯一权威定义。
- **(b) 超参坐实**：打印 ckpt 内的 metadata/config（run metadata、adapter 张量形状），坐实 adapter 的 **rank（预期 rank=16）与 alpha/scale 语义**——HF 侧 `provenance.json` 不含这两项，ckpt 与源码是唯一公开途径；若 alpha 确实不进 ckpt（Python 侧属性），按 `adapters.*.down/up` 形状 + 源码默认（rank=16, alpha=1.0, scale=alpha/rank=1/16）记录，并注明「alpha 系源码+形状推断、非 ckpt 实测」。
- **(c) 前向跑通**：按 Dodge 任务构造 augmentation 流（**21 维**，adapter 首层输入宽度必须精确对上，维度不符即 shape error——这是第一道可检查判据），连同 token 输入跑通 adapter + base 前向，**输出 29 维、有限值（无 NaN/Inf）**。Dodge 是四任务中唯一可纯离线复现的（参考动作=恒定站立，只需自造球轨迹）。
- **产物**：比对脚本、置换表、metadata 打印、前向数值快照，统一落服务器 `outputs/d065/`（脚本默认值）；结论行（可 grep 的 PASS/FAIL）落 tracker，同步件入 `apt_g1/outputs/sync/d065/`。

## 3. G1 LoRA 对照臂（cvgl，4096 envs）

（**本节 = v1 原形态**；09-21 发射前结构修正见 **§3a**，冲突以 §3a 为准——判据不变。）

**三臂同台、单变量**（评测口径与 D064 逐条一致，60s=3000 步）：

| 臂 | 动作通路 | 来源 |
|---|---|---|
| A | direct29：29 维关节目标直出 | **复用 D064**（不重训；ckpt 与评测在案） |
| B | 冻结 decoder：64 维 FSQ token → mean+alpha·std·a 仿射 → 冻结 decoder → 29 关节目标 | **复用 D064**（不重训） |
| C | **B + rank-16 零初始化 LoRA**：decoder 全部 7 个 Linear 各挂 adapter | 本实验新训 |

**C 臂规格（最小单变量版本）**：

- LoRA 挂 decoder **全部 7 个 Linear**（不是只挂输出层）；rank=16、alpha=1.0（scale=1/16，LoRA 惯例）；`down` 投影随机初始化、**`up` 投影零初始化** ⇒ 构造时 C 的策略与 B **逐比特一致**；
- **只训 adapter**：base（decoder）与其余全部网络参数冻结，只有 adapter 参数进优化器；
- **无感知条件化、无 augmentation 流**：剥掉 ViBe 的视觉提取器与首层条件口（本臂回答的是「低秩通道本身能不能消掉接口代价」，不是 ViBe 复现）；
- rank=16 远小于各层 `min(in,out)`（最小者末层 512→29 的 29），**不触发 ORCS 记录的「rank ≥ min(in,out) 静默退化为全秩 delta、scale=alpha」分支**；
- 训练后须断言 **base 权重与初始逐位一致（maxdiff==0）**（冻结范围的可检查化），decoder ONNX md5 入身份信封（锚 D064 在案值）。

**训练配方与 D064 完全一致**：官方 G1 rough manager-based 任务 + rsl_rl；官方课程 plane→rough；**5000 iters、4096 envs、lr 1e-3 + adaptive schedule（desired_kl 0.01）、clip 0.2、24 steps/env**；奖励/终止/事件全继承官方（不自创）；asymmetric critic；A/B 臂数据复用不作改动。C 臂若遇 4096 envs 类崩溃（D064 direct_s0 死点先例：envs 数绑定），按同一先例降 envs 并**在行内登记偏差**，不静默换配方。

## 3a. 修正记录（09-21，发射前）

**修正原因（结构性，非调参）**：G1 训练臂实现时发现，「C = B + LoRA、唯一变量」的**字面形态不可实现**——B 臂架构里 decoder 挂在 env 侧 `SonicActionTerm` 内，且整段解码在 `torch.no_grad()` 下运行（`apt_g1/isaac/sonic_action_term.py:285-291`：`with torch.no_grad():` @285 … `self.q_des = q_des.detach()` @290），decoder 位于策略计算图之外。后果：**PPO 梯度永远到不了挂在 decoder 内的 LoRA adapter ⇒ adapter 零梯度 ⇒ C 与 B 输出逐位相同（假阴性风险）**；按字面形态发射只会得到「LoRA 消不掉接口代价」的伪结论。owner 09-21 已授权按 ORCS（in-graph）形态重构。

**修正后 C 臂形态（G1 代码 v2）**：**A 臂架构（29 维关节空间动作、策略 std 冻结）+ μ 路径经 decoder（decoder 挂 rank-16 零初始化 LoRA）** = **E44 架构的低秩版**。env 侧动作/奖励/终止/课程与 A 臂逐字相同（同一 ActionTerm、同一 env 配置；policy 观测组末位多 930 维本体项 = decoder 输入契约，actor 输入经切片仍同维），**只换 policy 网络**：policy 前向内部走 token → decoder（带 LoRA）→ 29 维动作，decoder 在策略计算图内（可反传），base 权重仍冻结、只有 adapter 参数进优化器；挂载范围与零初始化约定同 §3。

**对照结构修正（判读语义随之变化，必须注明）**：

| 对照 | 是否单变量 | 说明 |
|---|---|---|
| **C vs A** | **是（单变量）** | decoder 在路径且可适配（C） vs 无 decoder（A）——回答「低秩适配后的 decoder 通路能否追平 29 维直出」 |
| **C vs E44** | **是（单变量）** | 低秩（C） vs 全秩（E44 全权重微调，当年把直行破坏成打转的稳健负结果）——回答「低秩是否避开全秩的破坏」 |
| B vs C | **否** | 动作空间不同（B = 64 维 token 直出 + env 侧冻结 decoder；C = 29 维关节空间直出 + policy 内 decoder）——B/C 只作**参考水位**，不得写成单变量结论 |

**判据不变**：主判据 = C 在 rough_paper 0.08 存活 **≥A−25%**（即把 A vs B 的 4.5–9.5× 差距收回 ≤1.33×）；辅判据 = det 推理 actual_vx 非近静止、flat vx 跟踪有效（评测命令通道先修复）；遗忘检查 = LoRA scale 置 0 后输出与原 decoder 逐位一致 + token 保真抽测。**原文以 §4 为准，本节不改判据语义，只改臂的实现形态。**

**trainable 两档登记（发射时二选一并写进行内，禁止事后换档）**：① **默认档 = policy + adapter**（from scratch，与 A 臂平行对照的主档）；② **adapter-only 档**（policy head 冻结，须以 `--init-head-from` 提供初始化；该档下 head 装载有键被跳过即报错退出，确认接受部分加载须显式 `--allow-partial-head`）。

## 4. 判据（预注册，发射前冻结）

- **主判据**：C 在 **rough_paper 0.08** 的 60s 固定档存活率 **≥ A−25%**（A 取 D064 A 臂在案值 0.75/0.79 为参照；等价形式 **A/C ≤ 1.33×**，即把 A vs B 的 4.5–9.5× 存活差**收回到 ≤1.33×**）。
- **辅判据（D064 教训）**：det 推理 **actual_vx 非近静止**、**flat vx 跟踪有效**（vx 0.4/0.6 两档）——D064 两臂 det 推理近静止（B≈0.002 / A≈0.018 m/s）且 flat vx RMSE 因「评测命令钉死通道失效」不采信；**本实验评测前必须先修复命令通道**（`vel_command_b` 被 resampling 覆写的戳写问题，涉未验证 API 面），修复后同一通道同时用于 A/B/C 三臂复评（口径统一，不复用失效数字）。
- **遗忘检查（两条）**：① **LoRA scale 置 0 后输出与原 decoder 逐位一致**（零初始化的构造性保证；训练后亦须复验，即「置零 ⇒ 回退到 B 的等价策略」）；② **token 保真抽测**（B/C 通路 token 统计对照，确认 adapter 未把解码器推离 token 流形）。
- 其余对照指标沿 D064 在案（课程 terrain level、rough_paper 0.06、L1–L9 出生行存活、flat vx RMSE），只报数不另设门。

## 5. 裁决（纲领 §6 分支 2）

纲领 §6 裁决四分支的第 2 条原文：

> 2. **适配明显更强**且无显著遗忘 → **允许转向适配**，不必为最初设定辩护。

映射到本实验：

- **C 明显更强且无显著遗忘**（主判据过 + 遗忘两条过）→ **允许转向适配路线**（本仓后续主线在冻结与受限适配之间重排；写作口径按 §1.4，不得写成「冻结路线被推翻」）；
- **C ≈ B**（主判据不过）→ **低秩消不掉接口代价**：文献综述解法③（LoRA/SPAR 整流受控改）在本接口上**关闭**，D064「接口 = 机制性瓶颈」结论**加固**（低秩维度已排除，剩余解释收窄为「离散 token 唯一入口」本身）；
- **C 崩**（训练不进/存活显著低于 B）→ **记录性负结果**：登记为「低秩适配在本配方本规模下不可用」，不推翻纲领任何分支（负结果与「简单规则已达学习水平」分支互不影响）。

**边界写法**（沿纪律）：结论限定在「本配方、本规模、本接口」，gate≠机制；不设中途改口径，若发生偏差（envs/iter 变动）在 Run 行登记。

## 6. 与主线关系

- **D062（作者 v1 300M）照走不误**：两条线并行、互为对照——**改上层 token 作者**（D060 语料 / D061→D062 阶梯）vs **改下层 decoder 权重**（本臂）。作者线与本臂共享同一套观测/评测语言（D064 栈），互不阻塞。
- D060 语料、D059 G0 结论、D061 门与既定判据不受本实验影响；本实验只新增文件（LoRA 模块 + 训练/评测接线），不改既有 A/B 臂行为。
- ORCS_SURVEY §4 建议 B（ORCS 权重作外部上界参照）与本臂不冲突，可后续另议；建议 C（照抄视觉线）不在本实验范围（我方无相机通道）。

## 7. 风险与限定

1. **不等于 ViBe 复现**：ORCS 的 LoRA 由**感知条件化**驱动（视觉提取器出 128 维向量注入首层），本臂**剥掉**该条件化与 augmentation 流——测的是「低秩通道本身」，不是 ViBe 的「感知→地形」能力；正结果只能支撑「低秩可消接口代价」，不能支撑「感知适配已成立」。
2. **D064 限定语四条照抄（原文，判读语义不得松动）**：①两臂 det 推理均近静止（diag actual_vx：B≈0.002 / A≈0.018 m/s）→ 存活差距的语义 = **同条件稳定性差距**，而非「行进间过地形能力」；②判据③ flat vx RMSE 未能正式测得——**评测命令钉死通道失效**（`cmd_vx_mean` ≠ 标称且随条件漂移 = resampling 覆写 `vel_command_b` 戳写），但 diag 已直证两臂命令不跟踪（actual_vx≈0），实质结论已有；**flat RMSE 实测值因通道失效不采信、数字落账备查**：decoder_s0 0.3978/0.5976、decoder_s1 0.4022/0.6023、direct_s1 0.3790/0.5790、direct_s0r3 0.3824/0.5807（vx=0.4/0.6 两档）；③A 臂 s0=2048envs 死点偏差（A s1@4096 单独已足支结论：0.75 vs 0.083–0.167）；④B 2seed / A@4096 1seed。
3. **许可与仓纪律**：ORCS ckpt 及 SONIC 权重均为 **NVIDIA Open Model License**（引用/再分发义务见 [ORCS_SURVEY](../surveys/ORCS_SURVEY.md) §7.1）——**ckpt 只留服务器 `outputs/`，不进本仓**；仓内只留比对脚本与结论。
4. **新代码路径风险**：C 臂新增「挂载点 + 零初始化」两处接线，须由 G0(c) 与「iter 0 输出 = B」双重自检兜底；训练后 base 逐位冻结断言兜底「只训 adapter」。
5. **归因边界**：本臂只检验「低秩」这一条修改路径；残差路线（E48 关闭线在 128envs 手搓配方前提下的重开评估）与延长训练排除「未收敛」解释，均属 D064 尾注另列的候选，不在本预注册范围。

## 8. 预算、产物与记录纪律

- **预算**：G0 零训练（纯前向，lab-ts，分钟级）；G1 单臂 cvgl 4090 池 4096 envs ≤5000 it（隔夜级）+ 三臂同批评测（含命令通道修复后的 A/B 复评）。
- **产物**：统一落服务器 `outputs/d065/`——G0 比对+置换+metadata+前向快照、C 臂训练 ckpt/log、三臂同批评测 JSON 分子目录；本仓镜像 `apt_g1/outputs/sync/d065/`；结论行落本仓 tracker。
- **记录纪律**：Run 行只进 [tracker/D.md](../tracker/D.md)（本行立项 + 结果追加）；判据发射前冻结、不调口径；single-variable 纪律（与 D064 唯一变量 = 是否挂 LoRA）；结论表述与 [HANDOFF/README](../../HANDOFF/README.md) §3 既有口径一致。

## 版本历史

- **v1（2026-09-21，本版）**：立项预注册——定位（D064 接口改造候选具体化 / 纲领 §6 对照路线外部同构实例 / 不作主假设证据）、G0 对齐冒烟门三小项、G1 三臂单变量对照（C = B + rank-16 零初始化 LoRA，只训 adapter、无感知条件化）、主/辅/遗忘三组判据、纲领 §6 分支 2 裁决映射、D062 并行关系、风险与限定（ViBe 非复现 / 限定语四条 / NVIDIA Open Model License 只留服务器）。依据 = owner 09-21 拍板执行 [ORCS_SURVEY](../surveys/ORCS_SURVEY.md) §4 建议 A。
- **v1.1（2026-09-21，G0 PASS 后发射前修正，§3a）**：G0 三门已在服务器全 PASS（明细见 [tracker/D.md](../tracker/D.md) D065 行尾注）；G1 训练臂实现发现 B 臂 decoder 在 env 侧 ActionTerm 且 `torch.no_grad()`（`sonic_action_term.py:285-291`）⇒ adapter 无策略梯度、C≡B 假阴性风险；owner 授权按 ORCS（in-graph）形态重构 C 臂为「A 臂架构 + μ 路径经 decoder（挂 rank-16 零初始化 LoRA）」=E44 架构低秩版，对照结构改写为 C vs A / C vs E44 单变量、B vs C 降参考水位；主/辅/遗忘判据原文不变；trainable 两档登记（policy+adapter 默认 / adapter-only 须 `--init-head-from`）；产物目录口径统一为 `outputs/d065/`。
