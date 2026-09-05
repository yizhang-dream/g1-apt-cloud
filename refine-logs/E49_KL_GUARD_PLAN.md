# E49-C KL 信任域守卫——退化调研结论与执行计划

> 【层位：专题日志域·执行计划｜↑ `refine-logs/README.md`（扇出树根地图）·
> `E49_STATUS_2026-09-05.md`（上一时点快照，§3c 诊断四步）｜↓
> `refine-logs/tracker/E.md`（Run 行唯一事实源）·`apt_g1/isaac/ppo_core.py`
> （实现本体）·`apt_g1/isaac/e49_kl_guard_test.py`（单测）】
>
> **状态：活跃**——E49-C 唯一执行依据。§A 调研结论冻结；§B 协议为预注册稿，
> 发射 E49-C2 前冻结，此后改动须进 §G 修订记录。
> 2026-09-06 定稿。裁决：owner（下一干预 = KL 信任域更新，不改奖励；固定
> log_std 初值/熵系数/训练器其余配置；不能只做 KL early-stop，必须含回滚；
> 先短探针定阈值，再单 seed）。

---

## §0 一页执行卡（发射从这里开始）

前置：E49-C0（实现+单测）已完成——见 §C.4 状态块。按顺序执行 C1→C2：

| 步 | 做什么 | 判什么 | 预算 |
|---|---|---|---|
| **C1 探针** | fix2 逐字配方 + `--kl-guard 1e9`（只测不拦）跑 300 it | 只收数据不定 PASS/FAIL：产出阈值 `thr` | ~6 min（fix2 实测 0.9 s/it） |
| **C1.5 定阈** | 用下方一行脚本算 `thr`，写进本表 §B.3 并冻结 | thr ∈ 合理带（§B.3）；带外 → 停，回 owner | 1 min |
| **C2 单 seed** | fix2 逐字配方 + `--kl-guard <thr>` 跑 1000 it + 双评 | 预注册 J1/J2/J3（§B.4）与分支判读（§B.6） | ~20 min + eval |
| **中止规则** | C2 发射后 60 it 内 rew<1.0 或 NaN 连发 → 杀；阈值过紧形态（§B.6 分支④）→ 回 §D.2 放宽 2× 重定 | — | — |

发射命令与监控判读的逐字命令见 **§D**；预注册判据全文见 **§B**。
对照基准 = `E49-A-fix2-s0`（同配方同 seed 无守卫，已有，不重跑）。

---

## §A 退化调研结论（2026-09-06，冻结）

数据源：四条 run 的 `train_log.json` 逐迭代提取（fix2 / lr1e4 / E49-B /
VAE 参照，服务器 `outputs/isaac_e49*`；本地提取件
`tmp/e49_investigation/`，gitignored）；代码核验在 HEAD c19bfe6 工作树。
外部锚点见 §F。

### A.1 核心改写：「KL/裁剪失控打崩策略」被日志时序否决

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
   - thr = `____`（P95）；P50=`__` P90=`__` P99=`__` max=`__`（C1.5 回填）

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
    （fix2 计数由分析脚本给出，写入本节回填栏：fix2 = `__` 次 /
    C2 = `__` 次）；
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

- [x] ppo_core.py 守卫逻辑 + expl_var（见 git diff）
- [x] train_apt_isaac.py CLI/hist/回显
- [x] e49_kl_guard_test.py 单测（≥6 用例）——服务器运行结果：**待运行**
  （部署与 sha256 对照随本轮提交执行；结果回填本栏与 tracker
  E49-kl-guard-impl 行，未 PASS 不发射 C1）
- [x] SCRIPT_MAP 登记

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
  完成、单测待服务器运行（结果回填 §C.4）；发射前 §B.3 阈值栏为空，属
  正常状态。
