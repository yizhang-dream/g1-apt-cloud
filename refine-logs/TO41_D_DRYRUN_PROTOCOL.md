# TO41 D 阶段 conformance dry-run 执行协议（D1/D2/D3，28-cell audit）

> 【层位 L3 执行协议】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`TO41_RUNG1_IMPL.md` §4（28-cell launch table）/ §6（launch sanity）/
> §8–§9（owner freeze review + 三十四轮授权）；specification 侧一切判据以
> `TO40C_PLAN.md` §10 为准，本文不得与其冲突｜ 状态：**活跃**
> （2026-09-02 三十四轮评审 9.8/10 授权开工；同日三十五轮评审 9.9/10
> **协议定稿 + implementation 开工**——注意语义是 *protocol
> authorized / implementation phase opened*，**不是** D 已启动/已通过；
> 同日三十六轮 owner 批准 implementation → Mode A runtime + independent
> checker 落码、negative tests 9/9（本机+lab-ts）、**28-cell D dry-run
> 执行完毕，D1/D2/D3 全 PASS**（report = `apt_g1/outputs/sync/to41_d/
> D_report`，lab-ts frozen env，checker 独立审计 failures 0）——
> execution freeze 待 owner 纯状态迁移；Rung 1 compute BLOCKED 不变；
> **09-02 三十七轮 owner 改判：暂不 freeze**——D decode-only PASS ≠ 完整
> 训练环境 plumbing，先过真实 `apt_flat_env.py` 的 L1–L4 launch sanity
> （= `TO41_LAUNCH_SANITY.md`，纯执行 gate，本文判据不动））。
> （执行 freeze 状态以此为准；上方三十六轮语义保留作历史。）

## 0. 对象、边界与当前状态板

**本文是 implementation 的第一份产物的预注册规格**：conditioning runtime
（仅 Mode A：`τ(v,C) = τ(v)`）的 28-cell conformance dry-run。dry-run 的
产出是**一份 audit artifact**（D1/D2/D3 conformance report），**不是实验
结果**——Rung 1 的任何科学结论都只能来自 execution freeze 之后的正式
compute。

**状态板（三十四轮裁定维持；冻结 / 未冻结严格分离）**：

```text
Specification          FROZEN*（TO40C_PLAN §10；reopen 仅限 owner）
Grid                   APPROVED（7-point：0.200–0.325 + 0.277 anchor）
Mode A                 LOCKED（τ(v,C) = τ(v)；Mode B 永久关闭）
τ(v) material          FROZEN（7/7，availability map 见 G_DOWN_SPEC §9）
G↓                     CLOSED / PASS（不再扩展，抽材料纪律见 §5）
D1/D2/D3               PASS（09-02 三十六轮 lab-ts 28-cell dry-run；
                       schema/D1/D2/D3A/D3B 全绿，failures 0）
D protocol             FROZEN（三十五轮 9.9/10 定稿；此后唯一允许变更
                       = 把 frozen specification 变成可执行代码）
Env launch sanity      NEXT（09-02 三十七轮 owner 裁定：D decode-only
                       PASS ≠ 完整训练环境 plumbing——先过 L1–L4 真实
                       env 接线验证 = TO41_LAUNCH_SANITY.md，再 freeze）
Execution freeze       BLOCKED pending launch sanity（三十七轮改判）
Main Rung 1 compute    BLOCKED
```

**28 cells ≠ 28 个独立 jobs**：`7 v × {C1,C2} × {τ ON, τ OFF} = 28` 是
**实验 cell 数**，不对应 28 个训练 / dircol / decoder job。dry-run 只做
**最小执行样本的 plumbing 验证**（manifest / 配置层 28 cell 全覆盖静态核对
+ 最小 runtime 实例化样本；样本量由 implementation 提出、在 report 中如实
记录）；正式 Rung 1 compute 才按实验设计启动相应训练/评估。

**执行顺序（三十四轮批准，不得跳步）**：

```text
implementation → 28-cell D dry-run → D1 → D2 → D3
    → audit artifact → 全 PASS → owner execution freeze → Rung 1 compute
```

## 1. D1 decoder invariance（检查面 ≥ weights unchanged）

**目标命题**：condition override 不改变 decoder computation semantics，
唯一允许的自由度 = 预定的 condition-selection 输入本身。

**必查字段清单（缺一即 FAIL，不得以「weights unchanged」单项替代）**：

1. checkpoint hash；
2. state_dict hash；
3. architecture identity（层结构/超参 identity，非仅类名）；
4. latent dimensionality；
5. input transform identity（tokenizer / 预处理管线 identity）；
6. normalization semantics（含统计量来源与数值）；
7. output tensor contract（shape / dtype / 数值范围约定）。

**已知风险（本清单的存在理由）**：weights 相同 + preprocessing 改了 →
decoder output 变了 → 仍被误报「decoder invariant PASS」。因此 D1 的 PASS
判据是**全字段清单 + 输出 contract 对照**，不是权重点查。

**runtime identity 附加项（三十五轮加严）**：`mode`（pointe / foot 类
接口开关）、variable layout、tensor shape 属 **D1/D3 的 runtime
identity**，不是普通日志字段。先例：`TO41_G_DOWN_SPEC.md` §「launch
audit incident（三十一轮）」已记录 `manifest fixed_params.mode=pointe
vs 源 mode=foot` 的 identity mismatch——weights 全同仍致行为不同。因此
这三项与七字段同等级别参与 PASS 判定，逐 cell 如实对照，缺记即 FAIL。

## 2. D2 assignment + same-τ identity（Mode A fingerprint）

**机械化判据**：∀v ∈ 7-grid，C1(v) 与 C2(v) 两个 cell 的 `tau_hash`
必须**完全一致**——即同一 target speed 的全部 4 个 cell
（C1/C2 × τ ON/OFF）共享同一 τ material identity。

**FAIL 语义（明确，不留解释空间）**：若某 v 出现
`C1 → H1, C2 → H2`（H1/H2 都是各自合法的 material），判定**不是**
「两个 τ 都有效」，而是 **Mode A 被实现成了 Mode B / joint treatment**
→ **D2 FAIL**。

assignment 对照对象 = mapping v2 YAML（14 rows 全交叉）；assignment
正确性（condition 期望值 vs runtime 实际值）与 same-τ identity 是 D2 的
两个组成检查，二者都过才算 PASS。

**lineage 伴记（三十五轮加严）**：D2 审计表逐 cell 不只写 `tau_hash`，
同时写 `tau_source_lineage`——若未来看到相同 hash，仍能确认两个 cell
引用的是**同一个冻结 artifact**（有 lineage），而非两个巧合相同的文件。
hash 相等 + lineage 缺失 = 记录不完整，判 FAIL。

## 3. D3 两条等式（分开证明，禁止合并）

1. `T_runtime(v,C) = T_mapping(v,C)` —— **assignment conformance**
   （C1/C2 是否按 v2 mapping 执行）；
2. `τ_runtime(v) = τ_frozen(v)` —— **material conformance**
   （runtime 是否真的在用冻结 τ(v)）。

二者分别验证、分别报告；**禁止**合并成一句「runtime conforms」。D3 PASS
的解释上限沿用 IMPL §3 重申：只支持「implementation conforms to
treatment specification」，不支持「conditioning is valid」，不支持 C1/C2
是两个真实 gait regimes。

**verdict 粒度（三十五轮加严）**：最终 JSON 不得只有一行 `D3: PASS`，
必须分别产出：

```text
D3A assignment_conformance: PASS/FAIL
D3B material_conformance:   PASS/FAIL
D3  (overall):              PASS/FAIL
```

这样任一失败都能立刻定位是 **condition plumbing**（D3A FAIL）还是
**material plumbing**（D3B FAIL）。D2 PASS + D3B FAIL = specification
没问题但 runtime 没忠实执行 mapping/material；D1 PASS + D2 FAIL =
decoder invariant 无恙但 conditioning 下 τ material 被错误变化（Mode A
被偷换成 Mode B / joint treatment）——层次划分照此诊断，不得互混。

## 4. audit artifact schema（dry-run report 逐 cell 必备字段）

每 cell 一行，单文件单 artifact（D1/D2/D3 合一份 report，不拆散）：

```text
cell                 # (v, C, τ_ff) 组合 id
target_speed         # treatment identity（冻结 grid 原值，禁止改标 realized）
decoder_condition    # runtime 实际生效 condition
tau_hash             # runtime 实际 τ material hash
tau_source_lineage   # 该 hash 对应冻结 artifact 的 lineage（三十五轮加严）
decoder_hash         # state_dict（及 checkpoint）hash
mode_layout_shapes   # mode / variable layout / tensor shape runtime identity
                     # （三十五轮升格，随 D1 判定，非普通日志字段）
assignment           # mapping v2 期望 condition + match 布尔
PASS/FAIL            # D1 / D2 / D3A / D3B 各自 verdict + D3 总 verdict
失败原因             # FAIL 时必填，指明违反本文哪条判据
```

**禁收字段（三十五轮加严）**：D 的 PASS criterion **不含任何
performance / locomotion quality 字段**（reward、walking quality、
stability 等）。audit artifact 允许的只有 execution success / schema /
hash / assignment 类 conformance 字段。dry-run 中若观察到「C1 的
rollout 比 C2 更平滑」一类现象，与 D conformance **无关**，不得写入
verdict、不得据此影响 PASS 判定——否则 D 会逐渐变成小型实验。D 只回答：
plumbing 是否忠实执行 specification。

**材料基线段（report 固定附段）**：7 材料 hash + v_realized + abs_err 全表
照录 `TO41_G_DOWN_SPEC.md` §9；**0.300 → 0.2925（abs_err 7.5e-03）、
0.325 → 0.3179（abs_err 7.1e-03）显眼标注**——语义恒为
*accepted under the pre-registered ±0.02 m/s realization tolerance*，
**不得**因这两个误差数字触发任何重新求解（那会把预注册容差门⑨事后变成
精度优化目标）。

## 5. G↓ CLOSED 后的材料纪律（execution integrity）

G↓ 已 CLOSED：不新增 downward source、不重新求低速、不扩展 seed、不改
canonical material。若 D 阶段发现某个低速 material 的 plumbing 有问题：

- 正确动作 = **报告 runtime/material conformance failure** → 修
  implementation → 重跑 D；
- **不是**回 G↓ 重做材料；重新打开 material freeze 需显式 owner reopen。
  否则 material freeze 的意义被破坏。

## 6. 诊断量纪律（abs_error 不进分析模型）

七字段诊断量（`target_speed` / `material_realized_speed` / `abs_error` /
`material_hash` / `source_lineage` / `decoder_condition` / `tau_ff`）是
**execution diagnostic**。三十四轮加严澄清（与三十三轮纪律合读）：

- treatment identity 恒为 `target_speed`（三十三轮解释纪律原文不变）；
- `abs_error` / `v_realized` 允许的用途仅限**预注册的 post-hoc 共变
  诊断**（检查效应是否随 realization error 系统变化）；
- **禁止**在 Rung 1 开始之后临时决定把 `abs_error` 升格为主分析模型的
  covariate adjustment / re-weighting——那会把 material realization
  error 重新引入 analysis model。诊断不进主模型，且不得中途加 stopping
  rule。

## 7. 保险丝（两条，D 阶段全程有效）

1. **D 发现 runtime bug → 修 implementation → 重跑 D**；**禁止为了让
   D PASS 去修改 frozen specification**。
2. implementation 发现 **Mode A 在真实代码里根本无法忠实实现** → 这不是
   普通 bug，而是 **genuine incompatibility → owner reopen**（流程走
   `TO41_RUNG1_IMPL.md` §3）。这是当前项目最重要的保险丝。
3. **spec 再解释熔断（三十五轮）**：implementation 过程中一旦发现某个
   已冻结字段需要「重新解释」才能继续写代码 → **立即停**，走 genuine
   incompatibility → owner reopen；**禁止顺手修协议**。三十五轮起协议
   本身不再是修改对象——implementation 阶段唯一允许的变更 = 把 frozen
   specification 变成可执行代码。

## 8. tracker 语义分离（三十四轮，防未来误读）

G↓ 的 4 个 run（`TO41-GD-*`，tracker/TO.md）= **material-generation /
coverage 结果（工程收束）**，**不是** Rung 1 scientific result。在 D 全
PASS + execution freeze + 28-cell compute 完成之前，任何叙述不得把 TO41
写成「已产生 Rung 1 研究结论」。

## 9. audit 独立性（三十五轮；checker 不消费被测代码的自报 PASS）

**禁止自证路径**：runtime 自己打印「assignment correct」→ checker 读这个
flag → 判 PASS——这是自证，Driver G2 修正教训（TO36 冻结验收哲学）同类。
正确路径：

```text
runtime
  ↓
actual assignment record（实际状态记录，非 verdict）
  ↓
independent checker（独立计算 / 独立读取）
  ↓
compare with frozen mapping v2 YAML + frozen material artifact
```

即：**D checker 尽量独立计算 / 读取实际状态（自己 hash、自己解析
YAML、自己对照 manifest），而不是消费被测代码自己生成的 PASS flag**。
runtime 侧产物只能是 *记录*（record），*判定*（verdict）只能出自
checker。

## 10. 执行环境与角色纪律（三十五轮）

1. **D 的实际执行只在 lab-ts frozen execution environment**（
   `/tmp/run_apt_isaac.sh` 包装，`.venv_isaac`）。本机（Windows，无
   venv）只允许：静态检查 / schema 校验 / SCRIPT_MAP 登记 / 文档；
   **禁止**在本地用不同 runtime 偷跑 D1/D2/D3 然后写「dry-run
   PASS」——否则本地 PASS + 服务器 FAIL 会重新引入环境差异。
2. **SCRIPT_MAP 登记角色标注**：conditioning runtime 标注为
   **state-changing execution code**；D checker 标注为
   **read-only audit**——checker 只读取与独立计算，禁止为了「修正发现
   的问题」顺手修改 material / mapping / 任何实验状态。发现问题的唯一
   正当出口 = report FAIL → 走 §7 保险丝。
3. **28 cells ≠ 28 jobs 的另一面**：dry-run 不产生任何
   performance / locomotion quality 性质的选择（见 §4 禁收字段）；
   audit 范围恒为 execution success / schema / hash / assignment。
