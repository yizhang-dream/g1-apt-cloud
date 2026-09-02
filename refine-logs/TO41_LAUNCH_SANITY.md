# TO41 Rung 1 launch sanity 执行协议（L1–L4，真实 env 接线验证，28-cell）

> 【层位 L3 执行协议】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`TO41_D_DRYRUN_PROTOCOL.md`（D 判据，FROZEN，本文不得与其冲突）／
> `TO40C_PLAN.md` §10（frozen specification）／ 三十七轮 owner 裁定（2026-09-02，
> tracker/TO.md）｜ 下游：lab-ts 28-cell sanity receipt + L report
> （`apt_g1/outputs/sync/to41_sanity/`，Run 行入 `tracker/TO.md`）｜
> 状态：**活跃**（三十七轮裁定：**暂不做 execution freeze**——D 全 PASS 证明的是
> decode plumbing，不是完整训练环境 plumbing；先做真实 `apt_flat_env.py` 的
> τ 注入/控制路径接线验证，通过后 owner 才做纯状态迁移 execution freeze；
> 主 Rung 1 compute 继续 BLOCKED）。

## 0. gate 定位与状态板

**本轮新增的不是第 37 轮 protocol、不是任何科学设计变更**（§10 设计、D 协议
、mapping、材料全部不动），而是一个**纯执行 gate**：D 的 decode-only dry-run
与"τ 注入真实 env"之间存在 execution gap——τ material 进错时间步、condition
override 被 natural bucketize 重新覆盖、两臂消费了另一份 buffer、reset 边界
恢复 natural assignment 等，都不会被 decode probe 发现。本 gate 回答且仅回答：

> **the treatment specification survives contact with the actual robot
> simulation environment.**

```text
G↓                           CLOSED / PASS
Material map                 FROZEN 7/7
Mode A                       LOCKED
D protocol                   FROZEN
D1/D2/D3                     PASS（三十六轮 lab-ts 28-cell dry-run）
L0 材料消费格式导出          PASS（gate 首格发现格式断链 → 既有 TO38 链
                             确定性导出 7 LUT + manifest，F11b 交叉验证
                             逐位 MATCH；§5b，09-03）
Env launch sanity（本文）    PASS（09-03 lab-ts 28-cell，checker
                             schema/L1/L2/L3/L4 全 PASS，failures 0）
Execution freeze             PENDING owner 纯状态迁移（launch sanity
                             PASS 前置已满足；freeze 不得同时修改
                             runtime/mapping/τ material/analysis protocol）
Rung 1 compute               BLOCKED（待 owner freeze）
Scientific result            NONE
```

**执行链（三十七轮裁定后）**：

```text
D1/D2/D3 decode conformance → L1–L4 env launch sanity → execution wiring PASS
    → owner execution freeze（纯状态迁移）→ Rung 1 compute
```

**28 cells ≠ 28 jobs**（沿 D §0 语义）：每 cell = 一次真实 env 最小执行
（300 控制步 ≈ 6 s，零策略动作驱动），无训练/评测/性能语义。

## 1. L1 — τ material consumption

对每个 target v：C1、C2（及 τ ON/OFF 共 4 cell）消费同一个 τ(v)，并记录
实际 consumer identity/hash：

1. **材料进入路径** = 冻结材料的 **L0 derived LUT**（§5b manifest）→ env
   既有通道 `cfg.to_ref_npz` → env `__init__` 自行 `np.load`（
   `apt_flat_env.py:319-328`，零改动）；
- **consumer identity** = `apt_flat_env.py:765-773 _apply_action (to_tau
  branch) ← _to_ref_lookup():776-787 ← self._to_tau`（receipt 逐 cell 记录
  该字符串 + buffer 快照）；
- **实际消费身份** = `env._to_tau` 缓冲规范哈希（episode 前后各一次）+
  per-call τ 序列 digest（ON 臂）；checker 独立 `np.load(tau_ref6)` 重算
  buffer 哈希对照；
- **Mode A env-level fingerprint**：同 v 4 cell buffer 哈希完全一致；
- **OFF 臂语义**：`to_tau=False` → 冻结消费点不执行（`n_tau_calls==0` 是
  预注册的"不注入"记录），材料身份照常装载并记录（D2 fingerprint 覆盖
  全 4 cell 的 env 延续）。

**OFF 臂载 τ(v) 材料的动力学中性**（实现裁定，非 spec 变更）：两臂同用
TO40C ctrl 配方形状（`to_ref=True, to_ref_obs_zero=True, to_ref_w=0`），
LUT 的全部消费者 = obs 块（置零）+ to_tau 分支（OFF 关闭）+ reward（权 0）
——材料内容对 OFF 臂动力学无可达路径，唯一 cfg 差 = `{to_tau}`（L3 机械
判定）。

## 2. L2 — override persistence

验证 `decoder_condition` 在整个执行过程内不被 natural `bucketize(cmd_vx)`
后续逻辑重新覆盖。实现机制即证明：override 位于 **decode 输入边界**
（实例级 shadowing `env._vae.decode`，冻结调用点 `:589`），冻结 bucketize
（`:583-586`）每步照跑，wrapper 每次 decode 调用记录 natural vb/db 并替换为
mapping bin——**逐 call 双记录 + boundary 前后计数**：

- 每 cell：`n_decode_calls == steps_done`；全部 call 的 applied == mapped；
  natural 分布 == checker 重算（`bucketize(v, linspace(0, vx_max, n+1)[1:-1])`
  的 numpy 等价，网格点无触界故无 ULP 敏感）；
- **恒定 cmd 纪律**：`cmd_vx` 每 control step 重申（episode 重置会用
  `U(vx_min, vx_max)` 重采样 commands——这是 owner 警示的"reset 把 condition
  状态改写"同类风险的实际机制，receipt 记录 `n_cmd_reassertions`）；
- **boundary persistence**：每 cell 至少 1 次强制 episode boundary（中段
  `jitter_and_reset`，eval 同款）+ 自然 term 重置计数；强制边界后
  `cond_calls_after > 0` 且全部 post-boundary call applied == mapped；
- episode `cmd_vx` 恒定 ⇒ natural bin 每 call 相同，interventional cell 的
  `n_override_changed == n_calls`（natural≠mapped 时）逐 call 可见。

## 3. L3 — ON/OFF isolation

同一 (v, C) 下 OFF/ON 两臂除预注册 τ_ff intervention 外无其它 treatment
改动：

- **cfg 快照 diff == `{"to_tau": False→True}`**（checker 机械判定；
  `to_tau_w` 两臂同为 1.0 显式值）；
- 两臂 decoder 身份（checkpoint/state_dict/arch）、材料文件哈希与 buffer
  哈希、condition mapping、seed 协议、步数与 boundary 协议、python/torch
  版本完全一致；
- decode 输出不参与 L3 判定（闭环下 z 随动力学演化，两臂输出不同是预期，
  不是 wiring 信号——与 D1 的 decode-probe 伴生检查区分）。

## 4. L4 — 28-cell receipt（来自真实 env 执行路径）

7×2×2=28 cell，receipt 必备字段（owner 三十七轮清单）：

```text
target_speed            # treatment identity（冻结 grid 原值）
decoder_condition       # mapping_lookup 的 condition id（mapped）
tau_hash                # τ(v) 材料文件 sha256
tau_source_lineage      # 冻结 artifact lineage（availability map + registry）
tau_ff                  # on/off
actual consumed material hash   # env._to_tau buffer 规范哈希（前后快照）
actual condition        # 逐 decode call 的 natural/applied vb/db
decoder identity        # checkpoint/state_dict（episode 前后）/arch 签名
execution receipt       # steps/boundaries/cmd 重申/auto reset 计数/版本
```

**禁收字段（沿 D §4 加严）**：reward / walking quality / stability /
achieved speed / h_min 一律不入 receipt 与 verdict——本 gate 只测接线，
不测性能。cell 执行失败 = 该 cell 无 completed receipt → L4 FAIL → 保险丝。

## 5. 接线纪律（本 gate 的机械核心）

1. **零 env 文件改动**：`apt_flat_env.py` 的 sha256 = mapping
   `preprocessing_hash` 冻结锚。接线全部为**实例属性 shadowing**
   （`env._vae.decode` / `env._to_ref_lookup`）+ 既有 cfg 通道，冻结类与
   冻结代码路径逐字不变。driver 执行前 preflight 三源哈希（env / vae /
   train arch）+ 7 材料可达，任一不符 fail-fast（env 被改动即 genuine
   incompatibility，禁止带病执行）。
2. **record / verdict 分离**（沿 D §9）：`env_wiring` / `launch_sanity`
   只产 record（per-call 记录/计数/哈希），全链封禁自报 verdict 字段；
   verdict 只出自 `l_checker`（不 import 被测两模块，独立解析器 + 独立
   npz 重算 + 独立自然 bin 重算）。
3. **canonical array hash 规范**（双实现同一 spec，selftest 交叉验证）：
   `sha256("shape={shape};dtype={dtype.str};data=" + float32 C-contiguous
   小端字节)`——float32 化对齐 env 冻结加载路径 `torch.from_numpy(...).float()`。
4. **cell 执行结构**（镜像 compute 的 per-cell 进程语义）：**每 cell 一个
   独立 python 进程**（`--cell-index 0..27`，服务器 bash 循环 28 次；
   AppLauncher 逐 cell 启动，receipt 落盘后 `os._exit` 硬退出——规避
   `DirectRLEnv.close()` 的 `sim.clear_instance()`、同进程重复建 env 的
   prim 冲突与 Isaac 退出挂死三个风险）。cell 内：新建真实 `AptFlatG1Env`
   （TO40C ctrl/t10 配方 × 该 v 冻结 τ(v) 材料 × latent-dir-bins + e39
   vae；`action_space=16` 零 z 动作驱动），`jitter_and_reset`（eval 同款）
   起止，preflight 三源哈希每 cell 进程都跑。
5. **执行环境**：仅 lab-ts frozen env（`.venv_isaac`，D §10.1 同款
   env-tag 机械闸）；本机只允许 static 配置层核对 / selftest / 登记。

## 5b. L0 — 材料消费格式导出（launch preparation；gate 首格发现，2026-09-03）

**gap 发现（gate 的第一项产出）**：cell 0 执行即暴露 `KeyError: 'q_ref6'`——
7 份冻结材料全部是 to36 hybrid dump 格式（X_left/X_right/U_left/U_right…），
而 env 冻结加载路径（`apt_flat_env.py:319-328`）需要 TO38 LUT 格式
（q_ref6/tau_ref6/pitch/z/heel_rel/T/v_avg）。**D decode-only dry-run 无法
发现这一断链**（D 只验材料身份哈希，不加载 q_ref6）——正是本轮 gate 存在
的理由（"完整训练环境 plumbing" ≠ "decode plumbing"）。

**定性**：缺失的 launch-preparation plumbing，**不是** genuine
incompatibility、**不是** material reopen——材料 canonical 身份 = 冻结 npz
一个字节不动（D1/D2/D3B 结论全部不动），TO38 既有生产链（to36 dump →
`to36_leg_to_drake.py world`（Drake FK，81 样本/相，确定性）→
`to38_export_ref.py`（numpy 重采样 + PHASE_PERM 符号映射，m_per_phase=60，
确定性））就是当年 to38_ref.npz 的实际生产路径。

**机械证据（三项）**：

1. **交叉验证**：F11b_flat（0.277）经新链重导 vs canonical `to38_ref.npz`
   （TO38/TO40 实际消费过的同一 LUT）——q_ref6/tau_ref6/pitch/z/heel_rel
   **逐位一致 max|diff| = 0.0**（T/v_avg 一致）；
2. **确定性**：v0.200 同输入两次导出——全部数组逐位一致（meta.world_src
   路径除外）；npz 文件字节级 sha 因 zip mtime 不可作复现锚，身份锚 =
   **数组级规范哈希**（lut_array_sha256）；
3. **周期闭合**：7 份 LUT wrap_gap_q 全部 0.0000。

**产物**：7 份 derived LUT + `lut_manifest.json`（v → source_artifact /
source_sha256 / lut_file / lut_array_sha256 / wrap_gap / T / v_avg）入仓
`apt_g1/outputs/sync/to41_sanity/luts/`（commit 8a6ee96）。工具侧唯一变更 =
`to36_leg_to_drake.py` world 子命令加 `--world-out`（默认 = canonical
to36_world_knots.npz 逐字节不变；commit e8bb406）。

**身份链（双层）**：冻结材料 npz（canonical，D 链锚定）--manifest
source_sha256--> derived LUT（env `cfg.to_ref_npz` 消费形式）--> env
`_to_tau` buffer。receipt 与 l_checker 的 L1 三层（材料/LUT/buffer）独立
重算对照。

## 6. 判据与保险丝

- **verdict 结构**：`schema_check / L1 / L2 / L3 / L4 / overall` 全 PASS =
  execution wiring PASS → owner execution freeze（纯状态迁移）。
- **保险丝 1（bug 路径）**：发现 implementation bug → 修 implementation
  → 重跑 launch sanity（必要时重跑 D1/D2/D3）。**禁止**为了让 gate PASS
  修改 frozen specification / mapping / 材料 / D 协议。
- **保险丝 2（incompatibility 路径）**：frozen specification 与真实 env
  无法一致实现（如冻结 env 结构使 Mode A 根本无法接线）→ 不是普通 bug，
  是 **genuine incompatibility → owner reopen**。
- **保险丝 3（spec 再解释熔断）**：实现中需要"重新解释"任何冻结字段才能
  继续 → 立即停 → owner reopen（沿 D §7.3）。

## 7. 执行记录（2026-09-02 落码 → 09-03 lab-ts 执行）

- selftest：本机 5/5 PASS（T-L0 双解析器交叉 / T-L1 wiring mock 单测 /
  T-L2b canonical hash 双实现交叉 / T-L3 合成正例 schema+L1–L4 全 PASS /
  Negative A–E）。**Negative D 揪出并修复 l_checker 一个真 bug**：缺
  receipt 时 L2 原为静默 `continue`（27/28 也能 L2 PASS）→ 已改为显式
  FAIL（覆盖洞必须级联）。lab-ts frozen env 复跑 5/5 PASS。
- static 配置层覆盖：28/28（`sanity_static_coverage.json`，local 口径），
  interventional 配对抽查正确（0.200 C2 = vb1 on natural vb0；0.325 C1 =
  vb0 on natural vb1）。
- **gate 首格发现并修复 L0 格式断链**（详见 §5b）：cell 0 首跑
  `KeyError: 'q_ref6'`——7 份冻结材料 = to36 hybrid dump 格式，env 冻结
  加载路径需要 TO38 LUT 格式。修复 = 既有 TO38 链确定性导出（材料身份
  不动），F11b 交叉验证逐位 MATCH + 数组级确定性 PASS + wrap_gap 全 0。
  **这条断链 D decode-only dry-run 结构上不可见——本轮 gate 存在理由的
  直接验证。**
- implementation 修复两次（保险丝 1 路径，均不触冻结工件）：①
  cfg_snapshot 的 terrain_seed 属性不存在；② forced boundary 记录改绝对
  calls_before（原增量语义恒 0）；③ receipt 补 state_dict_key_shapes
  （L4 签名表首跑 28×8 FAIL 后补）。
- **28-cell execute（lab-ts frozen env，per-cell 进程模型，commit
  d477562 入仓）**：28/28 receipt rc=0（每 cell 独立 python 进程 +
  AppLauncher；真实 `AptFlatG1Env` 300 控制步 ≈ 6 s + 中段强制 boundary；
  零 z 动作驱动；preflight 冻结三源哈希 28 进程次次通过；单 cell 墙钟
  ~10–20 s）。observe：ON 臂 n_tau_calls=1200（=300 步 × decimation 4，
  `_apply_action` 每 physics 子步消费一次）、OFF 臂 = 0；auto reset 每格
  自然发生并被记录。
- **independent checker：schema / L1 / L2 / L3 / L4 全 PASS，failures 0**
  （`L_report`，lab-ts `--materials-root` 三根）：
  - **L1**：7/7 v 三层 Mode A fingerprint——冻结材料 sha
    0038afb8…(0.200) / eb87ad84…(0.225) / 626b3407…(0.250) / 09c2915c…
    (0.275) / 3f239cbf…(0.277) / da6cb3a6…(0.300) / e7b4cdf4…(0.325)
    与 D 报告逐一吻合；LUT 文件+5 字段数组身份独立重算全中；
    buffer == LUT tau_ref6 重算；source 链闭合（manifest source_sha256 ==
    冻结材料 sha）。
  - **L2**：14 个 interventional cell（低带 C2 + 高带 C1）逐 call
    300/300 override 生效（含强制 boundary 与 auto reset 之后），14 个
    natural cell no-op；natural 分布 == checker 重算 bucketize（恒定
    cmd 每 step 重申生效）。
  - **L3**：14 对两臂 cfg snapshot diff **全部 == {to_tau}**（唯一预注册
    干预）；decoder 身份/材料身份/seed 协议两臂一致。
  - **L4**：28/28 冻结枚举 + env 源哈希 == mapping preprocessing_hash
    （**接线零 env 文件改动的机械证明**）+ decoder 签名表/episode 前后
    全等。
- 环境身份：env_tag=lab-ts；冻结三源（env ca39b76f… = preprocessing_hash
  锚 / vae f6adfc50… / arch b9edf163…）。
- 语义上限：L PASS = **treatment specification 在真实人形仿真执行路径上
  可忠实实现**；不构成任何 Rung 1 科学结论（禁收字段纪律全链无
  performance 字段）。**execution freeze 归还 owner 纯状态迁移。**
