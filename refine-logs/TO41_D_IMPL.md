# TO41 D 阶段 implementation 日志：Mode A runtime + independent checker

> 【层位 L3 实现日志】↑ `refine-logs/README.md`（扇出树根地图）｜
> 上游：`TO41_D_DRYRUN_PROTOCOL.md`（**D 判据唯一事实源**，FROZEN）／
> `TO41_RUNG1_IMPL.md`（实施章程）｜ 下游：lab-ts 28-cell dry-run report
> （产物入 `apt_g1/outputs/sync/to41_d/`，Run 行入 `tracker/TO.md`）｜
> 状态：**活跃**（2026-09-02 三十六轮 owner 裁定 implementation GO：
> 批准 Mode A runtime + independent checker 开工；28-cell dry-run 在
> 实现完成 + negative tests 全绿后进入；主 Rung 1 compute 仍 BLOCKED）。

## 0. 身份与范围

本文记录协议的可执行化（frozen spec → code）。**本阶段产物是代码 + 本机
静态验证证据，不是实验结果**；D1/D2/D3 verdict 全部 PENDING，唯一有效
来源是 lab-ts frozen environment 的 28-cell dry-run + independent checker
report（协议 §10.1：本机禁止偷跑 D 写 PASS）。

owner 三十六轮裁定的实现目标（原文口径）：**不优化实验、不优化机器人
表现；只实现一个 frozen-spec executor，以及一个不信任 executor 自己
verdict 的 read-only checker**：

```text
frozen spec → runtime → actual receipt → independent checker
```

## 1. 模块地图（SCRIPT_MAP §8 已登记）

```text
apt_g1/rung1/
├── mode_a_runtime.py   【state-changing execution code】Mode A 执行器
│                        CLI: --mode static（28-cell 配置层覆盖核对，本机可跑）
│                             --mode execute（decode-only dry-run receipt×28，仅 lab-ts）
├── d_checker.py        【read-only audit】D independent checker
│                        CLI: --receipts-dir --materials-root --env-tag
├── rung1_selftest.py   【工具】lookup 单元测试 + negative tests A–E（本机/lab-ts 均跑）
└── __init__.py
```

独立性设计（协议 §9 的落地）：

- **checker 不 import runtime**。frozen artifacts（mapping v2 YAML /
  source registry / G_DOWN_SPEC §9 availability map）由两个模块各自实现
  解析器，selftest T0 逐项交叉验证两套解析结果一致；
- runtime 产出 **record only**：receipt schema 封闭（顶层 10 字段）+
  全域封禁自报 verdict 字段（`pass/ok/verdict/assignment_ok/tau_ok/
  d*_pass/...` 黑名单，任何层级出现即 schema FAIL）——verdict 只能出自
  checker 重算；
- lab-ts 出 D report 必须 `--materials-root`：checker 对 7 份 material
  npz **独立重算 sha256**，与 receipt 记录对照（runtime 无法靠谎报哈希
  过关）；`--env-tag lab-ts` 在 Windows 上硬拒绝（协议 §10.1 机械闸）。

## 2. Mode A 契约的两个独立 lookup（owner 实现裁定 §4）

```text
mapping_lookup(v, arm)   → condition_id      # condition 轴（mapping v2，14 rows）
material_lookup(v)       → τ material        # material 轴（availability map，7 rows）
resolve_cell = 二者组合
```

- 纯查表，任何未命中 hard fail（KeyError→exit），**无 nearest /
  threshold /启发式选择的代码路径**；
- τ OFF cell 同样记录 material 身份（D2 的 Mode A fingerprint 覆盖
  同 v 全部 4 cell）；
- receipt 先于 decode 固定（immutable execution record first，owner
  实现裁定 §2）。

## 3. 协议条款 → 实现机制对照

| 协议条款 | 实现机制 |
|---|---|
| §1 D1 七字段清单 | receipt `decoder_identity` 全记录；checker 逐字段对照 mapping 冻结 hash（checkpoint / arch 源 / preprocessing / normalization）+ state_dict 签名（8 键 shape 期望表）+ 声明超参自洽 + latent dim（decoder.0 in-features=34=16+2+16）+ 输出 contract（shape [1,64]/float32/Tanh 值域） |
| §1 runtime identity（mode/layout/shapes 升格） | receipt `mode_layout_shapes` 独立块；checker 校验 decode 调用序（z, phase_sc(sin,cos), v_bin, d_bin）+ probe 输入 shape + mode 在场 |
| §1 state_dict 前后全等（IMPL §4 D1） | decode 前/后各算一次规范化 state_dict 哈希（sorted key+shape+dtype+bytes），逐 cell 相等 + 28 cell 全局相等 |
| §1 加载身份 | `load_state_dict(strict=False)` 的 missing keys 必须为空、unexpected keys ⊆ {encoder., mu., logvar.}（e39 checkpoint 只丢 encoder 侧） |
| §2 D2 same-τ fingerprint | ∀v：4 cell `tau_hash` 完全一致 + `tau_source_lineage` 在场且一致（缺 lineage 即 FAIL，协议 §2 加严） |
| §3 D3A / D3B 拆分 | 两个独立 check 函数、两份逐 cell 对照表、两个 verdict + D3 总 verdict；伴生机械检查：同 (v,C) 的 τ ON/OFF decode 输出 hash 必须相同（decoder 不应变，归 D1）；同 v 的 C1/C2 输出 hash 必须不同（condition 真正到达 decode，归 D3A） |
| §4 禁收 performance 字段 | receipt/verdict 全链无 reward/quality/stability 字段；checker 只认 execution success / schema / hash / assignment |
| §4 材料基线段 | report 固定附段照录 G_DOWN_SPEC §9 全表；0.300/0.325 行显眼标注 accepted-under-±0.02-tolerance（禁触发重解） |
| §9 禁自证 | §1 的 record/verdict 分离 + Negative D 用例 |
| §10.1 lab-ts-only | env-tag 机械闸 + local 报告文件名显式 `..._not_D_artifact` |
| §7.3 spec 再解释熔断 | 未触发（见 §6 观察项：只有不动工件的记录，无 reinterpretation） |

## 4. 28-cell dry-run 样本量提案（协议 §0 授权 implementation 提出）

- **静态层（本机已产）**：28/28 cell 经两套独立解析器 resolve 成功
  （`static_coverage.json`，env_tag=local）；
- **执行层（lab-ts）**：28 cell **全部**执行 decode-only receipt（7 v ×
  {C1,C2} × {τ ON,OFF}）。decode probe = 确定性输入（seed 20260902 的
  z∼N(0,1)@16 维 + walk-phase 1.234 rad 的 sin/cos + resolved bins），
  单次 decode 调用/ cell，无 Isaac rollout、无 dircol、无训练 job——
  28 cells ≠ 28 jobs（协议 §0）；
- **范围边界（如实记录）**：dry-run 验证 conditioning runtime 的
  assignment/material/decoder-identity/decode-contract plumbing；
  **τ 注入 env 的 exercise 不在 D 范围**（属 Rung 1 launch sanity，
  IMPL §6，且 env 文件被 preprocessing_hash 冻结，本阶段零改动）。

## 5. selftest 结果（negative tests 先于 dry-run，owner 实现裁定 §9）

本机 9/9 PASS，**lab-ts frozen env（python 3.10.20 / torch 2.5.1+cu124）复跑
9/9 PASS**（`sync/to41_d/selftest_report.json`；selftest 永不作为 D artifact）：

| 用例 | 构造 | 预期 = 实测（本机 + lab-ts） |
|---|---|---|
| T0 双解析器交叉一致 | runtime vs checker 各自解析三份冻结工件 | PASS |
| T1 lookup 单元 | 14 rows 逐项 + Mode A 恒等式（同 v 两 arm material 全等）+ 未命中 hard fail | PASS |
| T2 静态覆盖 | 两套实现 28 cell 枚举逐项相等 | PASS |
| T3 正例控制 | 未损坏 synthetic set | schema/D1/D2/D3A/D3B 全 PASS |
| **Negative A** | 某 cell τ hash 换另一合法 material | D2 FAIL + D3B FAIL（D1 不受累） |
| **Negative B** | 某 cell condition 改成错 arm | D3A FAIL（D1 不受累） |
| **Negative C** | decoder 超参 n_vbins 3→4 / state_dict shape 篡改 | D1 FAIL 且失败定位正确 |
| **Negative D** | 自报 `assignment_ok=true` 等 flag + 实际 assignment 不一致 | checker 仍 FAIL（schema+D3A 双FAIL，flag 未被消费） |
| **Negative E** | receipt 删 `tau_source_lineage` 字段 | schema FAIL + D2 FAIL（hash 相等也不足） |

synthetic 材料的 registry 冻结 sha256_16 锚不可本地复现（真 npz 在
服务器）——selftest 向 checker 喂 synthetic registry 副本（锚=合成文件
前缀）只验 checker **逻辑**；真锚（registry 三件 16-hex 前缀 + G↓ 四件
独立重算）只在 lab-ts D report 验证。

## 6. 观察项（不动冻结工件，仅记录）

1. `rung1_tau_dec_mapping.yaml` 头部 `freeze_status: generated-not-frozen`
   ——owner 冻结动作记录在 tracker/TO.md 三十三轮；implementation 不改
   冻结工件（保险丝 3），checker 把该字段作为 observation 照录 report。
2. eval 侧 decoder 类源 `apt_g1/isaac/token_window_vae.py` 在 mapping
   artifact 中无冻结 hash（mapping 只锚 train_token_vae_e39.py 与
   apt_flat_env.py）——D1 架构身份由 **state_dict 签名**（实例化真身的
   ground truth）+ 冻结调用点 + 冻结训练源三方覆盖，类源 sha 记入每份
   receipt 供追溯。
3. G↓ 四件材料（gdown_*.npz）此前只有截断哈希记于 tracker 叙事
   （09c2915c…/0038afb8…）；lab-ts D report 将首次落盘四件完整 sha256
   （runtime 记录 + checker 独立重算双源）。
4. Rung 1 compute 的 eval 栈集成（把 `mapping_lookup` 接入
   `apt_flat_env.py` 条件覆写）**不在本阶段**：env 为冻结 binning 源
   （preprocessing_hash），集成属 execution freeze 之后的 compute
   plumbing，届时复用同一 `mapping_lookup` 函数（conformance 随之继承），
   并过 IMPL §6 launch sanity。

## 7. lab-ts 执行指引（下一步，按序）

```bash
# 0) 服务器同步
ssh lab-ts "cd ~/ros2_data/g1-apt-cloud-sync && git pull"
# 1) negative tests（frozen env 下复跑，须 9/9）
ssh lab-ts "cd ~/ros2_data/g1-apt-cloud-sync && bash /tmp/run_apt_isaac.sh -m apt_g1.rung1.rung1_selftest"
# 2) 28-cell decode-only dry-run（receipt×28）
ssh lab-ts "cd ~/ros2_data/g1-apt-cloud-sync && bash /tmp/run_apt_isaac.sh -m apt_g1.rung1.mode_a_runtime --mode execute --env-tag lab-ts"
# 3) independent checker（独立重算材料哈希 → D report）
ssh lab-ts "cd ~/ros2_data/g1-apt-cloud-sync && bash /tmp/run_apt_isaac.sh -m apt_g1.rung1.d_checker --receipts-dir apt_g1/outputs/sync/to41_d/receipts --materials-root apt_g1/outputs --env-tag lab-ts"
```

（实际路径/wrapper 以 `HANDOFF/04_SERVER_GUIDE.md` 为准；材料 canonical
位置 `~/ros2_data/apt_g1/outputs/`，若从 sync clone 跑需以
`--materials-root`/`--vae-path` 显式指向 canonical 位置。）

**执行结果（2026-09-02，全部按上序完成）**：

- selftest：lab-ts 9/9 PASS；
- execute：28/28 receipt（`--materials-root` 三根：apt_g1/outputs +
  ~/ros2_data/outputs/gdown_targets + ~/ros2_data/outputs/gdown_smoke；
  e39 vae.pt 加载身份干净：missing 0 / unexpected 8 encoder 键）；
- checker：**schema/D1/D2/D3A/D3B 全 PASS，failures 0**（D2 fingerprint
  7 v same-τ 全中；D3B 独立重算 7/7 材料全中，registry 三件 16-hex 前缀
  吻合冻结锚，G↓ 四件完整 sha256 首次落盘：0038afb8… / eb87ad84… /
  626b3407… / 09c2915c…）；
- 产物 commit：b69a2ce（server push 经 SSH URL；HTTPS 非交互无凭据）；
- 环境身份：env_tag=lab-ts，python 3.10.20，torch 2.5.1+cu124，
  三份冻结源哈希（env/arch/vae.pt）执行前逐位验证通过。

## 8. 状态板（09-02 三十六轮收尾更新）

```text
Mode A runtime code        DONE（本机静态验证 + selftest 9/9）
Independent checker code   DONE（同上）
SCRIPT_MAP 登记            DONE（runtime=state-changing / checker=read-only）
Negative tests             PASS（本机 9/9 + lab-ts frozen env 9/9）
28-cell D dry-run          DONE（28/28 receipt，lab-ts，commit b69a2ce 入仓）
D1/D2/D3 verdict           PASS（report = sync/to41_d/D_report，failures 0）
audit artifact             已入仓（frozen 随 owner execution freeze）
owner execution freeze     PENDING（纯状态迁移，owner 裁定）
Main Rung 1 compute        BLOCKED（不变）
```
