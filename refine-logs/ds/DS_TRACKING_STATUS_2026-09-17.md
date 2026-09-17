# D059 G0 生死判别收束=甲：会话交接快照（2026-09-17）

【层位】L1 阶段状态/交接 | 上级：[refine-logs 索引](../README.md) · 上一快照：[DS_TRACKING_STATUS_2026-09-16](DS_TRACKING_STATUS_2026-09-16.md) · 纲领：[DS_TERRAIN_ADAPTER_CHARTER](DS_TERRAIN_ADAPTER_CHARTER.md) · 执行计划：[DS_TERRAIN_AUTHOR_PLAN](DS_TERRAIN_AUTHOR_PLAN.md)（v2，G0 已收束） · 预注册：[DS_CONTINUOUS_EXECUTION_PLAN](DS_CONTINUOUS_EXECUTION_PLAN.md) §5s · 判读口径：[HANDOFF/README](../../HANDOFF/README.md) §3

## 0. 三十秒版（新对话先读这段）

09-17 一天内跑完 D059 剩余两步并全线收束：**G0-② 转换冒烟 PASS**（12 段爬障语料，roundtrip 全优于 B4-lite 现役带；膝签名门两轮口径——字面四分位带门 2/12 判「语义错配」照 D057 G2 先例重锚 envelope 门 12/12）；**G0-③ oracle 回放判甲**（det 12132 双场景 72 run：门一存活 24/36=2/3 恰达冻结线、门二 realized 段级中位 1.20/剥离摔段 1.05 双超 R1 带 P25 0.428）——**地形线生死判别通过，全线开，下一步 = D060 数据发动机预注册（四源）**。甲下登记发现：4/12 快爬段（come_up）平地 3/3→rough 0/3 系统性早摔 = 地形条件稳定性边界，是 D060 作者纠偏预算的实证动机。本会话提交 dc84250（已推送）；工作树仍压 09-15 分域在途批次 ~150 文件，勿动勿整批提交。

## 1. 本日时间线（全落账）

| 时段 | 事件 | 落账 |
|---|---|---|
| 午前 | owner「开始」→ G0-② 执行：coder 选样 6 段冒烟（首轮）+ Explore 调研 G0-③ 前置（fork 接口/D030 地形不可序列化/R1 带数值/CVGL 模板） | 对话内 |
| 午前 | 首轮膝签名四分位带门 2/6 → 判读=门语义错配（联合带接受域仅 8.2%，度量典型形态非转换正确性）→ 照 D057 G2 454ab84 先例重锚 envelope 门；扩展 12 段+segments-json | tracker D059 行 |
| 午后 | fork `isaac/replay_token_terrain_d059.py`（--terrain+直度+无 vae_inputs 守卫+D059 身份串）；双 reviewer pass-with-notes；should-fix 全收（含门报表 None 安全+身份串+docstring 计数） | commit `dc84250` |
| 午后 | 判读两门制冻结（发射前）：门一存活≥2/3（≥24/36）∧ 门二段级 realized_med 中位≥R1 带 P25=0.428（R1 实测=段级中位 0.4665/存活 0.833）；跨池附则+偏离落账（mjlab D030→rough_paper） | tracker D059 行 |
| 12:15 | CVGL 部署（lab-ts sync pull→执行根 cp→rsync cvgl）+ 冒烟 det 12131 PASS（**EULA 修复首次运行时验证**、rough_paper 生成实跑、首 run 存活） | §4 命令 |
| 12:19-12:39 | 正式 det 12132 双场景串行 72 run，rc 0/0；watcher 收产物+算门 → **终判=甲** | §2 数值 |

## 2. G0-③ 判甲数值卡（判读对照，勿重复计算）

| 组 | 存活 | 段级 realized_med 中位 | 说明 |
|---|---|---|---|
| flat（plane，对照） | 35/36 = 0.972 | 1.0947 | 爬障剧本平地可演（8/12 come_up 段 ratio>0.9）=内容基线强 |
| rough_paper（判读对象） | **24/36 = 2/3 恰达冻结线** | **1.2001**（剥离 4 全摔段后存活 8 段中位 **1.0459**） | 两门全过；段级结构双峰干净（8×3/3 + 4×0/3，无部分存活段） |
| R1 平地带（参照） | 30/36 = 0.833 | 段级中位 0.4665，P25=0.428 | D048r 快段（v_med≥1.0）；爬障段分母 0.044-0.209 小 5-20 倍，跨池附则在案 |

**甲下登记发现**：4/12 快爬段（A062/A324/A316/A393，全 come_up）平地 3/3→rough 0/3 系统性早摔（fall_step 116-192 ≈ 2.3-3.8s，前冲 vx_med 0.95-1.36 把 ratio 撑到 4.9-10.5）=**地形条件稳定性边界**。机制=假设（±0.04 轻度粗糙不稳快爬步态；decoder 输出本身完好看平地臂），不外推。用途：D060 语料覆盖设计 + 作者纠偏预算的实证动机。

## 3. 各线状态与下一步

| 线 | 状态 | 下一步 |
|---|---|---|
| **D059 地形线** | **G0 收束=甲，D059 DONE** | **D060 数据发动机预注册**（DS_TERRAIN_AUTHOR_PLAN §6 四源：planner 命令标注生成/ds_bones 爬障 3.4k〔select_climb_d059.py 可复用，--per-tier 可扩〕/官方平地库/闭环收割；产量门=帧数×命令覆盖×地形覆盖）→ G2 作者 v0 ≤80M=D061 |
| D057 WBT | 只剩 G5 | 前提全备齐 CVGL（09-16 快照 §4 命令仍有效；EULA 变量必带） |
| D058 | CLOSED（09-17 owner） | 无 |

## 4. 接手命令速查

```bash
# G0-③ 产物（cvgl 主机 10.0.1.67 = cvglloginnode，非 lab-ts！）
ssh cvgl 'ls ~/gr00t/apt_g1/outputs/d059_terrain/'   # flat_rc.txt/rough_paper_rc.txt/两 log/两 replay_runs.json
# 本地镜像已拉回：tmp/d059_g03_results/（含 g03w_metrics.json 机器可读全量）
# 12 段语料：lab-ts ~/ros2_data/apt_g1/data/ds_bones/g1_d059_climb_smoke/（npz×12+manifest+select/gate/segments json）
# 选样复跑（D060 扩量）：select_climb_d059.py --per-tier N --segments-json-out ...（.venv_mjlab；gate 判读用 --gate-mode envelope）
# 转换复跑：convert_bones_g1_csv.py --list climbN_list.txt --calibration-json data/ds_bones/g1_b3p/calibration.json \
#   --out-dir ... --roundtrip --fresh-manifest --force   # .venv_mjlab（不是 venv_isaac！）
# CVGL Isaac 任务模板：tmp/d059_g03_main.yaml + ~/gr00t/d059_g03_main.sh（EULA 第 2 行级必须）
```

## 5. 本日新坑位（接手必读）

1. **det master/NAS 产物在 cvgl 主机（10.0.1.67，user zyz，cvglloginnode）非 lab-ts**——lab-ts 无 det 无 ~/gr00t；det CLI 直接 `ssh cvgl 'det ...'`（DET_MASTER=10.0.1.66:8080）。
2. **转换+roundtrip 须 `.venv_mjlab`**（venv_isaac 无 mujoco）——09-16 快照 §6 写反，以此为准。
3. 服务器执行根平铺无 build/ 子目录（09-15 分域只在本地仓）：build/ 下新脚本部署到执行根顶层；isaac/ 子目录服务器存在。
4. ssh 坑（本日实测三次）：非 heredoc 的 ssh 命令加 `-n` 防吞本地 stdin；但 `-n` 与 `bash -s <` / heredoc 互斥，二选一；bash 内 `$(cmd || echo <pending>)` 尖括号被当重定向。
5. det create 假死复现（"Preparing files... 0.0B"）——任务已建，另连 `det e ls | tail` 验证；det 状态与 rc 不可信，以 NAS 落盘 JSON 为准。
6. EULA 修复（`export OMNI_KIT_ACCEPT_EULA=YES`）**已通过 det 12131 冒烟运行时验证**——不再是"未经验证"状态。
7. 爬障系资产：膝签名 envelope 带门标准参照=tracker D059 行四分位表（n=3,402）；converter `class_of()` 对爬障 stem 落 class="come"（下游 class 聚合注意）；band/envelope 双 gate 默认同目录互踩 segments-json（显式 `--segments-json-out`）。
8. CVGL 4090 池当日无排队，G0-③ 全程 20min（72 run）；3090 池仍勿用（启动卡死三度坐实）。

## 6. 提交与产物索引

- commit `dc84250`（已推送四端）：select_climb_d059.py 入仓 + replay_token_terrain_d059.py fork 入仓 + SCRIPT_MAP 两行 + tracker D059 G0-② 落账；G0-③ 结果落账在本快照同批提交。
- 产物：NAS `~/gr00t/apt_g1/outputs/d059_terrain/`（md5：flat JSON `c6289e46…`/rough JSON `7a0ca8c4…`）；lab-ts `data/ds_bones/g1_d059_climb_smoke/`（segments json md5 `321c0f82…`）；本地 `tmp/d059_g03_results/`、`tmp/d059_artifacts_v2/`、运维脚本 `tmp/d059_g03_*.sh`（复现记录，保留）。
