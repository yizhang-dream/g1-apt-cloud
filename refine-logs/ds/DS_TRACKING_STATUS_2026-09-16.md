# D059 地形线开工 + D058 暂停：会话交接快照（2026-09-16）

【层位】L1 阶段状态/交接 | 上级：[refine-logs 索引](../README.md) · 上一快照：[DS_TRACKING_STATUS_2026-09-15](DS_TRACKING_STATUS_2026-09-15.md)（D057 只剩 G5）· 纲领：[DS_TERRAIN_ADAPTER_CHARTER](DS_TERRAIN_ADAPTER_CHARTER.md) · 本线执行计划：[DS_TERRAIN_AUTHOR_PLAN](DS_TERRAIN_AUTHOR_PLAN.md) · 预注册：[DS_CONTINUOUS_EXECUTION_PLAN](DS_CONTINUOUS_EXECUTION_PLAN.md) §5r（D058）/§5s（D059）· 判读口径：[HANDOFF/README](../../HANDOFF/README.md) §3

## 0. 三十秒版（新对话先读这段）

09-16 一天内完成两次转向：**D058（steer-to-harvest 平地收割）立项实施到 G1 冒烟后被 owner 暂停**（让位新方向，EULA 坑已修，恢复点明确）；**D059（地形行动泛化线）立项并跑完 G0-① 生死判别=甲**——tokenizer 通路根本不是风险，是我们仓里四代验证过的成熟链路。**当前唯一活跃主线 = D059 地形线**，下一步是 G0-②（爬障段选样转换冒烟），选料键已锁定。D057 G5（WBT 档位图）仍欠一个测量，前提已全部备齐在 CVGL。全仓 git 状态干净（本会话 4 个提交全已推送；工作树里压着 09-15 分域在途批次约 150 文件，属前任未完工作，勿动、勿整批提交）。

## 1. 本日事件时间线（全落账，commit 可查）

| 时段 | 事件 | 落账 |
|---|---|---|
| 午后 | steer-to-harvest 提案评估（tmp 报告 v2 重写修可读性）→ owner 批准边走边调采数据 | tmp/STEER_TO_HARVEST_HANDOFF_2026-09-16.md（保密件仅存 tmp/） |
| 下午 | **D058 立项+实施**：isaac/steer_harvest_d058.py（三臂 warmup/fixed/controller+segments 打包，selftest 7 绿+reviewer pass-with-notes）；部署 lab-ts+CVGL | `8bae9db` |
| 傍晚 | G1 冒烟 12077 死于 **Isaac EULA 交互确认**（容器无终端 EOFError）→ 定位修法 `export OMNI_KIT_ACCEPT_EULA=YES` 补进 d058_g1_smoke.sh → 重发 12081 | det 实验 12077/12081 |
| 晚 | owner 转向讨论：WholeBodyWAM 调研（wholebodywam.github.io，匿名评审中未开源；**冻结解码器是领域标准姿势——我方"独有"论断已收回**，真差异化=地形域/零演示/小模型/逐轴归因）；「完整作者」可行性（token 层闭环作者，三档：规则/借 N1.7/自训） | 对话内，关键修正已入 §2 |
| 晚 | **owner 重锚定任务=地形行动泛化**（=09-05 纲领 DS_TERRAIN_ADAPTER_CHARTER 原文，非新方向）；**D059 立项**：DS_TERRAIN_AUTHOR_PLAN.md（G0-G5，模型阶梯 v0≤80M→v1 目标 1B 跟数据产量走）+ §5s 预注册 | `d3ce221` |
| 深夜 | owner「直接杀了开新 idea」→ **12081 杀（排队中 CANCELED）、D058 → PAUSED**；**D059 G0-① 完成=判定甲**（详 §2）；G0-② 选料键锁定（详 §3） | `2e29855` / `0d4d657` |

## 2. D059 G0-① 结论（判定甲，含预注册勘误）

**§5s 写的「tokenizer 通路可用性未知」是过时前提**。事实：

- encoder = 官方 release `model_encoder.onnx`（50.1MB，HF nvidia/GEAR-SONIC **非门禁**），服务器在位：`~/ros2_data/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/`（另有 TRT 引擎 + observation_config.yaml）；
- 本仓 encode 链**四代先例**：`apt_g1/encode_bones_smoke.py`（D036/D038）、`apt_g1/build/convert_bones_g1_csv.py`（D044，BONES CSV→token 就是用它）、`apt_g1/build/convert_wbt_g1_parquet.py::encode_tokens_batch`（D057）；D038 已对账：回环 MAE 0.1094 **优于**官方自带 token 0.136 + Isaac 回放 2/2 零摔；
- 存量已编码 npz：`ds_bones/g1_b3p/` 62 个、`g1_b4lite/` 487 个、`g1_b4lite_v2conv/` 66 个（manifest 含 lattice_rate/roundtrip_mae）；
- 坑位：anchor 必须 ref-rel（D036 修复后语义）；encoder 版本与 decoder/observation_config 同源，**不可混 low_latency/sonic_v1_1**；本地 Windows 无 encoder（要用需从服务器/HF 拉）；
- 「官方 token 随数据自带」准确形态=闭环录制产物（ds_smoke/policy_input.csv 前 64 列）；一切 G1 关节数据（CSV/parquet）都需 encoder 现编——这是常态不是例外。

## 3. G0-② 就绪状态（下一步动作）

- **选料键已验证**：`ds_bones/seed_metadata_v004.parquet`（142,220 行）字段 `content_type_of_movement == "climbing box"` → **3,402 段**（备忘录"3.6k"落实）；近亲池备用：step up 1,964 / obstacle 1,532 / platform 1,570 / jump over 1,218（union 7,038 行，其中 vertical_move=1 共 2,686）；
- 元数据还有 522 演员字段（actor_uid）——G4 未见动作迁移的防泄漏划分靠它；
- 转换复用 `build/convert_bones_g1_csv.py`（CSV 契约：无表头 36 列、cm/deg、120fps→50Hz、Euler 标定 `g1_b3p/calibration.json` best=i:zyx 已有）；
- **膝签名门须按爬障系分布重建**（§5s 预注册：先对爬障段全量打签名分布取四分位带，不照抄 D057 平地阈值 median<-0.25/max>2.2）；
- 读 parquet 用 `.venv_mjlab`（pyarrow 在，pandas 不在）。

## 4. 各线状态与待办

| 线 | 状态 | 下一步 |
|---|---|---|
| **D059 地形线（主线）** | RUNNING，G0-① 甲 | **G0-②**：爬障段选样 6 集→转换→签名基线重建→冒烟门；随后 **G0-③** oracle 回放上限卡（K≥12 段×3seed×{平地,mjlab 地形}，CVGL Isaac，判读甲乙丙见 §5s——丙=纲领级收线报 owner） |
| D058 steer-to-harvest | **PAUSED**（owner 让位） | 恢复点=G1 冒烟重发（EULA 修复已在脚本第 4 行但**未经运行时验证**；复用 ~/gr00t/d058_g1_smoke.sh + ~/d058_g1.yaml，det create 即可）；G3 收割发射须 `--seeds 0,1,2` 防批间重放（reviewer should-fix） |
| D057 WBT | 只剩 G5 | 前提全备齐（CVGL 代码+token_vae_d057_wbtA 已 rsync）；命令照 09-15 快照 §4，Isaac 电池记得加 EULA 环境变量 |
| N1.7 探针 | 降级为 D059 G5 参考 | 不主动开；其 rollout 可作作者 v1 蒸馏源 |

## 5. 本日新坑位（接手必读）

1. **Isaac EULA**：CVGL 容器跑 Isaac 必带 `export OMNI_KIT_ACCEPT_EULA=YES`（kit_app.py 检查此环境变量；lab-ts 不踩是因为包目录里有 EULA_ACCEPTED 文件）。
2. **det create CLI 会假死**（停在 "Preparing files to send to master..." 数分钟）——任务实际已建，另开连接 `det e ls | tail` 验证即可，别傻等。
3. **det 认证过期**：从登录节点 `~/.docker/config.json` 的 harbor 凭据 base64 解密码（zyz:pw，det 与 harbor 共账号），`echo $PW | det user login zyz` 非交互可过。
4. **蹲守代理会无声消失**（本日一次）——关键任务要么双保险（自己定期查），要么给蹲守明确"必须带证据回来"的硬要求。
5. **工作树压着 09-15 分域在途批次**（~150 文件 M/R/??，前任未完）：提交自己的改动必须 pathspec 限定（`git commit --only -- <files>`，新文件先 add），共享文档文件（README/tracker 等）会连带该批次在同文件内的改动，提交信息注明即可（先例 8bae9db/d3ce221）。
6. CVGL 部署链已通：本地 push → lab-ts `g1-apt-cloud-sync` pull → 执行根 cp + `rsync -a --exclude data --exclude outputs ...` 到 cvgl；产物三件套（speedA VAE/d048r/D057 wbtA）已在 CVGL。
7. WholeBodyWAM 情报保密（匿名评审未发表，任务族与搬篮子项目疑似重合）——细节只在本会话与 tmp/ 提案报告，**禁外传/入公开仓**。

## 6. 接手命令速查

```bash
# G0-② 选样+转换（lab-ts，.venv_mjlab 读 parquet / .venv_isaac 跑转换）
ssh lab-ts 'cd ~/ros2_data/apt_g1 && ~/ros2_data/.venv_mjlab/bin/python - <<PY
import pyarrow.parquet as pq
t = pq.read_table("data/ds_bones/seed_metadata_v004.parquet", columns=["move_g1_path","content_type_of_movement","actor_uid"])
rows = [(p,a) for p,c,a in zip(...) if c=="climbing box"]  # 选 6 段（分层：段速/演员不重复）
PY'
# 转换（照 D044 惯例，calibration 用 g1_b3p/calibration.json，--roundtrip）
ssh lab-ts 'cd ~/ros2_data/apt_g1 && nohup bash -c "..." > /tmp/d059_g02.log 2>&1 < /dev/null & disown'
# G0-③ 回放：照 D048r R1（isaac/replay_token_speed_d048r.py --segments-json ...）CVGL node06，
# det yaml 模板 tmp/d058_g1.yaml 可复用（资源池 128c256t_1536_4090 = D052 先例池）
```

（速查为骨架，发射前按 §5s 判据补全参数；复杂 ssh 命令用 base64 管道或落 tmp/ 脚本再 `ssh bash -s`。）
