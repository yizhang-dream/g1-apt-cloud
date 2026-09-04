# DS 官方数据提权：SONIC 官方数据资产定位与 BONES-SEED 离线编码选项

> 【层位 L2 侧轴｜执行选项（2026-09-04 **owner 提权立项**：「立刻整理为文档并
> 提权，立刻寻找官方的数据」；资产盘点当日完成，执行待服务器会话）】↑
> `refine-logs/README.md`（扇出树根）｜上游：`DS_GAIT_MANIFOLD_PLAN.md`
> （Phase 1 采集主案，本文是它的数据放大器/备胎，不替代）、
> `LITERATURE_SURVEY_DS_MANIFOLD.md`（SONIC 论文精读 §5 = 本选项出处）｜
> 执行事实源：`tracker/D.md`｜状态：**活跃（数据定位完成，待下载+冒烟）**

---

## 0. 提权记录

- 前史：`docs/apt-rl-g1-gr00t-roadmap.md` 里程碑 3（"Download BONES-SEED G1
  data and inspect state/action format"）立项早期已列、**从未执行**。
- 09-04 SONIC 论文精读（`LITERATURE_SURVEY_DS_MANIFOLD.md` §SONIC 精读第 5 点）
  重新发现该选项，当时标注"未立项待 owner 裁定"；owner 当日裁定**提权**：
  整理成文 + 立即定位官方数据（本文 §1 = 定位结果）。

## 1. 官方数据资产清单（2026-09-04 定位）

### 1a. HuggingFace 模型仓 `nvidia/GEAR-SONIC`（36.6 GB，官方权重仓）

| 资产 | 大小 | 对本选项的意义 |
|---|---|---|
| `model_encoder.onnx` | 50.1 MB | **核心**：批量离线编码 token 的入口 |
| `model_decoder.onnx` | 40.9 MB | 已有（仓内 `gear_sonic_deploy/policy/release/`，server 侧现成） |
| `planner_sonic.onnx` | 774 MB | 本选项**不需要**（且被 Protect AI 打 PAIT-ONNX-200 标记，AV 扫描 0/74；不下载） |
| `observation_config.yaml` | 2.3 kB | B2 格式对齐的对照物 |
| `bones_seed_smpl/` | 7 部分 | **BONES-SEED 的 SMPL 格式镜像**（官方仓内） |
| `low_latency/` `sonic_release/` `sonic_v1_1/` | — | 检查点（本选项不需要，冻结底座不动） |
| `sample_data/` | 1 条 walk 序列 | **B2 冒烟起点**（官方快速起步样例） |

### 1b. 数据集 `bones-studio/seed`（BONES-SEED 本尊）

- **142,220 条标注动捕 / 288h / 522 演员 / 33 大类**（含自然语言描述与语义
  标注）；论文口径 = SONIC 611h/100M+ 帧训练数据的公开子集。
- **提供 SOMA 与 Unitree G1 双格式**——G1 重定向现成，无需自己做 retarget。
- License = `bones-seed-license`：**下载前须在 HF 页面核对是否需点同意协议**，
  引用/再分发条款进 §2 记录。
- 注意：`nvidia/GEAR-SONIC/bones_seed_smpl/` 是 SMPL 格式；**要 G1 格式去
  bones-studio/seed**，两条路径都留。

### 1c. 我方已有衔接资产（零新代码起点）

- `apt_g1/README.md:136`：server 侧已在用 `model_encoder.onnx`。
- `apt_g1/SCRIPT_MAP.md` `planner_sonic.py`：**官方三模型全栈复刻
  （规划器→encoder→decoder）已存在**，27 种 LocomotionMode 解锁关键 8 模式
  ——encoder 调用路径现成，本选项只需"喂 G1 动作进 encoder"，不需要写
  新推理栈。
- 质检机制现成：D001 token lattice 检验 + D034 Isaac oracle 回放（Phase 0
  机制，第 1 轮零迭代 PASS）直接复用。

## 2. 已知坑与边界（执行前必读）

1. **方向偏斜（仓内早有记录，`HUMAN_READABLE_COMPLETE_REPORT.md:859`）**：
   BONES-SEED 以直行/跑/舞为主，转向/横移方向天然稀少——**不能替代 Phase 1
   官方回路的 8 方向网格与过渡段采集**；它是速度段与规模的放大器。
2. **动态可行性未知比例**：动捕重定向非物理解。代理背书 = decoder 本是在
   Isaac 物理 PPO 中训练的跟踪器（真机 99.2%）；但必须走 **B3 oracle 回放
   抽检门**后才允许进数据集（复用 D034 机制，canonical env 零改动）。
3. **与 D033 数据卫生学不冲突**：D033 禁的是"planner 开环提取、无物理检验、
   context 自回归漂移"那条命令→token 路径；本选项是"官方 encoder 编码 +
   Isaac 回放检验"，走 D002/D034 质检框架，是两回事。
4. **encoder 输入格式**（论文 robot encoder 口径）：10 未来帧关节 pos+vel、
   Δt=0.1s（50Hz）、6D 旋转表示——B2 第一步拿 `observation_config.yaml` 与
   `planner_sonic.py` 的 encoder 输入逐字段对齐，先跑 `sample_data/` 冒烟。
5. **token 格式基线**：编码产物先过 D001 lattice 检验（k/16 格点），与官方
   回路 token 分布对照（mode 匹配段）再谈入集。
6. **大文件纪律**：数据只落 server `apt_g1/data/ds_bones/`（gitignored），
   仓内只进 tracker 行与本文档。

## 3. 执行设计（Phase B1–B4，预算 ~1 天，机时 <1h GPU）

- **B1 下载与盘点**（server lab-ts）：`hf download` 模板——
  `nvidia/GEAR-SONIC` 取 `model_encoder.onnx + observation_config.yaml +
  sample_data/`（<60MB）；`bones-studio/seed` 先取 G1 格式 1–2 个大类的
  抽样子集（locomotion/dance 优先，对应 G3 速度段目标）。核对 license 门。
- **B2 格式对齐 + 冒烟**：`observation_config.yaml` ↔ `planner_sonic.py`
  encoder 输入逐字段对齐；`sample_data` walk 序列 encode → token →
  decoder 回环 MPJPE sanity。
- **B3 抽检门（不设门不入集）**：每大类抽 ≥10 段 ×500 步 → Isaac oracle
  回放（D034 `AptFlatG1Env` 子类），记存活率 + realized 速度 vs 参考轨迹
  速度；类级存活 ≥95% 才准入（阈值可在首轮抽检后校准）。
- **B4 数据集构建**：通过门的大类全量编码 → `apt_g1/data/ds_bones/` npz
  （token/参考关节/类别标注/时长）；与 Phase 1 官方回路数据的合并策略
  三选一留 Phase 3 前裁定：①纯对照臂（bones-only 重训 VAE 看 held-out
  差异）②合并加权（bones 作速度段补充，官方回路作稳态/过渡/方向主料）
  ③探针（仅作流形内插验证材料）。
- **登记**：`planner_sonic.py` 已在 SCRIPT_MAP；若新写编码批处理脚本
  （如 `encode_bones_tokens.py`）须登记 SCRIPT_MAP。

## 4. 与 DS_GAIT_MANIFOLD_PLAN 的关系

- **不替代 Phase 1**：官方回路 4 族采集（~2.5h）仍是主案——它是唯一有
  物理检验的稳态+过渡段+方向网格来源；计划 Phase 0 已 PASS、Phase 1 解禁
  状态不变。
- 本选项定位三重：①G3 速度段放大器（BONES-SEED 含跑/舞高速材料，绕开
  官方回路 RUN 48% 执行衰减的数据侧问题）；②规模放大器（离线编码 vs 1:1
  实时，量级差 2–3 个数量级）；③Phase 1 工程受阻时的备胎。
- 时序建议：B1–B3 与 Phase 1 并行（B1 下载不占机时）；B4 的合并策略裁定
  排在 Phase 3 VAE 训练前。

## 5. 立即行动清单（下一服务器会话）

1. B1：下载 `model_encoder.onnx + observation_config.yaml + sample_data/` +
   bones-studio/seed G1 格式抽样（先 locomotion 大类）；记录 license 核对结果。
2. B2：格式对齐检查 + sample_data 冒烟 encode/decode 回环。
3. B3：首批 oracle 回放抽检（locomotion 类 ≥10 段）。
4. 回写：本文修订记录 + `tracker/D.md` 新 D 行（D035 起）。

## 6. Sources

- HF 模型仓：https://huggingface.co/nvidia/GEAR-SONIC （encoder 50.1MB /
  decoder 40.9MB / bones_seed_smpl ×7 / sample_data）
- BONES-SEED 数据集：https://huggingface.co/datasets/bones-studio/seed
  （142,220 条 / 288h / 522 演员 / SOMA+G1 双格式 / bones-seed-license）
- SONIC 论文：arXiv 2511.07820 = Science Robotics 11(117) eaed4592 (2026)
- 仓内前史：`docs/apt-rl-g1-gr00t-roadmap.md`（里程碑 3）、
  `docs/apt-rl-flat-ground-g1-repro.md`、`refine-logs/HUMAN_READABLE_COMPLETE_REPORT.md:859`

## 7. 修订记录（append-only）

- **09-04/05 夜 B1-lite + B2 执行回写（D036/D037，详见 tracker）**：
  ①**连通性**：huggingface.co 服务器直连不通，`hf-mirror.com` 可达（B 线下载一律走
  镜像 resolve URL）；服务器无 hf CLI，curl 直下可用。②**模型件零下载需求坐实**：
  HF root 的 `model_encoder.onnx`/`observation_config.yaml` 与服务器 deploy release
  **md5 逐字节全等**。③**license 门确认**：`bones-studio/seed` gated(auto)——需
  owner HF 账号在数据集页接受 bones-seed-license 并提供 read token 后方可下载；
  G1 格式为单体 `g1.tar.gz`（全量 142k csv 打包，无法部分下载），Locomotion 大类
  74,488 条（52%）。④**B2 冒烟三检验**：lattice 0 违例（encoder 内置 FSQ）；
  std 与官方一致但 mean-L2 3.16；回环 MAE 0.564 vs 官方 token 同 harness 0.136
  → **判别出编码路径真 bug**：锚定朝向块沿用 planner_sonic 的 apply_delta，样例
  初始 yaw −87° 时全部锚定向量被注入 ~90° 常量偏移（planner_sonic 站立起点
  apply_delta≈identity 掩盖了缺陷）；正确离线语义 = `btr = conj(bq[t])·bq[idx]`。
  ⑤**B3 预演 v1**：链路机械全通，2 seed ~1s 侧倒（yaw 类失败形态与 bug 三重
  印证）；新坑 = 本机 Isaac `sim_app.close()` 挂死，daemon 线程模式已回移
  canonical `oracle_token_replay_isaac.py`。⑥修复验证（ref-rel 锚定重编码 +
  回环 + Isaac 重跑）= **D038**，因 lab-ts 网络中断（Tailscale 直连断、hkg
  中继劣化）暂挂，恢复后执行。⑦新脚本登记：`encode_bones_smoke.py` /
  `roundtrip_official_control.py` / `isaac/replay_bones_tokens_isaac.py`。

- **09-05 凌晨 D038 修复证实 + 数据源改道 SMPL**：
  ①ref-rel 修复全指标逆转（详见 tracker D038）：回环 MAE 0.1094（优于官方
  token 0.136）、mean-L2 0.869、Isaac 2/2 全程零摔物理量复现 97%——
  **B2 编码路径关闭，B 线链路全通**。②**gated 通道受阻**：owner HF 登录
  失败；实测 hf-mirror 对 gated 仓库仅放行 README（200），实质文件 403——
  门禁在 HF 服务端执行，镜像无法绕过；OpenDataLab/ModelScope 查证均无
  主数据集镜像（ModelScope 仅 nv-community 衍生时长标注）。③**改道
  SMPL 官方镜像（零门禁）**：`nvidia/GEAR-SONIC/bones_seed_smpl/` 7 分卷
  32.3GB = BONES-SEED 的 SMPL 格式官方拷贝，encoder 原生 smpl 模式
  （observation_config encoder_modes[2]；`sample_data/smpl_filtered` 样例
  已在手：pose_aa 72 维 + transl + smpl_joints 24×3 @50Hz），编码产物同
  为 token，下游 decoder/Isaac 检验链路零改动——**G1 格式门禁不再是 B 线
  阻塞项**（owner 登录成功后可补 G1 通道作对照）。下载 09-05 启动
  （`data/ds_bones/b1_smpl/`，断点续传后台）。④下一步 = SMPL 模式 obs
  布局逆向（对照 g1 路径同法：官方样例字段 → 1762 维布局 introspection →
  冒烟三检验）→ B3 正式门（先 metadata 类目抽样、每大类 ≥10 段）。
