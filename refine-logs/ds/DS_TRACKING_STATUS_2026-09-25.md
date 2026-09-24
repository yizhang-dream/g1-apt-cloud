# D067 residual 八层防御栈 + r5 在途：会话交接快照（2026-09-25）

【层位】L1 阶段状态/交接 | 上级：[refine-logs 索引](../README.md) · 上一快照：[DS_TRACKING_STATUS_2026-09-18](DS_TRACKING_STATUS_2026-09-18.md) · 纲领：[DS_TERRAIN_ADAPTER_CHARTER](DS_TERRAIN_ADAPTER_CHARTER.md) · D067 预注册：[DS_CONTINUOUS_EXECUTION_PLAN](DS_CONTINUOUS_EXECUTION_PLAN.md) §5y · 判读口径：[HANDOFF/README](../../HANDOFF/README.md) §3

## 0. 三十秒版（新对话先读这段）

owner 09-23 目标改写（视觉→自主规划→**最快直行**，指令轴降级——D067 奖励移除
`track_lin_vel_xy_exp`/`track_ang_vel_z_exp` 即其落地）后全线执行况：

1. **D066 结案**：A0 前置门 C2/C3 PASS + C1 保真 PASS（位移判据被目标改写取代），
   §5w 发射锁解除。
2. **D062-R1 四臂+三轴电池毕**，两大发现：**规模轴转正**（四臂 top1 0.6620–0.6638
   全超 v0 0.6474）+ **行为层剧分化**（val 层四臂极差 0.0018 零分化，闭环却
   anchor 摔/chunk 高速+冻结双态/priv 唯一双稳/both 近静止）——token 源选择成为
   显式决策项。
3. **D067 三臂毕其二**：direct vs decoder 5000it×1024envs×max_forward，
   **mfwd 主奖励 0.449 vs 0.067 ≈ 7×**——token 通路代价的受控量化（同资产同配方
   唯一变量=动作通路）。
4. **residual 臂连闯四关在途**：OOM 阶梯→v0 模型 embedding_bag 64× 内存修复
   （d7194c9）→bf16 性能档（9f37617）→33% update 空转根因坐实→第八层 obs 源消毒
   （773cccb）；**现 r5 干净重跑 nohup 自持在途**（lab-ts，~57s/it，
   iter1250 里程碑≈09-25 22:15，5000 全量≈09-27 傍晚）。
   **接手第一步见 §2（核心节）。**

本会话代码侧落账 = commits `5d8afe2→773cccb`（八层防御栈，HEAD=773cccb 已推送）；
工作树仍压 09-15 分域在途批次，接手者勿整批 sweep（本次提交只动两文件，见 §7）。

## 1. 各线状态与下一步

| 线 | 状态 | 要点 |
|---|---|---|
| D066 A0 前置门 | **DONE·结案** | C2 四族地形特权可分 / C3 decoder md5 不变 / C1 保真 PASS（位移判据被 09-22 目标改写取代）；发射锁解除 |
| D062 作者 v1 | **R1 毕**（轴B 未跑，新目标下另定口径） | 规模轴转正+行为层剧分化两大发现；chunk@0.6 的 0.78 m/s 是全阵最高持续速度但 1.0 档冻结，intent-pin-max 语义存疑 |
| D067 三臂 | **direct/decoder 毕；residual r5 在途** | 干净二臂表：mfwd direct 0.449 / decoder 0.067（≈7×）；residual 待 r5 出数（§2 判读口径） |
| D065 LoRA 臂 | G0 PASS，G1 已撤回待顺位 | yaml 留 cvgl `~/d065_yaml/` + 本地 `tmp/d065_g1_lora_s{0,1}.yaml`，原样重发即可；算力让位 D067 |
| D064 底座 | DONE | 双因素实锤（接口机制瓶颈+配方规模伪影）；同栈八层修复推广待立项（§5） |

## 2. 在途与接手（核心节）

**residual_r5：lab-ts 服务器侧 nohup 自持**（与本对话解耦，会话切了不影响训练）。

- 进程：`train_g1_decoder_residual.py --arm residual`，pid 落
  `/tmp/d067_r5_train.pid`；nohup 日志 `/tmp/d067_residual_r5_nohup.log`。
- 输出目录：`~/ros2_data/apt_g1/outputs/d067_3arm/residual_r5/`。
- 速度：~57s/it（首段实测）；里程碑 **iter1250 ≈ 09-25 22:15**、
  **5000 全量 ≈ 09-27 傍晚**（以 train.log 实际为准）。
- 配置：1024envs×5000it×max_forward×seed0，token 源=author v0
  （ckpt md5 `00fbaed5…`），八层防御栈全开（`--obs-sanitize` 默认开）；
  发射信封=本地镜像 `apt_g1/outputs/sync/d067_3arm/residual/run.json` 同款
  （v2 格式，git_head b3b4042 起，r5 为 773cccb 码上重跑）。

**接手第一步命令**（判定存活与进度）：

```bash
# ① 进程在不在
ssh lab-ts "pgrep -af train_g1_decoder_residual"

# ② 训练进度（末行 Learning iteration = 当前 iter）
ssh lab-ts "tail -5 ~/ros2_data/apt_g1/outputs/d067_3arm/residual_r5/train.log"

# ③ 消毒/空转轨迹（purged_full、obs_sanitize 计数随 iter 的走向）
ssh lab-ts "tail -30 ~/ros2_data/apt_g1/outputs/d067_3arm/residual_r5/std_trajectory.log"

# ④ 若进程不在：先看崩没崩
ssh lab-ts "tail -40 ~/ros2_data/apt_g1/outputs/d067_3arm/residual_r5/main_loop_crash.log 2>/dev/null; tail -30 /tmp/d067_residual_r5_nohup.log"
```

**判读口径**（r5 出数后照此判，勿重复发明）：

1. **治本判读**：iter 500/800/1250 三点的 update 空转率（purged_full 行）
   对比 **r4 的 33%**——第八层 obs 源消毒把 −Inf 挡在 env 出口后，purged_full
   **应趋零**；若仍 33% 量级=消毒层没接住，回 §3 第八层排查。
2. **干净三臂表**：mfwd 主奖励 direct **0.449** / decoder **0.067** /
   residual r5 = **?**（r4 pilot 参照：33% handicap 下 mfwd 仍达 0.32–0.48
   ≈ direct 水平=救援信号已现；r5 干净跑应只更好）。
3. 判读完成后按仓库规则落账：tracker `D.md` D067 行追加 + 本快照标记收束。

**盯守**：本会话派出的 watcher 属旧会话（旧会话 watchers 不跨会话存活），
**新会话需重派 watcher 盯守**（轮询 §2 第一条命令三件套即可）。
发射脚本可复用：本地 `tmp/launch_d067_r3_v2.sh` 改 OUT 路径即可
（r5 实际发射脚本=同模板 `tmp/launch_d067_r5.sh`，含 md5 门+防重入+OOM 归档逻辑，
服务器 /tmp/ 同名）。

## 3. 八层防御栈清单（`train_g1_decoder_residual.py` 内，commits 5d8afe2→773cccb）

| 层 | 根因（一句） | 修法 | commit |
|---|---|---|---|
| 1 close 看门狗 | Isaac `close()` timeline 忙等死锁（`simulation_context.py:743` stop 回调无限 render），崩时假挂死吞异常 | safe_close 看门狗：daemon 线程 join 120s，超时 dump 全线程栈后 `os._exit` | `5d8afe2` |
| 2 σ 净化 | PPO σ 可训参数在 max_forward 下漂移瞬变 NaN（`normal.py:73 std>=0.0` 即崩；clamp 对 NaN 无效） | `where(isfinite, clamp(min=1e-4), 1e-4)` 保命带（夹紧仪器+σ 轨迹漂移日志=`922bf23`） | `250b7b1` |
| 3 update 探针 | update 内单步 NaN 无从定位（哪一步、哪个张量先坏） | 分步只读探针：pre_mid/loss_dict 逐 key/参数前后快照，定位=σ 29 维瞬变 | `4fb72d8` |
| 4 GAE 消毒 | advantage 归一化对单 NaN 零免疫（env 单点 NaN→critic value NaN→GAE 扩散→归一化放大全灭）；且 inference_mode 下就地写回被禁 | 消毒器 nan_to_num **重赋值**路径+storage.values 三张量全覆盖 | `4ad269d`+`3ca46e5` |
| 5 观测端消毒 | critic 用 critic_obs 现算重算路径 values 消毒管不到；env 单点 NaN obs 才是根源（env507/env495 两处确定性复现） | 消毒器上移观测端五路 attr，GPU 侧 isfinite 归约（大张量不拷贝） | `b3b4042` |
| 6 分块 adapter | author adapter 全 batch 完整前向单次分配 3.75GiB（1024envs×200帧×64×768），叠 Isaac 驻留必爆 12G | `--adapter-chunk-envs` env 维分块+window_len 全局 max 钉桩 | `5dd54f0` |
| 7 bf16 性能档 | adapter fp32 下限 ~80s/it、chunk128 实测 179s/it（10 天不可行）；790 TFLOPs/iter 结构性成本 | `--adapter-dtype bf16` 默认+chunk 128→1024，logits 边界升 fp32 保 argmax 口径 | `9f37617` |
| 8 obs 源消毒 | rollout 期 height_scan −Inf 源头=暴烈动作下 RayCaster 特定射线槽位打空（r4 实证 418 事件簇状分布，direct 0.055/it vs r4 0.33/it=6× 烈度放大），33% update 空转对 residual 不公平 | 动态子类包装 env.step/reset，obs dict 逐 key GPU nan_to_num（±inf→±1e4 保号）+计数入 std_trajectory | `773cccb` |

注：另有 v0 模型侧内存修复 `d7194c9`（fix(d062)，`emb(idx).mean(dim=2)`→
`F.embedding_bag`，64× 缩减）——属生产件首改，登记在发现①不在上表。

## 4. 关键发现登记（本次会话四条）

1. **v0 模型 `emb.mean` 64× 内存浪费**：`encode()` 两处
   `emb(idx).mean(dim=2)` 中间张量 (N,t,64,768) 满窗 chunk1024 达 37.5GiB
   任何卡不可行；`F.embedding_bag(mode=mean)` 数学恒等修复（实测峰值
   1220→27.7MiB=44×）；state_dict 零变化 ckpt 兼容，轴A/C 已回归全绿。
   对一切 author 消费普适（`d7194c9`）。
2. **rsl_rl advantage 归一化对单 NaN 零免疫**：单个 env 的 NaN 观测经
   critic→GAE 扩散→归一化放大 24576 全灭→梯度全染参数全毁。这是栈级
   脆弱点，非 residual 特有（`4ad269d`/`3ca46e5` 免疫接种）。
3. **height_scan −Inf 源头坐实**：暴烈动作下 RayCaster 特定射线槽位打空；
   418 事件全部落在 critic 组无噪 height_scan 段 [99,286)，簇状分布
   （99 列×84 / 115 列×40）判别坐实=物理压力假说（bf16 假说排除）；
   direct 同族 0.055/it vs residual 0.33/it=动作烈度 6× 放大（`773cccb`）。
4. **residual 救援信号已现**：r4 pilot 在 33% update 空转 handicap 下
   mfwd 仍达 0.32–0.48 ≈ direct 水平（0.449）——decoder(token)+r(s) 路线
   未被否，r5 干净跑是终判。

## 5. 待裁定/尾项池

- **r5 完成后**：三臂终判读（§2 口径）+ **是否 20000it 对齐 D064 步数**
  （现三臂 5000it=123M 环境步自洽对照，D064 对齐版留后续）→ 报 owner 裁。
- **D064/D065 同栈八层修复推广**：`train_g1_decoder.py` 等 D064 栈与
  D065 C 臂同款裸 close/σ 隐患，八层移植待立项（tracker D067 尾项在案）。
- **token 源 v1 变体候选**：D062-R1 电池后 token 源=v0 起步的预注册口径
  不变，但 v1 chunk（高速 0.78 m/s 带 1.0 档冻结双态）等变体是显式候选，
  待 r5 三臂表齐后一并裁。
- 两小尾巴：训练器 `--help` 触发 % 格式化崩溃；`judge_gate` c2 仍旧口径与
  停步判据分叉（`5aa0885` 登记）。

## 6. 坑位清单（接手必读）

1. **算力格局（当前唯一解）**：cvgl 4090 **禁用**（owner 09-23 指令只用
   3090/lab-ts；D067 两臂两投两卡挂死实证）→ 3090 池 Isaac 启动卡死前科
   （第四度坐实，12369 冒烟 0 iters）**不用** → **lab-ts 3060 唯一可用**，
   1024envs 峰值 3.9GB（3981/3945MiB 两次实测）。别再往 cvgl 投 Isaac 任务。
2. **sync 克隆缺 data/ 子树**：`~/ros2_data/g1-apt-cloud-sync` 是 git 克隆，
   `data/` 被 gitignore 不在树里——数据仍以 `~/ros2_data/apt_g1/data` 为准，
   别在 sync 克隆里找语料。
3. **执行根平铺快照坑**：`/tmp/run_apt_isaac.sh` 包装的执行根是平铺快照
   （缺 `training/` 等子树）；依赖包内模块的脚本（轴B 类、isaaclab 栈入口）
   必须从完整包根跑，平铺根会静默走兜底或解析错根。
4. **CRLF/管道退出码老坑**：Windows 侧改 shell 脚本防 CRLF 混入；服务器
   tee 管道吞非零退出码（已加 pipefail），判读只认日志行（如 SMOKE 行、
   Learning iteration 行）不认 rc。
5. 本机无训练 venv（workspace AGENTS 约定）：一切训练/评测经 ssh 到服务器；
   复杂命令 base64 管道传，防本地引号解析破坏。

## 7. 提交与产物索引

- **本次改动**：本快照 + `refine-logs/README.md` 登记行（pathspec 两文件提交，
  工作树在途批次勿 sweep）。
- **服务器在途**：`lab-ts:~/ros2_data/apt_g1/outputs/d067_3arm/residual_r5/`
  （train.log / std_trajectory.log / run.json；崩则 main_loop_crash.log）。
- **本地 sync 镜像**：`apt_g1/outputs/sync/d067_3arm/{direct,decoder}/`
  （run.json 发射信封 + std_trajectory.tail200.log，5000/5000 收尾全健康）、
  `…/residual/`（run.json + main_loop_crash.log=OOM 阶梯证据）。
- **代码**：八层防御栈 commits `5d8afe2→773cccb`（HEAD 已推送，
  origin/main 同步）；r4/r5 运行均在 `773cccb` 码上。
- **前序快照**：[09-18](DS_TRACKING_STATUS_2026-09-18.md)（特权缺口/DSMS），
  本快照期间其「当前接手入口」地位由本文接管。

## 修订记录（append-only）

- **2026-09-25 初版**：owner 09-23 目标改写后全线况交接——D066 结案/D062-R1
  毕/D067 二臂毕+residual 八层防御栈（5d8afe2→773cccb）/r5 干净重跑在途
  （接手命令与判读口径见 §2）。
