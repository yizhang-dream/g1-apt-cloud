# 训练栈提速专题：VAE 管线 + Isaac 热循环（语义不变验收制）

> 【层位 L2 侧轴｜专题（2026-09-05；触发 = LoongForge 百度百舸开源框架调研后
> owner 指令「开始动手，直接先修复并写文档记录」）】↑ `refine-logs/README.md`
> （扇出树根）｜数据：`tracker/D.md` D039｜代码：commit `8fa9982`（4 文件
> 185+/43-）。

状态：**活跃**（第一波已收束：D039 等价性全 PASS / 提速全零 / 两条预算口径
下修 / 默认值按实测回改；第二波候选重排见 §6）

## 1. 背景与动机

调研 [LoongForge](https://github.com/baidu-baige/LoongForge)（百度百舸全模态
训练框架，embodied 子系统对 VLA 声称 1.8–4.4× 训练吞吐加速）后的结论：其手段
中**单卡尺度可移植的是** torch.compile / CUDA Graph、fused 优化器、I/O 预取与
数据驻留、kernel 融合；**不可移植的是** 多卡分布式（HSDP/EP/多机重叠）。对
本项目（lab-ts 单卡 3060 12G、8C/16T）正确的路线不是引框架，而是清理自研栈里
的同类问题。代码审计发现两大病灶：

- **Isaac 热循环每控制步 CPU↔GPU 同步**：`_proprio_np()` 无条件把 930 维历史
  `.cpu().numpy()`（latent 模式下纯死代码）、VAE token 解码后 GPU→CPU→GPU
  往返、`rew.mean().item()` 逐控制步同步、PPO 每 minibatch ~6 次 `.item()` +
  逐参数 NaN 检查——policy 只是 2×256 MLP，计算喂不满 GPU，时间全耗在同步上。
- **VAE 训练（e39，~1h/30ep）小 kernel 风暴**：每 batch 4 次 fwd/bwd（主更新 +
  3 步对抗头）× 0.3M 参数小模型，数据集 17MB 本就全驻 RAM，瓶颈在 kernel
  发射而非数据或算力。

## 2. 第一波改动清单（commit `8fa9982`，2026-09-05）

### 2.1 VAE 线（`apt_g1/train_token_vae_e39.py`）

| 改动 | 内容 | 为什么快 | 为什么语义不变 |
|---|---|---|---|
| 全显存数据 | 数据集整体 `.cuda()`，DataLoader `num_workers=0, pin_memory=False` | 消掉 host↔device 逐 batch 搬运 | RandomSampler 的索引序列只由全局种子决定，与 workers 数无关 → batch 组成逐轮不变 |
| torch.compile | VAE 主模型 + 两个对抗头套 `torch.compile`（`--compile-mode`，默认 default） | 小 kernel 融合 | 只包训练循环内前向；state_dict/最终统计仍走原始模块（`vae.pt` 格式不变） |
| fused 优化器 | AdamW/Adam 全部 `fused=True`（老 torch 自动回退） | 优化器单 kernel | kernel 级数值重排，步数/超参/损失式不变 |
| 测试安全旗标 | `--epochs`（默认 30）/`--out-dir`（默认原目录） | — | 默认值与原硬编码一致；短训可重定向沙箱，不再威胁生产 `vae.pt` |
| `[SPEED]` 日志 | 每 epoch 墙钟/吞吐 + 训练结束总结行 | — | 纯诊断输出 |

逃生旗标说明：初版（commit `8fa9982`）默认全开；**D039 实测后默认值回改为
原始行为**（三项优化转 opt-in，后续 commit），理由见 §4.1。

### 2.2 Isaac 线（`apt_flat_env.py` / `ppo_core.py` / `train_apt_isaac.py`）

| 改动 | 内容 | 为什么快 | 为什么语义不变 |
|---|---|---|---|
| proprio 惰性化 | `_proprio_np()` 从 `_compute_q_des` 顶部无条件调用改为仅在 2 个真实消费分支（phase anchor 子分支 / router else 分支）计算 | latent 分支（现役主线）每步省一次 930 维 GPU→CPU 搬运 | 纯读函数、无 RNG 消耗，死分支跳过不改任何张量值；oracle replay 子类整体覆写 `_compute_q_des`，不受影响 |
| token 解码去往返 | latent 分支 decode 结果保持 GPU tensor 直送 `_decoder_obs_parts`（该函数放宽为 tensor/numpy 双入口） | 消掉每控制步 GPU→CPU→GPU | 往返只是 memcpy，同 dtype 同值逐位不变 |
| 奖励统计张量化 | `rew.mean().item()` 逐控制步 → `rew_sum += rew.sum()` GPU 累积，iter 末一次 `.item()` | 每 iter 少 T-1 次同步 | 均值口径不变（T×N 全体算术均值，env 数恒定） |
| NaN guard 单次同步 | 逐参数 `isfinite` → `torch._foreach_norm` 合并后一次 `isfinite().all().item()` | 每 minibatch N 次同步 → 1 次 | 梯度含 NaN/Inf ⟺ 其范数非有限；guard 在 clip 之后，有限梯度已缩到 max_norm 内无假阳性 |
| PPO 统计张量化 | 每 minibatch ~6 次 `.item()` → GPU 张量累积，epoch 末每 key 一次 | 每 epoch 少 ~6×minibatch 数次同步 | 均值定义逐 key 不变 |
| `--fused-adam` | PPOTrainer 加 `fused` 参数，**默认 False**，`--fused-adam` 开启 | （可选）优化器单 kernel | 默认关 = 与 E 系历史 run 严格可比 |
| `[SPEED]` 日志 | `--speed-log-interval`（默认 50）打 iter 墙钟/it/s | — | 纯诊断输出 |

注：`server_*.py` 三件 FORK **未同步改**（分叉警示，勿误合并）；`_build_commands_list`
逐 env 循环、`batched_router` CPU 路径、gate 边界 `cpu().tolist()` 属第二波。

## 3. 验证协议（预注册，避免事后挑口径）

- **等价性判据**（compile/fused 属 kernel 级浮点重排，不要求逐位相等）：
  - VAE：fast vs slow（slow = 全逃生旗标，即原计算路径）各 10 epoch，逐 epoch
    `tr_rec_mse / va_mse / kl / dir_acc / spd_acc` 对齐 = 末 epoch va_mse 相对
    偏差 <5% 且曲线同走势无系统性偏移；
  - Isaac：new vs old × {latent（128 envs × 100 iters，E27 模板旗标）、router
    （128 envs × 60 iters）}，同 `--seed 0`，old 用 git pull 前快照经
    PYTHONPATH 前置隔离运行；判据 = 逐 iter mean_rew/fall_rate 曲线同走势、
    无系统性偏移（Isaac 本身确定性良好，D034/D035 三 seed 逐位复现）。
- **速度判据**：VAE 以 `[SPEED]` 每 epoch 墙钟（快路径首 epoch 含 compile 预热，
  以稳态 epoch 计）；Isaac 以逐 iter dt 稳态中位数计。
- 生产保护：一切测试输出走 `*_d039_*` 沙箱目录；`outputs/token_vae_e39/`
  （TO42 在用）与 E 系 ckpt 零接触。

## 4. 验证结果〔D039，2026-09-05 夜，lab-ts〕

预检拦截-恢复、僵尸清理、依赖补齐（D1 `to42_gate.py`、D2 旧侧 `encoder/` 包、
A3 watcher 自身漏建旧包装）等运维事件见 §5。全部原始日志在服务器
`apt_g1/outputs/d039_*.log`（7 份）与 `outputs/token_vae_e39_d039_{fast,slow,fro,gpuonly}/`。

### 4.1 VAE 对照（e39，10 epochs）

| 配置 | 总墙钟 | 稳态/epoch | 稳态吞吐 | 备注 |
|---|---|---|---|---|
| 原始行为（CPU 数据+eager+非 fused） | 11.9s | ~1.2s | 52–54k samples/s | **基准，本来就不慢** |
| 全显存+fused（无 compile） | 13.3s | ~1.3s | ~47.4k | 慢 ~10% |
| 全显存+compile(default)+fused | 27.8s | ~1.5s | ~42k | ep1 含 compile 预热 14.6s；guard 开销净负 |
| compile `reduce-overhead` | 6s 即崩 | — | — | CUDA Graph 与训练循环跨 step 持有 compiled 输出不兼容（cudagraph_trees.py 报 overwritten） |

- **等价性 PASS**：三配置逐 epoch 曲线同区间同走向；末 epoch va_mse
  0.00650 / 0.00663 / 0.00664（slow/gpuonly/fast），相对差 ~2% < 5% 门槛。
- **对抗头终态 = 混沌诊断量（不作判据）**：fast（compile）final dir_head_acc
  0.1189 与生产 30ep run 的 0.1189 完全一致；两个 eager 测试跑均 0.691——
  模式化差异来自 compile 改变 RNG 流（inductor philox）导致对抗 min-max 终态
  分岔。heads 不入 `vae.pt`（生产件只有 VAE state_dict），科学上无影响。
- **判定**：e39 量级上三项优化全部中性偏负，**默认值已按实测回改为原始行为**
  （`--data-on-gpu`/`--compile-mode`/`--fused-opt` 全部 opt-in），旗标保留给
  VAE-L 量级再评估。
- **口径修正**：e39 全训 30ep 实测 ~40s 量级（原始路径），DS 计划 §218
  「VAE 训练 ~1h 级」估算虚高约 90×——VAE 线从来不是机器时瓶颈。

### 4.2 Isaac 对照（128 envs，同 seed 0，旧版 = git pull 前快照 PYTHONPATH 隔离）

| 组 | 新墙钟 | 旧墙钟 | 稳态 dt/iter | 首 iter 对比 |
|---|---|---|---|---|
| latent（E27 模板旗标，100 iters） | 95s | 90s | 双侧 0.8s | **逐位一致**（rew=1.612, loss=108.5717, kl=56.066559） |
| router（默认模式，60 iters） | 60s | 59s | 双侧 0.8s | loss 逐位一致（129.3299，ent 差 1e-4） |

- **等价性 PASS**：首 iter 逐位一致，其后小混沌发散（ rew 1.822 vs 1.831 量级），
  曲线同形态；同组 `policy_final.pt` 字节数完全相同（779018 / 735882）。
- **提速 = 零**（差值噪声量级）。判读：被删除的同步点在 128 envs 下单步成本
  μs 级，**真瓶颈是 Isaac 仿真步进本身**；§1 列举的剩余同步点
  （`_build_commands_list`/router/gate 边界）按同逻辑预期收益同样有限，降级。
- **口径修正**：2000 iters × 0.8s ≈ 27 min，DS 计划 §237「lab-ts ~4h
  （128 envs）」估算虚高约 8×——**预算表（§286-289）机器时口径需整体下修**
  （VAE-S 1h/VAE-L +3h/RL 4h 三行全部受影响）。
- 附带：四组全部自然退出（本轮零退出挂死）；新 `[SPEED]` 稳态 it_per_s=1.24；
  显存快照未捕获（单组 59–95s 短于轮询周期，训练中 GPU 无残留进程可证空载）。

### 4.3 第一波总判定

**等价性全 PASS、提速全零、两条预算口径下修、默认值按数据回改。** 科学产出
是证伪：本机该 workload 下，「热循环同步点」与「小 kernel 发射」均非瓶颈
（0.3M VAE 原始路径 52k samples/s、Isaac 双侧同 dt=0.8s）；LoongForge 式单卡
手段（compile/fused/数据驻留）在此量级无可挖余量。性能基建（旗标+沙箱+
[SPEED] 日志+预算修正）与负结果本身即第一波交付。

## 5. 事故与运维记录（2026-09-05 夜）

- **oracle replay 僵尸进程清理**：D039 预检发现 GPU 被两个
  `oracle_token_replay_isaac.py` 进程占 2170MiB（09-04 21:58 / 22:39 启动，
  已 5h+/4h+）。证据链：两者输出 JSON（`ds_phase0/oracle_replay_isaac{,_d035}.json`）
  在启动后 ~3 分钟即写完且 5 小时零增长、gate PASS、数值与台账 D034/D035 记录
  完全一致（mean_vx 1.6657 / ratio 1.6125）、日志冻结在最终 gate 行、进程却以
  129% CPU 空转 → 判定为前夜 B 线会话遗留的完成态僵尸（Isaac 退出挂死的
  空转变体），kill 后 GPU 释放（337MiB 桌面残留 / 0% util）。
- **同步克隆 untracked 豁免**：`g1-apt-cloud-sync` 存在 untracked
  （`apt_g1/outputs/sync/to40_cond_*`、`__pycache__`、`output/`），已核实与
  incoming 无路径冲突，pull 照常；遗留数据文件归 owner 处置，本专题不动。
- Isaac 短训按 D037 已知退出挂死预案处理：final ckpt 落盘 + 日志停长 5 分钟
  → kill 记录在案的 PID。

## 6. 第二波候选（D039 后重排：真瓶颈 = 仿真步进吞吐与 eval 串行）

1. **num_envs 扫描**（升为 Isaac 首位）：128 已验证，向上探 256/512 的显存
   天花板与 dt 曲线——小 policy 下每 iter 样本量近似随 env 数线性涨，是唯一
   有真实上限空间的吞吐杠杆（注意：改变 update batch 语义，属实验设计非纯优化）。
2. **eval 并行化**：`eval_apt_isaac.py` num_envs=1 串行（A/B/C/D 全测时间翻倍
   级）→ 批量 env 或多进程（D030 六进程并行模式现成）；按 §4.2 口径修正后
   eval 可能比训练本身更耗墙钟，优先级升高。
3. TF32 试验（`torch.set_float32_matmul_precision('high')`）：compile 日志提示
   未启用；动数值精度，须配等价性门再开。
4. VAE-L / BONES-SEED 数据管线预案：288h/5180 万帧 f32 ~13GB+，31G RAM 勉强，
   预打包单 memmap/pt + `persistent_workers` + `prefetch_factor`；VAE-L 量级
   重评 `--data-on-gpu/--compile-mode/--fused-opt` 三旗标（e39 量级结论不外推）。
5. `--fused-adam`（Isaac，默认关）与剩余热循环同步点向量化：D039 判读后
   预期收益有限，降级为顺手做。

## 7. 速查：旗标一览（D039 后默认值）

| 脚本 | 旗标 | 默认 | 说明 |
|---|---|---|---|
| train_token_vae_e39 | `--data-on-gpu` | 关 | opt-in 全显存数据（e39 量级实测中性偏负） |
| train_token_vae_e39 | `--compile-mode {none,default,reduce-overhead}` | none | e39 量级 compile 净负；reduce-overhead 与本循环不兼容 |
| train_token_vae_e39 | `--fused-opt` | 关 | opt-in fused 优化器 |
| train_token_vae_e39 | `--epochs` / `--out-dir` | 30 / outputs/token_vae_e39 | 测试沙箱（勿覆盖生产 vae.pt） |
| train_apt_isaac | `--fused-adam` | 关 | 保持 E 系历史 run 可比 |
| train_apt_isaac | `--speed-log-interval` | 50（0 关） | `[SPEED]` 日志间隔 |
