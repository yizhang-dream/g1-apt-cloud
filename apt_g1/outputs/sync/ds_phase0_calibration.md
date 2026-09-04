# DS Phase 0 校准报告 —— Isaac 执行保真度（oracle token 回放）

> 【层位 L4 产物｜DS_GAIT_MANIFOLD_PLAN §2 前置门；Run 行 = `tracker/D.md`
> D034；判据与动机见计划 §2】执行会话 2026-09-04（lab-ts）。

## 结论（门判定）

**PASS（第 1 轮，零对齐迭代）**：3/3 seed 完成、零摔倒，realized vx 均值
**1.6657 m/s** ÷ 官方回路 realized 1.033 m/s = **实现率 1.61 ≥ 0.9**。
→ G3 全速度段目标成立；Phase 4 解锁「cmd U(0,1.5) 第二臂」分支；Phase 1 解禁。

## 方法

- **数据**：D033 `drive_run_probe` 官方回路 RUN 录音复用（`/tmp/ds_smoke`
  三件套，零采集机时）。token 窗 = 行 [1048, 4048)（1s 参考位移 >0.7m 自动
  定位起始），3000 行 @ 50 Hz；**lattice 违例 0**（64 维 token 全部在 k/16
  格点上）。
- **回放**：`isaac/oracle_token_replay_isaac.py`（SCRIPT_MAP §9）——
  `AptFlatG1Env` 子类旁路 policy/VAE/router，token 直进冻结
  `SonicTorchDecoder` → q_des = default + action×scale；env 自持 10 帧闭环
  decoder history（D002 协议 Isaac 版）；canonical env 零改动。
  `jitter_and_reset` 与 E 系评测同源（rng 1000+seed）。
- **分母**：`--official-vx 1.033`（D033 base_sim 探针 62m/60s）。
  **注意**：`target_motion.csv` col0 是 planner 参考轨迹（本窗实测
  2.086 m/s = 裸 planner 2.12 同级），**不是** realized——拿它当 realized
  会把官方回路速度高估一倍。

## 结果

| seed | steps | fall | h_min | disp_x (m) | vx_x (m/s) |
|---|---|---|---|---|---|
| 0 | 3000 | - | 0.691 | 99.80 | 1.663 |
| 1 | 3000 | - | 0.691 | 100.01 | 1.667 |
| 2 | 3000 | - | 0.691 | 100.02 | 1.667 |

disp_norm ≈ disp_x（99.96–100.07，直线跑无 yaw 漂移）；h_min 三 seed 全等
（刚性步态，出生抖动不敏感）。

## 执行层衰减排序修正（D032/D033 → D034）

同一 RUN token 流、同一冻结 decoder，三个执行栈对 planner 参考（2.086 m/s）
的实现率：

| 执行栈 | realized | 实现率 |
|---|---|---|
| **Isaac**（gear_sonic 物理配置 + 隐式 PD，200Hz 物理 / 50Hz 控制） | 1.666 | **79.8%** |
| 官方 WBC sim 回路（D033 探针） | 1.033 | 48.7% |
| 我方 MuJoCo harness（10Hz 在线重规划 + 自配 PD，D032） | 0.37 | 17.5% |

D033 的「执行层衰减」风险在 Isaac **不成立**——harness 的 0.37 归因进一步
收窄到 harness 自身执行配置（PD 增益/力矩限制/10Hz 重规划边界扰动），与
token/decoder 无关。

## Caveats（gate ≠ 机制）

1. PASS 判的是「Isaac RL 底板能执行官方回路快步态 token」（Phase 4 前提），
   **不构成**「Isaac 物理 = 官方 WBC 物理」：Isaac 比官方回路快 61%，两套
   realized 互不外推；引用速度数字时必须注明执行栈。
2. 未验证足底接触真实性：硬 PD 追踪下的高 realized 可能含足打滑成分（P2
   抽检，不阻塞 Phase 1：给 `rollout_log_joints.py` 加 oracle 模式后渲染
   核验足触地 + 关节跟踪误差；若打滑显著，Isaac 速度上限声明打折）。
3. 只测 RUN 前向单命令；SLOW/HAPPY（尤其 23 号跳舞形态）在 Isaac 的表现由
   Phase 2/3 质量门与 held-out 检验覆盖，不在本判定范围内。

## 打滑核验（D035，09-04 追加；owner 定为最关键项）

同一 oracle 回放加足底审计：contact 代理 = 足框世界高 < min_z+0.02（URDF
`left/right_ankle_roll_link`），测接触期足水平速度分布 + 双足占空 + 步频 +
关节跟踪 MAE；**预注册判定 = 较差足中位接触速度 <0.15 m/s 诚实**。

| seed | 中位接触足速 | p90 | slip_frac(>0.2) | 单足占空 | 步/秒/足 | q 跟踪 MAE | 判定 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0377 | 0.918 | 0.058 | 0.383/0.424 | 1.35 | 0.1714 | **HONEST** |
| 1 | 0.0381 | 0.964 | 0.060 | 0.383/0.424 | 1.35 | 0.1713 | **HONEST** |
| 2 | 0.0379 | 1.030 | 0.060 | 0.384/0.424 | 1.35 | 0.1714 | **HONEST** |

解读：①接触期足速中位 ≈ 0.04 m/s（钉地），无系统打滑；②单足占空 ~0.4 →
双足合计 ~0.8、**~20% 周期腾空 = 真跑步**；③步频折算步长 1.666/2.7 ≈
0.62 m 合理；④q 跟踪 MAE 0.171 rad——PD 非硬吸附，1.67 m/s 不是「刚性拖拽
贴目标」伪影；⑤p90 ~1 m/s 为落地/蹬伸瞬间，占少数（slip_frac 6%）。

**结论：Isaac 的 1.6657 m/s 是物理诚实的跑步步态**；三 seed 与 D034 逐位
复现（确定性佳）。渲染抽检免做。外部证据面见 `refine-logs/DS_S2R_EVIDENCE.md`。

## 复现

```bash
# lab-ts（Isaac 启动 ~2 min + 3 seed × 60s）
cd /home/cvgluser/ros2_data && nohup bash /tmp/run_apt_isaac.sh \
  /home/cvgluser/ros2_data/apt_g1/isaac/oracle_token_replay_isaac.py \
  --headless \
  --out /home/cvgluser/ros2_data/apt_g1/outputs/ds_phase0/oracle_replay_isaac.json \
  > /home/cvgluser/ros2_data/apt_g1/outputs/ds_phase0/replay.log 2>&1 \
  < /dev/null & disown
```

产物：`outputs/ds_phase0/{oracle_replay_isaac.json, replay.log}`、
`data/ds_phase0/run_tokens.npz`；本目录 JSON 为同步副本
（`ds_phase0_oracle_replay_isaac.json`）。
