# APT-RL × SONIC × G1 实验总结（MuJoCo + Isaac 两阶段）

> 更新：2026-08-13。完整过程记录见 `MUJOCO_APT_LOG.md`、`DISTILL_EXPERIMENT.md`、
> `EXPERIMENT_TRACKER.md`、`ISAAC_APT_LOG.md`。
> 阶段总结（2026-08-13：优先级链收尾 + 方向 A/B/C 定论）见
> `STAGE_SUMMARY_2026-08-13.md`。

## 一句话结论

**在平坦地面上，冻结的相位路由器 + SONIC 解码器（aux=0）就是最优策略**：60s 行走、500N 扰动、
68s 命令切换、跳跃全部通过；而所有 learned-aux 变体（11 组训练）都只做到了“存活但质量下降”
（超速+偏航漂移），无先验的 vanilla RL 在同等甚至 2.5 倍算力下完全无法稳定行走。
论文的 aux/latent 机制价值在“先验覆盖不到的工况”（地形/强扰动），平坦地面无法体现。

## 为什么做这个实验

用 SONIC 的 FSQ token + 官方闭环数据替换 APT-RL 论文的 2D 轨迹优化 + TVAE 先验，
在 Unitree G1 上复现论文的“latent 动作 + 辅助动作 + 2Hz 门控”流水线，验证蒸馏先验是否必要、
RL 复用是否有效。

## MuJoCo 阶段（已完成，结论：基础设施不足）

### 控制器线

- 冻结相位路由器 token 先验 + PPO 训练 aux：7 个 RL 变体全部劣于 aux=0 基线。
- 原因：单进程 MuJoCo PPO（即使 4 个顺序采样 env）样本效率不足，评测时 aux 破坏蒸馏先验
  （vx 0.83→0.2–1.0）。
- **判定：无法继续**（缺并行向量化 RL 基础设施）。

### 数据线

- 闭合周期 token 数据集：闭合误差 0.04–1.38 → 0.00000（判据达成）。
- 复测：闭合路由 A 持平、B/C 变差；相位 AR 自反馈漂移更差。
- **判定：判据达成但无益**——漂移根源在控制器层缺闭环修正，闭合先验不能替代。

## Isaac 阶段（已完成，结论：机制验证齐全）

### 基础设施

- Isaac Lab 2.1.0 + Isaac Sim 4.5（Python 3.10 via uv），RTX 3060 12GB。
- 踩坑记录：ensurepip 缺失、flatdict/pkg_resources、poetry-core、CRLF、`simulation_app.close()` 挂死。
- 自建 DirectRLEnv：G1 + 平面地形 + 冻结相位路由器 + torch SONIC 解码器（ONNX 转 torch，动态 batch）。
- 训练 64 envs：0.7s/iter。

### 实验矩阵（11 组训练）

| 变体 | A 60s 行走 | B 扰动 | C 切换 | D 跳跃 |
|---|---|---|---|---|
| **冻结路由器 aux=0（E1 基线）** | **3/3，47m** | 12/12 | 3/3 | 3/3 |
| aux gate-on（E2） | 3/3，3–11m | 12/12 | 3/3 | 3/3（vx 0.57） |
| aux gate-off（E6） | 3/3，16–32m | 12/12 | 3/3 | 3/3（vx 0.76） |
| aux+扰动+正则（E7b） | it_600：0/3 全倒 | - | - | - |
| aux 修正门控（E13） | 3/3，34–37m | 12/12 | 3/3 | 3/3（vx 0.44） |
| aux 修正门控+扰动（E14） | 3/3，5.7–12.6m | - | - | - |
| phase+aux 联合 RL（E3 系列） | 不走路（vx≈0.04） | - | - | - |
| phase warm-start（E8） | 3/3，vx 0.99，7–19m | 12/12 | 3/3 | 3/3（vx 0.76） |
| **vanilla RL 800it（E9）** | **0/3 立即倒** | - | - | - |
| **vanilla RL 2000it（E11）** | **0/3 立即倒** | - | - | - |

### 关键机制结论

1. **蒸馏先验必要**：同等预算 vanilla RL 无法行走；先验版 800 iters 内全项通过。
2. **aux 在平坦地面无正向收益**：全部变体存活率达标但超速+偏航漂移；与论文“aux 服务地形/扰动”逻辑一致。
3. **离散 token 相位选择无法替代 TVAE 潜空间**：恒定相位=静态姿势，MLP PPO 无法从零发现
   步态振荡器（E3 系列）；warm-start 可注入行走但 RL 微调后期退化（E8）。
4. **2Hz 门控实现偏差**：naive 自由运行 tick 被策略当钟表（E2 vs E6）；论文的 2Hz 信号绑定步态决策。
5. **PPO 后期不稳定**：多个变体在 600–800 iters 后收益下降/KL 爆增，中段 checkpoint 更优。

## 下一步建议（按优先级）

1. 地形课程（论文 7 类地形）——aux 的真正价值场景；Isaac env 已可扩展。
2. 修正 2Hz 门控为“步态决策绑定信号”（组切换时才 tick），对比 E6。
3. 用 TVAE/连续潜空间替代离散 token 相位（论文原始 latent 结构），配 SONIC 解码器。
4. 真机/部署链（C++ ZMQ 1280B 头）留待仿真方向确定后。

## 2026-08-12 续：地形 / 数据泛化 / 感知（详见 ISAAC_APT_LOG.md、DATA_GENERALIZATION_LOG.md）

### 地形（E15/E16）

- 修复地形可复现性：`HfRandomUniformTerrainCfg` 的高度场用**全局 np.random**，
  `TerrainGeneratorCfg.seed` 无效；改为 env 创建前 `np.random.seed(terrain_seed)`。
- 先验盲走鲁棒性曲线（固定 seed 0）：0.04 3/3、0.06 3/3、0.08 1/3、0.10 0/3。
- E15（gate-fixed aux + 0.04→0.08 课程，无感知）：0.08 0/3，0.06 2/3 但位移只剩
  1.7–9.7m——纯 proprio aux 无正向价值。
- E16（aux + 特权 elevation map 9×9@0.15m）：0.06 0/3、0.08 0/3——加地图反而
  更差。结论：**elevation 需要 latent/gait 选择通道才有表达载体**（论文的地图
  价值经 latent 动作 + gait logit 传导）；只进 aux 关节偏移通道无效。用户记忆
  正确：论文先给特权地图（教师），后训练感知复原地图。

### 数据/网络泛化（v8/v8c/v9 + oracle）

- v8（共享相位网络）失败：proprio→phase 强分组相关，池化后相位误差 1.6–2.3 rad。
- v8c（per-group 相位 + 共享 token 解码器）部分成功：walk_fwd 3/3，但 walk_back/
  turn/strafe 差于 v6 离散原型；连续回归解码器离开训练流形，闭环误差累积。
- exp3 补采 walk 6 个缺失方向（14,580 步，0 跌倒）→ v9 重建：walk_fwd/back 3/3；
  新方向 bin5 3/3（降级成前进）；bin1/2/3/6/7 0/3。
- **oracle 上限**：walk 6 方向官方 token 回放全部 95–270 步内倒——教师本身在
  本 MuJoCo env 做不了稳定 walk+turn。v9 已到/超过教师上限。
- 回答"是否需要更好的数据/网络"：数据覆盖必要不充分；连续回归网络会离开流形，
  离散原型+fallback 更稳；要突破上限需换带力矩标注的数据+力矩解码器（论文
  原方案）或换环境（Isaac）。

### 产出物

- 脚本：`apt_g1/{train_phase_router_v8,v8c,v9}.py`、`eval_battery_v8/v9.py`、
  `drive_exp3.py`、`build_exp3_dataset.py`、`oracle_walk_bins.py`、
  `isaac/elevation_map.py`、`make_terrain_fig.py`、`run_e15/e16_*.sh`。
- 数据：`apt_g1/data/exp_all3/`（68,093 步）、`exp3_raw/`。
- 模型：`apt_g1/outputs/distill_v8/v8c/v9/`、`isaac_e15_s*/`、`isaac_e16_s*/`。
- 结果：`outputs/terr_*.json`、`terrain_summary.png`、`eval_battery_v8/v9.json`。

## 关键文件

- `apt_g1/isaac/`：env、批量路由器、torch SONIC 解码器、PPO、训练/评测脚本。
- `apt_g1/outputs/isaac_*`：全部训练日志、评测 JSON、checkpoint。
- `refine-logs/ISAAC_APT_LOG.md`：Isaac 阶段完整留档（含每次踩坑）。
- `apt_g1/outputs/isaac_sample_efficiency.png`：样本效率对比图。

## 2026-08-12 夜续（MuJoCo 粗糙地形 + 力矩解码器 + E21a）

- **MuJoCo 本地 hfield 鲁棒性曲线**（修复 MuJoCo 3.11 hfield 碰撞语义后）：
  0.00–0.02 3/3 → 0.03 2/3 → 0.04 0/3 → 0.06 0/3 → 0.08/0.10 0/3；按坡度
  对齐与 Isaac 阈值一致。产出 `rough_mujoco_sweep.json` + 重渲染视频。
- **力矩级解码器（负面结论）**：从闭环 PD 力矩标签训练 phase→torque 解码器，
  论文式控制（tau_dec + PD(q_default)）能站住但不前进（vx≈0.03），混合式
  （token PD + tau_dec）1.3–1.5s 倒。**PD 标签不是论文 TO 那种规划前馈，
  无法替代 token 位置路径**——这是 E2–E20c aux 全负面的机制根源。
- **E21a（进行中）**：gate + anti-stop + 特权地图（rough 0.06 训练），检验
  "地图经步态选择通道产生价值"（对应论文/unitree_rl_mjlab 主流接法）。

## 2026-08-12 深夜续（按用户优先级链：先平坦 → 再闭环）

- **优先级 2（平坦命令完备性）完成**：
  - 审计 v9 全部命令组（20 组 ×3 seeds）：稳定=idle、slow 0.2 bin0/1/2/6、
    slow 0.6 bin4、walk bin1/4、jump；不稳定=walk bin0/2/3/5/6/7、slow
    bin4/5、stealth。
  - 把 bin5 降级固化为显式回退表（`router_fallback.py`，数据驱动锚点）。
  - **回退全命令 battery：24/24 命令 3/3×20s 无跌倒**；slow_fwd→slow 0.6
    组（vx 0.56）；walk_back 方向降级（bin1 锚点）为诚实限制。
  - 60s+ 切换马拉松 S1 2/3（长运行后 walk_back 段 ~29.7s 倒）——多命令长跑
    的闭环复合误差残余。
- **优先级 3（Isaac aux 判据实验）完成，判据未达成**：
  - E22a（E13 风格修正门控 + aux）：A aux 34–36m < 阈值 42.9m，B 持平，
    C/D aux 更差 → 不达标。
  - E22b（+aux L2/rate 正则 + 紧 yaw）：A aux 0.9–4.3m（原地振荡）→ 更差。
- 定论：SONIC 位置 token 先验 + aux 关节偏移管道下 aux 无正向价值；
    复现论文 aux 正向价值需力矩级解码器 + 真实 TO/逆动力学力矩数据。

## 2026-08-13（三方向：力矩级数据 / 连续潜空间 / 千级并行）

- **方向 A（力矩级数据，完成）**：用 mj_inverse 从重放轨迹算 ID 力矩
  （27k 行，1/99 百分位裁剪），phase→ID 力矩 val MAE 4.13 N·m（PD 标签版
  ~9.4）；混合式 ×0.2/0.3 平坦 3/3 与基线持平（15.2–15.4m），但论文式
  （纯 ID 前馈）2.5s 倒、粗糙地形无增益。根因：我们的 ID 力矩来自带位置
  跟踪控制器的重放，不是论文 TO 那种自洽规划力矩。方向 A 单独不能解锁
  论文式控制。
- **方向 B（连续潜空间，完成）**：相位插值读取（相邻原型按相位分数线性
  插值）MuJoCo 闭环 3/3 与离散基线持平、phase→token 连续；接入 Isaac 训练
  E23（warm-start 相位 RL），评测 A 3/3 存活但位移 0.8–2.0m（原地振荡/
  画圈）、C 切换 0/3 → **判据不达标**。连续读取代平滑不改流形上限，奖励
  压力（anti-stop/进度）才是关键（E20c gate+anti-stop 已证）。
- **方向 C（千级并行，评估完成）**：官方正解 = unitree_rl_mjlab
  （MuJoCo-Warp，默认 4096 envs），是从零学速度跟踪的独立轨道；当前 Isaac
  管道 3060 上 64→128 envs 吞吐仅 +61%（2,130→3,425 samples/s），显存
  2.8GB / GPU 62% 未用满，瓶颈在迭代更新/同步开销。服务器外网通、
  warp-lang 已装，mjlab 可装；12GB 需 1024–2048 envs 试跑。
- 最新日志：`DATA_GENERALIZATION_LOG.md` 13–15 节、`ISAAC_APT_LOG.md`
  E23 节、`EXPERIMENT_TRACKER.md` 方向 A/B/C 收尾表。
