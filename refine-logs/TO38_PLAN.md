# TO38 收束：RL 稳定器叠加 TO 参考（E47 管线 + F11b 注入）——分支一：可消化，低速带增益 50×

> 【层位 L2 侧轴｜设计定稿：TO38（2026-08-31 定稿并开跑）】↑
> `refine-logs/README.md`（扇出树根地图）｜上游：`LEG_LEVEL_TO_REPORT.md`
> （TO36 收束报告，§3 指明正路）、`tracker/TO.md`（TO36/TO37/TO38 Run 行）｜
> 设计过程：2026-08-30 grill-me 三项拍板 → 2026-08-31 autoscirub rubric
> 审查（3 PASS / 6 FAIL，6 条均为「补最小规范」级，已全部并入本页）。
> **结果（08-31 收束）**：双臂 lab-ts 训完 + 14 组评测，主指标合并配对
> 差分 −0.0507（δ=0.03）→ 分支一「TO 参考 RL 可消化」：cmd 0.2 主臂
> 跟踪误差 0.002 vs 对照 0.102（50×，对照臂低速空洞）；代价 = 中速带
> 弧线走（TO 2.4s 时钟 vs 解码器步频冲突显形，disp 7.4 vs 33m）。
> Run 行详见 tracker/TO.md TO38 节；后续正路 = cmd 联动缩放/TO37 解族
> 条件化。

## 0. 问题与结论口径（三分支）

TO36 C 门证明开环回放不可行（1.84 s 倒）；TO38 检验 **E47 从零 RL 管线叠加
TO36 F11b 解参考后，TO 数据能否被闭环消化**。配对 A/B（同机同 seed）：

| 分支 | 判定条件 | 结论 |
|---|---|---|
| a 显著优于 ctrl | 主指标配对差分超等效边界 | 「TO 参考 RL 可消化」（低速带增益） |
| a ≈ ctrl | 差分在边界内且跟踪误差不降 | 「注入被忽略」（信噪比不足，E48 同族） |
| a 劣于 ctrl | 差分反向超边界 | 「TO 参考与解码器先验冲突」（负结果） |

任一臂 floor 失败（见 §4）→ 只报「失败/不可判定」，不进三分支结论。

## 1. 方案（b+c：observation 注入 + cmd 门控 reward shaping）

基线 = E47 精确配方：`--latent-mode --latent-vae-path e39/vae.pt
--latent-speed-bins --latent-dir-bins --latent-kl-prior zero
--progress-scale 1.0 --heading-scale 0.4`，2000 iters。

TO 数据 = F11b 单速度（v 0.277 m/s，T 2.4 s，A 门膝可行 + B 门双验证）。
数据链：`to36_world_knots.npz` →（`apt_g1/to38_export_ref.py`，逐相
角色→SONIC 列重排 + B 门符号，首尾闭合检查 wrap_gap=0）→ `to38_ref.npz`
（M=120 LUT：q_ref6/tau_ref6/pitch/z/heel_rel + meta）。

**关键坑（已修）**：world npz 的 q6 列序按角色排（[支撑 A/K/H, 摆动
H/K/A]），且符号按角色（支撑 [-1,-1,-1]、摆动 [+,+,-]）——两相列序互为
镜像，直接拼接会得到「单腿摆动」的畸形参考（实测 L/R 膝行程 0.31 vs
1.33 不对称暴露）；修正后 wrap_gap=0、行程对称。

### obs 注入（12 维，两臂 obs 维度一致）

独立 TO 时钟 ψ 自由跑（周期 2.4 s，reset 随机初相；**不与解码器 walk
时钟共钟**——pca 步频 0.255 rad/step ≈ 0.49 s/周期 vs TO 2.4 s/stride，
共钟会把解码器步态压出 VAE 流形）：

`[sinψ, cosψ, q_ref6_rel(6), pitch_ref, z_ref, heel_x_rel, heel_z]`
（q_ref6_rel 减 SONIC default，与 proprio 同口径；heel 为摆动脚相对骨盆）

### reward（1 项，cmd 门控）

`r_to = w_to · exp(−Σ(q_sag − q_ref(ψ))²/σ²) · exp(−(cmd_v − 0.277)²/gate²)`

v1 超参：w_to=0.3、σ²=0.1（σ≈0.32 rad）、gate²=0.0036（±0.06 m/s）。
非矢状自由度不跟踪（2D→3D 投影缺口：TO 只管矢状面，其余交给解码器先验）。
不加 τ 前馈（C 门已证脆弱）、不加摆动脚 reward 项（落点信号只进 obs）。

### 两臂定义（差异唯一归因于注入内容）

| 臂 | obs 块 | r_to | 时钟/LUT |
|---|---|---|---|
| TO38a 主臂 | 真实参考 | w=0.3 | 加载 |
| TO38b 对照臂 | **零块** | w=0 | **同样加载**（时钟照跑，诊断口径一致） |

## 2. 执行约束（rubric 补）

- 两臂同一 commit、同一 seed=0、同机（g1-train 4090D）顺序运行；
  运行间隔记录 `nvidia-smi`/负载状态；两臂完整命令行 + 配置 diff
  随 Run 行记入 tracker。
- cmd 采样 U(0,0.8) 与全部随机化（reset 抖动/相位随机）两臂一致
  （同 seed + 同代码路径保证）。
- 256 envs × 2000 iters ×2 臂；先 a 后 b。

## 3. 评测协议（rubric 补）

- **A 60s 存活**：3 seed（eval seed 0/1/2），cmd 固定 {0.2, 0.277, 0.35}
  三点低速带 + {0.5} 高速对照，各 3 次重复 → 每臂 4 cmd × 3 seed。
- **B 500N 推扰**：t=5s 时刻、+x 向 0.5s 脉冲，cmd 0.277，3 seed ×
  3 重复；恢复步数 = 推扰后 |vx−cmd| 回到 0.1 内的步数。
- **平滑度**：|q̈|（jvel 差分，50 Hz，60s 全窗均值）与 action rate
  （|z_t − z_{t−1}|，同窗）。
- **跟踪归因**：逐步记 ψ、q_ref6、q_sag（eval 时 50 Hz 全量落盘）；
  六关节 RMSE 按相位 8 箱分箱报告；两臂同 ψ 时钟同表计算。

## 4. 决策表（rubric 补）

- **floor（两臂各自，失败即报失败）**：A 60s 存活 ≥6 s、h_min ≥0.6、
  disp >0.5 m（C 门口径）+ 防蹲蹭。
- **主指标**：低速带（cmd∈{0.2,0.277,0.35}）vx 跟踪误差
  （|vx − cmd| 60s 均值，seed×cmd 聚合为每臂均值±std）。
- **配对差分**：同 seed 同 cmd 逐对作差；等效边界 δ=0.03 m/s
  （~10% cmd）；差分均值 >δ 为显著优于/劣于，|差分|≤δ 为等效。
- **次指标**（主指标判定后解释用）：矢状跟踪 RMSE、h_min 分布、
  B 恢复步数、平滑度。
- floor 失败的臂不参与三分支（§0），只报失败 + 归因方向。

## 5. 产物清单（rubric 补）

commit sha / 两臂完整命令与 seed / `to38_ref.npz`（含 meta）/
两臂 ckpt 目录 / 评测原始 JSON / 配对汇总表 —— 云机
`/workspace/g1-apt-cloud/output/`（canonical 在云，收束时索引回
`HANDOFF/03_OUTPUTS_INDEX.md`）；Run 行进 `tracker/TO.md`，
`EXPERIMENT_TRACKER.md` 行数同步，SCRIPT_MAP 登记
`to38_export_ref.py`（MODULE）与 env/train/eval 的 to_ref 注入。

## 6. 云开发机速查（交接保留）

- 实例 `g1-train`（`di-20260830193312-zbfs7`）；开机
  `POST /dev-api/api/dev-instance/{id}/start`（已验证），
  状态/SSH 信息 `GET /dev-api/api/dev-instance/list`；
  ssh `ssh -p <port> root@<ip>`（2026-08-31: 101.126.139.122:9120）。
- 代码 `/workspace/g1-apt-cloud`（工作区版本同步，含 e39 VAE + decoder）；
  跑训练 `cd /workspace/g1-apt-cloud && ACCEPT_EULA=YES
  /workspace/isaaclab/isaaclab.sh -p train_cloud.py …`（isaaclab.sh
  不在 PATH，用全路径；`--out` 会触发 Kit 启动 pybind 错误——
  该错误实为系统性，见 Run 行，绕法：默认输出目录 + 显式改名）。
- 费用：算力 ¥5.4/时 + 存储 ¥0.05/时（关机也计费，不用了
  `DELETE /dev-instance/{id}`）。
