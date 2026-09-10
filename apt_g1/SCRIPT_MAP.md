# apt_g1 脚本索引（SCRIPT_MAP）

> 【层位：代码侧轴入口，1 脚本 = 1 行】主轴同粒度层：`refine-logs/tracker/`
> 系列文件（L3 Run 台账·事实源，`EXPERIMENT_TRACKER.md` 为总索引）｜
> ↓更细：脚本源码与文件头注释（含 `_archive/` 归档）。
> 生成于仓库整理阶段（2026-08-13）。逐文件标注每个脚本的角色、用途与对应实验。
> 实验号对照见 `refine-logs/EXPERIMENT_TRACKER.md` 与 `HANDOFF/02_EXPERIMENT_HISTORY.md`。

## 分类标记

| 标记 | 含义 |
|---|---|
| **CANONICAL** | 现行/最终版，复现核心结果所需，保留在顶层 |
| **ARCHIVE** | 被取代的旧版本或已判死路的探索性脚本，已移入 `_archive/`（可恢复） |
| **FORK** | `server_*` 服务端分叉版本，与非 server 版同源但独立演进，原位保留 |
| **DEV** | 诊断/冒烟/数值校验小工具，原位保留 |
| **MODULE** | 被其它脚本 import 的库模块（非入口） |

---

## 1. `apt_g1/` 顶层 —— CANONICAL（保留在顶层）

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `train.py` | 入口 | MuJoCo 平坦地 APT-RL 训练主入口 | R 系列 |
| `evaluate.py` | 入口 | 对保存的 APT policy 做确定性 MuJoCo rollout | 通用 |
| `export_reference_tokens.py` | 工具 | 用官方 encoder mode-0 导出参考运动 SONIC token（**2026-08-14 修复 anchor 朝向 offset：584→601，漏了 root_z/anchor_single 的 17 维零填**） | 数据基础 |
| `replay_reference_track.py` | 评测 | **官方参考跟踪闭环复刻**：每步读实时基座四元数在线编码 encoder 观测→token→冻结解码器→关节目标，稳定追踪参考动作（squat/kick/lunge 等，全程不倒） | 动作库扩展 |
| `planner_sonic.py` | 评测 | **官方三模型全栈复刻**（规划器→encoder→decoder）：规划器 ONNX 输入 4 帧 qpos+模式命令→输出未来 qpos 轨迹→FK 转 motion→encoder→token→decoder；27 种 LocomotionMode 已解锁关键 8 模式（idle/slow_walk/walk/run/stealth/squat/kneel/crawl 全栈闭环 fall=None） | 动作库扩展 |
| `terrain_generalize_test.py` | 评测 | **地形泛化测试（goal 第 ④ 部分）**：8 模式 × {平地, rough 0.08} 对比，输出 fall / adv / h_rms / jp_rms（姿态跟踪误差），直接度量冻结解码器"平地→粗糙"泛化 | MQ08 |
| `planner_closed_loop.py` | 评测 | **官方 10Hz 闭环规划器复刻**：每 5 步用 live qpos 重出轨迹（context=4×当前 live 状态），支持运行中切换步态模式 + 粗糙度触发(walk→stealth)；证明重规划是地形泛化机制 | MQ09 |
| `closed_loop_sweep.py` | 评测 | **闭环盲重规划的地形振幅扫描**：均匀凸包地形(无平坦中心) × 振幅 0.08–0.20 × 3 种子，量化官方 planner 盲重规划的真实泛化边界(~0.12–0.14)，发现对地形 seed 高度敏感 | MQ10 |
| `closed_loop_levers.py` | 评测 | **闭环"杠杆"测试**：height 命令(弱) + 步态模式 walk/stealth/crawl 在边界 amp 0.14 × 3 种子；发现 crawl 3/3 不倒、walk 0/3 → 步态模式是真杠杆 | MQ11 |
| `srb_to.py` | 数据生成 | **G1 双足 SRB 轨迹优化器（CasADi）**：复刻 APT 论文"Impulse scale-based TO"（2D 单刚体 q=(x,z,θ) + 半正弦 GRF + 动量守恒 Eq.1 + 周期成本 Eq.2 + IPOPT 求解），产出周期性 CoM 轨迹 + GRF；**TO04 起支持 duty factor d**（walk d=1.0 / run d=0.5 带腾空相）+ 垂直速度 zd0 周期项（`grf_amp(d)`、`integrate_step`） | TO 数据 |
| `srb_to_torque.py` | 数据生成 | **SRB TO → 2 连杆腿 IK → 关节力矩**：把 SRB 的 CoM+足底反力映射成 hip/knee 角 + 力矩（τ=Jᵀf 静态力矩臂），产出 (state, torque) 数据（IK 往返自检 err≈0）；TO04 起 roll_out 支持 (v,T,d) 并积分腾空相；TO05 加 ankle 力矩（CoP 杠杆臂 L_HEEL/L_TOE） | TO 数据 |
| `train_torque_decoder_srb.py` | 训练 | **SRB TO 力矩解码器可学性**：MLP(sinφ,cosφ,v,d)→(hip,knee,ankle 力矩)，跨 walk+run 两步态 + 6 速度；报告 per-gait MAE + 跨步态泛化（train walk→test run）；ankle 用 CoP 杠杆臂模型 | TO 数据 |
| `eval_torque_srb.py` | 评测 | **SRB TO 力矩开环回放（G1 MuJoCo 力矩闭环）**：τ = sign·τ_SRB(2Hz 相位钟) + kp·(zero-token q_des − q) − kd·q̇ 施加到 sagittal 关节（act 0/3/4 与 6/9/10）；TO06 负结果——SRB 力矩太简化不驱动 G1 走路 | TO06 |
| `drive_exp3.py` | 数据采集 | 脚本化官方闭环采集缺失的 walk 方向 | D 系列 / exp3 |
| `drive_ds_smoke.py` | 数据采集 | **DS 重采线冒烟驱动**：RUN(3) 模式 + movement_speed 速度轴阶梯（WALK 基线/RUN 默认+阶梯/SLOW_WALK 阶梯，段间 idle+fall 计数）；setup 镜像 `/tmp/setup_exp3.sh` 模式 | D029 / DS_RECOLLECT_PLAN |
| `ds_mode_terrain.py` | 评测 | **DS mode×terrain 矩阵 runner**（planner_closed_loop 参数化 fork）：21 非静态 mode 全集 × mjlab 官方论文地形（内存 MjSpec 组装 + from_xml_path patch 注入 env，零 XML 落盘；rough 对称化=坑为必要难点）；hurdle/gap 自建标注 self-built-paper-params；每格 JSON 行输出 | D030 / DS_RECOLLECT_PLAN |
| `encode_bones_smoke.py` | 评测 | **B 线 B2 冒烟**：官方 GEAR-SONIC sample_data（BONES-SEED 格式 pkl，joblib+zlib，`dof` 判定 MuJoCo 序 30Hz）离线编码进冻结 encoder（planner_sonic g1 布局 1762 维，30→50Hz 线性重采样）→ SONIC token (2003×64)；三项检验：lattice 违例率（0.0，同 oracle_token_replay 容差）/ vs ds_smoke WALK 基线分布对照（mean-L2 3.16）/ decoder oracle 回环 MAE（0.564 rad，弱于默认站姿基线 0.223，开环分布偏移，负结果不阻塞）；JSON 落 `data/ds_bones/b2/smoke_result.json` | D036 / BONES-SEED |
| `scan_smpl_metadata.py` | 数据构建 | **B 线 M1 metadata 扫描**：bones_seed_smpl 镜像 131,455 pkl 全量并行扫描（multiprocessing，断点续扫）→ 逐文件 CSV（类目/演员/帧数/时长/速度分位/transl 转向/根 yaw/体尺/up 轴判定）+ 汇总 JSON（类目×时长×速度表、run 758 实长定量、方向偏斜定量、_M 镜像副本记账）——官方数据独走的数据分析主力 | D040-prep / DS_OFFICIAL_DATA_PLAN M1 |
| `build_b4lite_map_labels.py` | 数据构建 | **B4-lite S2 前半：描述名→语义族映射层**（服务器 venv_isaac CPU 跑，7s）：读 seed_metadata_v004.parquet 142,220 段→归一描述类 4,067（D040 NAME_RE 同口径）→owner 裁定① 10 语义族关键词规则映射（优先级序取更具体者；置信度 high=名称单族/med=名称多族/low=仅描述字段/other）；产出 `data/ds_bones/g1_b4lite/{desc_family_map.json, map_stats.json, desc_family_per_segment.csv}`；首版覆盖率 87.6%、other 12.4%，sideway 系列归 lateral、jump_sideway 归 forward_jump；temporal_labels 缺失→协议 §7.2 fallback（整段粗标签+置信度降级） | D045 / DS_B4LITE_ACTION_SPLIT_PLAN §1.3 §6-S2 |
| `build_b4lite_candidates.py` | 数据构建 | **B4-lite S2 后半+S3+S4：候选池选择+五维标签+清洗台账+划分冻结**（服务器 venv_isaac CPU 跑，4s，DEV）：读 L1a 段级族 CSV + parquet 官方标注，硬条件筛候选（时长 360–7200 帧/props 排除进扩展池/is_neutral 排除/同 take 选母不选镜像/每演员每族≤2），五维标签规则打标（移动/上肢/姿态/接触/时间，label_sources 三 bool，manual=false 待抽看）；三层清洗台账（L1 空占位待 S5 回填，L2 逐段原因，L3 保留——A005 显式登记 boundary_ref）；S4 划分（owner 6/2/2 族角色+配额 6×18/2×10/2×12，演员 md5 mod 10 三桶 80/10/10，T-fam 族演员与训练/开发 8 族零交集，训练族∩测试桶→T-seg）+泄漏检查三项（镜像同集/T-fam 演员交集/跨 set 报告）+每族抽 5 段 translate 差分轨迹核实（v_med/v_p90/航向总量，120fps cm→m，矛盾打 flag）；产出 `g1_b4lite/{candidates,candidates_stats,cleaning_ledger,split_assignments,leak_check}.json`；首跑 110/152 入选+1 边界段：dance_rhythm/lateral/slow_walk 三族硬条件过筛后凑不满配额（0/0/6 段，缺口如实回传未放宽条件）；**S5 修订（owner 裁定 2026-09-07，重跑生效）**：① lateral 族 locomotion 恒「侧移」（渲染复核 jog_sideway 系 filename 词根误标前进/混合，覆写族真值）；② EXCLUDED_STEMS 5 段 L2_label_mismatch 移出首版（A020 原地站立/A465 orange justice 防泄漏/crawl 三连窗 A125-A127），ledger L2 逐段+split excluded 标注+manifest exclusion 字段语义 → 最终 147 段（152-5）；③ A005 首尾 ~2 帧 T-pose 校准帧登记（L3 detail+quality_notes，不处理） | D045 / DS_B4LITE_ACTION_SPLIT_PLAN §1.1 §2 §3 §6-S2/S3/S4/S5 |
| `build_b4lite_gate_inputs.py` | 数据构建 | **B4-lite S5 输入构建三合一**（服务器 python3 纯 stdlib）：`bridge` 模式=split_assignments+candidates→`labels_split_bridge.json`（CB `--labels-json` 格式：family/labels_5dim/label_confidence/label_sources；键名调和 movement→locomotion、metadata→trajectory，后者=协议「文件名/轨迹/人工」三级来源的官方 parquet metadata 实现）+`csv_list.txt`（152 段绝对路径→CB `--list`，一次跑完防 D044 坑① manifest 覆盖）；`gate-split` 模式=manifest+划分→`gate_A/`(t_role∈{train,T-seg},108 段)+`gate_B/`({dev,T-fam,boundary_ref},44 段) 各带 npz 副本+子 manifest（重写 npz_path），不改门代码实现双门并行（t003 防线：分目录防撞）；`ledger-l1` 模式=转换失败段回填 cleaning_ledger L1 条目（协议 §2 记录不清洗） | D045 / DS_B4LITE_ACTION_SPLIT_PLAN §5 §6-S5 |
| `encode_smpl_smoke.py` | 评测 | **B 线 B2-s 冒烟（首实验 D040）**：SMPL 格式样本走 encoder mode 2（smpl）离线编码——布局取自 deploy C++ obs registry + observation_config.yaml（1762 维三模式共用；smpl_joints[922:1642) + 锚定[1642:1702) + wrists[1702:1762)，mode 头 obs[0]=2）；配对判别 = 同一运动 robot_filtered 的 g1-mode ref-rel tokens（D038 验证版）做参照，候选矩阵 {identity, y2z 变换}×{ref-rel, refheading 锚定}，判据 = lattice + 配对 mean-L2 + decoder 免环境回环 MAE（g1 参照 0.109 rad）；wrists 置零（镜像无 robot dof）+ 真腕对照行；JSON 落 `data/ds_bones/b2_smpl/` | D040 / DS_OFFICIAL_DATA_PLAN §3.1 |
| `retarget_smpl_g1.py` | 数据构建 | **M2 stage1 约定标定**：配对官方样本（robot/smpl/soma 三格式同 motion）标定 SMPL 镜像→G1 全部约定——根位置 Umeyama（尺度 0.773 残差 3mm、会话→机器人全旋转 R）、SMPL 24 关节名经 soma 命名骨架消歧（Mixamo 风格）、根方位生产式解 = 骨架几何（髋轴+脊柱轴）+ 常数校正四元数（vs 官方 quat 均差 4.64°）；G1 FK 对应表 | D041 / DS_OFFICIAL_DATA_PLAN M2 |
| `retarget_ik_pilot.py` | 数据构建 | **M2 stage2 IK 重定向试点**：逐帧全位姿 IK（根位置=Umeyama 精确固定、根方位=官方/skel+常数校正两模式、29 dof 逐关节 bounds=官方包络±0.25、位置+9 骨方向残差）；腕 6 dof 位置不可观测→钉官方常数；验证 = dof MAE vs 官方重定向 + g1-mode 回环 + Isaac 冒烟；结果 dof MAE 0.248/回环 0.233/Isaac 2/2 零摔（D041） | D041 / DS_OFFICIAL_DATA_PLAN M2 |
| `convert_bones_g1_csv.py` | 数据构建 | **D044 CSV→obs 转换器（B3' 现役入口，官方 G1 数据线）**：BONES-SEED G1 原生 CSV（36 列=Frame+root translate XYZ(cm)+root rotate 欧拉(deg)+29 dof 语义名(deg)，120fps，D043 全解）→ 语义名断言映射（29 列=MJ 序，m2i→Isaac 序）→ 120→50Hz 线性重采样（quat 半球对齐 lerp）→ `encode_bones_smoke` 零漂移 import 的 build_obs 1762 维 g1-mode ref-rel → 冻结 encoder tokens；**欧拉轴序不假设**——`--calibrate` 对官方配对 pkl（210531 walk_forward_amateur_001__A001 双格式并存）扫 6 内在欧拉约定取旋转误差最小；`--sample N` 类级抽样（walk/jog|run/jump/dance|dancing 正则、演员去重、排 `_M` 镜像）→ manifest.json（npz_path，供 `b3p_gate_isaac.py` 消费）；`--roundtrip` 免环境 decoder oracle 回环 MAE（sonic_history 镜像 smoke check3：default-relative 位置/归一 last_actions/有限差分体轴 ω/重力方向）；**D045 扩展**：meta 新增嵌套 `b4lite` 块（协议 §5 四问字段：source/frames/actor_group/semantics/kinematics/quality/split/exclusion/flat_replay，npz 内嵌 meta 与 manifest 双写同构、向后兼容 D044 读者）+ `--labels-json`（stem 命中则合并语义标签，缺省全 null）+ `--tar-version` | D044 / DS_OFFICIAL_DATA_PLAN B3' |
| `backfill_b4lite_manifest.py` | 数据构建 | **B4-lite S5→S6：gate 回填 + manifest 冻结（v2）**：join `b3p_gate_isaac.py` 的 gate_result.json（per_segment `{stem}__seed{seed}` 按 stem 聚合，`__seed` 尾部 rpartition 防截断 stem 内 `__`）→ 每段 `b4lite.flat_replay`（survived=全 seed completed 与 / n_seeds / fall_steps 逐 seed 列表 / fall_during_playback 任一 / steps_budget / q_track_mae_med / realized_path_ratio_med / fall_form 定性留人工）写 v2 manifest；**L3 判定回填**（survived=false → quality.layer3=recorded + boundary_set=true + t_role 预填 boundary_ref，协议 §2「执行失败=能力边界记录不清洗」）；同产 boundary_set.json（owner 裁定③双轨，逐条 stem/actor/family/fall_steps/q_track_mae_med/evidence_note）；无 gate 条目的段 flat_replay 保持 null 不伪造；纯 stdlib 零依赖；**S6 扩展（--split）**：读 S4 冻结划分回填 `b4lite.split.{set_role,t_role}` + repr_participation 冻结声明（vae=适配训练(以冻结 manifest 为准) / norm_stats=B4-lite 样本内标定 / sonic_pretrain 不可排除）+ boundary_ref 段 A005 T-pose 观察 note + 排除段 exclusion 防御写入 + **npz 内嵌 meta 同构重存**（meta.b4lite=manifest 条目，D044 坑①）；**L3 判据升级**（owner 裁定 2026-09-07）：3-seed 中 ≥2 fall=系统性入 boundary_set + t_role 预填 boundary_ref，1/3 fall=seed-variance 仅 quality_notes 不入集（单 seed fall 保守收录；numpy 仅 --split 路径 lazy 引入，否则仍零依赖） | D045 / DS_B4LITE_ACTION_SPLIT_PLAN §5 §6-S5/S6 |
| `build_b4lite_reports.py` | 数据构建 | **B4-lite S6 报表双产（只读 join，纯 stdlib）**：吃冻结 manifest_v2 + S4 划分 + candidates_stats + 台账 + 双门/3-seed gate JSON → ① `gate_summary_d045.json`（overall + 10 族逐族 n/survived/rate/PASS-FAIL（≥95% 且 n≥10）+ T-seg/dev/T-fam/boundary 桶存活率 + 3-seed 逐段 verdict（≥2 fall=systematic/1/3=seed_variance）+ boundary 明细，A005 单列不入族聚合）；② `publication_table_d045.json`（候选池 124,609 + 逐层排除逐原因计数（L2 硬条件/mirror_dedup/L2_label_mismatch 裁定 5 段/L1 转换失败/L3 边界收录）+ 最终 147 段覆盖表（10 族×n/演员数/五维标签分布）+ 三测试划分冻结计数 + 三表示参与声明 + 已知噪声登记（lateral 标签修正含前后 diff 段清单/A005 T-pose 帧/渲染复看存疑 stem）+ 原始双门与 3-seed 门数字） | D045 / DS_B4LITE_ACTION_SPLIT_PLAN §2 §6-S6 |
| `build_b4lite_vae_inputs.py` | 数据构建 | **D046 B4-lite VAE v1 输入构建**（服务器 venv_isaac python，秒级；--split train/dev/tfam 三次调用）：manifest_v2 `set_role×t_role` 双过滤排除 boundary_ref 3 段（train 86→84、dev 21→20，T-fam 22）→ 逐帧速度 = trans_m XY 中心差分（dt=0.02）+5 帧滑动平均 → 8-bin 方向标签（段首 θ0=前 10 有效帧方向圆均值、bin0=±22.5°=前向〔与 exp_all3 bin4=前向约定不同〕、平滑速度 <0.05 m/s 帧继承前 bin、纯站立段全 bin0 并记录）+ mode=族 id 固定序 0-9（mode==2=fast_walk_run 恰喂 E39 phase-PCA/v-bin 机制）；train 另做族平衡过采样（补齐到最大族预算 forward_jump 14,011 帧，stem 轮转整段复制+末段截断，确定性无随机数，因子入 meta）+ `segment_bounds.npy` 段边界表（配 train_token_vae_e39.py 掩码实现窗口不跨段）+ `norm_stats_d046.npz`（只从 84 段原始帧）；产出 `g1_b4lite/vae_v1/{token,mode,angle_bin,segment_bounds}.npy+build_meta.json`（bin 规则/过采样因子/stem↔帧区间/逐族 bin 直方图）+ `dev/`、`tfam/` 子目录 | D046 |
| `eval_b4lite_vae.py` | 评测 | **D046 B4-lite VAE 预注册判据评测（J1–J4，只前向不训练）**：确定性 mu 前向、合法窗口帧（与训练同 segment_bounds 掩码）、dev/tfam 重放 train 的 pca.npz+vbin 分位（不重拟合统计量）；自动扫 run1/vae_ep*.pt 出 train/dev 重建曲线+最末快照终判：J1 dev/train 重建 MSE 比（≤1.5 预注册线）、J2 dev dir_head(8-bin)>12.5% 与 speed_head(3-bin)>33% 双随机线+逐族分桶、J3 指标有限性、J4 T-fam 基线数；落 `run1/metrics_d046.json` | D046 |
| `build_b4lite_mode_map.py` | 数据构建 | **D046 v2：27-mode 词表全量重标 + 子集裁定 + v2 候选池/划分/泄漏（一脚本六产物）**（venv_isaac，~2min）：抄录 localmotion_kplanner.hpp LocomotionMode 27 值→子集规则预注册（①映射覆盖 union≥200 段 ②非静态 ③纲领范围；道具 GUN/OBJECT_CARRYING+爬行域 out_scope；STEALTH_WALK_2/RANDOM_PUNCH 判别性预剔除）→ 8 mode 入子集（SLOW_WALK/WALK/RUN/FORWARD_JUMP/STEALTH_WALK/INJURED_WALK/LEDGE_WALKING/HAPPY_DANCE_WALK）；映射规则优先级序互斥解析（分解优先：上肢=UL 正交轴{none,sym,asym}非 mode；lateral/turn=WALK×方向旗〔hpp movement_direction 3D 单位向量语义〕；start_stop/posture→intermediate；纯站立 gesture→none 剔除），置信度三档照 D045；142,220 段逐段打 (mode|intermediate|none, UL, direction, conf)→`mode_map_v2/{mode_map.json, relabel_per_segment.csv, relabel_stats.json}`；候选池=D045 硬条件（import build_b4lite_candidates 单一事实源）+none/intermediate 剔除规则+D045 boundary 3 段承接排除，配额 min(18,池深)/T-fam 12，intermediate 材料 12 段 w=0.3；T-fam=SLOW_WALK+INJURED_WALK（梯子+池深≥10 门：STEALTH_WALK 池 7 被门槛）；泄漏三项照 D045 全过；产 `candidates_v2/split_v2/leak_check_v2.json`+`csv_list_v2.txt`/`labels_v2.json`（新段转换器输入，npz 落 v2 专用 g1_b4lite_v2conv/ 不碰 v1）；v1 147 段对照（86 存活/46 剔除/16 转 intermediate/48 新增）；**v2.1（同日二次裁定）**：静态类仅 IDLE(0) 入集作 mode 间转换公共枢纽——子集规则静态例外+IDLE 词表扩纯站立词表（stand/idle/wait/still/stationary/neutral）+映射规则⑤反转（纯站立→IDLE、站立+上肢→(IDLE,UL)），纯站立门 `_idle_veto`（名称+全字段合文本 loco/其他 mode/posture 词否决、起止过渡词不否决）；IDLE 退出描述字段循环仅兜底（防字段序抢跑）、posture 补 stands/gets up 形式；产物目录 `mode_map_v21/`（*_v21.json 六件）不覆盖 v2（v2.1 重标：IDLE 14,835 段/池 5,116/选 18=10sym+8asym、inter→IDLE 130，其余 mode 池深与 v2 持平） | D046(v2/v2.1) / owner 2026-09-07 两次指令 |
| `build_b4lite_vae_inputs_v2.py` | 数据构建 | **D046 v2 VAE 训练输入四件套**（venv_isaac，秒级）：读 candidates_v2.json 133 段（121 mode+12 intermediate），npz 双目录解析（v1 npz/ 只读+v2conv/）；复用 build_b4lite_vae_inputs.py 的 frame_speed/angle_bins 代码路径（bin 规则同款：bin0=段首前向、5 帧平滑、v_thresh 0.05）；mode 数组=embed idx（8 子集 mode 按 hpp id 序+intermediate 殿后，表落 build_meta）+ 新增 ul 数组（none/sym/asym 段级常量）；train 84 段 42,281 帧→按 mode 预算平衡过采样 83,195 帧（STEALTH 因子 9.53× 如实记录）+ `window_keep_mask.npy`（intermediate 窗口 keep 0.3=owner 低权重连通性落法，seed=0）+ `norm_stats_d046v2.npz`（只算 train 原始帧）；产 `vae_inputs_v2/{token,mode_id,ul,angle_bin,segment_bounds}.npy+build_meta.json+dev/ tfam/`（dev 8 段=bucket8、tfam 24 段=T-fam）；**v2.1**：MODE_HPPO+IDLE(hpp 0)，embed idx 按 hpp 序（IDLE=0/WALK=2/intermediate=9→n_modes=10），读 candidates_v21.json，产 `vae_inputs_v21/+norm_stats_d046v21.npz`（train 99 段 52,356 帧→过采样 96,319〔IDLE 因子 1.30×〕→91,348 窗，**sym 9,917 窗首次非零**；dev/tfam 与 v2 同）；**v3.1（D046b-R3）**：oversample_by_mode 返回 source_idx（过采样后每段→源原始段下标）并落盘 `segment_source.npy` + build_meta 增 oversample_identity 块（segment_source_stems）与 segment_detail.actor——留出评估按源段分组的身份依据（同一原始动作及其派生副本同组） | D046(v2/v2.1) / D046b-R3 |
| `build_d048n_vb_speed_labels.py` | MODULE | **D048n 速度锚定 vb 标签构建**（numpy-only，服务器秒级）：源 npz trans_m 逐帧后向差分 ×fps（v[t]=‖trans[t]−trans[t−1]‖×fps、v[0]=v[1] 镜像 E39 rate 口径；默认 XY 平面同 D046 frame_speed）→ 参考帧（mode==--ref-mode，默认 2=v21 SONIC WALK 前向 walk 族，与 train 脚本 rate 边界掩码同口径）速度 1/3、2/3 分位 edges → vb=clip(digitize(v,edges),0,2) 全帧；段对齐走 segment_source.npy / holdout_ident.derive_segment_sources 确定性重放（三重 checksum 不符即 raise）+ tokens 逐帧 md5 抽查 ≥3 段；产 `<inputs-dir>/vb_speed.npy`+`vb_speed_meta.json`（edges/counts/label_policy/checksum/源 npz md5 清单），配 train_token_vae_e39.py --vb-npy 消费 | D048n |
| `build_d048p_vb_edges_calib.py` | MODULE | **D048p edges 命令带反标定 vb 标签构建**（D048n 校准版，numpy-only）：import 复用 build_d048n_vb_speed_labels 的段对齐/速度/ checksum 实现（单一事实源，不复制），单变量=edges 来源——参考帧（mode==--ref-mode，默认 2，D048n 同口径）速度分布上网格搜索 (e1,e2)：e1∈[0.10,0.50]、e2∈[0.50,1.50]、步长 0.01、e1<e2，最小化 Σ_i (mean_i(v_ref)−t_i)²（t=--target-means 默认 0.20,0.45,0.65 等权，§5h 预注册一次不动），约束三档参考帧占比各 ≥--min-frac（默认 0.10，向量化=排序前缀和分解，与逐对暴力枚举同口径）；无可行解→打印 v_ref 分位数表(5/25/50/75/95)+可行域分析、落 vb_speed_calibration_rejected.json 并非零码退出（负结果路径）；产物与 D048n 同格式（vb_speed.npy+vb_speed_meta.json，meta 增 calibration 块：target_means/min_frac/网格/选中 edges/三档 mean/三档 frac/top-5 候选，label_policy 改校准口径，experiment=D048p），配 train_token_vae_e39.py --vb-npy 消费；自测 tmp/d048p_test/build_d048p_selftest.py（T1 暴力枚举一致/T2 min-frac 拒绝/T3 负结果退出/T4 全帧 vb=digitize） | D048p |
| `pick_fast_segments_d048r.py` | MODULE | **D048r 快段清单构建**（numpy-only，本机/服务器均可秒级）：v2 g1 npz 双目录（DEFAULT_NPZ_DIRS 同 D048n，先到先得、stem 去重）全池逐段算 v_med——import 复用 build_d048n.frame_speed_bwd（trans_m 后向差分×fps、v[0]=v[1] 同一实现不复制）取段中位（D048n/D042 口径）→ 门槛 --min-vmed 1.0 m/s、取 top --top 12（不足取实际数，§5i 预注册分支）；全池 v_med 分位数分布随 json 落盘供该分支判读；产 `data/d048r/d048r_fast_segments.json`（segments[] = stem/npz_path/v_med/v_p90/frames/fps），配 `isaac/replay_token_speed_d048r.py --segments-json` 消费；自测 tmp/d048r_test/pick_fast_segments_selftest.py（合成 fixture：排序/门槛/双目录去重路径/字段/top 截断） | D048r |〔R2 canon 臂条件标签=v21 语料 walk_phase_rate+canon vbin_meta edges [0.085,0.141] digitize，生成式记录于产物 vb_canon_rule_v21.meta.json；hotfix2=load_vae 训练侧完整类 strict=True（R2 encode 全链必需）〕
| `build_exp3_dataset.py` | 数据构建 | 合并 exp1+exp2+exp3 raw → `exp_all3`（68,093 步） | D026 |
| `recover_torque_data.py` | 工具 | 为已录 SONIC 闭环数据恢复 PD 力矩标签 | 方向 A |
| `recover_id_torque.py` | 工具 | 用 mj_inverse 重放相位路由器恢复逆动力学力矩 | 方向 A |
| `train_torque_decoder.py` | 训练 | 训练论文式力矩解码器 (phase+cmd → 12-d 腿力矩) | 方向 A |
| `eval_torque_paper.py` | 评测 | MuJoCo 闭环评测论文式力矩控制 | 方向 A |
| `train_phase_router_v9.py` | 训练 | **v9 相位路由器**：从 exp_all3 重建（19 命令组） | D-蒸馏最终 |
| `train_token_vae_e27.py` | 训练 | **E27 相位条件化 token VAE** | E27 |
| `train_token_vae_e31.py` | 训练 | **E31 训练 SpeedPhaseTokenVAE**（速度条件化，3 档速度 bin） | E31 |
| `train_token_vae_e35.py` | 训练 | **E35 训练 DirSpeedPhaseTokenVAE**（方向条件化，8 档方向 bin） | E35 |
| `train_token_vae_e37.py` | 训练 | **E37 方向解耦 VAE**（对抗 dir_head z→8，adv 3.0 + 类平衡 CE） | E37 |
| `train_token_vae_e39.py` | 训练 | **E39 双解耦 VAE**（dir_head z→8 + speed_head z→3 双对抗头；**D046 扩展全向后兼容（默认=原始行为）：--data-dir 参数化 + segment_bounds.npy 在场时启用不跨段窗口掩码（向后窗 10 帧剔除跨段窗口）+ --ckpt-every 每 N ep 快照 + --bin0-forward 语义标注 + heads.pt 落盘**；**D048n 增 --vb-npy 外部 vb 标签（默认关=旧口径零变化，pca/dbin 落盘照旧）**） | E39 / D046 / D048n |
| `train_token_vae_e43.py` | 训练 | **E43 快区加权方向解耦**（fast_extra 2.0，快 bin 行 3× 挤压） | E43 |
| `train_token_vae_e39_v2.py` | 训练 | **D046 v2：E39 fork +mode/UL 条件轴**（对 e39 canonical md5 56676924 最小 diff，逐行可审计）：可选 mode_embed(9,8)/ul_embed(3,8) 与 speed/dir embedding 同维拼接进 decoder 条件，缺省关闭时 state_dict/前向与原版逐位一致（--parity-check 对账 PASS）；可选 ul_head(z→3 对抗 CE，adv_ul 3.0)；--inputs-dir 吃 vae_inputs_v2 四件套，窗口按 segment_bounds 切不跨段（v1 同款），window_keep_mask 落 intermediate 0.3 权重，dev=冻结划分目录（v-bin edges 复用 train）；**dev/tfam phase 复用 train 投影基 pmean/V2（phi_rate_from，不自算 PCA——dev 无 WALK 帧时 per-split PCA 得空集 NaN 的修正）**；--walk-mode-idx 1 指定 WALK 为 phase/v-bin 基（v1 用 mode==2）；run：100ep/217s 零 NaN，落 `vae_v2/run1/`（vae.pt+每 10 ep ckpt+meta.json 含曲线）；**v2.1 零 diff**：n_modes=mode_id.max()+1 按 max_id+1 建座自动 10（IDLE embed idx=0 直接索引无需改码），--walk-mode-idx 2；run `vae_v21/run1/` 100ep/251.5s 零 NaN | D046(v2/v2.1) |
| `probe_vae_v2.py` | 评测 | **D046 v2 评测探针（owner 判据更正版，只前向）**：J1 dev/train 重建比+J8 域偏移定性；J2′ 条件通路可控性探针（替代作废的 head 阈值版 J2）：dev mu≥200 窗固定、遍历 db8/vb3/mode 条件解码，对 train 前向按真值条件分组的 token 质心做最近质心判定（≥2× 随机线）；J5 生成侧 mode 混淆矩阵（对角占优）；J6 UL 同法+asym-vs-none 二分（零支持类=无训练窗 embedding 不参与质心、单独登记——v2 池 sym 0 窗的教训）；J7 有限性；head acc 降级描述性统计；内嵌 v1 对照块；落 `run1/metrics_d046v2.json`；**v2.1**：默认指 vae_v21/run1+vae_inputs_v21，J6′ 三元矩阵含 sym 分支（sym 池非空后新增可测项）+sym_vs_none 二分，新增 **J9 枢纽指标**（IDLE 重建 MSE 单列+mode 扫描 IDLE 对角命中，只报数不下结论）+v2_comparison 对照块，落 `run1/metrics_d046v21.json` | D046(v2/v2.1) |
| `probe_vae_cross.py` | 分析 | **D046b 交叉 z 探针（单轴交叉转向矩阵，判别条件通路真琴键 vs 贴纸——输出跟 z 走还是跟条件走）**：每次只交叉一个条件轴，z 取条件值 i 的训练窗 mu、被测轴置 j、其余轴保持窗口真值，输出对 train 自条件前向按真值分组质心做最近质心判定 → M[i][j]=P(命中 j)（mode8×8/UL3×3/vb3×3/db8×8 各轴独立）；预注册判读量 cond_follow/z_follow/chance=1/k 三选一判定（cond≥2×chance→条件可用 / z≥3×cond→z 主导 / else 部分可控），行门槛≥150 训练窗（T-fam 零训练窗 mode 不入阵如实登记）；J9 hub 复测=mode 矩阵 IDLE 行（命令离开）/IDLE 列（命令进入）；附 IDLE z vs locomotion z 分布诊断（loco 基 PCA top8 逐 PC shift+std 比，排除「IDLE z 无信息」平凡解释）；与 probe_vae_v2 同加载/窗口/质心口径同快照；产物 metrics_d046b.json+summary_d046b.txt 落 `<run-dir>/../probe_cross/`；CVGL det 首跑 48c96t_512_3090 池 88s；**v2 修正版（owner 评审 2026-09-07）**：z_follow 改 off-diag P(pred==i)（v1 对角均值实现作废 z/c=1.53）+行列/argmax 名一律经 values[t] 映射（v1 mode 轴错位）+质心改 train 窗末真 token（y=decode 目标空间；v1 自重建均值未校准）+新增基线 P(命中 j|z_i,cond_i) 与逐轴 mean delta、diag 可辨识率+(mode,UL) 支持集分层 trained/unseen 两套矩阵+支持矩阵产物；产物 probe_cross_v2/{metrics_d046b_v2.json,summary_d046b_v2.txt}；det 10256 32s；**v3 二轮修正版（owner 评审 2026-09-07）**：训练支持集升四元 (mode,UL,vb,db)（R1 二元低估 OOD：IDLE 进入列 5,263 窗仅 465 窗四元有支持）四轴全分层+真 token 段级留出识别门（留出宏识别<2*chance→readout_unidentifiable）+verdict 措辞降级（撤「架构问题证实」，z_diag 收窄「未观察到所测投影方差塌缩」）；产物 probe_cross_v3/{metrics_d046b_v3.json,summary_d046b_v3.txt}；**v3.1 三轮修正版（owner 评审 2026-09-07）**：⑨留出分组改按源原始段（v3 按过采样后段序号奇偶，构建器过采样副本跨两侧——v21 材料 99 原始段展开 215 段、39 段跨两侧，同源窗抬高留出识别；v3.1 副本继承源段组别，身份=segment_source.npy 或 oversample_plan 三重 checksum 重放）+stem_hash/stem_halves 固定对照分组敏感性；⑩留出门加覆盖检查（缺类判覆盖不足，不得静默缩小分类任务后对原任务随机线放行）；产物 probe_cross_v3_1/{metrics_d046b_v3_1.json,summary_d046b_v3_1.txt}（v3 产物保留作泄漏口径存档） | D046b / D046b-R1 / D046b-R2 / D046b-R3 |
| `holdout_ident.py` | MODULE | **D046b-R3 留出识别与过采样身份工具（numpy-only）**：derive_segment_sources（从 build_meta 的 oversample_plan 确定性重放过采样计划，给过采样后每段标注源原始段下标；copies 计数+逐段帧数+总帧数三重 checksum 不符即 raise）+ holdout_splits（主分组=源段下标奇偶〔build 顺序=stem 升序，副本继承源段〕+stem_hash/stem_halves 对照）+ nearest_centroid_holdout（resub+留出最近质心识别，覆盖门=缺类判不可估计不缩小任务）；被 probe_vae_cross v3.1 与 d046b_r3_regression_test 共用（现役函数单一实现，测试不得复制逻辑） | D046b-R3 |
| `d046b_r3_regression_test.py` | 测试 | **owner 评审三轮反例回归覆盖（纯 numpy，全打现役函数）**：T1 过采样身份重放（真实 oversample_by_mode 复制→derive 重放→副本 token 逐位一致/copies 双向计数/篡改 raise）+T2 v3 副本跨侧缺陷性质+源段分组零跨侧+T3 缺类 0.5 误放行反例被覆盖门拦截（含全覆盖可分对照）+T4 同窗比值完整播放 1.0（v2 口径 1.0101 对照）/提前终止/hold/退化/越界防御；R2 教训落实：测试不复制实现逻辑 | D046b-R3 |
| `probe_vae_disentangle.py` | 分析 | E37 解耦探针：fresh 线性分类器 z→8 方向 vs 多数类 | E37 |
| `probe_vae_e39.py` | 分析 | E39 双探针：fresh 分类器 z→8 方向 + z→3 速度 vs 多数类 | E39 |
| `probe_vae_e39_bins.py` | 分析 | **per-bin 方向泄漏探针**（按速度 bin 分组，`[out_dir]` 参数可探 e43） | E40 归因 |
| `probe_fastbin_data.py` | 分析 | **数据侧探针**：快 bin 数据的方向分布（直行占比/左右平衡） | E40 归因 |
| `probe_vae_vbin_semantics.py` | 分析 | **D048m decode 档位语义离线探针**：同 z/相位扫 vb∈{0,1,2} 解码，测 token 流相位变化率排序（E39 标签生成量 walk_phase_rate，去 mode 掩码全帧当 walk）+档间 token 距离——判别 D048l 档位→速度闭环反转（b1→0.61/b2→0.45，与名义档位反向）发生在 VAE 条件层还是闭环执行层；预注册四分叉 monotone_up（条件层排序正常，反转归因执行层）/inverted（条件层自身反转）/flat（vb 不动节奏）/mixed（看 per_z），相邻相对差 10% 门；decode 调用/加载逐字镜像 apt_flat_env latent 分支（decode 第 1 参=策略 z、第 2 参=walk clock 的 sin/cos、db 固定前向档 4、vb 整流常量=force_vbin 语义），VAE 维数从 state_dict 形状反推（canonical e39=env 默认）；纯 torch 零仿真 CPU 可跑，产物 JSON（args/vae_md5/code_md5/per_z/aggregate/verdict）+stdout 表 | D048m |
| `eval_mjlab_fwd.py` | 评测 | **mjlab 从零策略原生任务评测**（自家 sim、60s 直行命令、[seed steps noise video]） | M-FROM0 |
| `replay_render_mujoco.py` | 渲染 | **MuJoCo 3D 真实模型 offscreen 渲染**（回放 Isaac rollout npz → mp4，MUJOCO_GL=egl） | 渲染管道升级 |
| `plot_latent_cmp.py` | 分析 | **E27–E30 对比图**（产出 `outputs/latent_cmp.png`） | E30 |
| `plot_paper_figures.py` | 分析 | **汇总图表六件套 fig1–fig6**（对照阶梯 / 速度-直行 Pareto / 地形形状边界矩阵 / 规划器线 / TO 战役 / E48 残差），数据优先读服务器评测 JSON、缺失项嵌 TRACKER 台账数字并标 J/T 来源，产出 `outputs/figs/` | 文档补全轮（2026-08-27） |
| `animate_skeleton.py` | 渲染 | **2D 骨架动画**（已降级为调试辅助，被 `replay_render_mujoco.py` 取代） | 渲染 |
| `router_fallback.py` | 评测 | 相位路由器的稳定性门控命令解析（回退表） | 优先级 2 |
| `flat_battery_fallback.py` | 评测 | 带 StableResolver 的全命令空间平坦地 battery | 优先级 2 |
| `switch_marathon_fallback.py` | 评测 | 60s+ 命令切换马拉松（经 StableResolver） | 优先级 2 |
| `interp_router_test.py` | 评测 | 连续潜空间：相位插值原型读取测试（v9 路由器） | 方向 B |
| `oracle_walk_bins.py` | 评测 | 新 walk 方向数据的 oracle 上限检查 | D028 |
| `eval_battery_v9.py` | 评测 | **v9 battery**：per-group 相位路由器目录闭环评测 | D-蒸馏最终 |
| `make_depth_dataset.py` | 数据 | 生成本地 depth → 特权 elevation 数据集（P2-lite） | 感知 |
| `train_depth_student_gru.py` | 训练 | P2-lite v2 深度学生（CNN+GRU+BPTT，最新版） | 感知 |
| `train_perception_distill.py` | 训练 | 感知蒸馏 demo（论文 stage 4 机制） | 感知 |
| `make_rough_xml.py` | 工具 | 构建本地粗糙地形 MJCF（heightfield） | 地形 |
| `make_terrain_fig.py` | 工具 | 地形实验汇总图（survival vs noise） | 地形 |
| `rough_render.py` | 渲染 | MuJoCo 粗糙地形路由器评测 + 视频（本地） | 地形 |
| `render_reel_v9.py` | 渲染 | v9 路由器 highlight reel（含新 walk 方向） | 渲染 |
| `rough_sweep_smooth.py` | 评测 | v9 walk 在平滑本地 hfield 的鲁棒性扫描 | 地形 |
| `stress_test.py` | 评测 | 60s+ 压力测试（蒸馏 PhaseRouterEncoder，无弹力带） | 优先级 2 |
| `perturb_eval.py` | 分析 | 扰动上限：oracle token + k 维偏 1 级的闭环存活 | 分析 |

## 2. `apt_g1/` 顶层 —— ARCHIVE（已移入 `_archive/`）

| 脚本 | 归档原因 |
|---|---|
| `train_phase_router.py` | 被 `_v9` 取代（相位路由器系列初版） |
| `train_phase_router_v2.py` / `_v21` / `_v23` / `_v4` / `_v5` / `_v8` / `_v8c` | 相位路由器演进中间版，最终采用 `_v9` |
| `eval_battery_v2.py` / `_v21` / `_v23` / `_v4` / `_v5` / `_v6` / `_v7` / `_v8` | battery 演进中间版，最终采用 `_v9` |
| `train_distill.py` / `train_distill2` / `train_distill3` / `train_distill4` | BC token 回归（D003），闭环 20-30x 复合误差，被相位路由器取代 |
| `eval_final.py` / `eval_final_v2.py` / `eval_final3.py` | 旧原型 battery，被 `eval_battery_v9` 取代 |
| `knn_eval.py` | kNN 记忆蒸馏初版，保留最新 `knn_eval2.py`（motion-matching） |
| `render_reel_local.py` / `render_reel_local_v6.py` | 旧版/v6 版 reel，保留 `render_reel_v9.py` |
| `rough_sweep.py` / `rough_sweep_slow.py` | 旧粗糙扫描，保留 `rough_sweep_smooth.py` |
| `flat_battery.py` | 无 fallback 的平坦命令审计，保留 `flat_battery_fallback.py` |
| `train_token_vae.py` | 早期 token VAE，被 `train_token_vae_e27.py` 取代 |
| `train_vae_lite.py` / `eval_vae_lite.py` / `train_token_seq_vae.py` | TVAE-lite / 序列 VAE 尝试，均未成功 |
| `train_depth_student.py` | 深度学生初版，保留 GRU 版 |
| `build_v6.py` | v6 专用构建，已过时 |
| `train_phase_ar.py` / `train_apt_phase.py` / `train_router.py` | 被取代的训练尝试（相位自回归 / APT-phase / 路由器蒸馏 v2） |
| `train_knn_mlp.py` | kNN 重标签 + MLP，死路 |
| `train_dagger_slow.py` | slow_fwd 的 DAgger-lite，死路 |
| `proto_variants.py` | 边缘组原型变体调参，探索性 |
| `stress_isolate.py` | walk_back 隔离测试，保留 `stress_test.py` |
| `build_closed_cycles.py` | 闭合周期数据（D 系列），闭合误差 0.00000 但无益 → 死路 |
| `eval_distill.py` | BC 蒸馏闭环评测（已判失败） |
| `eval_closed_router.py` | 闭合周期路由器重评测（死路） |
| `eval_apt_aux.py` | MuJoCo 端 APT aux 闭环评测（R 系列已终止）；Isaac 端等价物见 `isaac/eval_apt_isaac.py` |
| `motion_dataset.py` | （原 `data/`）孤儿源码，无 importer；移入归档以纳入版本控制 |

> `knn_eval2.py` 保留在顶层（kNN 最新版），但其结论（kNN 记忆蒸馏原理可行）已被相位路由器超越，仅供对照。

## 3. `apt_g1/isaac/` —— Isaac Lab 训练栈（全部原位保留）

| 脚本 | 角色 | 用途 |
|---|---|---|
| `__init__.py` | MODULE | 包初始化 |
| `apt_flat_env.py` | MODULE | Isaac Lab DirectRLEnv 平坦地 APT 环境（G1）；**D048h 增 `cfg.res_stats`（默认 False=零开销）+ `reset_res_stats()/pop_res_stats()`：按实际执行残差（冻结置零后、clamp 前）逐步累积 \|res\| 分位池/sat/near_sat/逐步差分/分关节均值，`pop` 收割并清零（eval `--res-stats` 逐局调用）；2026-09-09 修正：增 `_rs_signed_sum` 累加器与 `pop` 新键 `res_joint_signed_mean[29]`（带符号逐关节均值），`res_joint_mean` 实为各关节绝对值均值（历史键勿改名，120 份 JSON 可比性），此前「29 关节系统性正偏」系误读该键**；**D048j 增 `cfg.progress_cap_cmd`（默认 False=零变化）：progress 项封顶上界 1.0→当前命令（改调 `reward_terms.progress_bonus`），REW_CONTRACT_VER 改实例属性（`__init__` 里 `3 if progress_cap_cmd else 2`，类属性 2 保留为默认身份），`_last_rew_terms` 在 progress 分支激活时补 `progress` 键（--diag-log 分项同窗）**；D048l 增 cfg.force_vbin（评测干预覆写 vb，训练恒 -1）；**D049b 增 `cfg.vb_from_policy`（默认 False=零变化）：动作 [z(16),aux(12),vb_logits(3)]=31 维，末 3 维 logits softmax=软权重 w 直接作 latent decode 的 vb_soft（复用 D048q 通路、来源改策略动作，自然分桶 vb 仅日志对照）+ obs +3 w 反馈（`_last_vb_w`，reset 回零）+ 与 to42_sel/latent_residual 互斥、与 force_vbin/force_vbin_soft **可组合**（§5j b 臂 force 电池复测：decode 优先级 硬档>软混>策略w>自然分桶，`_resolve_decode_vb_mode` 纯函数；force 覆写时 w 不进 decode 但照常进 obs/日志）+ 前置校验 `_check_vb_from_policy` 纯函数，super 前用传入 cfg——3162292 教训（tmp/d049_test 经 ast 提取两函数单测真身）** |
| `apt_flat_env_vanilla.py` | MODULE | Vanilla RL 基线环境（无 SONIC 先验，E9/E11 对照） |
| `batched_router.py` | MODULE | 向量化相位路由器 encoder |
| `ckpt_identity.py` | MODULE | **D048h ckpt 配置身份信封**：`build_identity/save_ckpt/load_ckpt/verify_ckpt_identity/file_md5`——新格式 ckpt = `{"state_dict","ckpt_identity"}`（format=1：res_scale/res_clip/res_l2/res_freeze_steps/latent_mode/latent_residual/obs·action 维度/rew_contract/env_sha256/vae_md5/decoder_md5/git_head/it/entry/timestamp）；`load_ckpt` 兼容旧纯 state_dict（返回 `(sd,None)`）；`verify` 逐键比较，**2026-09-09 owner 裁定硬化：format=1 信封 7 个 VERIFY_KEYS（res_scale/res_clip/latent_residual/obs_dim/action_space/vae_md5/decoder_md5）在 expect 与 ckpt 两侧均必填非 None，任一侧缺失即失配条目点名（`xxx: missing on ckpt/eval side (required for format=1)`），不再静默跳过（身份信封必须自证完整）；legacy（无身份块/format 不符）维持缺键跳过不误报**；只依赖 torch+stdlib（无 isaaclab）。train 三保存点写身份块（7 键全供给），eval 加载时失配即 exit 6（D048h） |
| `elevation_map.py` | MODULE | 特权局部 elevation-map 观测 |
| `reward_terms.py` | MODULE | **D048j 奖励分项纯函数模块**（只依赖 torch，无 isaaclab——测试 import 不拉 Isaac 栈）：`progress_bonus(vx, cmd, cap_cmd)`——cap_cmd=False 返回 `clamp(vx,0,1)`（与 apt_flat_env 旧内联公式逐位一致）；True 返回 `min(clamp(vx,0,1), clamp(cmd,min=0))`（封顶上界=当前命令，静态最优恰为 cmd，消除 cmd=0.4 下静态最优≈0.534 的结构性超速偏置：vx=0.62 处每步反高 +0.033 → -0.187，stillness_vx_scale=0.05 计入）。被 `apt_flat_env.py` progress 分支消费（cfg `progress_cap_cmd`，默认 False=零变化；开启时 REW_CONTRACT_VER 2→3），单测 `test_progress_cap_reward.py`（2026-09-10 登记） |
| `sonic_decoder_torch.py` | MODULE | 纯 torch 重实现的 SONIC 解码器（ONNX→torch） |
| `sonic_decoder_isaac.py` | MODULE | Isaac Lab 批量化 SONIC 解码器（ONNX Runtime） |
| `terrain_cfg.py` | MODULE | 地形配置辅助（plane/rough/stairs/stones/discrete + **2026-08-15 G0 新增 `rough_paper`（论文形状：对称±噪声、0.2m 粗格）/ `rough_sym`（对称±噪声、0.1m 格，坑 vs 格子解耦对照）**） |
| `token_window_vae.py` | MODULE | token VAE 三件：**E27 PhaseTokenVAE + E31 SpeedPhaseTokenVAE（+速度条件）+ E35 DirSpeedPhaseTokenVAE（+方向条件）**，冻结解码器供 RL |
| `decft_policy.py` | MODULE | **E44 解码器微调策略**：E39 z头 → 冻结 VAE → token → **可训练 SONIC 解码器** → 29-d 关节目标动作（PPO 评分梯度直达解码器 + 官方解码器漂移正则） |
| `ppo_core.py` | MODULE | 向量化 PPO（含论文式训练附加项；E44 增加 `decoder_ft` 分支与 `decoder_reg_coef`；**E49 修复：GAE 边界 done|trunc 都切断递推且不自举 + aux_executed=False 时 aux 不进 log_prob/entropy + 真 epoch 循环 + approx_kl/clip_frac/act_std 指标，stats 键 `kl` 更名 `kl_prior`**）；**D048i 增 `kl_step_guard` 步级解析 KL 看门+资格检查（`kl_diag_gaussian_reverse`/`kl_categorical_reverse` 解析 KL(old‖new) + `ksg_*` 快照/还原/资格检查/看门步方法族，update() 内独立 `elif ksg_on` 分支，默认关=冻结版行为不变）**；**D049b 增可选 `vb_head`（`nn.Linear(hidden,3)`，gate 头完全同模式：Categorical 采样进 act()/update()/ksg_joint_logp/post_update_kl 的联合 log_prob 与熵，ksg 仅顺带记录 kl_vb 不进联合目标；旗标关=不建头零变化）** |
| `train_apt_isaac.py` | 入口 | 训练 APT（相位路由器先验 + aux）策略；TO42 修订 v4 增 `--ppo-minibatch`（默认 512 = 既有行为不变；2048envs 操作点用 4096）；**E49 增 `--token-mode/--token-phase-obs/--token-alpha/--token-bound/--token-stats`（直出 64d token，无 VAE）；E49 修复轮增 `--ppo-epochs`（默认 1 = 历史单遍）+ latent/token/to42 置 `aux_executed=False` + vx 拆 fwd（机体系带符号）/spd（模长，hist `vx` 键不变）双口径 + hist 增 approx_kl/clip_frac/act_std + `policy_it_0.pt` 初始快照**；**D048h：三 ckpt 保存点（it_0/50it 阶梯/final）改 `ckpt_identity.save_ckpt` 带 ckpt_identity 配置身份块（git_head/env_sha256/rew_contract 与 train_log 同源复用，vae/decoder md5 资产加载后各算一次），`--resume` 兼容解包新格式**；**D048i：`--kl-step-guard` 步级解析 KL 看门+资格检查（与 `--kl-guard` 互斥 argparse 报错；hist 增 14 个 ksg 键仅开启时初始化；资格违约逐条打印后 exit 8、连续死轮存 `_deadround` 诊断 ckpt 后 exit 7，两路径退出前均落盘 train_log）** |
| `eval_apt_isaac.py` | 入口 | A/B/C/D 评测；**E49 增 token-mode 同款旗标；E49 修复轮增 `--init-policy`（未训练初始化对照，`--checkpoint` 随之转 optional）；D048f 阶段0（metrics_contract=d048f_stage0）增：任务成功指标（vx_rmse/yaw_err_int/lat_max/fwd_max/survived_budget/task_success 冻结门）+逐局初态指纹（md5 配对核验）+每 entry 命令刷新+显式失败纪律（缺/坏 ckpt、空结果非零退出，不退回随机模型）+`--impulse-n`（term 分支探针）**；**D048h：ckpt 身份核验（新格式 ckpt 逐键比对 res_scale/res_clip/latent_residual/维度/vae·decoder md5，失配 exit 6 在跑局前拒绝；legacy WARNING 继续）+ `--res-stats`（env 侧残差执行统计逐局 res_diag 落盘 + 顶层聚合，metrics_contract=d048h_resdiag；2026-09-09：res_diag 增 `res_joint_signed_mean` 带符号逐关节键、顶层 `_agg_res_diag` 同步聚合）+ out 新增 `ckpt_identity` 块与 cfg `decoder_md5` 键**；**2026-09-09：ident_expect 补默认 apt 路径 `action_space`（14/31）——format=1 双侧必填硬化后 eval 侧 7 键不可再缺**；D048l：--force-vbin 强制速度档评测干预（eval_interventions 记录）+yaw_err_deg/heading_gate_pass 单位防误读 |
| `rollout_log_joints.py` | 入口 | **无相机 rollout → npz**（base 位姿 + 29 关节角，SONIC order，供 `replay_render_mujoco.py` 渲染） |
| `eval_fast.py` | 入口 | 守护式评测（只跑请求的 A/B/C/D 段） |
| `render_walk.py` | 渲染 | 从 APT Isaac 环境渲染短行走视频 |
| `inspect_decoder.py` | DEV | 检查发布版 SONIC 解码器 ONNX 图 |
| `parity_decoder.py` / `parity_layers.py` / `parity_onnx2torch.py` | DEV | ONNX↔torch 解码器数值一致性校验 |
| `check_isaac.py` / `smoke_isaac.py` | DEV | Isaac venv 导入检查 / 环境冒烟 |
| `e49_smoke.py` | DEV | **E49 直出 token 模式不变量冒烟**（obs 维度 / decoder 收到的映射 token / 反馈槽=原始 a / B 臂 φ obs 逐位 / 初始动作统计；2026-09-05 登记） |
| `e49_gae_test.py` | DEV | **E49 训练器修复确定性测试**（纯 torch CPU，无 isaaclab 依赖，仓库根 `PYTHONPATH=. python apt_g1/isaac/e49_gae_test.py`：朴素 GAE 对拍 / 手算边界小例 / done+trunc+last_value 切断不变性 / aux_executed=False 剔除不变量 / num_epochs step 计数 / approx_kl+clip_frac+kl_prior 指标 sanity；六用例全 PASS exit 0；2026-09-05 登记） |
| `e49_kl_guard_test.py` | DEV | **E49-C KL 信任域守卫单测**（纯 torch CPU，无 isaaclab 依赖，仓库根 `PYTHONPATH=. python apt_g1/isaac/e49_kl_guard_test.py`：解析对角高斯 KL 公式 vs torch.distributions 对拍 / kl_guard=None 默认关闭零回归 / 极小阈值全回滚+连续回滚断路 / 巨阈值探针形态零回滚 / grow lr 回复路径 / expl_var 口径；六用例全 PASS exit 0；2026-09-06 登记） |
| `test_ckpt_identity.py` | DEV | **D048h ckpt 身份信封单测**（纯 torch+tempfile，无 isaaclab 依赖，**服务器 .venv_isaac 运行**：仓库根 `PYTHONPATH=. python apt_g1/isaac/test_ckpt_identity.py`：带身份块存取往返 / verify 全匹配空清单 / 单键失配检出（res_scale/res_clip/latent_residual/obs·action 维度/vae·decoder md5，无误报键）/ legacy 纯 state_dict→(sd,None) / 损坏文件异常路径（eval exit 3 对应）/ legacy 缺键跳过（2026-09-09 硬化后仅限非 format=1 身份）/ file_md5 稳定+缺失容错 / format=1 必填硬化拒绝用例（裸 `{"format":1}` 对错误 res_scale 的完整 expect 被拒且点名缺失 / ckpt 侧缺 vae_md5 点名 / expect 侧缺键同样失配）/ 双侧完整回归（全匹配空清单、res_scale 失配单条、1e-9 浮点容差不变）；**十一用例全 PASS exit 0**；2026-09-09 登记，同日硬化更新 7→11 用例） |
| `test_kl_step_guard.py` | MODULE/test | **D048i 步级 KL 看门+资格检查单测**（纯 torch CPU，无 isaaclab 依赖，**服务器 .venv_isaac 运行**：仓库根 `PYTHONPATH=. python apt_g1/isaac/test_kl_step_guard.py`：KL(old‖new) 闭式 vs 手算标量+torch.distributions oracle 双对拍（含与 rsl_rl 口径 kl_diag_gaussian 的方向性区分）/ 同分布 KL=0（对角高斯+gate 离散头）/ ksg_snapshot/ksg_restore 参数+Adam 状态（exp_avg/exp_avg_sq/step）逐位精确还原（还原前先证第二步确有位移）/ 超大 lr 1e6 极小策略看门全程拒绝（参数零位移+lr_scale 触底 2^-6+资格无违约）/ 中等 lr 以 0.5×KL 探针阈回退后 lr_scale<1 接受 / 篡改 logp_old 资格检查函数级检出 / guard-off vs guard-on 核心统计逐位一致+键集合检查（零回归）；七用例全 PASS exit 0；2026-09-09 登记） |
| `test_progress_cap_reward.py` | MODULE/test | **D048j progress 封顶单测**（纯 torch CPU，无 isaaclab 依赖，服务器/本机均可跑：仓库根 `PYTHONPATH=. python apt_g1/isaac/test_progress_cap_reward.py`：cap-off 与旧公式 `clamp(vx,0,1)` 逐位一致（网格+随机+单点 vx=0.6→0.6）/ cap-on 超速封到 cmd（0.6→0.4、旧上界 1.0→0.4）/ 低于封顶不裁（0.3/0.399 原值）/ cmd=0 或负→0、负 vx→0 / cmd 张量逐元素 [0.2,0.4,0.6]/ cmd 标量广播（float 与 0-d）/ 静态最优 argmax 钉（v∈[0,1] 步长 0.001 网格：cap-on=0.4±0.002、cap-off∈[0.50,0.57]）/ f(0.62)-f(0.4) 偏置反转钉（+0.033→-0.187，stillness_vx_scale=0.05 计入）；八用例全 PASS exit 0；2026-09-10 登记） |
| `dbg_path.py` | DEV | 诊断 sys.path / PYTHONPATH |
| `server_apt_flat_env.py` | **FORK** | `apt_flat_env.py` 的服务端分叉（同源，body 已分叉） |
| `server_train_apt_isaac.py` | **FORK** | `train_apt_isaac.py` 的服务端分叉 |
| `server_eval_apt_isaac.py` | **FORK** | `eval_apt_isaac.py` 的服务端分叉 |

> **`server_*` 三件**与非 server 版 docstring 逐字相同但 body 差异显著
> （125/86/160 行 diff），是部署到服务器后独立演进的副本。交接包的 run-command
> 引用的是非 server 版（`train_apt_isaac.py` 等）。**哪套为"正统"未判定**，
> 待用户确认；在此之前原位保留两套。

## 4. 库模块（`sonic/` `encoder/` `envs/` `policies/`）

| 模块 | 内容 |
|---|---|
| `sonic/` | `sonic_wrapper.py`（SONIC 封装）、`token_vae.py`、`token_seq_vae.py`、`apt_manager_env.py` |
| `encoder/` | `phase_router_encoder.py`（蒸馏相位路由器）、`phase_ar_encoder.py`；导出 `PhaseRouterEncoder` |
| `envs/` | `g1_flat_env.py`、`mujoco_g1_flat_env.py`（MuJoCo G1 平坦环境） |
| `policies/` | `apt_policy.py`、`phase_aux_policy.py` |

## 5. 配置（`configs/`，23 个 yaml）

平坦地主线的历代配置。CANONICAL 为 `flat_g1_walk_noband.yaml`（最佳冻结零 token 行走）
与 `flat_g1_reference_aux*.yaml`（参考 token + aux 系列）。其余为各种尝试
（jointvae / seqvae / skill / vae16 / residual / ref_band_anneal 等），
保留供历史复现，不单独归档。

## 4. TO 力矩线脚本（TO01–TO22，2026-08-14/16 增补登记）

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `srb_to.py` / `srb_to_torque.py` | 数据生成 | SRB TO（CasADi）→ 2 连杆 IK → 力矩 | TO01–TO05 |
| `kinematic_gait_id.py` / `foot_gait_id.py` | 探针 | 关节空间/足空间运动学步态 + 手动全模型 ID 力矩（`M@qacc+qfrc_bias−qfrc_constraint`） | TO08/TO09 |
| `train_torque_decoder_gait.py` | 训练 | phase→τ_clean 解码器；**2026-08-16 起新暴露 `compute_gait_full`（含规划轨迹 Q/Qd）** | TO10 |
| `train_torque_decoder_to36.py` | 训练 | **TO37b（2026-08-30）**：TO36 dircol 解族→条件化解码器 MLP(sinφ,cosφ,v,phase)→6D τ（MJCF 符号，B 门 sign 映射内嵌）；留一速度泛化 + 全量 MAE，对照 TO10 基准 | TO37 |
| `eval_torque_gait.py` / `eval_torque_nmp.py` | 评测 | 力矩闭环冒烟；**2026-08-16 起支持 `--track-gait`（PD 跟踪规划轨迹）与 CoM 踝反馈** | TO11/TO15/TO18–TO20 |
| `eval_torque_srb.py` | 评测 | SRB 力矩前馈闭环（负结果） | TO06 |
| `train_aux_rl.py` / `probe_full_id_torque.py` | 训练/探针 | 力矩级 aux RL（负结果）/ τ_clean 符号探针 | TO17/TO07 |
| `planar_biped_model.py` / `nmp_biped.py` | 模块 | 平面 5 连杆模型 / 直接配点 NMP | TO12–TO14 |
| `wbc_gait.py` | 评测 | **TO23–TO33（现役主线）**：QP-WBC + LIPM/质心 MPC 参考层 + 全机制（梯形侧摆/捕捉落脚/加宽支撑/姿态正则/CoP 盒/knee-guard/H 任务）+ 诊断套件；TO32 修复锥约束空洞 bug 后加宽脚掌首破 8s，`--foot-halfy`+`APT_SCENE`（`foot_gait_id.py`）支持脚宽参数化 | TO23–TO33 |
| `lipm_gait_id.py` | 评测 | **TO21/TO22（新）**：LIPM 周期轨道（解析初值）→ IK（含平脚踝规划 `ankle=−hip−knee`）→ 全模型 ID → 闭环（`--ff-scale/--ankle-kp-boost/--stab-*` 扫参 + 逐关节诊断） | TO21/TO22 |
| `to36_leg_to_drake.py` | DEV | **TO36 v1 solve + B/C 门宿主**：v1 dircol（load/solve，D2 判死留作负结果对照）+ **D5 起 B/C 门现役**：`world`（Stage A，.venv_drake：hybrid foot 解→世界系 81 样本/相，骨盆解析加度+FK 目标）、`verify`（Stage B，.venv_mjlab：B 门双验证——基座行消去法 ID 解跟/尖 λ，绕 TO08 mj_inverse bug；符号映射 FK 搜索；D5 判定：effort PASS/数字口径 FAIL/支撑链 −5 N·m 归因 = URDF↔MJCF CoM 差 1.4 cm）、`closedloop`（C 门：矢状 τ 前馈+gait PD+bias/stab 归因参数）。两 venv 分段运行，过程坑见 tracker TO36-D5 | TO36 |
| `to36_common.py` | 模块 | **TO36 共享件（2026-08-29 决策 C 案抽出）**：v1 原样迁移的 resolve_model/build_plant/KnotKinematics 等模型构造与工具（D1 行为基线），v1 与 hybrid 共用 | TO36 |
| `to36_hybrid_dircol.py` | DEV | **TO36 现役主线（D3 起）**：hybrid 双相位 dircol 周期步态——pointe（5 体 pin）/foot（7 体全掌 weld+踝，真脚几何 URDF 解析）双模式；分离式 Hermite–Simpson（defect+插值）+ 相位接口 P 映射（foot 图映射含倒装踝双翻转，只在双脚平贴交集精确）+ 刚体冲击（辅助变量 λ）。**2026-08-30 修复两真 bug：#16 接口切片错位（假缝，此前全部收敛解作废）、#17 压缩式 HS 缺中点插值约束（混叠伪解：配点自洽但 knot 间能量漂移 45 J）**。配套**审计验收制**（IPOPT 证书仅供参考：每级过 HS/冲击/接口<1e-6 + 相内能量漂移<2 J + 冲击不产能）+ 重力斜坡同伦（slope_deg）+ --guess-npz 链式热启动 + --retries 混合重启 + --v-cap。**foot 平地刚性 A 门达成**（v 0.318 m/s，drift 1.7 J，审计采纳；F9 解备份 to36_hybrid_gait_F9.npz）；pointe 平地解（v 1.701）产自带洞转录、复核未过被 foot 取代。**D5 膝盒修正（F11）**：原对称 ±2.0 放行膝反屈（F9 映射后超真实限位 [−0.087,2.88] 至 46°，C 门根因）→ 相位感知盒（支撑 [−2.88,+0.087]/摆动 [−0.087,2.88]+踝收紧），初值支撑膝负弯曲 | TO36 |
| `to36_setup_drake_env.sh` | DEV | 服务器 `.venv_drake` 建 env 脚本（无 sudo：`--without-pip`+get-pip 引导 + `pip install drake`） | TO36 |
| `to38_analyze.py` | 评测 | **TO38（2026-08-31）**：双臂配对差分分析——读 `to38{a,b}_eval_*.json`，floor 检查 + 低速带 vx 跟踪误差配对差分 + 三分支判定（决策表在 TO38_PLAN §0/§4）；支持多 ckpt 配对（每臂 best=各自窗口最优） | TO38 |
| `to38_export_ref.py` | 模块 | **TO38（2026-08-31）**：`to36_world_knots.npz` → RL 注入用紧凑 LUT `to38_ref.npz`（M=120：q_ref6/tau_ref6/pitch/z/heel_rel + meta）。**关键：world npz q6 列序按角色排**（[支撑 A/K/H, 摆动 H/K/A]，两相互为镜像、符号按角色）——本脚本做逐相角色→SONIC 重排 + B 门符号 + 周期闭合检查（wrap_gap）。配套注入代码：`isaac/apt_flat_env.py` 的 `to_ref*` cfg（12 维 obs 块 + cmd 门控矢状跟踪 reward + 独立 ψ 时钟）、`train_apt_isaac.py`/`eval_apt_isaac.py` 的 `--to-ref*` CLI | TO38 |
| `to40c_analyze.py` | 评测 | **TO40C（2026-09-01）**：三臂（ctrl/t10/t05）配对差分分析——读 `{arm}_eval_{ck}*_a{cmd}.json`，floor 检查（completed/h_min≥0.6/disp>0.5）+ 门开带 vx 跟踪误差配对差分（δ=0.03 等效边界）+ 路径效率（disp/(v_speed·60s)，<0.5 判绕圈）+ 2×2 交叉注入诊断（{arm}_eval_{ck}_x{on|off}_a0277.json）；判定逻辑见 `refine-logs/TO40C_PLAN.md` §5。配套注入代码：`isaac/apt_flat_env.py` 的 `to_tau*`（kp 从 sim 读取） | TO40C |

## 6. `refine-logs/tools/` —— 文档树工具（2026-08-29 增补登记，非实验代码）

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `tree_check.py` | DEV | 实验记录扇出树完整性闸门（仿 mini Biosphere `doc_tree_check.mjs`）：挂树/实存/链接三项检查，任一失败 exit 1；树根与规则见 `refine-logs/README.md` | 文档基建（无实验号） |

## 7. `apt_g1/` artifact 工具（2026-09-01 增补登记，非实验代码）

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `gen_tau_dec_mapping.py` | DEV | 生成 Rung 1 treatment mapping artifact（`configs/rung1_tau_dec_mapping.yaml`，**schema v2 全交叉**：7 speeds × {C1,C2} = 14 rows，十二轮 B reopen 后 supersedes v1 恒等物化）：cfg 默认值从冻结 `apt_flat_env.py` 正则提取，bin 算子与冻结代码逐字同源（torch bucketize right=False），同输入重跑 byte-identical；产物 pre-run-only（无 observed/expected 字段），内建 gate A schema 自检；验收 gate 见 `refine-logs/TO40C_PLAN.md` §10.8、B 链状态见 `TO41_RUNG1_IMPL.md` | TO40C→Rung 1 |
| `to41_material_driver.py` | DEV | τ(v) material campaign driver：`validate` 子命令做全门集判定（G1 字段/G2 solver terminal success/G345 审计镜像 `_audit_pass`/G6 NaN-Inf/G7 速度容差 0.02/G8 配置身份），输出两字段 accounting JSON（solver_terminal_status ⊥ material_status）；`run` 子命令待 hot-start source 冻结后启用。规格与状态见 `refine-logs/TO41_RUNG1_IMPL.md` §5 | TO41（Rung 1） |

## 8. `apt_g1/rung1/` —— Mode A conditioning runtime + D independent checker（2026-09-02 登记）

D 阶段执行协议（`refine-logs/TO41_D_DRYRUN_PROTOCOL.md`，FROZEN）的可执行化。
**角色口径（协议 §10.2）**：runtime = **state-changing execution code**；checker =
**read-only audit**（只读取+独立计算，禁改 material/mapping/任何实验状态，
发现问题的唯一出口 = report FAIL → 协议 §7 保险丝）。checker 不 import runtime
（双解析器独立实现，selftest 交叉验证）。产物目录 `apt_g1/outputs/sync/to41_d/`。

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `rung1/mode_a_runtime.py` | 入口（**state-changing**） | Mode A 契约 `τ(v,C)=τ(v)` / `C=T_mapping(v,C)` 的执行器：mapping lookup（(v,arm)→condition_id）与 material lookup（v→τ material）**两个独立函数**，先落 immutable execution record 再 decode；CLI `--mode static`（28-cell 配置层覆盖核对，本机可跑）/`--mode execute`（decode-only dry-run receipt×28，**仅 lab-ts**，`--env-tag` 强制口径）。receipt 只含 record 字段，无任何 verdict/PASS/ok 字段（协议 §9） | TO41 D（Rung 1） |
| `rung1/d_checker.py` | 审计（**read-only**） | D independent checker：自带独立解析器读冻结 mapping v2 YAML / source registry / G_DOWN_SPEC §9 availability map，重算哈希（材料文件独立 sha256）、重算 D1（七字段+mode/layout/shapes）/D2（same-τ fingerprint+lineage）/D3A/D3B verdict；receipt schema 封闭 + 全域封禁自报 verdict 字段；禁收 performance 字段（协议 §4）；本机运行恒标 `--env-tag local`，lab-ts D 报告必须 `--materials-root` | TO41 D（Rung 1） |
| `rung1/rung1_selftest.py` | 工具（自测） | checker 逻辑自测（synthetic receipt，永不作为 D artifact）：T0 双解析器交叉一致 / T1 lookup 单元（Mode A 恒等式）/ T2 静态覆盖 28/28 / T3 正例控制 / **Negative A–E**（τ 换 hash→D2+D3B FAIL；condition 错 arm→D3A FAIL；decoder 超参/shape 篡改→D1 FAIL；自报 PASS+实际不一致→仍 FAIL；lineage 缺失→schema FAIL）。dry-run 前必须全绿 | TO41 D（Rung 1） |

## 8b. `apt_g1/rung1/` —— launch sanity（真实 env 接线验证，2026-09-02 三十七轮登记）

L1–L4 纯执行 gate（判据唯一事实源 = `refine-logs/TO41_LAUNCH_SANITY.md`；
D protocol FROZEN 不变）的可执行化：把冻结 Mode A runtime 接入**未改动的**
真实 `apt_flat_env.py` τ 注入/控制路径，28 cell 只测接线、不测性能。
**接线纪律**：零 env 文件改动（env sha256 = mapping `preprocessing_hash`
冻结锚，checker 机械校验）——condition 轴 = 实例级 shadowing `env._vae.decode`
（冻结 bucketize 每步照跑，wrapper 记 natural/applied 双记录）；τ 轴 =
`cfg.to_ref_npz` 既有通道 + 实例级探针 `_to_ref_lookup`（:768 唯一消费点）。
l_checker 不 import launch_sanity/env_wiring（audit 独立性同 D §9）。
产物目录 `apt_g1/outputs/sync/to41_sanity/`。

| 脚本 | 角色 | 用途 | 对应实验 |
|---|---|---|---|
| `rung1/env_wiring.py` | 模块（**state-changing**） | Mode A→env 接线 shim：`ConditionOverrideHandle`（decode 调用点拦截，per-call natural/overridden 双记录）/`TauConsumptionProbe`（τ 消费点拦截，per-call digest+buffer 快照）/`canonical_array_sha256`（冻结数组身份规范化，checker 侧独立实现交叉验证）。只产 record，无 verdict 字段；Rung 1 compute 的 eval 侧接线届时复用同一实现（conformance 随之继承） | TO41 L（Rung 1） |
| `rung1/launch_sanity.py` | 入口（**state-changing**） | 28-cell 真实 env 接线驱动：`--mode static`（配置层覆盖核对，本机）/`--mode execute --cell-index N`（**仅 lab-ts，per-cell 进程模型**——每 cell 独立 python 进程 + AppLauncher，receipt 落盘后 os._exit 硬退出，服务器 bash 循环 28 次；新建真实 AptFlatG1Env（TO40C ctrl/t10 配方 × τ(v) 材料 × e39 vae，两臂唯一 cfg 差={to_tau}），jitter_and_reset + 恒定 cmd_vx 每步重申 + 中段强制 episode boundary + 零动作驱动真实 step 循环，不调 env.close()（sim.clear_instance/挂死 gotcha））。冻结锚 preflight（env/vae/arch 三源哈希 + 7 材料可达，Isaac 启动前 fail-fast，每 cell 进程都跑） | TO41 L（Rung 1） |
| `rung1/l_checker.py` | 审计（**read-only**） | L1–L4 independent checker：L1 τ material consumption（buffer 哈希 = checker 独立 np.load 重算 + 4 cell/v Mode A fingerprint + ON 消费/OFF 零注入）/ L2 override persistence（逐 call natural vs applied + checker 重算自然 bucketize + boundary 后 persistence）/ L3 ON/OFF isolation（两臂 cfg diff == {to_tau}）/ L4 28-cell receipt（冻结枚举 + 冻结 env 源哈希锚 + decoder 签名表）。schema 封闭 + 封禁自报 verdict 字段 + 禁收 performance 字段；本机恒 `--env-tag local`（报告文件名显式 not_L_artifact） | TO41 L（Rung 1） |
| `rung1/l_selftest.py` | 工具（自测） | sanity 自测（synthetic receipt + mock env，永不作为 L artifact）：T-L0 双解析器交叉一致 / T-L1 wiring handle mock 单测（override 逐调用/契约 fail-fast/digest 确定性）/ T-L2b canonical hash 双实现交叉 / T-L3 合成正例全 PASS / **Negative A–E**（τ buffer 换 hash→L1+L3 FAIL；boundary 后 applied 回退→L2 FAIL；两臂混入额外 cfg 差→L3 FAIL；receipt 缺件→全级联 FAIL；自报 verdict→schema FAIL）。execute 前必须全绿 | TO41 L（Rung 1） |

## 8c. `apt_g1/rung1/` —— Rung 1 正式评测栈（execution freeze 后 compute，2026-09-03 三十九轮登记）

| 脚本 | 角色 | 用途 | 实验号 |
|---|---|---|---|
| `rung1/eval_cell.py` | 入口（**state-changing**） | 28-cell × train_seed 正式评测驱动（per-cell 进程模型同 launch_sanity；`--cell-index 0..27 --train-seed 0/1`，receipt 落盘后 os._exit）：两 lookup 原样复用 mode_a_runtime（Mode A τ(v,C1)=τ(v,C2) 机械继承）+ build_cell_cfg 原样复用（两臂 cfg diff=={to_tau} 继承）+ env_wiring 复用（override 每 call 双记录 → 60s rollout 的 reset 后 persistence）；policy = selection manifest 固定的单一 ckpt（checkpoint selection 与 eval condition 隔离）；每 eval seed jitter_and_reset + 恒定 cmd 每步重申 + 确定性策略动作 + termination 即停；eval seeds = {0,1,2} 预注册清单；`--smoke` 隔离目录。receipt = rung1-eval-receipt/v1（outcome 聚合为 record，verdict 只出自 eval_checker） | TO41 R1（Rung 1） |
| `rung1/select_checkpoint.py` | 分析（**read-only**） | checkpoint 机械选择：各臂 train_log 50-iter 窗口最优 → ckpt=窗口末 `policy_it_{N}.pt`（tie 取最小窗口序；TO40C §4 预注册规则逐字机械化，四臂对称执行）；产出 `ckpt_selection.json`（driver 消费 + checker 比对的唯一 ckpt 身份源）；另录 policy_final.pt 作稳健性对照（非 primary）。同一 (arm,seed) 全部 14 cells 共用同一 ckpt | TO41 R1（Rung 1） |
| `rung1/eval_checker.py` | 审计（**read-only**） | eval receipts 独立审计（不 import 被测代码；mapping/LUT/数组哈希/natural bucketize 审计侧独立实现）：G1 28×seed 覆盖精确 / G2 每 (arm,seed) 单 ckpt == selection manifest（C1→A/C2→B 或 ON/OFF 各选各的 = FAIL）/ G3 消费 LUT 数组身份==冻结 manifest / G4 每 (v,seed) 四 cell τ 消费身份唯一且==冻结 LUT（Mode A env 层证明）/ G5 override 全 call 生效 + call 数语义（ON=steps×decimation，OFF=0）+ buffer 恒定 / G6 归一化 cfg 全局唯一 + on/off diff=={to_tau} / G7 target∈冻结 grid + assignment==mapping lookup / G8 边界簿记 / G9 eval seeds==预注册 / G10 outcome 字段完整。只判 conformance，不做科学统计（双线纪律） | TO41 R1（Rung 1） |
| `rung1/eval_selftest.py` | 工具（自测） | eval 栈自测（synthetic 28-receipt 场景 + 仓内真实 LUT 副本，本机可跑）：T1 正例全 PASS / **N1 错 τ→G3 + driver preflight hard fail** / N2 override 覆写→G5 / N3 未授权 cfg→G6 / N4 coverage 缺口→G1+G4 / N5 OFF 臂 τ 泄漏→G5 / N6 ckpt 混用→G2 / N7 probe 记录不完整→G5。正式 eval 前必须全绿 | TO41 R1（Rung 1） |
| `rung1/eval_diagnose.py` | 分析（**read-only**） | 四十轮三连后 owner 裁定 (c) 诊断：仅消费已入仓 receipts + effect_table_v1（零新增执行，纯 stdlib 本机可跑）。五块：V 方差分解（eval-seed vs train-seed，读数取 effect_table per-eval-seed 配对差=主指标原生粒度；receipt 聚合字段与 err60s 口径不可互推，仅作形态证据）/ S eval seed identity 审计（jitter 代码事实 + 逐 episode bit-level 差异）/ N natural-vs-interventional（natural_vb_distribution 逐格提取 + bucketize 复算 + Δ_cond 双段拼接判定）/ D C1-C2 comparability（OFF 臂 support 区间重叠机械判定）/ W 分叉裁决（owner 预注册树映射，决定权在 owner）。产出 `sync/to41_eval/diagnosis_v1.{json,txt}` | TO41（四十一轮诊断） |

## 8d. `apt_g1/isaac/to42_*` + 仓根 `to42_cloud_wave.py` —— TO42 学习型 regime selection 栈（2026-09-03 owner 开跑授权登记）

| 脚本 | 角色 | 用途 | 实验号 |
|---|---|---|---|
| `isaac/to42_gate.py` | MODULE | 论文 gait-gate 语义在 {vb0,vb1} 二元 regime 上的纯 torch 状态机（2Hz 决策边界采纳 / 边界间锁存 0.5s / gate 布尔只在真切换步；fbkt 模式 = 每步 clamp(bucketize(cmd),0,1) 且 gate 恒静、策略位被忽略；reset 自然 bin 中性起步）——env 消费与 G0 自检是同一份代码；被 `apt_flat_env.py` cfg 门控加载（`to42_sel="off"` 时零接触） | TO42 |
| `isaac/to42_selftest.py` | DEV | G0 纯 torch 自检（**负例先行**，本机 CPU 全绿）：边界错位可察觉 / 非法参数拒绝 / fbkt 偏离即失败 / 非边界切换的坏实现可被抓到 / 冻结公式对照 torch.bucketize / fbkt 随机流逐位 / lsel 边界-锁存-布尔语义 / 策略 gate_k=2 头 + PPO gate 分支有限且梯度可达 | TO42 |
| `rung1/to42_eval.py` | 入口（**state-changing**） | 单 (arm×v×train_seed) cell 正式评测（per-cell 进程 + receipt 落盘后 os._exit）：harness 逐字继承 TO41（jitter rng(1000+seed) / 恒定 cmd 每步重申 / 确定性策略 / episode_length_s=120 → 60s 无 auto-reset / eval seeds {0,1,2}）；cfg = TO41 ctrl 臂形状 + `to42_sel`；receipt = to42-eval-receipt/v1（err60s / vx / disp / h_min + **selection 时间线 b64** + 切换步 + 策略选择头 p(vb1) 均值）；`--smoke` 隔离目录 | TO42 R1 |
| `rung1/to42_select.py` | 分析（**read-only**） | ckpt 机械选择：50-iter 窗口 argmax 规则逐字复用 `select_checkpoint.select_run`，臂集合 {lsel,fbkt}×{s0,s1}；manifest = to42-ckpt-selection/v1（同一 (arm,seed) 全 7 v-cells 共用同一 ckpt） | TO42 R1 |
| `rung1/to42_checker.py` | 审计（**read-only**） | eval receipts 独立审计（**先审计后分析，不读行为指标**）：C1 28-receipt 覆盖精确 / C2 84 episodes completed 零 fall / C3 每 (arm,seed) 单 ckpt == manifest / C4 env 源哈希 + vae sha + to42 cfg 跨 receipt 一致 / G0a fbkt 时间线逐位 == 自然 bin 且 gate 恒静 / G0b lsel 切换 ⊆ 2Hz 边界（t%25==0）。verdict 唯一出自本文件 | TO42 R1 |
| `to42_vram_probe.py`（仓根） | DEV | **L20 2048envs 显存/速度探针**（修订 v4 entry gate）：真实配方跑 3 iters + nvidia-smi 2s 采样，判据 peak ≤ 46G 且 rc=0 → `TO42_VRAM_PROBE` JSON 行；PASS 才发全链（防 OOM 中断） | TO42 R1 修订 v4 |
| `to42_cloud_wave.py`（仓根） | 入口（**state-changing**） | 云端 wave 编排（flux gm-run 入口；单 A10 pod 内全链 fail-fast，`--stages` 可子集重入）：G0 自检 → 双臂冒烟训练（30it）+ Isaac 级 G0 冒烟 eval（行内断言）→ 4 runs 全训（E47 配方 + ctrl 旗标 + τ 恒 OFF）→ ckpt 选择 → 28-receipt eval → checker → err60s 效应表（descriptive）→ 产物打包 `output/to42/to42_artifacts.pt`（平台 ckpt 发现通道）+ `TO42_RESULT_JSON` stdout 摘要。**v3 = 动态流水线（owner 09-04「还不够极限」）：训练完成即增量选择并动态注入该臂 7 评测格（评测与训练重叠，mid-band 优先）/ 显存自适应放行（nvidia-smi free − 90s 预留）/ 冒烟零训练（随机 init ckpt 验 wiring）/ bundle 原子写仅主线程**。**修订 v4 操作点（owner 09-04 规模指令）：2048 envs × 500it × minibatch 4096（论文式大并行，样本预算 4×）@ L20 48G（ESKU000005），训练并发 1（36G/臂），全程估 ~3.5–4.5h；发全链前置 = to42_vram_probe PASS** | TO42 R1 |

> **canonical 文件的 TO42 增量**（cfg 门控、默认 `"off"` = 行为与 TO41 逐位一致）：
> `apt_flat_env.py`（`to42_sel/to42_hold_steps/to42_n_sel` cfg + To42Gate 状态机 +
> decode vb 覆写 + obs 追加 [sel_state, gate_bool] + reset 自然 bin 起步）、
> `train_apt_isaac.py`（`--to42-sel/--to42-hold-steps` 旗标 + action 16→17 +
> policy gate_k=2 + buf["gate"] 槽位，PPO gate 分支原样复用）。
> **TO41 九项冻结清单不受影响**（TO41 复现路径全部走 off 默认值）。

## 9. DS 步态流形线（2026-09-04 登记；计划 = `refine-logs/DS_GAIT_MANIFOLD_PLAN.md`）

| 脚本 | 角色 | 用途 | 实验号 |
|---|---|---|---|
| `isaac/oracle_token_replay_isaac.py` | 入口 | **Phase 0 Isaac 执行保真度校准**：官方回路 RUN 录音（`/tmp/ds_smoke/policy_input.csv` 前 64 列 token，D033 录音复用不重采）→ `AptFlatG1Env` 子类旁路 policy/VAE/router，token 直进冻结 `SonicTorchDecoder`→q_des（D002 协议 Isaac 版；env 自持 10 帧闭环 history，canonical env 零改动）；自动定位 RUN 起始行（1s 窗位移 >0.7m 且后 20s 均速 >0.8）+ lattice 合法性抽检 + official_vx 自算；3 seed × 60s 回放，门判据 mean realized vx / official vx ≥ 0.9 → PASS（<0.9 对齐执行参数 ≤3 轮，仍不达降级 G3）；token npz 落 `data/ds_phase0/`，JSON 落 `outputs/ds_phase0/`；**含 D035 足底滑移审计**（contact=足框高 min_z+0.02 代理，接触期足速中位/p90/占空/步频/q 跟踪 MAE，预注册三档判定） | D034/D035 / Phase 0 |
| `roundtrip_official_control.py` | 评测 | **B2 判别对照（D036 control arm）**：官方闭环 WALK token（ds_smoke `policy_input.csv` 前 64 列，events.json walk_fwd_60s_baseline 窗 rows [1896,4896) 3000 行 @50Hz）过与 B2 完全同构的 decoder oracle 回环 harness（quat 工具/check3 机器/定窗全部 import 自只读的 `encode_bones_smoke.py` 防漂移；oracle proprio 取 `target_motion.csv` 官方 planner 参考：col0-2 root xyz + col3-6 root quat wxyz + col7-35 共 29 关节 MuJoCo 序（默认角 corr 0.72 + FK 足高双判据），50Hz 原生不重采样）；实测官方 MAE **0.136** < 同窗默认站姿基线 0.212 << 我方 B2 0.564 → **0.564 不是 harness 指标量级，B2 离线编码路径有真问题**（官方最差轴 waist_roll/pitch 0.41–0.45 + 右髋 pitch 0.53；我方 top5 全在髋 6/7/2/1 轴 1.3–1.9 rad）；JSON 落 `data/ds_bones/b2/control_official.json`；**根因修复注记（2026-09-04 代码审阅）**：encode 锚定块沿 planner_sonic 的 heading-norm apply_delta，样例 pkl 初始 yaw≈−87° → 全部 anchor 注入 ~90° 恒定世界 yaw（planner_sonic 参考首帧=站立 → delta≈identity，缺陷不可见），解释 D036/D037 的 yaw 类误差 1.3–1.9 rad 与侧倒摔倒；encode_bones_smoke.py 已加 `--anchor {ref-rel,heading-norm}` 默认 ref-rel（btr=conj(q_t)·q_idx 沿参考相对旋转：闭环唯一自洽形式 + yaw 不变量；f=0 anchor=identity sanity 已内置；v1 tokens 不覆盖，v2 存 `tokens_*_anchorrefrel.npy`）；**v2 ref-rel 重编码实测（2026-09-04）：roundtrip MAE 0.1094 rad——优于官方 token 同 harness 0.136，基线 0.223，v1 0.564；worst 轴全 ≤0.25（waist pitch 0.253/16/13、右髋 pitch 0.231），yaw 类 1.3–1.9 rad 误差消失；f=0 锚定 sanity = 0.00e+00；mean-L2 vs official walk 3.16→0.869；lattice 违例仍 0**；v2 JSON 落 `data/ds_bones/b2/smoke_result_anchorrefrel.json` | D036 / DS plan-B B2 |
| `isaac/replay_bones_tokens_isaac.py` | 入口 | **B3 预演（D037，预演非门）**：B2 离线编码 token（`data/ds_bones/b2/tokens_walk_forward_amateur_001__A001.npy` 2003×64，lattice 违例 0）按 D034 同构 oracle 回放链路闭环执行（`AptFlatG1Env` 子类旁路 policy/VAE/router → 冻结 `SonicTorchDecoder`→q_des，env 自持 10 帧闭环 history，`jitter_and_reset` 站姿，2 seed × 40s）；参考轨迹复用 B2 pkl（import `encode_bones_smoke` 同源 loader+重采样；关节序映射延后到 AppLauncher 之后——`gear_sonic...g1` import 需 sim 已启动，曾在此崩）；记录 fall 步/h_min/h_end/路径长/位移/均速/对参考 q 跟踪/PD 跟踪/摔倒姿态+高度轨迹（term 步不计入指标——DirectRLEnv 在 term 步内 auto-reset，否则 disp/h_end 被 reset 站姿污染）；实测 **2/2 摔（fall step 51/48 ≈1.0s）**：h 先冲高 0.763→0.835 再 ~0.8s 单调坍缩至 0.20/0.24（躯干翻转 96°/77° 主轴 ±x 侧倒型，disp_x −0.29/−0.41 m，路径仅 0.77/0.73 m vs 参考 16.84 m）；q 跟踪 vs 参考 0.736/0.727 rad、PD 跟踪 0.438/0.443 rad（D035 官方 token 回放 PD 仅 0.171）——链路机械上全通（token→decoder→env→物理→终止判定），内容上 token 步态质量差，与 D036 判读一致；两 seed 逐位可复现；JSON 落 `data/ds_bones/b3_rehearsal/rehearsal.json`；**坑：sim_app.close() 本身会挂死，已改为 daemon 线程 + 30s 超时 + os._exit(0)**；**v2（anchor=ref-rel）重跑实测【修复证实】：2/2 全程 40s 零摔（2003/2003 步，h_min 0.725 / h_end 0.779），路径长 16.33/16.34 m vs 参考 16.84 m（97%），均速 0.408 vs 0.420 m/s（97%）；q 跟踪 vs 参考 0.0705/0.0704 rad、PD 跟踪 0.1055/0.1042 rad（优于 D035 官方 token 回放 0.171）；绕圈未闭合净漂移 disp 4.40/4.32 m（对标物=路径长）；两 seed 逐位可复现；JSON 落 `data/ds_bones/b3_rehearsal/rehearsal_anchorrefrel.json`** | D037 / DS plan-B B3 |
| `isaac/b3p_gate_isaac.py` | 入口 | **B3' 正式门驱动（D044）**：吃 `convert_bones_g1_csv.py` 的 manifest（每段 npz：tokens (n,64)@50Hz + jp_isaac 参考 + trans_m）→ 类级 ≥10 段 × max(n_rows,500) 步 Isaac oracle 回放（D034/D037 同构 BonesReplayEnv 子类：旁路 policy/VAE/router，token→冻结 decoder→q_des，env 自持 10 帧闭环 history，jitter_and_reset 站姿起步，零 aux 动作；token 末行夹紧 = hold 相，fall_step 分离播放相/hold 相）；单 env 复用循环全部段（episode_length_s 按最长段设置），类级存活 ≥95% 且 n≥10 逐类独立判门；daemon close + os._exit 防挂死；**D046b 修正轮 instrumentation（2026-09-07）**：per_segment 新增 playback_path_m / playback_path_ratio（播放相路径及与全参考比）/ hold_disp_m 分列——旧 realized_path_ratio 分子累计到播放+hold 结束/摔倒，与完整参考不同时间窗（D047 注记：A101 参考 258 步 vs 实际 500/199 步），新列供未来门做同窗比值；不回刷历史 run；**v2 聚合修正（owner 评审二轮 2026-09-07）**：类级 n_segments 改唯一 stem 计数（旧版按回放条数计数，look 4 段×3 seed 曾记 12 段假过门）+主判 seed=min(seeds)（预注册缺省 seed0，加 seed 不得跨 ≥10 段门槛）+playback_path_ratio 改同窗比值（分母截断至实际有效步数，playback_t_used/total 落盘）；**v3 比值修正（owner 评审三轮 2026-09-07）**：同窗比值双侧同取 n_pb-1 个间隔——重置位姿无参考对应帧，实际侧首间隔（抖动起步沉降）扣除（playback_first_step_m 单列留存、playback_intervals 落盘），参考侧 trans_m[:n_pb]（v2 完整播放时实际 n_pb 个间隔 vs 参考 n_pb-1 个，系统性偏高 ~1/(n_rows-1)；100 步匀速例 1.0101）；计算收入 numpy-only `isaac/playback_window.py`（回归测试打现役函数）；D047 已按独立重组裁决保留、历史 run 不回刷 | D044 / D047 / D047-R1 / D046b-R3 / B3' 门 |
| `isaac/playback_window.py` | MODULE | **D046b-R3 播放相同窗路径比（numpy-only）**：playback_same_window_ratio(traj_pb, trans_m_xy)——时间对应约定=token 行 t 第 t 步消费、step 后状态↔参考行 t（与 q_track_mae_vs_ref_rad 同口径），重置位姿无参考对应帧故双侧均不含首间隔；实际=traj_pb[1:] 差分、参考=trans_m[:n_pb] 差分，各 n_pb-1 个间隔；ratio guard 参考路径 ≤0.5 m→None；被 b3p_gate_isaac v3 与 d046b_r3_regression_test 共用 | D046b-R3 |
| `isaac/replay_token_speed_d048r.py` | 入口 | **D048r R1/R2 快段闭环回放（decoder 本体上限 / VAE 重建代价三角测量）**：吃 `pick_fast_segments_d048r.py` 段清单（或单 --npz），骨架抄 oracle_token_replay_isaac（AptFlatG1Env 子类旁路 policy/VAE/router、token→冻结 decoder、env 自持 10 帧闭环 history、jitter_and_reset 站姿起步、Isaac 前 fail-fast、daemon close+os._exit、done 步不计指标）；20s×--seeds（默认 0,1,2）/段，token 末行夹紧=hold 相；双口径指标 mean_vx_fwd（净前向位移/时长，初始航向系）+ mean_speed_path（路径长口径，b3p 同式）+ realized_ratio（闭环 vs 参考 v_med）+ 播放相同窗比（复用 playback_window v3）+ survived（未摔且 h_min≥0.40）+ fall_step；--token-source orig=R1 原始 token / vae_recon=R2：token_window_vae DirSpeedPhaseTokenVAE 与 env 同源加载（strict=False）、build_windows 窗口、phase=pca.npz pmean/V2 投影（walk_phase_rate 投影法不自拟合）、vb/db 段帧级标签走 build_d048n.align_segments 确定性对齐+逐帧 md5 抽查（对不齐显式退出，D046b-R3 纪律）、encode 默认确定性 mu（--z-mode sample 可切）；聚合 per 段跨 seed 中位+per token_source 汇总（ceiling_vx_fwd=§5i decoder 本体上限）；产物 `<out>/d048r_<tag>/replay_runs.json` | D048r |
| `isaac/z_sweep_cem_d048r.py` | 入口 | **D048r R3 z 常向量 CEM（接口-主动层 decoder 上限包络）**：覆写 env `_compute_q_des` 不走策略，直接 tokens = VAE.decode(z_const, sin/cos(walk clock), vb_const, db_const)（clock 推进抄 env `_latent_phase` 固定 pca 步频、起步钉 0；db_const 默认 4=+x 前向档；vb 走 --vb-list 0,1,2 每档独立 CEM）；CEM=--pop 128（=num_envs 每 env 一个 z∈ℝ16 候选）×--iters 5×精英 top 16%（k=20），首轮 z~N(0,I) 逐 env 独立采样，每轮 20s（1000 步）rollout，fitness=净前向位移/20s（初始航向系；摔倒 env=-1 记 fall_step、终末 upright<0.9 判负，upright=exp(-g_xy²/0.1) eval 同式），下一轮 z~N(mean_elite,(std_elite+0.1²)I)（协方差对角=std_elite+0.01，塌缩时 std→0.1 防塌缩 floor）；停机规则=某档首轮存活率<0.5 停该档落盘分布事实（§5i）；CEM 更新核心=顶部纯 numpy 函数（torch 延迟 import，本机可单测 tmp/d048r_test/cem_selftest.py 七用例：精英选择/k 取整/哨兵排序/floor/重采样统计/种子确定性/平手稳定）；种子三线落盘（cem numpy/torch/jitter 规则）可复现；产物 `<out>/d048r_r3_<tag>/{z_sweep_summary.json, z_history.npz}`（每轮 fitness 分布/终轮 best z 16 维/vx*=终轮 best fitness/三档包络=接口上限） | D048r |
