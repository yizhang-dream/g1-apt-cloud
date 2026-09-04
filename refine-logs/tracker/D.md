# Tracker 系列：D（D 系列（蒸馏 Distillation 线 + Stress Test；另有 D021–D028 四行混排在 E.md 的『地形/数据泛化/感知』节））

> 【层位 L3｜Run 台账·系列文件（数据唯一事实源）】↑ `refine-logs/EXPERIMENT_TRACKER.md`（总索引）与 `HANDOFF/02_EXPERIMENT_HISTORY.md`（L2 阶段史）｜↓ `HANDOFF/03_OUTPUTS_INDEX.md` → 服务器 `outputs/`（L4）｜≈ `apt_g1/SCRIPT_MAP.md`（代码轴）。
## Distillation Experiment (2026-08-11, see DISTILL_EXPERIMENT.md)

| Run ID | Purpose | Variant | Data | Metric | Status | Result |
|--------|---------|---------|------|--------|--------|--------|
| D001 | Expert data collection | official closed loop, no band, 4 modes | 20,838 ctrl steps (50Hz) | - | DONE | tokens == token_state.csv (max diff 0.0); encoder output on k/16 lattice |
| D002 | Harness validation | oracle token replay in apt_g1 env | same | survival 400-600 steps | DONE | idle/slow/walk/jump stable -> harness OK (walk vx~0.8) |
| D003 | BC regression | MLP / GRU / deep / transformer / AR | 20,838 | val per-dim 60-72% | DONE | closed loop all fall (3-10s); compounding: closed-loop token MSE 20-30x open-loop |
| D004 | AR-delta (teacher forcing) | prev token + delta regression | 20,838 | val per-dim ~100% | DONE | exposure bias; closed loop falls faster (~1-2s) |
| D005 | Random-error tolerance | oracle tokens + k dims +/-1 level | - | survival | DONE | up to 8/64 dims no effect -> failure is systematic, not decoder sensitivity |
| D006 | kNN memory distillation | nearest official state -> token | train rows only | survival 600 steps | DONE | idle/slow/walk/jump all stable -> state->token learnable in principle |
| D007 | Phase classification router | fixed-period bins + classifier | 20,838 | phase acc | DONE | idle/slow OK; walk/jump fail (1Hz replan breaks fixed period) |
| D008 | Phase regression router | PCA circular phase + MLP(sin,cos) + 40 prototypes + EMA0.3 | 20,838 | - | DONE | idle 3/3, slow 3/3, walk 3/3 (vx 0.81-0.83, 16.2-16.6m/20s), jump 1/3; 40s switch episode passed |
| D009 | Command-switch episode | idle->walk->idle->slow->jump->idle | - | survival 40s | DONE | complete, h_min 0.74 |

## Distillation Phase 2 (2026-08-11, exp2 + routers v2-v5)

| Run ID | Purpose | Variant | Data | Metric | Status | Result |
|--------|---------|---------|------|--------|--------|--------|
| D010 | exp2 data collection | backward/turns/strafes/more jump/stealth, no band | 32,675 steps, 0 falls | - | DONE | merged exp_all = 53,513 steps, 5 modes |
| D011 | router v2 (merged, angle bins) | (mode,speed,8-dir-bin) groups | 53,513 | survival 20s x3 | DONE | walk_fwd 3/3, walk_back 3/3, idle regressed, turns stand/stumble |
| D012 | router v2.1 (density filter + 2D metrics) | NN-distance outlier filter; fdir=mdir fix | 53,513 | survival 20s x3 | DONE | idle fixed 3/3; walk 3/3 both dirs; slow standing (vx~0.01) |
| D013 | router v3 (density filter per group) | same | 53,513 | survival | DONE | slow prototypes still standing (vx 0.009) -> data heterogeneity issue |
| D014 | router v4 (slow=exp1 only) | slow group restricted to exp1 | 53,513 | survival | DONE | slow vx 0.07-0.16, 2-3/3 |
| D015 | router v5 (slow=exp1 phase1 only) | slow group = first exp1 slow phase | 53,513 | survival | DONE | slow 2/3 (vx 0.07-0.25); walk fwd/back 3/3 (16-17m); switch 58s passes; jump 1/3; turns/strafes/stealth at or near oracle ceiling |
| D016 | Oracle ceiling check | official turn/strafe token replay | - | survival | DONE | turn bins 1/6/2 fall ~200 steps -> distilled cannot beat teacher; env caps curved-motion ceiling |

## Distillation Phase 3 (2026-08-11, proto tuning + DAgger)

| Run ID | Purpose | Variant | Data | Metric | Status | Result |
|--------|---------|---------|------|--------|--------|--------|
| D017 | Prototype tuning sweep | mean/median/nearest x B=40/64 for marginal groups | 53,513 | survival 20s x3 | DONE | jump median/B40 -> 3/3; turn_right mean/B64 -> 3/3; turn_left/strafe_left nearest/B40 -> 3/3 with real motion |
| D018 | Final router (v6) | best per-group config | 53,513 | survival 20s x3 | DONE | idle 3/3, slow 2/3, walk fwd/back 3/3 (0.83/-0.78 m/s), jump 3/3, turns 3/3, strafes 3/3, stealth 0/3 (=oracle), 58s switch passes |
| D019 | DAgger-lite for slow | student states + kNN phase relabel, retrain slow net | 2,543 new samples | survival | DONE | regressed to standing (vx 0.01); not adopted; weak-rhythm gaits need cleaner phase labels or RL |
| D020 | Stealth oracle check | official stealth token replay | - | survival | DONE | oracle falls at step 361 -> stealth 0/3 is teacher-bound, not a distillation gap |

## Stress Test (2026-08-12, encoder consolidation)

| Run ID | Purpose | Variant | Horizon | Metric | Status | Result |
|--------|---------|---------|---------|--------|--------|--------|
| D021 | Single encoder module | PhaseRouterEncoder (group select + EMA + Command.from_vxvy) | - | API | DONE | unified encode() validated end-to-end; matches inline eval (vx 0.83) |
| D022 | 60s straight walk (fwd/back) | long single-command runs | 60s x3 | survival, disp | DONE | walk fwd 3/3 (50.9-52.0m), walk back 3/3 (48.7m) |
| D023 | Disturbance grid | 200/500N impulses x 4 dirs x 3 seeds during walk | 45s x24 | survival, recovery | DONE | 21/24 complete; recovery 0.02-2.6s; 3 seed-dependent late falls |
| D024 | Command-switch marathon | 68s mixed schedule x 3 seeds | 68s x3 | survival | DONE | 0/3; falls at jump (2 seeds) / walk_back (1 seed); earlier 58s pass was a 20s episode-length artifact |
| D025 | Isolation | walk_back 60s; walk40->idle->jump | - | survival | DONE | walk_back 3/3 standalone; jump-after-40s 2/3 (h_min~0.21) -> jump under prolonged running is the residual fragility |


## Recollect line (2026-09-04, DS_RECOLLECT_PLAN.md; owner 方向转向：停 TO 线主攻数据重采)

| Run ID | Purpose | Variant | Data | Metric | Status | Result |
|--------|---------|---------|------|--------|--------|--------|
| D029 | SONIC 全动作组核查 + RUN/速度轴冒烟 | deploy LocomotionMode 枚举考古（27 mode）+ pty 键盘驱动 9 段：WALK 基线 60s / RUN(3) 默认 60s / RUN '0' 阶梯 ×2/×4/×6 各 30s / SLOW_WALK 阶梯 ×2/×4/×6 各 30s | ds_smoke: commands 42,020 行 + policy_input 21,015 行 + target_motion 21,015 行（lab-ts apt_g1/data/ds_smoke/） | falls=0 全段; target_vel 阶梯实测 | DONE | **RUN mode 从未采过（exp1/2/3 只按过 1/2/4/5）但完全可用**：默认 60s 零摔（target_vel 1.5），速度阶梯 1.5→1.7→2.1→2.7 全部站住；SLOW_WALK 阶梯贴 0.8 上限；WALK 键盘路径速度恒 -1 不可调。deploy 枚举：SLOW_WALK 0.1–0.8 / WALK 0.8–2.5 / RUN 2.5–7.5（键盘钳 1.5–3.0），三段相接 = 论文 trot/bound 结构同构物。材料上限修正：vx 0–1.0 是采集子集事实，非 SONIC 系统上限 |
| D030 | mode×terrain 资格赛 Phase A | 21 非静态 LocomotionMode 全集 × {flat, rough_mid(论文形状对称±0.04/0.2m粗格,mjlab官方生成器)} × 3 seed；harness=ds_mode_terrain.py（planner_closed_loop 参数化 fork，10Hz 重规划 + ONNX planner 直调任意 mode；mjlab 地形内存组装注入 env；episode 上限 20s=1000步）；6 进程并行 lab-ts | 126 格 × JSON 行（adv/h_min/h_end/fall） | survive = fall==null 或 h_end≥0.55；fell = h_min<0.40 | DONE | **Phase A 判读**（126/126 全格，episode 20s 上限）：
①rough_mid（论文形状中档）对多数 mode 不构成威胁——18/21 mode rough 存活
≥2/3，WALK rough 前进 10.24m≈flat 10.31m（MQ09 walk≈flat 在论文形状上复现）；
②**唯一被 rough 显著打击 = mode 3 RUN**（flat 3/3 adv8.54 → rough 1/3 adv6.86、
全场唯一 h_min 0.22 真摔）= 速度换地形脆弱；③**前进排序（rough_mid）**：
WALK(2) 10.24 ≈ HAPPY_DANCE(23) 9.06 > RUN(3) 6.86(1/3) > SCARE(26) 3.94 ≈
FORWARD_JUMP(17) 3.61 > SLOW(1) 3.18 ≈ INJURED(19) 2.92 > STEALTH(18) 1.76 >
ZOMBIE(24) 1.18；④**两个惊喜**：HAPPY_DANCE_WALK 在 rough 上 9.06m 全存活
（超 flat 8.44）；FORWARD_JUMP flat 0/3→rough 3/3 且 adv 3.61（跳跃在坑洼地
反而有效——直接支持「跳跃越障」假设）；⑤crawl(8)/elbow(14)/ledge(20)/拳系
(11-16)/stealth_2(22)/gun(25)：存活但 20s adv<1m（极慢/原地）——crawl 的
抗倒价值场景（MQ11 amp0.14 档）本档未触发，Phase B/C 加档验证。冒烟锚：walk×rough_mid 300步 fall=null adv 3.42；flat walk 20s adv 10.45m（走完大半 16m 场地）。benchmark 选型：论文原生不可得（项目页无代码无地形），主基准=mjlab 官方地形库（Isaac Lab 同源），hurdle/gap 自建标注（DS_RECOLLECT_PLAN §3.5） |
| D030b | Phase B 全矩阵 | 11 入选 mode × 6 崎岖地形类（stairs/stones/discrete mjlab 原生 + highstep/hurdle/gap 自建论文参数）× 3 seed × 20s，6 并行 | 198 格 | 同 D030 口径 | DONE | **〔gap_mid 数据作废——几何 bug：两板 center±4.8+half4.8 在 x=0 恰好接触，缝宽实为 0（证据=gap 成绩与 flat 逐位一致 walk 10.35 vs 10.31）；修复后 B2 补跑〕**。其余五类判读：①**stairs_mid(0.18m 级) 无威胁**——全 mode 3/3，WALK 全速 10.29m 上下台阶（G1 腿长优势）；②**stones_mid = 首个真分化地形**：WALK 0/3 全摔（h_min 0.21）/ RUN 1/3，SLOW/JUMP/INJURED/STEALTH/23/26 全 3/3——快步态踩空、慢步态+跳跃存活；③discrete(0-0.16m) 无威胁（RUN 1/3 复现脆弱）；④**hurdle_mid(0.4m 栏) = 全 mode 硬墙**：无一通过（最高 adv 0.7m），快步态被绊倒（WALK 2 摔/RUN 1 摔/HAPPY 1 摔）、其余被挡不摔；⑤**highstep_mid(0.45m 高台) 同为硬墙**：全 mode 在墙前被挡（adv 截断 ~5m 无摔）。**结论：planner 27-mode 步态库的翻越上限 < 0.4m 栏 / < 0.45m 台**——论文 hurdle 训练上限 0.792m 远在此上；B2 高度扫描（0.1/0.2/0.3m × WALK/RUN/JUMP/HAPPY）定量测各 mode 翻越上限 |
| D030c | gap 修复重跑 + hurdle 高度扫描 | gap_mid 真几何（0.8m 实缝）11 mode × 3 seed；hurdle_h10/h20/h30（0.1/0.2/0.3m 栏）× {WALK,RUN,JUMP,HAPPY} × 3 seed | 69 格 | 同 D030 | DONE | **①gap 修复版全线通过**：WALK 3/3 跨沟 adv10.41、RUN 7.94/HAPPY 9.01/JUMP 3.28/SLOW 3.13/INJURED 3.59 全 3/3——0.8m 沟=落点跨度问题，G1 步幅自然覆盖，无需感知（论文 gap 上限 1.5m 未测）；**②hurdle 扫描零通过且低栏更险**：没有任何 mode 通过任何高度（最高 adv1.30m<第一栏距离）；**0.1m 矮栏把 WALK/RUN/JUMP/HAPPY 全部真摔**（h_min 0.20，脚尖绊停前转力学），0.3m 高栏反而不摔（撞停被挡 adv~0.5）——盲重规划无抬脚感知，正障碍翻越能力=0；**③最终结论：27-mode 步态库在论文七地形中档的可通行域 = rough/discrete/stairs/gap（多 mode 可过，WALK 全速）+ stones（需选步态：慢步态 or JUMP）；正障碍（栏/高台）零通过**——「攻破论文崎岖地形」缺的不是步态（27 个全试完），是感知驱动的越障动作（planner 无高度图输入，接 MQ11「运动学盲重规划」与 MQ12「ADAPTING/感知→选步态」结论，与论文 perceptive+jump 技能侧对应） |
| D031 | 单栏相位诊断（owner 时机假设检验） | 单根 0.1m（及 JUMP 加测 0.2m）栏 @x=0；出生点扫描 -6.0→-3.0 步长 0.25（13 档=13 个到达相位）× {WALK, RUN} h10 + {JUMP} h10/h20 × 2 seed，30s/格 | 104 格；判据 x_end>0.5=过栏 | DONE | **owner 时机假设证实（对 WALK/RUN）**：同一 WALK 步态，出生相位差 0.25m 即分三态——过栏（10/26 ≈38%，过栏点 x_end 0.5–0.77 踉跄通过）/ 栏前摔（h_min 0.20–0.23 脚尖绊停前转）/ 栏前停（x_end 0.27–0.44 触栏止步不摔）；RUN 同构 10/26。**0.1m 栏不是绝对墙，是相位彩票**——planner 盲重规划不知道栏位，摆动腿跨栏全凭落点运气。**JUMP 失败机制不同**：0/52 通过且多数 run 根本未到栏（x_end -4.7~0.5，h_min 0.5–0.67 平地跳停推进不足——FORWARD_JUMP 是小幅前跳非跑酷跳远，30s 平地仅前进 ~1m）；到栏的少数相位也 0.2m 栏前摔（h_min 0.26）。**结论：①迈大步步态本身具备跨 0.1m 的摆动能力，缺的是相位对齐；②JUMP 缺的是幅度/推进，不是时机；③感知最小价值目标 = 把 38% 相位彩票变确定通过（高度图→相位同步触发），完整版 = 论文 perceptive gait selection（感知条件化 regime/时机选择，与 TO42 learned selection 汇合）** |
| D032 | RUN 平地速度 bug 三层诊断 + stones 死点修正（owner 质疑触发） | ①ds_mode_terrain 加 --target-vel：RUN flat × {-1,1.5,2.5} 闭环对比；②planner ONNX 裸输出轨迹 root 速度直测；③stones 逐 seed x_end 复查 | 闭环 3 格 + 裸 planner 5 配置 + 明细复查 | realized vs planner vx 比 | DONE | **①RUN「慢」不是数据层 bug——planner ONNX 裸输出 RUN 轨迹 root 前进 2.12 m/s（真跑步材料），但 MuJoCo 蒸馏闭环只实现 0.37 m/s（17%）；执行层速度衰减系统性存在：WALK 0.91→0.51（56%）/SLOW 0.54→0.15（28%）——速度越快衰减越狠（候选根因：PD 位置控制相位滞后/腾空相无锚定/obs 采样率压缩高速轨迹；旁证：D002 oracle walk 回放 0.8≈planner 0.91 的 88%，闭环比 oracle 慢）；②target_vel ONNX 行为非单调：-1≈2.5（逐位同轨迹）>1.5（1.27 m/s）——显式传值需校准，默认用 -1 哨兵；③〔修正 D030b 判读〕WALK stones 0/3 三 seed 全死 x≈0.5（0.67/0.47/0.50 超集中）=中央安全岛（x∈[-0.5,0.5]）东侧出口——**WALK 实际穿过 ~5.4m 石头阵未摔，死点是平台↔石头阵过渡带**；23 号死的 seed 亦死 x=-0.59 西入口——踏石杀手=过渡带非单石（与 D031 栏相位同构）；RUN stones 存活格走 5.9m，此前判读过严。**设计影响：RUN 高速段材料有效（2.12 m/s）流形设计维持；执行层衰减列为新风险条目（真机 deploy 回路对照待查）；过渡带风险强化采集设计中过渡段的必要性** |
| D033 | 官方 deploy 回路 RUN realized 对照（D032 衰减定位收口） | base_sim.py 临时 DSPOS 探针（100 步/print root x/z，用后已还原）+ drive_run_probe 官方回路 RUN 60s | sim_probe.log 763 探针 | RUN 窗口 x 位移/时长 | DONE | **衰减两层定位完成**：planner ONNX 裸输出 2.12 m/s → **官方 deploy 回路（WBC sim）~1.0 m/s（62m/60s，48%；z 大体 0.7+ 一次 0.21 低谷踉跄、0 falls）** → 我们 MuJoCo harness 0.37 m/s（17%）。即：官方 WBC 执行栈能实现 planner 的一半（比我们 harness 好 3×，差在执行配置：我们 apt_g1 env 位置式 PD+50Hz vs 官方完整 WBC）；另一半衰减在更深的物理/解码链（腾空相、decoder 平滑）。**设计含义：①数据继续走官方回路采集不变（记录的就是真实可执行行为 ~1 m/s 跑步）；②执行层衰减从风险条目升格为 Isaac env 校准任务（RL 环境若只有 0.37 上限则学不出快步态，需向官方 WBC 执行配置看齐）；③"直接从 planner 开环提取数据"再次否决——官方回路的物理检验正是把 2.12 压到 1.0 的那道闸，绕过它数据就含不可执行轨迹** |
| D034 | Phase 0 Isaac 执行保真度校准（DS_GAIT_MANIFOLD_PLAN §2 前置门，owner 指定最先执行） | D033 drive_run_probe 官方回路 RUN 录音复用（不重采；token 窗 = 行 [1048,4048) 3000 行 @50Hz，1s 参考位移 >0.7m 定位起始，lattice 违例 0）→ `oracle_token_replay_isaac.py`（SCRIPT_MAP §9）：AptFlatG1Env 子类旁路 policy/VAE/router，token 直进冻结 SonicTorchDecoder→q_des，env 自持 10 帧闭环 history（D002 协议 Isaac 版；canonical env 零改动），jitter_and_reset 与 E 系评测同源，3 seed × 60s | oracle_replay_isaac.json + run_tokens.npz（`outputs/ds_phase0/`，JSON 副本入 sync/）+ replay.log | realized vx / 官方回路 realized 1.033（D033 探针）≥ 0.9 | DONE | **PASS（第 1 轮，零对齐迭代）**：3/3 完成零摔（h_min 0.691 三 seed 全等，disp_norm≈disp_x 无漂移），vx 1.663/1.667/1.667 → 均值 1.6657，**实现率 1.61**。执行层衰减排序修正（对 planner 参考 2.086 m/s）：**Isaac 79.8% > 官方 WBC 回路 48.7% > 我方 MuJoCo harness 17.5%**——harness 的 0.37 归因进一步收窄到 harness 自身执行配置（PD 增益/力矩限制/10Hz 重规划边界扰动），与 token/decoder 无关；**gate≠机制 caveat：Isaac 比官方快 61%，两套 realized 互不外推**；足底接触真实性列 P2 抽检不阻塞（硬 PD 高 realized 或含打滑成分，需给 rollout_log_joints 加 oracle 模式后渲染核验）。**设计影响：G3 全速度段成立，Phase 4 解锁 cmd U(0,1.5) 第二臂分支，Phase 1 解禁** |
| D035 | Phase 0 打滑核验（owner 定为最关键项；触发面 = `DS_S2R_EVIDENCE.md`） | D034 同一 oracle 回放加足底审计：contact 代理 = 足框世界高 < min_z+0.02（URDF `left/right_ankle_roll_link`），测接触期足水平速度分布 + 双足占空 + 步频 + 关节跟踪 MAE；**预注册判定 = 较差足中位接触速度 <0.15 m/s 诚实 / <0.4 部分打滑 / ≥0.4 滑行** | oracle_replay_isaac_d035.json（sync 副本）+ replay_d035.log | 3 seed 逐足中位接触速度等 8 指标 | DONE | **HONEST（3/3）**：中位接触足速 0.0377–0.0381 m/s（钉地无系统打滑），p90 0.92–1.03（落地/蹬伸瞬间少数帧），slip_frac(>0.2)=6%；单足占空 0.38–0.42 → 双足合计 ~0.8、**~20% 周期腾空 = 真跑步**；步频 1.35 步/s/足 → 步长 ≈0.62 m 合理；q 跟踪 MAE 0.171 rad（PD 非硬吸附，排除刚性拖拽伪影）。**结论：Isaac 1.6657 m/s = 物理诚实跑步步态；三 seed 与 D034 逐位复现（确定性佳）；渲染抽检免做**。Isaac 速度上限声明解除打滑保留意见 |

## B 线：官方数据（2026-09-04/05 夜，DS_SONIC_OFFICIAL_DATA.md；BONES-SEED × 官方 encoder 离线编码）

| Run ID | Purpose | Variant | Data | Metric | Status | Result |
|--------|---------|---------|------|--------|--------|--------|
| D036 | B1-lite 模型件校验 + B2 官方样例离线编码冒烟 + 判别对照 + 锚定 bug 根因定位 | ①hf-mirror 镜像下载（**huggingface.co 服务器直连不通、hf-mirror 可达；服务器无 hf CLI 用 curl resolve 直下**）：GEAR-SONIC sample_data 6 pkl + encoder/observation_config——**md5 与 deploy release 逐字节全等**（服务器侧本就官方件）；②`encode_bones_smoke.py`：官方 walk_forward_amateur 样例（BONES-SEED 格式：joblib+zlib pkl，dof 1202×29 **MuJoCo 序** 30Hz 双证据判定[默认角相关 0.715 vs 0.018 + FK 足底检验]，root_rot xyzw→wxyz）→ planner_sonic g1 布局 1762 维 obs（30→50Hz 线性重采样）→ 冻结 encoder → token 2003×64；③判别对照 `roundtrip_official_control.py`：官方 ds_smoke WALK token（rows [1896,4896)，target_motion.csv 37 列探明=root xyz+quat wxyz+29 关节 MuJoCo 序）过**同一**回环 harness | `data/ds_bones/b1/`（6 pkl+encoder/config+api.json）、`b2/`（smoke_result.json + control_official.json + tokens/q_des npy） | lattice 违例率；decoder 回环帧对齐 MAE vs 站姿基线；mean-L2 vs official walk | DONE（修复验证 = D038） | **①lattice 违例 0.0（encoder 内置 FSQ 天然 k/16 格点，官方 token 同 0）；②逐维 std 与官方几乎一致（0.113/0.108）但 mean-L2 3.16、我方 token 带大幅直流偏置；③回环 MAE 0.564 ≈ 2.5× 站姿基线 0.223，而官方 token 同 harness 0.136 < 自身基线 0.212（decoder 是「帮忙」的）→ 判定编码路径真 bug，误差集中髋 yaw/roll 1.3–1.9 rad（官方最差 0.53）；④根因（代码审阅定位）：锚定朝向块沿用 planner_sonic 的 apply_delta（初始 heading 归一化）——planner_sonic 站立起点 apply_delta≈identity 掩盖缺陷，样例初始 yaw −87°（沿 −y 行走）→ 全部锚定向量被注入 ~90° 常量偏移；正确离线语义 = `btr = conj(bq[t])·bq[idx]`（三重依据：deploy 闭环完美跟踪时精确退化 / 双端施加 apply_delta 时严格相消 / yaw 不变量）。⑤B1 阻塞项：**bones-studio/seed 确认 gated(auto)**——需 owner HF 账号接受 bones-seed-license + 提供 read token 才能下载；g1 格式 = 单体 `g1.tar.gz`（142k csv 全量打包，Locomotion 74,488 条占 52%） |
| D037 | B3 预演 v1：离线编码 token → Isaac oracle 闭环回放（bug 印证 + 链路机械验证） | `isaac/replay_bones_tokens_isaac.py`（D034 同构：AptFlatG1Env 子类旁路 policy + SonicTorchDecoder + env 自持 10 帧闭环 history；复用件 import 不改 canonical），v1（bug 版）tokens 2003 帧，2 seed × 40s | `data/ds_bones/b3_rehearsal/rehearsal.json` + log | fall 步 / h_min/h_end / 路径长 / 均速 / q 跟踪 | DONE（v1；v2 修复版见 D038） | **链路机械全通**（token npy → 冻结 decoder → 闭环物理 → 终止判定，零报错、进程干净退出）；**两 seed 同模式 ~1s 侧倒**：fall 51/48 步，h 0.763 → 0.17s 冲高 0.835 → 单调坍缩 0.20/0.24，末帧 root 翻转 96°/77° 侧倒型，均速 0.74 / 路径 0.77/0.73 m vs 参考 16.84——yaw 类失败形态与 D036 锚定 bug 完全吻合（第三重印证）。**新坑（已处置）：本机 Isaac `sim_app.close()` 本身挂死**（两次实测，os._exit 执行不到）——daemon 线程 + 30s 超时 + os._exit 模式写入 D037 脚本并回移 canonical `oracle_token_replay_isaac.py`。预演非门：单段慢速折返业余步态摔倒不定罪 B 线（正式 B3 门 = 每大类 ≥10 段存活 ≥95%） |
