# TO36+ 腿级 TO（Drake dircol）设计定稿

> 【层位 L2 侧轴｜计划：腿级 TO 新线设计（活跃）】↑ `refine-logs/README.md`
> （扇出树根地图）｜上游口径：`FINAL_REPORT.md` §9 剩余方向 1、
> `docs/g1_fullbody_trajectory_optimization_roadmap.md`（技术路线调研）｜
> Run 行事实源：`tracker/TO.md`（TO36 起）。
> 状态：**收束（2026-08-30，时间盒用满）**。本文档是 2026-08-29 grill-me
> 写前访谈的产物：五项决策由用户逐题拍板，含可否决的默认实现细节与
> 降级阶梯。执行结果与三门判定见收束报告 `LEG_LEVEL_TO_REPORT.md`、
> Run 行见 `tracker/TO.md` TO36 段：A 门达成（F11b 平地+刚性+膝可行
> 0.277 m/s）、B 门双验证执行完毕（支撑链 −5 N·m 归因 = URDF↔MJCF
> CoM 差 1.4 cm）、C 门负结果（开环回放 1.84 s，缺落足级稳定层——
> capture +40% 是正向线索）。

## 1. 背景与动机

- `FINAL_REPORT.md` §9：**「G1 全身/腿级 TO（非 SRB）」是唯一真正剩下的大方向**——
  解锁「力矩级 aux 正向价值」的正路。TO06 已证明 SRB（2D 单刚体）力矩量级
  （hip 峰值 25–76 N·m）远小于 G1 行走所需（~100–300 N·m），是模型选择错误。
- 调研底稿：`docs/g1_fullbody_trajectory_optimization_roadmap.md`（五类做法对照
  + 最小可行第一步 + 风险清单）。本计划在其基础上按访谈决策调整了两点：
  求解器栈改 **Drake dircol**（弃 CasADi 自建），模型本体改**真实 G1 MJCF 降维
  平面实例**（弃手写 2D 简化模型）。
- 已判死路线不重蹈（`FINAL_REPORT.md` §7）：PD 力矩解码器、SRB 力矩直接前馈、
  开环重放补数据等均不在本线内。

## 2. 访谈确认的五项决策（2026-08-29，grill-me 逐题拍板）

| # | 决策点 | 结论 |
|---|---|---|
| 1 | 方向 | 腿级 TO 第一步（力矩级 aux 正向的前置门），实验号续 **TO36** 起 |
| 2 | 栈 | **Drake dircol**（弃 CasADi 自建；代价：服务器新装 `.venv_drake`，不复用 `srb_to.py` 骨架） |
| 3 | 模型本体 | **真实 G1 MJCF 降维平面实例**：Drake 载入真实 MJCF，锁非矢状 DOF + 平面基座；**真脚掌**（足底简化跟/尖两接触点）+ **踝主动**（≈9 DOF：平面基座 3 + 双腿 hip/knee/ankle_pitch 各 3）；惯量/限位自动继承真实值，TO06 病根自动修复 |
| 4 | DoD 三门 | A：dircol 周期解且平均前进速度 ≥0.2 m/s；B：hip/knee 峰值力矩 100–300 N·m；C：43-DOF MuJoCo 闭环存活 ≥6s + h_min ≥0.6（防蹲蹭：真走 0.76 vs 作弊 0.20）+ 世界系 disp >0.5 m（防 body 系假速度）。**三门全过才算第一步成功** |
| 5 | 时间盒 | **5–7 人日 + 降级阶梯**；到期未全过 → 写负结果报告收束（对齐 TO 线纪律） |

### 降级阶梯（按卡点触发，任一门卡 >1.5 天降级）

- MJCF 解析不过 → 转 URDF（MJCF→URDF 转换或删减不兼容元素）→ 再退回
  「自建 2D 模型装真惯量」（roadmap §2 原方案，用户已否决为首选但保留为降级档）。
- A 门卡（dircol 不收敛）→ 先点足锁踝出周期解，再逐步加回脚掌/踝。
- C 门卡 → 区分归因：「TO 轨迹本身不好」vs「闭环投影/缺稳定层」（TO18–22 已证
  缺反应式稳定层时好轨迹也活不长），必要时加简单 LIPM 反馈稳层再测，不得混归因。

### 2.1 第二轮 grill-me 增补（2026-08-29 晚，hybrid 启动前四项拍板）

> 触发：D2 首版架构判死（见 §3 修订）后，「真脚首选 / 点足兜底」的阶梯顺序
> 与新风险格局冲突，就 hybrid 双相位的启动参数再访谈一轮。

| # | 决策点 | 结论 |
|---|---|---|
| 6 | 首版形态 | **点足锁踝先行**（每相 5 DOF：支撑 pin + 双腿 hip/knee，照 MIT kneed compass gait 模板）；真脚+踝是「pin 换 weld + 解锁踝」的纯增量升级，不推翻已验证代码。理由：D2 证明风险在 NLP 架构（hybrid/impact/周期边界），先在最小模型上一次验证 |
| 7 | 本轮出口 | **真脚版 A 门（周期解 ≥0.2 m/s 口径不变）**；点足周期解是内置里程碑而非出口。卡壳按降级阶梯收窄、记负结果。DoD 不稀释：点足解天生缺踝力矩，对下游力矩级解码器数据价值不足 |
| 8 | 时间盒 | **双卡点制**：点足里程碑 ≤2 人日（爆了降级：减 DOF / 弱连续 impact / 简化初值）；真脚升级 ≤2 人日（爆了退回点足口径收轮）。总盒 5–7 人日不变 |
| 9 | 代码组织 | **新文件 `to36_hybrid_dircol.py` + 抽 `to36_common.py`**（用户选 C 案，AI 推荐的 A 案被否）；v1 solve 保留标注判死，不移 `_archive`（负结果证据） |

附带拍板的默认（AI 提案、随确认通过）：impact 照 §3 原案（点足 = 模板角动量
冲击，真脚 = 全掌刚体冲击），不做弱连续；不强行 v=0 原地踏步，摆动离地
clearance 同伦阶梯直接起步；整周期两相位（不做镜像对称假设）。

## 3. 默认实现细节（访谈中由 AI 定、用户可否决）

> **2026-08-29 D1/D2 实现修订**（详见 tracker/TO.md TO36 执行记录）：
> ① 模型栈实测为 MJCF 不兼容（Drake 只吃 .obj 网格）→ 走 URDF 降级档，
> 且把「锁非矢状 DOF」升级为字面 weld 降维（bbox 钉 0 会触发 IPOPT
> "too few degrees of freedom"——消元固定变量后动力学等式行仍保留）；
> ② 平面基座用 PlanarJoint 字面实现（浮基+平面化约束每 knot 变量 32 vs
> 等式 37，结构性欠自由度）；③ 首版「钉足运动学约束 + 无约束动力学」被
> 判不可行（接触反力缺失，inf_pr 发散），下一步转 hybrid 双相位 dircol
> （支撑脚 per-phase weld，underactuated compass-gait 模板）。

1. **接触建模**：接触当**解析约束**（支撑相足点位置/速度=0 + 摩擦锥/摆动脚离地
   path constraint，MIT underactuated / Posa 风格），不用 Drake 接触求解器
   （hydroelastic/timestepping dircol 是研究级天坑）。
   〔D2 修订：解析约束须配 per-phase weld 的相位 plant 才能动力学自洽，见上〕
   〔D5 修订（2026-08-30）：行程盒须**相位感知**——支撑链关节与 MJCF 坐标
   反号（P 映射 flipped 行），对称膝盒 ±2.0 曾放行膝反屈（F9 映射后超真实
   限位 [−0.087, 2.88] 至 46°，C 门闭环根因）→ F11 起支撑膝 [−2.88, +0.087]/
   摆动膝 [−0.087, 2.88]，踝同步收紧 [−0.524, 0.873]〕
2. **impact 处理**：优先照 kneed compass gait 模板用角动量守恒冲击约束；
   收敛不良退回 timetable 弱连续（roadmap §4.1 原方案）。
3. **B 门双验证**：dircol 解内 τ + 服务器 MuJoCo `mj_inverse` 对同一轨迹复核
   （`recover_id_torque.py` 已验证 mj_inverse 可用），两套量级一致才算过。
   〔D5 修订（2026-08-30 执行完毕）：mj_inverse 因 TO08 浮基 bug 弃用，改
   基座行消去法（λ 由无驱动基座 6 行解出，跟/尖两点与 weld 同构）。结果：
   摆动链一致 PASS、支撑链 −5 N·m 系统差归因 = URDF↔MJCF 整机 CoM x 差
   1.4 cm（已定量）；数字口径 100–300 FAIL——该区间系 TO06 时代对「真走
   需求」的估计，F9 慢拖步态（T=2.4 s）力矩 6.7–39.5 N·m 为其诚实量级，
   提速轮（F10 系）才是上探该区间的路径（被审计拦截暂缓）〕
4. **数据接口**：解出的周期轨迹 (q, dq, τ, 接触序列) 存 .npz，格式对齐
   「力矩级解码器训练数据」，直接衔接 TO 数据→力矩解码器→aux 正向主线。
5. **环境**：服务器 `/home/cvgluser/ros2_data/.venv_drake`（Python 版本以勘察为准，
   Drake wheel 需匹配）；无 sudo，按 `.venv_isaac` 惯例
   `python3 -m venv --without-pip` + `curl get-pip.py` 引导；`pip install drake`。
   本机不装任何依赖。
6. **流程合规**：动手前 evo 检索已做（2026-08-29，0 命中无先例冲突）；Run 行追加
   `tracker/TO.md`；脚本登记 `SCRIPT_MAP.md`；实验结论（含负结果）先写 evo
   （group `gr00t-apt`）再收尾。

## 4. 执行计划（5–7 人日）

| 天 | 内容 | 出口判据 |
|---|---|---|
| D1 | `.venv_drake` 建 + G1 MJCF 载入 Drake + 降维/锁定冒烟 | MultibodyPlant 可微分，锁定后 DOF 数 = 预期 |
| D2–3 | dircol 两相位周期解（A 门） | IPOPT 收敛，周期边界残差收敛，步速 ≥0.2 m/s |
| D4 | B 门：解内 τ 提取 + mj_inverse 复核 | hip/knee 峰值 100–300 N·m，两套一致 |
| D5–7 | C 门：43-DOF MuJoCo 闭环冒烟（矢状 τ 投影 + 其余 PD 稳住，防作弊指标记录） | 存活 ≥6s + h_min ≥0.6 + disp >0.5 m |

## 5. 风险表

| 风险 | 等级 | 缓解 |
|---|---|---|
| Drake MJCF parser 对 G1 MJCF 兼容性未验证 | 高 | D1 首测；URDF 转换退路；自建 2D 兜底 |
| dircol 对初值/模式序列敏感（多体+接触约束） | 高 | SRBD 周期解（`srb_to.py` 现成）做初值；先减配点/点足 |
| 平面解迁回 3D 闭环时缺横滚/肩臂补偿致倒 | 中 | C 门口径已定为「短暂不倒+量级对」；失败按 §2 降级阶梯归因 |
| 锁定 DOF 的反作用力矩被忽略（锁≠无扰动） | 中 | 闭环冒烟里被锁 DOF 用 PD 稳住即为该假设的直接检验 |
| Drake wheel 与服务器 Python 版本不匹配 | 低 | 勘察后选 wheel；必要时 uv/conda 独立解释器（无 sudo 限制内） |
