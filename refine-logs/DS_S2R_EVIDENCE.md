# DS 线调研：Isaac 训练 → 真机执行 gap 的外部证据综合

> 【层位 L2 侧轴｜调研（2026-09-04；owner 指令「搜索论文与资讯，判断 Isaac 训练
> 到底会不会影响真机」，触发点 = D034 Isaac 79.8% vs 官方 WBC-sim 48.7% 的执行
> 不对称）】↑ `refine-logs/README.md`（扇出树根）｜上游：`DS_GAIT_MANIFOLD_PLAN.md`
> §2 / `tracker/D.md` D034、D033、D032｜结论归属：行动裁定见 §4（owner 已裁）。

## 1. 问题

Isaac 底板比官方 WBC-sim 回路「更快」（D034：同一 RUN token 流 realized
1.67 vs 1.03 m/s）。担心：Isaac 上训得动的东西，真机是不是执行不了？

## 2. 代码考证：真机执行层是什么

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`：
  输出通路 = `G1Deploy::LowCommandWriter`，**500 Hz 直接发布 Unitree
  `unitree_hg::msg::dds_::LowCmd_`**（HG_CMD_TOPIC，关节级位置/速度/增益命令，
  unitree_sdk2 DDS）。即真机消费 decoder q_des 的**关节级伺服命令**。
- "WBC" 的全部配置代码住 `gear_sonic/utils/mujoco_sim/`（wbc_configs、
  unitree_sdk2py_bridge）——**是仿真侧构件**（run_sim_loop 内模拟机器人底层
  的层），真机上不存在这道独立关卡。
- 含义：三个执行栈 = Isaac（关节级隐式 PD）/ WBC-sim（mujoco_sim 桥接层）/
  **真机（Unitree LowCmd 关节伺服，类别上更接近 Isaac）**。真机 realized
  是从未测过的第三列，排序 Isaac > WBC-sim 不预测真机位置。

## 3. 外部证据

1. **SONIC 官方就是在 Isaac Lab 上训的（直接实锤）**：NVlabs
   GR00T-WholeBodyControl README 自带 `IsaacLab 2.3.2` badge，训练入口明确
   "Train / finetune SONIC → Isaac Lab's Python env"（服务器仓库同款可查）。
   SONIC 论文 = *Supersizing Motion Tracking for Natural Humanoid Whole-Body
   Control*（1 亿+ 动捕帧、~42M 参数、ICLR 2026），官方部署真机 G1；**我们
   立项前自己也在真机上跑通过 SONIC**（proj2605.md 前置史）——「Isaac Lab 训
   → deploy LowCmd → 真机 G1」这条完整链路在本项目硬件上已存在成功先例。
2. **APT-RL 论文（我们复刻的对象）**：*Agile perceptive multiskill locomotion
   for quadrupedal robots in the wild*（Science Robotics adz7397，KAIST；
   arXiv:2607.13579）：仿真中 action pretraining 训可复用 skill 库 → 学得的
   策略转移真四足 wild 复杂地形成功。「sim 训 → 真机用」是该论文的成立前提。
3. **ASAP（RSS 2025，CMU+NVIDIA，arXiv:2502.01143）**：精确量化了
   **IsaacGym → real G1** 的动力学鸿沟：gap 真实存在、**非均匀分布在不同
   关节**（执行器动力学为主体）；「delta action model」（真机轨迹学残差修正
   动作）把跟踪误差降 52.7%，优于 SysID / domain randomization / delta
   dynamics 基线；代码开源（LeCAR-Lab/ASAP）。含义：①我们担心的 gap 是本领域
   公认、被精确刻画、有标准解的问题；②gap 的主体是**执行器动力学**（增益/
   延迟/力矩限幅），不是「真机多一道更严的控制器」。
4. **G1/H1 生态的普遍模式**：ExBody（RSS 2024）/ ExBody2 / HumanPlus（Stanford，
   CoRL 2024）/ HoST（RSS 2025）全部 = Isaac 训 + **关节位置目标动作空间**
   （如 G1 29-d normalized joint position targets + PD，与本项目
   decoder→q_des→PD 同构）+ 真机部署。「仿真偏乐观、真机偏保守」是这类管线
   的默认状态而非异常。打滑的标准度量与处理 = **足底接触期足速度**（contact
   期低足速惩罚/审计，位置控制 sim 的特有伪影）。

## 4. 综合判断与行动裁定

**判断**：影响**存在但非致命**。方向可预期（真机 realized 大概率低于 Isaac
口径数字，主因执行器动力学非均匀差异）；同构管线（Isaac 训 + 关节位置目标 +
G1 真机）在 SONIC 官方、ExBody 系、HumanPlus、ASAP 全线走通；且本项目流形
内容 = 官方回路执行过的 token，动作面被冻结 VAE + decoder 限死——RL 的
可利用空间被压到「已被执行过的运动」邻域，风险形态是**偏好偏移**而非
「发明真机不可执行的运动」。

**owner 裁定（09-04）**：不来回横跳，**计划维持不变，先在 Isaac 上训好**；
**打滑核验升级为当前最关键工作项**（判别设计：oracle 回放中加足底接触期
滑移速度测量 + MuJoCo 渲染抽检，标准度量见证据 4）；反向桥出口门（Round 1
Q1）暂不立项、留档于此。若未来上真机：ASAP delta action model 是现成的
最后一段路方案。

## Sources

- [SONIC 项目页（NVlabs GEAR-SONIC）](https://nvlabs.github.io/GEAR-SONIC/)
- [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
  （README：IsaacLab 2.3.2 训练依赖）
- [nvidia/GEAR-SONIC checkpoints](https://huggingface.co/nvidia/GEAR-SONIC)
- [APT-RL: Science Robotics adz7397](https://www.science.org/doi/10.1126/scirobotics.adz7397)
- [APT-RL preprint arXiv:2607.13579](https://arxiv.org/abs/2607.13579) ／
  [项目页](https://skillquadsr.github.io/)
- [ASAP: arXiv:2502.01143（RSS 2025）](https://arxiv.org/abs/2502.01143) ／
  [代码 LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP)
- [ExBody arXiv:2402.16796](https://arxiv.org/abs/2402.16796) ／
  [ExBody2 arXiv:2412.13196](https://arxiv.org/html/2412.13196v2)
- [HumanPlus arXiv:2406.10454](https://arxiv.org/html/2406.10454v1)
- [G1 位置目标动作空间示例](https://huggingface.co/hardware-pathon-ai/unitree-g1-phase1-locomotion)
- [足滑惩罚/接触期足速度综述 arXiv:2308.12517](https://arxiv.org/html/2308.12517v2)
