"""D064: manager-based velocity 栈承载冻结 SONIC decoder 的 env cfg 家族（G1）。

基于官方 ``LocomotionVelocityRoughEnvCfg`` 继承链（经官方 ``G1RoughEnvCfg``，
其 ``G1Rewards`` 覆写原样继承：终止罚 -200、速度权重 1.0/2.0、feet_air_time_positive_biped
等），只做三类改动，使「decoder 臂 vs direct 臂」之间唯一自变量 = 动作通路：

  1. scene.robot 替换为 apt_flat_env.py 同一 G1 articulation cfg
     （``gear_sonic.envs.manager_env.robots.g1.G1_CYLINDER_MODEL_12_DEX_CFG``，直接
     import 同一对象 = 零漂移），双臂同资产是归因设计的前提。
  2. ActionsCfg 两变体：
     - ``DecoderActionsCfg``：单 term ``SonicDecoderActionTermCfg``（模块
       ``sonic_action_term``，action_dim=64，frozen SONIC ONNX decoder）。
     - ``DirectActionsCfg``：官方 ``mdp.JointPositionActionCfg``（29 关节位置目标）。
  3. Observations 增 ``critic`` 特权组（policy 全项的去噪原值版，rsl_rl 惯例组名）；
     另设 ``make_env_cfg`` 地形工厂（rough/plane/stairs/stones/discrete）。

------------------------------------------------------------------------------
【继承未改清单】（行号均指 tmp/isaac_ref/ 下的官方源码拷贝）
  - scene.terrain（rough 臂）：官方 ROUGH_TERRAINS_CFG generator 原样
    （velocity_env_cfg.py:44-62）；max_init_terrain_level=5、marble 视觉材质不动。
  - height_scanner / contact_forces / sky_light：官方原样
    （velocity_env_cfg.py:66-82；scanner prim 由官方改到 torso_link，rough_env_cfg.py:112）。
  - commands：官方 G1 命令范围原样（rough_env_cfg.py:146-148：vx (0,1)、vy 0、wz (-1,1)）。
  - rewards：``G1Rewards`` 全部继承（rough_env_cfg.py:20-100）+ 官方 post_init 权重/asset_cfg
    覆写（rough_env_cfg.py:132-143），本文件只做下述「资产适配」三处（见已改清单 R1-R3）。
  - terminations / events / curriculum(rough) / decimation=4 / sim.dt=0.005（50 Hz 控制）：
    官方原样（velocity_env_cfg.py:150-272, 296-321; rough_env_cfg.py:114-129, 151）。
    50 Hz 与 apt_flat_env.py:173-180（episode 20 s、decimation=4、dt=1/200）一致。
  - observations.policy：官方八项原样（velocity_env_cfg.py:119-146）。

【已改清单】（每点附依据）
  A1. scene.robot：官方 G1_MINIMAL_CFG（rough_env_cfg.py:111）→ apt_flat_env 同一
      G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
      （apt_flat_env.py:201-203；"/World/envs/env_.*/Robot" 与 ENV_REGEX_NS 同一正则，
      此处沿用 manager-based 惯例写法）。资产逐字段出处：
      - spawn：UrdfFileCfg（fix_base=False、replace_cylinders_with_capsules=True、
        {ASSET_DIR}/robot_description/urdf/g1/main.urdf、activate_contact_sensors=True，
        gear_sonic robots/g1.py:200-223；ASSET_DIR 定义 g1.py:9）；
        rigid_props / articulation_props / joint_drive(零增益) g1.py:206-222。
      - init_state：pos z=0.76，joint_pos hip_pitch=-0.312 / knee=0.669 /
        ankle_pitch=-0.363 / elbow=0.6 / shoulder(+left roll)=0.2、right roll=-0.2，
        joint_vel=0（g1.py:224-237）——即 SONIC 默认角（apt_flat_env.py:69-78
        SONIC_DEFAULT_ANGLES_MUJOCO 经 G1_MUJOCO_TO_ISAACLAB_DOF 重排后逐项一致）。
      - soft_joint_pos_limit_factor=0.9（g1.py:238）。
      - actuators 与 PD 增益：legs/feet/waist/waist_yaw/arms 五组 ImplicitActuatorCfg
        （g1.py:239-358），增益常量 ARMATURE_*/STIFFNESS_*/DAMPING_*（g1.py:11-27，
        10 Hz 自然频率、阻尼比 2.0；踝 stiffness=2×STIFFNESS_5020=28.50125，
        与 apt_flat_env.py:328-329 to_tau_kp 注释口径一致）。
      - 关节名单：29 DOF 全驱动（G1_ISAACLab_ORDER，gear_sonic joint_utils.py:10-40；
        legs 12 + waist 3 + arms 14，无手指），无未驱动关节。
  R1. rewards.joint_deviation_arms 的 asset_cfg.joint_names 官方式含
      ".*_elbow_pitch_joint"/".*_elbow_roll_joint"（rough_env_cfg.py:68-75），本资产
      肘为单 dof ".*_elbow_joint" 且无 elbow_roll → 改写为本资产实际存在的 14 个手臂
      关节（= g1.py:303-311 actuator 正则逐字）。改写 asset_cfg 正则是官方自己的
      适配手法（rough_env_cfg.py:137-143 对 dof_acc/dof_torques 同样操作）；
      func/weight(-0.1) 不动 = 非自创奖励。
  R2. rewards.joint_deviation_fingers = None（rough_env_cfg.py:78-95 的手指关节本资产
      不存在）：贡献恒为 0，置 None 与官方去 term 手法一致（rough_env_cfg.py:115-116,
      133）。29 DOF 资产无手指（joint_utils.py:10-40 全集可查）。
  R3. rewards.joint_deviation_torso 的 "torso_joint"（rough_env_cfg.py:99）本资产命名为
      "waist_yaw_joint"（g1.py:294-301）→ 同一物理关节改名字符串；weight -0.1 不动。
      是否扩到 waist_roll/pitch 属实验决策，此处不做。
  O1. observations.critic 特权组：policy 八项同序、全部去噪原值——即 base_lin_vel 与
      height_scan 用原值（无 noise；height_scan 另去 clip），其余项同步去噪
      （enable_corruption=False）。解读说明：派发语义「policy 全项 + base_lin_vel +
      height_scan 原值」中后两者已是 policy 项，故按特权 critic 惯例实现为
      「policy 全项的原值版」，而非再追加重复项。rsl_rl 侧按 IsaacLab 2.x 惯例以组名
      "critic" 消费（obs_dict["critic"]，缺组时回落 policy，不影响本 cfg 单独使用）。
  AC. ActionsCfg 两变体（见类 docstring）；decoder 臂下 obs 的 actions 项自动反映
      64 维 raw：mdp.last_action 返回 env.action_manager.action（observations.py:512-521），
      该缓冲存的是 term 切分前的原始动作（action_manager.py:318-339），总维数 =
      各 term action_dim 之和（action_manager.py:228-246）——单 term(64) 时即 64 维
      raw，无需改任何 obs 配置。direct 臂下同机制自动为 29 维。
  T1. make_env_cfg 地形工厂：rough=官方课程原样；plane=官方 G1FlatEnvCfg flat 语义
      逐字（flat_env_cfg.py:19-41：terrain_type="plane"、generator=None、关
      height_scanner 与 obs.height_scan、关课程，外加官方 flat 奖励/命令覆写）；
      stairs/stones/discrete = 逐字段复用 apt_g1/isaac/terrain_cfg.py:149-217 的 Hf
      配方组装 TerrainImporterCfg（curriculum 关：terrain_levels=None +
      generator.curriculum=False）。height_scan 走 RayCaster、与地形类型无关
      （mesh_prim_paths=["/World/ground"] 与四种 TerrainImporterCfg 的 prim_path 一致，
      terrain_cfg.py:194）——这是本栈相对旧 elevation_map 缺口的结构性解法。

【版本兼容纪律（2026-09-19 环境实测更正：cvgl 实际为 isaaclab 2.1.0 /
isaacsim 4.5.0.0；tmp/isaac_ref/ 是 v2.0.0 快照，作 API 地基）】
- 继承不重写：凡从官方 LocomotionVelocityRoughEnvCfg/G1RoughEnvCfg 原样继承的
  字段（scene.terrain、height_scanner/RayCasterCfg、contact_forces、sensors 更新
  周期等——2.1.0 里 RayCasterCfg 参数名有 ray_alignment/attach_yaw_only 类变动）
  本文件一律不在子类重指定，靠继承规避版本名漂移。plane 变体只把官方继承来的
  ``height_scanner``/``height_scan`` 整项置 None（flat_env_cfg.py:22-23 同款），
  不触碰其构造参数。
- 确需覆写/显式构造处，注释均标「v2.0.0 参考=file:line，2.1.0 需冒烟核对」：
  scene.robot 与奖励 asset_cfg 覆写（_adapt_g1_sonic_asset）、critic 组的 mdp 项、
  DirectActionsCfg 的 JointPositionActionCfg、Hf 地形配方（_hf_terrain_importer_cfg，
  字段与服务器实测过的 terrain_cfg.py:104-147,193-217 同源，属 2.1.0 已验证面）。
- reward 适配（R1-R3）对 cfg.rewards.* 用直接属性访问（不做 hasattr 静默跳过）：
  若 2.1.0 的官方 G1Rewards 改了 term 名，宁可首启即报错也不静默漏适配。
服务器运行环境 cvgl；import 路径 ``isaac.*``（= apt_g1/isaac 包，PYTHONPATH 含
apt_g1，apt_flat_env.py:40-47 同款）与 ``gear_sonic.*`` 由 /tmp/run_apt_isaac.sh 的
PYTHONPATH 提供（HANDOFF/04_SERVER_GUIDE.md §3）。

【gym 任务 id 命名建议】（与 tmp/isaac_ref/g1_init.py 注册风格一致；本文件不执行
注册——训练/评测脚本用 make_env_cfg 直构，需要 gym 工作流时在注册模块里照抄模板，
见文件尾注释）：
    Isaac-Velocity-G1-Sonic-Rough-v0            G1SonicDecoderEnvCfg
    Isaac-Velocity-G1-Sonic-Rough-Play-v0       G1SonicDecoderEnvCfg_PLAY
    Isaac-Velocity-G1-Sonic-Rough-Direct-v0     G1SonicDirectEnvCfg
    Isaac-Velocity-G1-Sonic-Rough-Direct-Play-v0 G1SonicDirectEnvCfg_PLAY
    其余地形经工厂：Isaac-Velocity-G1-Sonic-{Plane,Stairs,Stones,Discrete}-v0
    （后缀顺序约定： terrain -> Direct -> Play；默认 rough 省略进 id 时写作 Rough）。

用法：
    from isaac.g1_velocity_decoder_env import make_env_cfg
    cfg = make_env_cfg(terrain="rough", num_envs=4096, action="decoder",
                       token_stats="path/to/token_stats_g1.npz")
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import (
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfSteppingStonesTerrainCfg,
    MeshRandomGridTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
# 官方继承入口（v2.0.0 参考；模块路径在 2.1.0 未变——manager_based 布局与
# config/g1 目录两版一致，仍需服务器首启冒烟核对）。
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ObservationsCfg,
)

# 同一 G1 资产：直接复用 apt_flat_env.py:201 所用对象（零字段漂移）。
from gear_sonic.envs.manager_env.robots import g1

# D066 C2 门（§5w）climb_box 配方的高度档单一事实源在 terrain_cfg.py
# （CLIMB_BOX_HEIGHT_RANGE，对齐 D060 语料 climbing_box{height:0.5}）；此处只
# 引用常量、不复制数值，防两处漂移（双兼容 import 同 sonic_action_term 惯例）。
try:
    from isaac.terrain_cfg import CLIMB_BOX_HEIGHT_RANGE
except ImportError:  # pragma: no cover（本机走此支，需 apt_g1 可导入）
    from apt_g1.isaac.terrain_cfg import CLIMB_BOX_HEIGHT_RANGE

# 冻结 decoder 的自定义 ActionTerm cfg（同批 D064 模块；action_dim=64）。
try:  # 服务器执行根平铺 import / 仓库根包 import 双兼容（仓内既有惯例，
    # 与 sonic_action_term.py:293-296 同构）
    from isaac.sonic_action_term import SonicDecoderActionTermCfg
except ImportError:  # pragma: no cover（本机走此支，需 apt_g1 可导入）
    from apt_g1.isaac.sonic_action_term import SonicDecoderActionTermCfg

# D065 C 臂（v2 in-graph）：decoder 迁进 policy 后，env 侧只负责提供 decoder 的
# **本体历史观测**（930 维，raw 无噪声）——即 B 臂 ActionTerm 的 assemble 内容。
try:
    from isaac.sonic_lora_policy import PROPRIO_DIM, sonic_proprio_hist
except ImportError:  # pragma: no cover（本机走此支）
    from apt_g1.isaac.sonic_lora_policy import PROPRIO_DIM, sonic_proprio_hist

__all__ = [
    "DecoderActionsCfg",
    "DirectActionsCfg",
    "G1SonicObservationsCfg",
    "G1SonicDecoderEnvCfg",
    "G1SonicDecoderEnvCfg_PLAY",
    "G1SonicDirectEnvCfg",
    "G1SonicDirectEnvCfg_PLAY",
    "LoRAPolicyObservationsCfg",
    "G1SonicLoRAPolicyEnvCfg",
    "make_env_cfg",
]

# 冻结 SONIC decoder ONNX（相对服务器执行根；apt_flat_env.py:207-209 同一默认值）。
SONIC_DECODER_ONNX = "gear_sonic_deploy/policy/release/model_decoder.onnx"

# 29 DOF 全驱动关节名单（与 gear_sonic joint_utils.G1_ISAACLab_ORDER 同一全集）；
# 此处用正则表达式集合表达（= 资产 actuators 正则，g1.py:239-358），不重复硬编码名单。
G1_ARM_JOINT_PATTERNS = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]  # g1.py:303-311 逐字；14 关节（left/right × 7）

G1_WAIST_YAW_JOINT = "waist_yaw_joint"  # = 官方 G1 的 "torso_joint"（R3，见模块 docstring）


##
# Actions：两变体
##


@configclass
class DecoderActionsCfg:
    """Decoder 臂：策略动作 = 64 维 raw token 坐标 → 冻结 SONIC decoder → 关节目标。

    cfg 字段（同批 sonic_action_term 模块的 SonicDecoderActionTermCfg）：
    - asset_name="robot"：作用于本家族替换后的 gear_sonic G1（A1）。
    - onnx_path：冻结 decoder；默认与 apt_flat_env.py:207-209 同一路径。
    - token_stats：官方 g1-mode token 的 mean/std/rate npz（必填，"" 会显式失败——
      与 train_apt_isaac.py:445 的 assert 同纪律）；不猜路径。
    - token_alpha=1.0、token_bound="tanh"：E49-A 受限值域稳定化（映射
      token = mean + alpha*std*tanh(a)，apt_flat_env.py:275-284；E49 无界臂曾漂出
      可解码区致步态崩塌，tanh 臂为收敛后的规范配置）。可经 make_env_cfg 覆写。
    """

    sonic_token: SonicDecoderActionTermCfg = SonicDecoderActionTermCfg(
        asset_name="robot",
        onnx_path=SONIC_DECODER_ONNX,
        token_stats="",
        token_alpha=1.0,
        token_bound="tanh",
    )


@configclass
class DirectActionsCfg:
    """Direct 臂：官方关节位置动作（对照组），官方动作通路逐字。

    scale=0.5 的对齐考虑：
    - 官方 velocity 栈 ActionsCfg 即 scale=0.5、use_default_offset=True
      （velocity_env_cfg.py:112），官方 G1 rough env 未覆写 actions
      （rough_env_cfg.py 全文无 actions 项）→ direct 臂 = 官方 G1 velocity 动作通路
      逐位一致，这是「动作通路单变量」里 direct 侧的锚点，不取 SONIC 的逐关节
      scale（apt_flat_env.py:55-67）以免引入第二个变量。
    - use_default_offset=True 时 offset = default_joint_pos
      （joint_actions.py:155-156）；本资产 default 即 SONIC 默认角（A1/g1.py:224-237），
      两臂共享同一站立默认位，动作零点对齐。
    - joint_names=[".*"]：解析为本资产全部 29 个驱动关节
      （find_joints，joint_actions.py:61-63；全集 = G1_ISAACLab_ORDER，
      joint_utils.py:10-40），即「名单同资产」。
    """

    # v2.0.0 参考=velocity_env_cfg.py:112；JointPositionActionCfg 字段 2.0/2.1 同名，
    # 2.1.0 需冒烟核对（首训打印 action_manager 表即可）。
    joint_pos: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


##
# Observations：官方 policy 组 + critic 特权组（O1）
##


@configclass
class G1SonicObservationsCfg(ObservationsCfg):
    """policy 组官方原样（velocity_env_cfg.py:119-146）+ critic 特权组。

    critic 组 = policy 八项同序的原值版（无噪声；height_scan 另去 clip）。
    actions 项在 decoder 臂自动为 64 维 raw（AC，见模块 docstring 依据行号）。
    """

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations（rsl_rl 惯例组名 "critic"）。

        v2.0.0 参考=velocity_env_cfg.py:119-146 同名 mdp 项；这些 mdp 观测函数名
        在 2.1.0 未变，组名 "critic" 的 rsl_rl 消费路径 2.1.0 需冒烟核对。
        """

        # observation terms (order preserved = policy 组同序)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 原值（policy 版带 Unoise，velocity_env_cfg.py:124）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )  # 原值：无 noise 亦无 clip（policy 版 clip=(-1,1)，velocity_env_cfg.py:134-139）

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    critic: CriticCfg = CriticCfg()


##
# 资产/奖励适配（A1、R1-R3；两臂共用）
##


def _adapt_g1_sonic_asset(cfg: G1RoughEnvCfg) -> None:
    """在官方 __post_init__ 之后替换资产并做奖励 asset_cfg 适配（幂等）。

    必须在 super().__post_init__() 之后调用：官方先写入 G1_MINIMAL_CFG 与
    G1Rewards 参数（rough_env_cfg.py:107-151，v2.0.0 参考），此处覆写为最终态。
    注意：对 cfg.rewards.* 用直接属性访问——若 2.1.0 官方 G1Rewards 改了 term 名，
    首启即 AttributeError 暴露契约漂移（2.1.0 需冒烟核对），不做 hasattr 静默跳过。
    """
    # A1: 双臂同资产 —— 与 apt_flat_env.py:201-203 同一对象（零漂移）。
    cfg.scene.robot = g1.G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # R1: 手臂关节正则适配（官方适配先例 rough_env_cfg.py:137-143）。
    cfg.rewards.joint_deviation_arms.params["asset_cfg"].joint_names = list(G1_ARM_JOINT_PATTERNS)
    # R2: 无手指资产，置 None（官方去 term 先例 rough_env_cfg.py:115-116, 133）。
    cfg.rewards.joint_deviation_fingers = None
    # R3: torso_joint -> waist_yaw_joint（同一物理关节）。
    cfg.rewards.joint_deviation_torso.params["asset_cfg"].joint_names = G1_WAIST_YAW_JOINT


##
# Environment configurations
##


@configclass
class G1SonicDecoderEnvCfg(G1RoughEnvCfg):
    """Decoder 臂（rough 官方课程原样）：64 维 token -> 冻结 SONIC decoder。"""

    observations: G1SonicObservationsCfg = G1SonicObservationsCfg()
    actions: DecoderActionsCfg = DecoderActionsCfg()

    def __post_init__(self):
        # 官方链：G1Rewards 覆写 + G1 专属 post_init（rough_env_cfg.py:107-151）
        # + 基础 post_init（velocity_env_cfg.py:296-321）。
        super().__post_init__()
        _adapt_g1_sonic_asset(self)


@configclass
class G1SonicDirectEnvCfg(G1RoughEnvCfg):
    """Direct 臂（rough 官方课程原样）：官方 JointPositionAction（对照）。"""

    observations: G1SonicObservationsCfg = G1SonicObservationsCfg()
    actions: DirectActionsCfg = DirectActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        _adapt_g1_sonic_asset(self)


##
# D065 C 臂（v2 in-graph）：A 臂动作通路逐字 + policy 组追加 930 维 decoder 本体历史观测
##


@configclass
class LoRAPolicyObservationsCfg(G1SonicObservationsCfg):
    """policy 组 = 官方八项（逐字继承）+ 末位追加 `sonic_proprio`（930 维）。

    - 追加项 = decoder 的**本体输入契约**（token 64 之外的 930 维，见 sonic_action_term.py:24-34
      的通道序）；值由 `sonic_lora_policy.sonic_proprio_hist` 用 **SonicActionCore 同一份代码**
      （历史环形缓冲 + assemble）产出，raw 无噪声——因为它是 decoder 的输入而不是策略输入，
      加噪/缩放会与 B 臂 ActionTerm 的输入分布不一致。
    - 末位追加（dataclass 字段序）保证策略侧切片 `obs[:, :official_dim]` 恰为官方八项，
      即 token head 的输入与 A/B 臂 actor 输入逐字同维（见 sonic_lora_policy.SonicLoRAPolicy）。
    - critic 组不动（`CriticCfg` 显式列项，不受 policy 组新增项影响）⇒ 特权 critic 与 A/B 逐字同构。
    """

    @configclass
    class PolicyCfg(G1SonicObservationsCfg.PolicyCfg):  # type: ignore[misc]
        # 末位追加（继承字段在前 ⇒ 该项在拼接序末位）
        sonic_proprio = ObsTerm(
            func=sonic_proprio_hist,
            params={"action_term_name": "joint_pos", "asset_name": "robot"},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1SonicLoRAPolicyEnvCfg(G1SonicDirectEnvCfg):
    """D065 C 臂（v2）：env 侧 = Direct 臂逐字（同一 `DirectActionsCfg` 对象类）+ 本体历史观测项。

    与 `G1SonicDirectEnvCfg` 的唯一差异 = `observations.policy` 末位多一项 930 维
    `sonic_proprio`；动作通路（JointPositionActionCfg scale=0.5/use_default_offset=True）、
    奖励、终止、课程、命令、critic 组全部逐字继承 ⇒ 「C vs A」的单变量 = **decoder 是否在
    动作路径上且其权重可否适配**（decoder 由 policy 侧持有，见 sonic_lora_policy.py）。
    """

    observations: LoRAPolicyObservationsCfg = LoRAPolicyObservationsCfg()

    def __post_init__(self):
        super().__post_init__()  # Direct 臂链：G1Rewards/资产适配/DirectActionsCfg 全部照旧


##
# 变体配方（官方逐字）
##


def _apply_flat_semantics(cfg: G1RoughEnvCfg) -> None:
    """plane 变体 = 官方 G1FlatEnvCfg flat 语义逐字（flat_env_cfg.py:19-41，
    v2.0.0 参考；只整项翻转/置 None 官方继承来的字段，不重指定构造参数，
    2.1.0 需冒烟核对项仅 terrain_type/terrain_generator 属性名——两版同名）。"""
    # terrain / height scan / curriculum（flat_env_cfg.py:19-25）
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.height_scanner = None
    cfg.observations.policy.height_scan = None
    cfg.observations.critic.height_scan = None  # 本栈新增的特权组同步关闭
    cfg.curriculum.terrain_levels = None
    # Rewards（官方 flat 配方，flat_env_cfg.py:27-37）
    cfg.rewards.track_ang_vel_z_exp.weight = 1.0
    cfg.rewards.lin_vel_z_l2.weight = -0.2
    cfg.rewards.action_rate_l2.weight = -0.005
    cfg.rewards.dof_acc_l2.weight = -1.0e-7
    cfg.rewards.feet_air_time.weight = 0.75
    cfg.rewards.feet_air_time.params["threshold"] = 0.4
    cfg.rewards.dof_torques_l2.weight = -2.0e-6
    cfg.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
    )
    # Commands（flat_env_cfg.py:39-41）
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


def _hf_terrain_importer_cfg(kind: str, seed: int | None) -> TerrainImporterCfg:
    """stairs/stones/discrete 的 TerrainImporterCfg。

    配方逐字段取自 apt_g1/isaac/terrain_cfg.py:149-217（Hf 子地形 + generator 参数
    8x8/边 20/10x20/0.1/0.005/0.75/无缓存 + average 摩擦材质、无视觉材质）；
    terrain_cfg.py 本就在 cvgl（isaaclab 2.1.0）服务器实跑过，故本配方属 2.1.0
    已验证面，2.1.0 需冒烟核对风险低（v2.0.0 参考=同文件，字段两版同名）。
    seed 语义同 terrain_cfg.py:19（默认 0 = 固定地形，跨 run 可复现）。
    """
    if kind == "stairs":
        sub_terrains = {
            "stairs": HfPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.04, 0.08),
                step_width=0.35,
                platform_width=1.0,
                border_width=0.25,
            ),
        }
    elif kind == "stones":
        sub_terrains = {
            "stones": HfSteppingStonesTerrainCfg(
                proportion=1.0,
                stone_height_max=0.06,
                stone_width_range=(0.25, 0.4),
                stone_distance_range=(0.3, 0.5),
                holes_depth=-0.5,
                platform_width=1.0,
                border_width=0.25,
            ),
        }
    elif kind == "discrete":
        sub_terrains = {
            "discrete": HfDiscreteObstaclesTerrainCfg(
                proportion=1.0,
                obstacle_height_range=(0.05, 0.10),
                obstacle_width_range=(0.15, 0.35),
                num_obstacles=10,
                platform_width=1.0,
                border_width=0.25,
            ),
        }
    elif kind == "climb_box":
        # D066 C2 门（§5w）四族之一：与 terrain_cfg.make_terrain_importer_cfg
        # ("climb_box") 同配方（官方 MeshRandomGridTerrainCfg boxes 原语 +
        # CLIMB_BOX_HEIGHT_RANGE 高度档），仅供本工厂经 make_env_cfg(terrain=
        # "climb_box") 接入 height_scan；数值唯一事实源=terrain_cfg.py 常量。
        sub_terrains = {
            "climb_box": MeshRandomGridTerrainCfg(
                proportion=1.0,
                grid_width=0.45,
                grid_height_range=CLIMB_BOX_HEIGHT_RANGE,
                platform_width=2.0,
            ),
        }
    else:
        raise ValueError(f"unknown hf terrain kind {kind!r} (expect stairs/stones/discrete/climb_box)")
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=seed,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains=sub_terrains,
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


def _apply_play_recipe(cfg: G1RoughEnvCfg) -> None:
    """官方 PLAY 配方逐字（rough_env_cfg.py:156-180，v2.0.0 参考；属性翻转型
    覆写，2.1.0 需冒烟核对仅属性名——num_envs/episode_length_s/commands ranges
    两版同名）。

    含官方 PLAY 的命令覆写（lin_vel_x=(1,1) 等）——非 rough 地形与 play 组合时
    命令范围同样取 PLAY 值（工厂层约定，见 make_env_cfg docstring）。
    """
    # make a smaller scene for play
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 2.5
    cfg.episode_length_s = 40.0
    # spawn the robot randomly in the grid (instead of their terrain levels)
    cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    # disable randomization for play
    cfg.observations.policy.enable_corruption = False
    # remove random pushing
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


@configclass
class G1SonicDecoderEnvCfg_PLAY(G1SonicDecoderEnvCfg):
    """Decoder 臂 rough Play（官方 PLAY 配方；gym 注册入口用，脚本侧建议走工厂）。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_play_recipe(self)


@configclass
class G1SonicDirectEnvCfg_PLAY(G1SonicDirectEnvCfg):
    """Direct 臂 rough Play（官方 PLAY 配方）。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_play_recipe(self)


##
# 地形变体工厂
##


def make_env_cfg(
    terrain: str = "rough",
    num_envs: int = 4096,
    action: str = "decoder",
    play: bool = False,
    onnx_path: str = SONIC_DECODER_ONNX,
    token_stats: str = "",
    token_alpha: float = 1.0,
    token_bound: str = "tanh",
    seed: int | None = 0,
) -> G1RoughEnvCfg:
    """按 (terrain, action, play) 组装 env cfg 实例。

    - terrain="rough"：官方课程与地形原样（velocity_env_cfg.py:44-62, 269-272；
      generator.curriculum 由官方 post_init 置 True，velocity_env_cfg.py:314-321）。
    - terrain="plane"：官方 flat 语义（_apply_flat_semantics，flat_env_cfg.py:19-41）。
    - terrain in {"stairs","stones","discrete"}：terrain_cfg.py:149-217 的 Hf 配方，
      课程关（terrain_levels=None + generator.curriculum=False，官方置 False 先例
      rough_env_cfg.py:170）；height_scan RayCaster 保留（与地形类型无关，T1）。
    - terrain="climb_box"（D066 C2 门新增，§5w）：官方 MeshRandomGridTerrainCfg
      boxes 原语、高度档 CLIMB_BOX_HEIGHT_RANGE（terrain_cfg.py 单一事实源，
      对齐 D060 语料 climbing_box{height:0.5}）；课程关同 Hf 族。既有族配方与
      默认值零改动（单变量纪律）。
    - action="decoder"|"direct"|"lora_policy"：decoder/direct 两臂唯一差异在 actions；
      lora_policy（D065 C 臂 v2）= direct 逐字 + policy 组末位追加 930 维本体历史观测
      （decoder 由 policy 侧持有，env 侧无 decoder action term）。
    - play=True：官方 PLAY 配方（rough_env_cfg.py:156-180）叠加在地形变体之后；
      含官方 PLAY 的命令覆写（非 rough 地形 + play 时命令也取 PLAY 值，工厂层约定）。
    - num_envs 在类默认/PLAY 值之后最终生效。
    - decoder 臂的 onnx_path/token_stats/token_alpha/token_bound 由此透传
      （token_stats 必填，DecoderActionsCfg docstring）。
    - seed 只作用于 stairs/stones/discrete 的 Hf generator（None = 不固定）；
      rough 臂保持官方 ROUGH_TERRAINS_CFG 原样，不做 seed 覆写。
    """
    if action == "decoder":
        cfg = G1SonicDecoderEnvCfg()
    elif action == "direct":
        cfg = G1SonicDirectEnvCfg()
    elif action == "lora_policy":
        cfg = G1SonicLoRAPolicyEnvCfg()
    else:
        raise ValueError(f"unknown action {action!r} (expect 'decoder'/'direct'/'lora_policy')")

    if terrain == "rough":
        pass  # 官方课程原样
    elif terrain == "plane":
        _apply_flat_semantics(cfg)
    elif terrain in ("stairs", "stones", "discrete", "climb_box"):
        cfg.scene.terrain = _hf_terrain_importer_cfg(terrain, seed)
        # 课程关（官方置 False 先例 rough_env_cfg.py:170；官方 post_init 的
        # curriculum 耦合逻辑在 velocity_env_cfg.py:314-321，须在其后覆写）。
        cfg.curriculum.terrain_levels = None
        if cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = False
    else:
        raise ValueError(
            f"unknown terrain {terrain!r} (expect rough/plane/stairs/stones/discrete/climb_box)"
        )

    if play:
        _apply_play_recipe(cfg)

    cfg.scene.num_envs = num_envs

    if action == "decoder":
        cfg.actions.sonic_token.onnx_path = onnx_path
        cfg.actions.sonic_token.token_stats = token_stats
        cfg.actions.sonic_token.token_alpha = token_alpha
        cfg.actions.sonic_token.token_bound = token_bound

    return cfg


##
# gym 注册建议（不在此执行；模板与 tmp/isaac_ref/g1_init.py:14-23 风格一致）
#
# import gymnasium as gym
#
# gym.register(
#     id="Isaac-Velocity-G1-Sonic-Rough-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point":
#             f"{__name__}.g1_velocity_decoder_env:G1SonicDecoderEnvCfg",
#         "rsl_rl_cfg_entry_point": "<D064 runner cfg>",  # 单独的 PPO cfg，不在本文件范围
#     },
# )
#
# 同式：Isaac-Velocity-G1-Sonic-Rough-Play-v0 -> G1SonicDecoderEnvCfg_PLAY
#       Isaac-Velocity-G1-Sonic-Rough-Direct-v0 -> G1SonicDirectEnvCfg
#       Isaac-Velocity-G1-Sonic-Rough-Direct-Play-v0 -> G1SonicDirectEnvCfg_PLAY
# 其余地形（Plane/Stairs/Stones/Discrete）建议由训练/评测脚本经 make_env_cfg 直构；
# 若需 gym id，按 "Isaac-Velocity-G1-Sonic-<Terrain>[-Direct][-Play]-v0" 扩展。
##
