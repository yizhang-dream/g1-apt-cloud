"""Author v0 (D061a) 闭环评测：把 D061a author 模型挂成 Isaac Lab 策略消费 64 维动作。

分层（与 apt_g1/isaac/sonic_action_term.py 同构：Layer 1 纯 torch，本机可测；
Layer 2 isaaclab import 集中在函数内）：

Layer 1（纯 torch/numpy，零 isaaclab / onnx import，本机 Windows 可 import 可测）
  - 契约常量（WIN/TOKEN_DIM/STATE_DIM/CTX_DIM/命令哨兵/地形表）
  - policy obs 8 项切分 ``slice_obs_term``（flat=134 / rough=321 两套 layout）
  - 关节序映射（MuJoCo <-> IsaacLab/SONIC，硬编码 perm + 运行时一致性 ASSERT）
  - 状态装配 ``build_state``（36 = jp_mj 29 + quat_wxyz 4 + trans 3）
  - 命令 -> 13 维 ctx ``ctx_from_command``
  - ``AuthorPolicyAdapter``：200 帧滚动窗 + 自回归 1 步延迟 + 逆 token 仿射
  - ckpt 装载 ``load_author_ckpt``（以 ckpt 内 model_cfg 为准）+ 契约断言

Layer 2（isaac）
  - ``--run-cond`` 子进程体：AppLauncher -> 建 flat env -> 装载 author ckpt ->
    adapter 闭环 rollout -> 打一行 ``AXIS_COND <json>``
  - 父进程：逐条件 spawn 子进程（Popen encoding=utf-8）-> 汇总身份信封 + JSON

------------------------------------------------------------------------------
契约对照表（policy obs 8 项 -> author 输入）
------------------------------------------------------------------------------
policy obs 组（apt_g1/isaac/g1_velocity_decoder_env.py:29 注明的官方八项，
velocity_env_cfg.py:119-146）通道序即下表顺序；**无 root 四元数/位置**，
故 quat/trans 必须从 ``robot.data.root_quat_w``(wxyz) / ``root_pos_w - env_origins``
另取（见本模块 docstring 末尾风险 R3）。

+-------------------+--------+------------------------------+-------------------------------+
| policy obs 项     | 宽度   | author 侧消费                | 处理                          |
+===================+========+==============================+===============================+
| base_lin_vel      |   3    | 不消费（占位切分）           | 仅用于 layout 校验            |
| base_ang_vel      |   3    | 不消费（占位切分）           | d060 语料差分速度，见 R4      |
| projected_gravity |   3    | 不消费（占位切分）           | 同上                          |
| velocity_commands |   3    | ctx[0..3] 来源（vx/vy/wz）   | 见 ctx_from_command           |
| joint_pos         |  29    | state[0:29] 的原始量         | joint_pos_rel + default 还原  |
| joint_vel         |  29    | 不消费（占位切分）           | R4                            |
| actions           |  64    | 不消费（占位切分）           | 闭环里由 adapter 自回归替代   |
| height_scan       |  187   | ctx[11]（噪声档）/R2 来源    | flat 臂几何=平面、读值≈0      |
+-------------------+--------+------------------------------+-------------------------------+
合计 flat=3+3+3+3+29+29+64=134；rough=+187=321（= D064 双臂 ckpt 的 obs 维）。

author 输入 -> 闭环输出（每个控制步一次）:
  state (N,36) = jp_mj(29, 绝对角) + root_quat_wxyz(4) + root_trans_m(3)
                 ^ 由 joint_pos(rel, SONIC/IsaacLab 序) + SONIC default，经
                   MuJoCo<->IsaacLab perm 重排（见 ISAAC_TO_MUJOCO_PERM）
  codes (N,200,64) int64 = FSQ 码**索引**（= 真码域值 − code_min；真码 value =
                           (索引 + code_min) / token_scale，v0 语料 scale=16）
  intent  = codes 本身（自条件，d060_windows.py:19-22 口径）
  ctx  (N,13) = command(4) + has_truth(4) + terrain one-hot(3) + (height, noise)(2)
  前向取最后 t 帧（t = n_valid，上限 WIN=200）-> 取末帧 logits argmax
  -> idx + code_min = 码 -> tok = 码/16 -> 逆仿射 a = (tok - mean)/(alpha*std)
     （token_bound="tanh" 时先 atanh(clip(z,-1+eps,1-eps))）-> (N,64) 原始动作

------------------------------------------------------------------------------
与既有脚本的复用点
------------------------------------------------------------------------------
- ``train_g1_decoder._import_heavy``：App 起来后的唯一重型 import 入口
  （torch/gym/ManagerBasedRLEnv/make_env_cfg/SonicDecoderActionTerm[Cfg]/
  file_md5），避免双份 sibling 双兼容 import 漂移（train_g1_decoder.py:501-578）。
- ``train_g1_decoder._find_decoder_term``：按 isinstance 找 action term，
  类名兜底防 isaac.* vs apt_g1.isaac.* 模块身份分裂（train_g1_decoder.py:368-391）。
- ``eval_g1_decoder._stamp_command``：flat 臂把命令钉死为 (vx,0,0)
  （eval_g1_decoder.py:172-178）——本脚本直接 import 复用。
- ``eval_g1_decoder._file_md5 / _git_head / _parse_csv_* / COND_TIMEOUT_S``：
  身份信封与 CSV 解析同款（eval_g1_decoder.py:41-67）。
- ``eval_g1_decoder._eval_one_main`` 的 flat 条件配方（bring-up 顺序）：
  **建 rough cfg 再换平面几何**（保 obs=321，避免 actor 首层 size mismatch；
  eval_g1_decoder.py:283-322）。本脚本逐条照抄该配方（差异见假设 A2）。
- ``AuthorV0Transformer``（apt_g1/training/train_author_v0.py:388-433）：
  ckpt 装载时 import 其类，以 ckpt 内 model_cfg 重建（train_author_v0.py:513-522）。
- ``sonic_action_term.SonicActionCore`` 的 token_mean/std、sonic_default/scale、
  token_alpha/token_bound：从 action term 实例上直接读（不重复加载 npz）。

------------------------------------------------------------------------------
服务器运行命令模板（cwd 由 /tmp/run_apt_isaac.sh 设为 GR00T-WholeBodyControl）
------------------------------------------------------------------------------
  # 0) 冒烟（100 步，8 env）——先过这步再跑全量
  cd /home/cvgluser/ros2_data && nohup bash /tmp/run_apt_isaac.sh \\
      apt_g1/isaac/eval_author_v0_decoder.py --ckpt <d061a/ckpt_final.pt> \\
      --token-stats <g1_token_stats.npz> --battery flat --smoke \\
      --out-json outputs/d061b_axis_b_smoke.json \\
      > logs/d061b_smoke.log 2>&1 < /dev/null & disown

  # 1) 正式电池（64 env x 2 vx x 1 seed）
  cd /home/cvgluser/ros2_data && nohup bash /tmp/run_apt_isaac.sh \\
      apt_g1/isaac/eval_author_v0_decoder.py --ckpt <d061a/ckpt_final.pt> \\
      --token-stats <g1_token_stats.npz> --battery flat --num-envs 64 \\
      --seeds 0 --vx-cmds 0.4,0.6 --ep-steps 3000 \\
      --out-json outputs/d061b_axis_b.json \\
      > logs/d061b_axis_b.log 2>&1 < /dev/null & disown

  # 本机（无 isaac，不 import 任何 isaac 模块）
  python apt_g1/isaac/eval_author_v0_decoder.py --selftest-adapter   # -> AXIS_B_SELFTEST_PASS
  python apt_g1/isaac/eval_author_v0_decoder.py --dry-run            # -> AXIS_B_DRYRUN_OK

------------------------------------------------------------------------------
风险清单
------------------------------------------------------------------------------
R1（import 身份分裂）：服务器 PYTHONPATH 同时含仓根与 仓根/apt_g1，同一文件会被
  加载成 isaac.* 与 apt_g1.isaac.* 两个模块身份，isinstance 恒 False 时静默拿到
  错的 action term（cvgl 冒烟第二轮根因）。对策：主支走 isaac.*（sibling 整组
  一起走，见 _import_heavy），term 查找用 train_g1_decoder._find_decoder_term
  的类名兜底并打 [WARN]；本脚本另对 term 取到的 token_mean/std 做形状断言。
R2（同名 sibling 解析）：本脚本自身的 ``_import_heavy`` 复用 train_g1_decoder；
  若该模块被加载成 apt_g1.isaac.train_g1_decoder，其内部主支仍走 isaac.*，
  功能不受影响，但身份信封里的 env 模块 md5 必须按**实存文件路径**记录（本脚本
  ``--dry-run``/父进程按 REPO_ROOT/apt_g1/isaac/*.py 计算，避免同名双份歧义）。
R3（policy obs 无根位姿）：quat 取 ``robot.data.root_quat_w``（wxyz，与 d060 语料
  的 root_quat_wxyz 同口径）、trans 取 ``root_pos_w - env_origins``（env 局部系，
  与语料 trans_m 的“根相对世界原点”口径**可能不同**；见假设 A4）。三者任一被
  IsaacLab 改口径会静默污染 state 前三段之外的全部帧，故闭环里对 state 做
  finite 断言并按步抽样打印诊断。
R4（state 速度信息缺失）：d060 语料 state 36 维**不含 jv**（d060_windows.py:37-41
  明确“jv 是否并入留 D061 训练侧决定”）。闭环侧同样不喂 joint_vel/base_lin_vel，
  速度只能靠窗内因果差分——与语料同构，但意味着 author 在闭环里拿不到显式速度。
R5（自回归误差累积）：闭环 token 来自 author 自身 argmax 的**离散码**，一步错码
  会进下一帧窗（自回归）。本脚本固定 1 步延迟（假设 A3）并在窗口未满时用
  neutral 码 + 站立帧填充（假设 A5）——前 200 步的 state/token 分布与训练语料
  （真实运动学窗）不同，前 4s 指标不可比，判读只看 t>200 步后的段。
R6（子进程化）：同进程反复建 env 会悬挂（train/eval 两侧既有告诫），故每条件
  独立子进程；Popen 必须显式 encoding="utf-8"（train r5 实证：C locale 下
  text=True 默认 ASCII 解码遇 UTF-8 崩）。

------------------------------------------------------------------------------
显式假设（指令含糊处的解释，均落成常量/ASSERT/打印，便于事后改判）
------------------------------------------------------------------------------
A1 关节 perm：G1_MUJOCO_TO_ISAACLAB_DOF 以**硬编码副本**落在 Layer 1（其源模块
   顶部 import isaaclab，本机不可导入）；副本合法性由 assert_joint_perm 自检，
   与权威源的一致性只能在服务器侧人工/CI 核对（见 R2）。若后续 g1.py 改动该
   常量，闭环 state 会静默错配——这是本脚本最大的单点假设。
A2 flat 臂关观测噪声：--run-cond 显式置 observations.policy.enable_corruption=False。
   eval_g1_decoder 的 flat 条件未显式关（依赖播放/cfg 缺省）；author 臂消费 policy
   obs 去建 state，带噪 obs 会污染 state，故这里必须关——属**有意偏离** D064 配方。
A3 自回归 1 步延迟：t 步用 state[t] + **上一步预测的码**，预测 code[t+1] -> a[t]。
   与语料"窗内真实码、逐帧 next-token"在稳态下同构，但首帧用 neutral 码填充。
A4 state 语义：jp_mj 取**绝对**关节角（语料源 1/2/3 = kinematics，d060_intake_src14.py
   注明源 4 才是 q_des 目标）；根位姿取 env 局部系（root_pos_w - env_origins），
   与语料 trans_m 的参考系**未核对**（闭环无全局绝对位置概念，这是刻意的选择）。
A5 窗口未满期填充：t = min(max(n_valid), WIN) 下限 1；未满期缓冲用 standing_state_row
   + neutral 码（使 tok ≈ mean）填充，n_valid 控制喂入帧数。
A6 token_scale 固定 16：d060 语料 meta 记 token_scale=16（d060_windows.py:24-26），
   author ckpt 的 model_cfg **不含该字段**，故脚本用常量 TOKEN_SCALE_DEFAULT=16 并在
   JSON/日志里显式回写；若未来语料换网格，须改常量或给 ckpt 补 meta。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # 以 `python apt_g1/isaac/eval_author_v0_decoder.py` 直跑时 sys.path[0] 是脚本
    # 目录而非仓根（train_g1_decoder.py:531-534 同款处理）。
    sys.path.insert(0, str(REPO_ROOT))

# 复用 eval_g1_decoder 的纯工具件（该模块 import 期不 import isaac，本机实测可导入）。
try:  # noqa: E402
    from isaac import eval_g1_decoder as egd
except ImportError:  # pragma: no cover（本机走此支）
    from apt_g1.isaac import eval_g1_decoder as egd

COND_TIMEOUT_S = egd.COND_TIMEOUT_S
DEFAULT_VX = egd.DEFAULT_VX

# ---------------------------------------------------------------------------
# Layer 1: 契约常量 + 纯 torch 件（零 isaaclab / onnx import）
# ---------------------------------------------------------------------------
WIN = 200                  # 4s 窗 @50Hz（train_author_v0.py:75 / d060_windows.py:98）
TOKEN_DIM = 64             # FSQ token 维（train_author_v0.py:76）
STATE_DIM = 36             # jp_mj 29 + quat_wxyz 4 + trans_m 3（train_author_v0.py:77）
CTX_DIM = 13               # 4(命令) + 4(has_truth) + 3(terrain one-hot) + 2(height/noise)
N_JOINTS = 29              # SONIC / G1 body 关节数
TOKEN_SCALE_DEFAULT = 16   # d060 语料 token_scale（d060_windows.py:24-26）
SENTINEL = -1.0            # D033 无真值哨兵（train_author_v0.py:78）

COMMAND_FIELDS = ("target_vel", "movement_direction", "mode", "height")
TERRAIN_TYPES = ("plane", "rough_paper", "climbing_box")
TERRAIN_DEFAULT_PARAMS = {
    "plane": {},
    "rough_paper": {"noise": 0.04},
    "climbing_box": {"height": 0.5},
}

# policy obs 八项（官方 velocity_env_cfg.py:119-146；顺序即拼接顺序）
POLICY_TERM_WIDTHS = (
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", N_JOINTS),
    ("joint_vel", N_JOINTS),
    ("actions", TOKEN_DIM),
    ("height_scan", 187),
)
POLICY_OBS_DIM_FLAT = 134   # 无 height_scan（_apply_flat_semantics 置 None）
POLICY_OBS_DIM_ROUGH = 321  # 含 height_scan（= D064 双臂 ckpt 的 obs 维）

# G1_MUJOCO_TO_ISAACLAB_DOF 的**硬编码副本**（来源
# gear_sonic/envs/manager_env/robots/g1.py:94-...；该模块顶部 import isaaclab，
# 本机不可导入，故 Layer 1 只能硬编码 + 运行时 ASSERT 一致性，见 R2/假设 A1）。
# 语义：perm[i] = IsaacLab/SONIC 序第 i 个关节的 MuJoCo 下标。
# 用法同 sonic_action_term.py:365 `SONIC_DEFAULT_ANGLES_MUJOCO[perm] -> Isaac 序`。
G1_MUJOCO_TO_ISAACLAB_DOF = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
    24, 18, 25, 19, 26, 20, 27, 21, 28,
)
_MJ2ISAAC = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF, dtype=np.int64)
ISAAC_TO_MUJOCO_PERM = np.argsort(_MJ2ISAAC).astype(np.int64)  # isaac 序 -> mujoco 序

# SONIC default angles in **MuJoCo 序**（sonic_action_term.py:351-360 同一常量；
# 同一原因硬编码）。经 _MJ2ISAAC 重排得 IsaacLab/SONIC 序版本。
SONIC_DEFAULT_ANGLES_MUJOCO = np.asarray(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)
VALID_TOKEN_BOUNDS = ("none", "tanh")


def assert_joint_perm(perm: np.ndarray | None = None) -> None:
    """ASSERT 关节 perm 是 0..28 的合法排列（硬编码副本的自检，假设 A1）。"""
    p = ISAAC_TO_MUJOCO_PERM if perm is None else np.asarray(perm)
    if p.shape != (N_JOINTS,) or sorted(p.tolist()) != list(range(N_JOINTS)):
        raise AssertionError(f"关节 perm 不是 0..{N_JOINTS - 1} 的排列：{p.tolist()}")


def joint_perm_digest() -> dict:
    """硬编码关节序常量的内容摘要（写进身份信封，判读时可与权威源核对）。"""
    payload = json.dumps(
        {"mujoco_to_isaaclab": [int(x) for x in G1_MUJOCO_TO_ISAACLAB_DOF]},
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "mujoco_to_isaaclab": [int(x) for x in G1_MUJOCO_TO_ISAACLAB_DOF],
        "md5": hashlib.md5(payload).hexdigest(),
    }


def sonic_default_isaac() -> np.ndarray:
    """SONIC default angles in IsaacLab/SONIC 序 (29,)（sonic_action_term.py:363-365 同式）。"""
    return SONIC_DEFAULT_ANGLES_MUJOCO[_MJ2ISAAC].astype(np.float32)


def sonic_default_mujoco() -> np.ndarray:
    """SONIC default angles in MuJoCo 序 (29,)（= 站立帧 jp_mj 段）。"""
    return SONIC_DEFAULT_ANGLES_MUJOCO.astype(np.float32)


assert_joint_perm()


# --------------------------------------------------------------- obs 切分
def policy_obs_layout(obs_dim: int) -> tuple[tuple[str, int], ...]:
    """按 obs 总维返回该组的 (term, width) 表（134=flat 无 height_scan / 321=rough）。"""
    if int(obs_dim) == POLICY_OBS_DIM_FLAT:
        return tuple(t for t in POLICY_TERM_WIDTHS if t[0] != "height_scan")
    if int(obs_dim) == POLICY_OBS_DIM_ROUGH:
        return tuple(POLICY_TERM_WIDTHS)
    raise ValueError(
        f"policy obs 维须 ∈ {{{POLICY_OBS_DIM_FLAT}, {POLICY_OBS_DIM_ROUGH}}}，"
        f"收到 {obs_dim}（布局表见本模块 docstring 契约对照表）"
    )


def slice_obs_term(obs: torch.Tensor, term: str, obs_dim: int | None = None) -> torch.Tensor:
    """从 policy obs 末维切出单项（(..., obs_dim) -> (..., width)）。

    按 POLICY_TERM_WIDTHS 的拼接顺序累加偏移；obs_dim 缺省取 obs.shape[-1]。
    """
    d = int(obs.shape[-1]) if obs_dim is None else int(obs_dim)
    if int(obs.shape[-1]) != d:
        raise ValueError(f"obs 末维 {obs.shape[-1]} 与 obs_dim={d} 不符")
    lo = 0
    for name, width in policy_obs_layout(d):
        if name == term:
            return obs[..., lo : lo + width]
        lo += width
    raise KeyError(f"policy obs 无此项 {term!r}（dim={d} 的项：{[n for n, _ in policy_obs_layout(d)]}）")


# --------------------------------------------------------------- state 装配
def build_state(
    joint_pos_isaac: torch.Tensor,
    quat_wxyz: torch.Tensor,
    trans_m: torch.Tensor,
    *,
    jp_absolute: bool = True,
    default_isaac: torch.Tensor | None = None,
) -> torch.Tensor:
    """(N,29)+(N,4)+(N,3) -> (N,36) float32 state（jp_mj + root quat_wxyz + root trans）。

    ``joint_pos_isaac`` 为 IsaacLab/SONIC 序（= G1_ISAACLab_ORDER）29 关节角；
    ``jp_absolute=False`` 时按其为 ``q - default`` 相对量，用 ``default_isaac``
    还原绝对角（policy obs 的 joint_pos 项即该相对量，mdp.joint_pos_rel）。
    随后经 ISAAC_TO_MUJOCO_PERM 重排到语料的 jp_mj 段（d060_windows.py:302-322
    的 state_from_kinematics 布局）。

    假设 A4：语料 jp_mj 为**绝对**关节角（源 1/2/3 = kinematics），故闭环同用绝对角。
    """
    jp = torch.as_tensor(joint_pos_isaac, dtype=torch.float32)
    q = torch.as_tensor(quat_wxyz, dtype=torch.float32)
    tr = torch.as_tensor(trans_m, dtype=torch.float32)
    if jp.shape[-1] != N_JOINTS:
        raise ValueError(f"joint_pos 末维需 {N_JOINTS}，得 {tuple(jp.shape)}")
    if q.shape[-1] != 4:
        raise ValueError(f"quat_wxyz 末维需 4，得 {tuple(q.shape)}")
    if tr.shape[-1] != 3:
        raise ValueError(f"trans_m 末维需 3，得 {tuple(tr.shape)}")
    if jp.shape[:-1] != q.shape[:-1] or jp.shape[:-1] != tr.shape[:-1]:
        raise ValueError(f"三件 batch 形状不一致：{tuple(jp.shape)} {tuple(q.shape)} {tuple(tr.shape)}")
    if not jp_absolute:
        if default_isaac is None:
            default_isaac = torch.as_tensor(sonic_default_isaac(), dtype=torch.float32, device=jp.device)
        jp = jp + default_isaac
    perm = torch.as_tensor(ISAAC_TO_MUJOCO_PERM, dtype=torch.long, device=jp.device)
    jp_mj = jp.index_select(-1, perm)  # isaac 序 -> mujoco 序
    st = torch.cat([jp_mj, q, tr], dim=-1)
    if st.shape[-1] != STATE_DIM:
        raise AssertionError(f"state 维需 {STATE_DIM}，得 {st.shape[-1]}")
    if not bool(torch.isfinite(st).all()):
        raise AssertionError("state 含非有限值（R3：检查 root_pos_w / env_origins / joint_pos 来源）")
    return st


def standing_state_row(device="cpu", dtype=torch.float32) -> torch.Tensor:
    """站立帧 (36,)：jp = SONIC default(mujoco 序)、quat = (1,0,0,0)、trans = 0。

    对应 sonic_action_term.SonicActionCore.reset 的站立帧语义（历史重填）。
    """
    jp = torch.as_tensor(sonic_default_mujoco(), dtype=dtype, device=device)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    tr = torch.zeros(3, dtype=dtype, device=device)
    return torch.cat([jp, q, tr])


# --------------------------------------------------------------- ctx
def ctx_from_command(
    vx: float,
    vy: float = 0.0,
    wz: float = 0.0,
    *,
    terrain: str = "plane",
    terrain_params: dict | None = None,
    min_speed: float = 1.0e-3,
) -> np.ndarray:
    """env 命令 -> 13 维 ctx（序同 train_author_v0.ctx_from_record）。

    [0:4]   command = (target_vel, movement_direction, mode, height)；无真值填 -1 哨兵
    [4:8]   has_truth（float 0/1）
    [8:11]  terrain one-hot（plane / rough_paper / climbing_box）
    [11:13] (height, noise) —— TERRAIN_DEFAULT_PARAMS 叠加 terrain_params

    映射（契约摘要口径）：target_vel = vx（has_truth=True）；movement_direction =
    atan2(vy, vx)，水平速度 < min_speed 时判无方向 -> 哨兵 + has_truth=False；
    mode/height 无真值 -> 哨兵 + has_truth=False。wz 不进 ctx（语料无此项），
    仅由调用方打印诊断。
    """
    if terrain not in TERRAIN_TYPES:
        raise ValueError(f"terrain 非法 {terrain!r}（允许 {TERRAIN_TYPES}）")
    speed = float(np.hypot(float(vx), float(vy)))
    has_dir = speed >= float(min_speed)
    cmd = {
        "target_vel": float(vx),
        "movement_direction": float(np.arctan2(float(vy), float(vx))) if has_dir else SENTINEL,
        "mode": SENTINEL,
        "height": SENTINEL,
    }
    ht = {
        "target_vel": True,
        "movement_direction": has_dir,
        "mode": False,
        "height": False,
    }
    for k in COMMAND_FIELDS:
        if not ht[k]:
            cmd[k] = SENTINEL  # 无真值一律哨兵（d060_windows.make_command 同款）
    onehot = [0.0, 0.0, 0.0]
    onehot[TERRAIN_TYPES.index(terrain)] = 1.0
    p = dict(TERRAIN_DEFAULT_PARAMS[terrain])
    p.update(terrain_params or {})
    hp = [float(p.get("height", 0.0)), float(p.get("noise", 0.0))]
    out = [cmd[k] for k in COMMAND_FIELDS] + [1.0 if ht[k] else 0.0 for k in COMMAND_FIELDS] + onehot + hp
    vec = np.asarray(out, dtype=np.float32)
    if vec.shape != (CTX_DIM,):
        raise AssertionError(f"ctx 维需 {CTX_DIM}，得 {vec.shape}")
    return vec


# --------------------------------------------------------------- 环境索引
def resolve_env_index(env_ids, num_envs: int):
    """把 reset 钩子可能的 env_ids 形态解析成 (索引, 行数)。

    与 sonic_action_term.py:79-104 的 ``resolve_env_index`` **同式复制**（那份所在
    模块尾部 import isaaclab，Layer 1 不可引用）：slice 必须原样直通（不能过
    torch.as_tensor），仅 list/Tensor 走 as_tensor。
    """
    if env_ids is None:
        return slice(None), int(num_envs)
    if isinstance(env_ids, slice):
        if env_ids.start is None and env_ids.stop is None:
            return env_ids, int(num_envs)
        n = len(
            range(
                env_ids.start if env_ids.start is not None else 0,
                env_ids.stop if env_ids.stop is not None else int(num_envs),
                env_ids.step if env_ids.step is not None else 1,
            )
        )
        return env_ids, n
    t = torch.as_tensor(env_ids, dtype=torch.long)
    return t, int(t.numel())


# --------------------------------------------------------------- ckpt
def load_author_ckpt(ckpt_path: str | os.PathLike, device: str | torch.device = "cpu"):
    """装载 D061a author ckpt -> (model.eval(), model_cfg)。**配置以 ckpt 为准**。

    ckpt 结构见 train_author_v0.py:513-522：
    {format, step, model: state_dict, model_cfg: {d_model,n_layer,n_head,ffn,
     vocab,code_min,state_dim,win,token_dim}, ...}。
    契约断言：win == model.pos.shape[1] - 2（pos 参数含 n_ctx=2 前缀）、
    state_dim == STATE_DIM、token_dim == TOKEN_DIM。
    """
    _ensure_repo_root_on_path()
    from apt_g1.training.train_author_v0 import AuthorV0Transformer  # 纯 torch 模块，本机可导入

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_cfg" not in payload or "model" not in payload:
        raise RuntimeError(
            f"ckpt 结构不符（需 train_author_v0.save_ckpt 产物，含 model/model_cfg 键）："
            f"{ckpt_path}；实得键 {list(payload)[:8] if isinstance(payload, dict) else type(payload).__name__}"
        )
    cfg = dict(payload["model_cfg"])
    for k in ("vocab", "code_min", "win", "state_dim", "token_dim", "d_model", "n_layer"):
        if k not in cfg:
            raise RuntimeError(f"ckpt model_cfg 缺键 {k!r}（实得 {sorted(cfg)}）")
    if int(cfg["state_dim"]) != STATE_DIM:
        raise AssertionError(f"ckpt state_dim={cfg['state_dim']} != {STATE_DIM}")
    if int(cfg["token_dim"]) != TOKEN_DIM:
        raise AssertionError(f"ckpt token_dim={cfg['token_dim']} != {TOKEN_DIM}")
    if int(cfg["win"]) != WIN:
        raise AssertionError(f"ckpt win={cfg['win']} != {WIN}（window 契约冻结）")
    model = AuthorV0Transformer(
        int(cfg["vocab"]),
        state_dim=int(cfg["state_dim"]),
        d_model=int(cfg["d_model"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg.get("n_head", 12)),
        ffn=int(cfg.get("ffn", 4 * int(cfg["d_model"]))),
        grad_ckpt=False,
    )
    n_pos = int(model.pos.shape[1]) - int(model.n_ctx)
    if n_pos != int(cfg["win"]):
        raise AssertionError(
            f"pos 表长 {model.pos.shape[1]} - n_ctx {model.n_ctx} = {n_pos} != win={cfg['win']}"
        )
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    model.to(device)
    return model, cfg


def _ensure_repo_root_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def author_token_stats(mean: torch.Tensor, std: torch.Tensor, *, floor: float = 1.0e-3):
    """token 统计形状/下限截断（sonic_action_term.py:443-453 同式），返回 (mean, std)（CPU）。

    统一落 CPU：decoder term 上的统计在 GPU（cvgl），Layer 1 侧需 numpy 互通。
    """
    m = torch.as_tensor(mean, dtype=torch.float32, device="cpu").reshape(-1)
    s = torch.as_tensor(std, dtype=torch.float32, device="cpu").reshape(-1)
    if m.shape != (TOKEN_DIM,) or s.shape != (TOKEN_DIM,):
        raise ValueError(f"token mean/std 须为 ({TOKEN_DIM},)，收到 {tuple(m.shape)}/{tuple(s.shape)}")
    return m, torch.clamp(s, min=float(floor))


# --------------------------------------------------------------- Adapter
class AuthorPolicyAdapter:
    """D061a author 模型 -> Isaac Lab 64 维动作空间的闭环适配器（纯 torch）。

    缓冲（与语料窗口同布局）：
      - ``codes`` (N, WIN, 64) int64：FSQ 码**索引**（= 真码域值 − code_min）
        （窗口布局与语料同构；forward 消费的是索引，逆仿射时再加回 code_min）
      - ``states`` (N, WIN, 36) float32
      - ``ctx`` (N, 13) float32；``n_valid`` (N,) long 每 env 重置后的有效帧数

    每控制步：``step(state_row, ctx=...)``
      1. 滚动推入本步 state 行 + **上一步预测的码**（自回归 1 步延迟，假设 A3）
      2. 取最后 t 帧（t = min(max(n_valid), WIN)）-> model(codes, states, codes, ctx)
         （intent = codes = 自条件）-> 末帧 logits argmax -> idx
      3. tok = (idx + code_min) / token_scale -> 逆仿射
         a = (tok - mean) / (alpha * std)；token_bound="tanh" 时再过 atanh(clip)
      4. 返回 (N, 64) float32 原始动作（可直接喂 env.step）

    ``reset(env_ids)`` 把该批 env 的缓冲重填站立帧 + neutral 码、n_valid 归零
    （窗口未满时用该填充，假设 A5）。
    """

    def __init__(
        self,
        model,
        *,
        code_min: int,
        vocab: int,
        token_mean: torch.Tensor,
        token_std: torch.Tensor,
        num_envs: int,
        device: str | torch.device = "cpu",
        token_alpha: float = 1.0,
        token_bound: str = "none",
        token_scale: int | float = TOKEN_SCALE_DEFAULT,
        win: int = WIN,
        state_dim: int = STATE_DIM,
    ) -> None:
        if token_bound not in VALID_TOKEN_BOUNDS:
            raise ValueError(f"token_bound 须 ∈ {VALID_TOKEN_BOUNDS}，收到 {token_bound!r}")
        if int(win) != WIN:
            raise AssertionError(f"win 冻结为 {WIN}，收到 {win}")
        if int(state_dim) != STATE_DIM:
            raise AssertionError(f"state_dim 冻结为 {STATE_DIM}，收到 {state_dim}")
        if TOKEN_DIM != 64:
            raise AssertionError(f"token_dim 冻结为 64，实为 {TOKEN_DIM}")
        n_pos = int(model.pos.shape[1]) - int(getattr(model, "n_ctx", 2))
        if n_pos != int(win):
            raise AssertionError(
                f"model.pos 表长 {model.pos.shape[1]} - n_ctx {getattr(model, 'n_ctx', 2)} "
                f"= {n_pos} != win={win}"
            )
        if int(vocab) <= 0:
            raise AssertionError(f"vocab 须 > 0，收到 {vocab}")

        self.model = model
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.win = int(win)
        self.code_min = int(code_min)
        self.vocab = int(vocab)
        self.token_scale = float(token_scale)
        self.token_alpha = float(token_alpha)
        self.token_bound = token_bound
        self.token_eps = 1.0e-6
        mean, std = author_token_stats(token_mean, token_std)
        self.token_mean = mean.to(self.device)
        self.token_std = std.to(self.device)
        self._scale = torch.as_tensor(
            float(token_alpha) * std.numpy(), dtype=torch.float32, device=self.device
        )  # alpha * std

        # neutral 码 = 使 tok ≈ mean 的码（窗口未满期的填充，假设 A5）
        neutral_tok = mean.numpy() * self.token_scale
        neutral_idx = np.rint(neutral_tok).astype(np.int64) - self.code_min
        neutral_idx = np.clip(neutral_idx, 0, self.vocab - 1)
        self.neutral_idx = torch.as_tensor(neutral_idx, dtype=torch.long, device=self.device)

        self.codes = self.neutral_idx.view(1, 1, TOKEN_DIM).expand(self.num_envs, self.win, TOKEN_DIM).clone()
        self.states = (
            standing_state_row(self.device)
            .view(1, 1, STATE_DIM)
            .expand(self.num_envs, self.win, STATE_DIM)
            .clone()
        )
        self.ctx = torch.zeros(self.num_envs, CTX_DIM, dtype=torch.float32, device=self.device)
        self.n_valid = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_idx = self.neutral_idx.view(1, TOKEN_DIM).expand(self.num_envs, TOKEN_DIM).clone()

    # ------------------------------------------------------------ windows
    def set_ctx(self, ctx) -> None:
        """写 ctx (13,) 或 (N,13)（逐 env 同值广播）。"""
        t = torch.as_tensor(np.asarray(ctx), dtype=torch.float32, device=self.device)
        if t.ndim == 1:
            if t.numel() != CTX_DIM:
                raise ValueError(f"ctx 维需 {CTX_DIM}，得 {t.numel()}")
            self.ctx[:] = t.view(1, CTX_DIM)
        elif t.shape == (self.num_envs, CTX_DIM):
            self.ctx[:] = t
        else:
            raise ValueError(f"ctx 形状须 (13,) 或 ({self.num_envs},13)，得 {tuple(t.shape)}")

    def window_len(self) -> int:
        """本步前向的帧数 t = min(max(n_valid), WIN)（下限 1；见假设 A5）。"""
        t = int(self.n_valid.max().item()) if self.num_envs else 0
        return max(1, min(t, self.win))

    def _push(self, state_rows: torch.Tensor) -> None:
        """滚动推入 state 行 + 上一步预测码（clone 防同张量重叠视图赋值）。"""
        self.states[:, :-1] = self.states[:, 1:].clone()
        self.states[:, -1] = state_rows
        self.codes[:, :-1] = self.codes[:, 1:].clone()
        self.codes[:, -1] = self._last_idx
        self.n_valid = (self.n_valid + 1).clamp(max=self.win)

    # ---------------------------------------------------------------- step
    def step(self, state_rows: torch.Tensor, ctx=None) -> torch.Tensor:
        """推入本步 state（N,36）-> 前向 -> 逆仿射 -> (N,64) 原始动作。"""
        st = torch.as_tensor(state_rows, dtype=torch.float32, device=self.device)
        if st.shape != (self.num_envs, STATE_DIM):
            raise ValueError(f"state_rows 形状须 ({self.num_envs},{STATE_DIM})，得 {tuple(st.shape)}")
        if ctx is not None:
            self.set_ctx(ctx)
        self._push(st)
        t = self.window_len()
        codes_w = self.codes[:, self.win - t :]
        states_w = self.states[:, self.win - t :]
        ctx_w = self.ctx
        with torch.no_grad():
            logits = self.model(codes_w, states_w, codes_w, ctx_w)  # (N,t,64,vocab)
            if logits.shape[1] != t or logits.shape[2] != TOKEN_DIM or logits.shape[3] != self.vocab:
                raise AssertionError(
                    f"author 前向输出形状异常：{tuple(logits.shape)}（期望 "
                    f"({self.num_envs},{t},{TOKEN_DIM},{self.vocab})）"
                )
            idx = logits[:, -1].argmax(dim=-1)  # (N,64)
        self._last_idx = idx
        tok = (idx.to(torch.float32) + float(self.code_min)) / self.token_scale
        z = (tok - self.token_mean) / self._scale
        if self.token_bound == "tanh":
            z = torch.atanh(z.clamp(-1.0 + self.token_eps, 1.0 - self.token_eps))
        act = z.to(torch.float32)
        if act.shape != (self.num_envs, TOKEN_DIM):
            raise AssertionError(f"动作形状异常 {tuple(act.shape)}")
        if not bool(torch.isfinite(act).all()):
            raise AssertionError("逆仿射产出非有限值（检查 token_mean/std/alpha 与 ckpt code_min/vocab）")
        return act

    # --------------------------------------------------------------- reset
    def reset(self, env_ids=None) -> None:
        """站立帧 + neutral 码重填该批 env，n_valid 归零。"""
        idx, _n = resolve_env_index(env_ids, self.num_envs)
        if isinstance(idx, torch.Tensor):
            idx = idx.to(self.device)
        self.states[idx] = standing_state_row(self.device)
        self.codes[idx] = self.neutral_idx
        self._last_idx[idx] = self.neutral_idx
        self.n_valid[idx] = 0


# --------------------------------------------------------------- selftest
def run_selftest_adapter() -> None:
    """纯 torch 自测（本机可跑，不 import 任何 isaac 模块）。

    覆盖：obs 切分两套 layout / state 装配与 perm / ctx / ckpt 往返装载 /
    adapter 前向形状与 finite / 逆仿射往返 / tanh clip / reset / 满窗不越 pos 表。
    """
    _ensure_repo_root_on_path()
    from apt_g1.training.train_author_v0 import AuthorV0Transformer

    torch.manual_seed(0)
    # --- 1) policy obs 切分（两套 layout）---
    for dim in (POLICY_OBS_DIM_FLAT, POLICY_OBS_DIM_ROUGH):
        obs = torch.randn(2, dim)
        got = {n: slice_obs_term(obs, n).shape[-1] for n, _ in policy_obs_layout(dim)}
        assert sum(got.values()) == dim, (dim, got)
        assert got["joint_pos"] == N_JOINTS and got["actions"] == TOKEN_DIM, got
        assert ("height_scan" in got) == (dim == POLICY_OBS_DIM_ROUGH), got
    try:
        slice_obs_term(torch.randn(2, POLICY_OBS_DIM_FLAT), "height_scan")
        raise AssertionError("flat layout 不应含 height_scan")
    except KeyError:
        pass

    # --- 2) 关节 perm / state 装配 ---
    assert_joint_perm()
    jp = torch.randn(3, N_JOINTS)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
    tr = torch.zeros(3, 3)
    st = build_state(jp, q, tr)
    assert st.shape == (3, STATE_DIM), st.shape
    assert torch.allclose(st[:, :N_JOINTS], jp.index_select(-1, torch.as_tensor(ISAAC_TO_MUJOCO_PERM))), "perm 重排失效"
    st_rel = build_state(jp, q, tr, jp_absolute=False)
    assert torch.allclose(st_rel[:, :N_JOINTS], (jp + torch.as_tensor(sonic_default_isaac())).index_select(-1, torch.as_tensor(ISAAC_TO_MUJOCO_PERM)))
    assert standing_state_row().shape == (STATE_DIM,)

    # --- 3) ctx ---
    c = ctx_from_command(0.4, terrain="plane")
    assert c.shape == (CTX_DIM,) and c.dtype == np.float32, c
    # vy=0 且 speed=0.4 >= min_speed -> movement_direction 有真值 0.0（has_truth=1）
    assert abs(float(c[0]) - 0.4) < 1e-6 and float(c[1]) == 0.0 and float(c[2]) == float(SENTINEL)
    assert c[4:8].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert c[8:11].tolist() == [1.0, 0.0, 0.0] and c[11] == 0.0 and c[12] == 0.0
    c2 = ctx_from_command(0.4, 0.2, terrain="rough_paper")
    assert abs(c2[1] - float(np.arctan2(0.2, 0.4))) < 1e-5 and c2[5] == 1.0
    assert c2[8:11].tolist() == [0.0, 1.0, 0.0] and abs(c2[12] - 0.04) < 1e-6
    c0 = ctx_from_command(0.0)
    assert c0[1] == np.float32(SENTINEL) and c0[5] == 0.0

    # --- 4) ckpt 往返 ---
    tmp_dir = REPO_ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tmp_dir / "_axis_b_selftest_ckpt.pt"
    vocab, code_min = 27, -14
    model0 = AuthorV0Transformer(vocab, state_dim=STATE_DIM, d_model=32, n_layer=2,
                                 n_head=2, ffn=64, grad_ckpt=False)
    torch.save(
        {
            "format": "d061a.v0",
            "step": 0,
            "model": model0.state_dict(),
            "model_cfg": {
                "d_model": 32, "n_layer": 2, "n_head": 2, "ffn": 64,
                "vocab": vocab, "code_min": code_min, "state_dim": STATE_DIM,
                "win": WIN, "token_dim": TOKEN_DIM,
            },
        },
        ckpt_path,
    )
    try:
        model, cfg = load_author_ckpt(ckpt_path)
        assert cfg["vocab"] == vocab and cfg["code_min"] == code_min
        assert int(model.pos.shape[1]) - 2 == WIN

        # --- 5) adapter 闭环前向 ---
        n_env = 4
        mean = torch.linspace(-1, 1, TOKEN_DIM)
        std = torch.full((TOKEN_DIM,), 0.5) + 0.01
        ad = AuthorPolicyAdapter(
            model, code_min=code_min, vocab=vocab, token_mean=mean, token_std=std,
            num_envs=n_env, device="cpu", token_alpha=1.0, token_bound="none",
        )
        ad.set_ctx(ctx_from_command(0.4))
        assert ad.codes.shape == (n_env, WIN, TOKEN_DIM) and ad.codes.dtype == torch.long
        assert ad.states.shape == (n_env, WIN, STATE_DIM)
        ad.reset()
        assert int(ad.n_valid.max()) == 0 and ad.window_len() == 1
        a = ad.step(st[0:1].repeat(n_env, 1))
        assert a.shape == (n_env, TOKEN_DIM) and a.dtype == torch.float32 and torch.isfinite(a).all()
        assert int(ad.n_valid.max()) == 1 and ad.window_len() == 1
        # 自回归：第二步的 codes 末帧 = 上一步 idx
        prev_idx = ad._last_idx.clone()
        ad.step(st[0:1].repeat(n_env, 1))
        assert int(ad.n_valid.max()) == 2 and ad.window_len() == 2
        assert torch.equal(ad.codes[:, -1], prev_idx), "自回归 1 步延迟失效"
        # 满窗（t=WIN）不越 pos 表
        ad.n_valid[:] = WIN
        a_full = ad.step(st[0:1].repeat(n_env, 1))
        assert ad.window_len() == WIN and a_full.shape == (n_env, TOKEN_DIM) and torch.isfinite(a_full).all()
        # reset 后窗口归零
        ad.reset(torch.tensor([0, 2]))
        assert ad.n_valid[0].item() == 0 and ad.n_valid[1].item() == WIN

        # --- 6) 逆仿射往返（码域 [-14,12] 内取点，避免 clip 干扰断言）---
        ad2 = AuthorPolicyAdapter(
            model, code_min=code_min, vocab=vocab, token_mean=mean, token_std=std,
            num_envs=1, device="cpu", token_alpha=1.0, token_bound="none",
        )
        tok_target = ad2.token_mean * 0.5  # 必落 [-0.875, 0.75] 有效码域
        a_in = (tok_target - ad2.token_mean) / ad2._scale
        idx = torch.round(tok_target * TOKEN_SCALE_DEFAULT).to(torch.long) - code_min
        assert bool(((idx >= 0) & (idx < vocab)).all()), f"selftest 取点越码域：{idx.tolist()}"
        tok_q = (idx.to(torch.float32) + code_min) / TOKEN_SCALE_DEFAULT
        a_out = (tok_q - ad2.token_mean) / ad2._scale
        assert torch.allclose(a_out, a_in, atol=1.0 / TOKEN_SCALE_DEFAULT / (std.min() + 1e-9)), "逆仿射往返偏差过大"
        ad3 = AuthorPolicyAdapter(
            model, code_min=code_min, vocab=vocab, token_mean=mean, token_std=std,
            num_envs=1, device="cpu", token_alpha=1.0, token_bound="tanh",
        )
        # tanh 臂：越界码 -> clip 到 (-1+eps,1-eps) 后 atanh，输出有限且被 |atanh(1-eps)| 限住
        z_raw = (tok_q * 10.0 - ad3.token_mean) / ad3._scale
        z_clip = torch.atanh(z_raw.clamp(-1.0 + ad3.token_eps, 1.0 - ad3.token_eps))
        bound = float(np.arctanh(1.0 - ad3.token_eps))
        assert bool(torch.isfinite(z_clip).all()), "tanh 臂逆变换产出非有限值"
        assert float(z_clip.abs().max()) <= bound + 1e-6, "tanh 臂未 clip 到 (-1+eps,1-eps)"
    finally:
        try:
            ckpt_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Layer 2: isaac（以下所有 import 都在函数体内，模块 import 期零 isaac 依赖）
# ---------------------------------------------------------------------------
def _import_terrain_cfg():
    """terrain_cfg 双路径 import（同 train_g1_decoder._import_heavy 的 sibling 处理）。"""
    try:
        from isaac.terrain_cfg import make_terrain_importer_cfg
    except ImportError:  # pragma: no cover（本机走此支）
        from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg
    return make_terrain_importer_cfg


def _import_heavy() -> SimpleNamespace:
    """App 起来后的重型 import（复用 train_g1_decoder._import_heavy，单份组装防漂移）。"""
    try:
        from isaac import train_g1_decoder as td
    except ImportError:  # pragma: no cover（本机走此支）
        from apt_g1.isaac import train_g1_decoder as td
    return td._import_heavy()


def _import_train_g1_decoder():
    try:
        from isaac import train_g1_decoder as td
    except ImportError:  # pragma: no cover（本机走此支）
        from apt_g1.isaac import train_g1_decoder as td
    return td


# ------------------------------------------------------------- 身份信封
def _env_module_md5() -> dict:
    """env/action-term 实存文件的 md5（按 REPO_ROOT 下的真实路径算，见 R2）。

    note-6：补收关节序权威源 ``robots/g1.py``（G1_MUJOCO_TO_ISAACLAB_DOF 的来源）
    与 ``env_utils/joint_utils.py``——SF-2 的运行时 ASSERT 之外，身份信封同时记录
    这两个源文件的 md5，改动可追溯。
    """
    return {
        "g1_velocity_decoder_env.py": egd._file_md5(str(REPO_ROOT / "apt_g1" / "isaac" / "g1_velocity_decoder_env.py")),
        "sonic_action_term.py": egd._file_md5(str(REPO_ROOT / "apt_g1" / "isaac" / "sonic_action_term.py")),
        "sonic_decoder_torch.py": egd._file_md5(str(REPO_ROOT / "apt_g1" / "isaac" / "sonic_decoder_torch.py")),
        "eval_author_v0_decoder.py": egd._file_md5(str(Path(__file__).resolve())),
        "gear_sonic/robots/g1.py": egd._file_md5(str(REPO_ROOT / "gear_sonic" / "envs" / "manager_env" / "robots" / "g1.py")),
        "gear_sonic/env_utils/joint_utils.py": egd._file_md5(str(REPO_ROOT / "gear_sonic" / "envs" / "env_utils" / "joint_utils.py")),
    }


def build_identity(ckpt_path: str, model_cfg: dict | None, onnx_path: str = "") -> dict:
    """身份信封：script/git/ckpt md5 + model_cfg + env 模块 md5（obs/action 维由子进程补）。

    ``onnx_path``：CLI 请求的 SONIC decoder ONNX（空 = 走 make_env_cfg 工厂默认）。
    SF-1 的记录口径——信封里同时给出 requested 与 effective，判读可见实际生效路径。
    """
    return {
        "script": str(Path(__file__).resolve()),
        "script_md5": egd._file_md5(str(Path(__file__).resolve())),
        "git_head": egd._git_head(),
        "ckpt": str(ckpt_path),
        "ckpt_md5": egd._file_md5(str(ckpt_path)) if ckpt_path and os.path.isfile(str(ckpt_path)) else None,
        "model_cfg": dict(model_cfg) if model_cfg else None,
        "env_module_md5": _env_module_md5(),
        "joint_perm": joint_perm_digest(),
        "onnx_path": {
            "requested": str(onnx_path) if onnx_path else "",
            "consumed": bool(onnx_path),
            "effective": str(onnx_path) if onnx_path else "factory-default(make_env_cfg, SONIC_DECODER_ONNX)",
        },
        "obs_dim_assumption": {"flat": POLICY_OBS_DIM_FLAT, "rough": POLICY_OBS_DIM_ROUGH},
    }


def _read_model_cfg(ckpt_path: str) -> dict | None:
    """父进程侧读 ckpt 的 model_cfg（不进 isaac；失败返回 None 并打 WARN）。"""
    if not ckpt_path or not os.path.isfile(str(ckpt_path)):
        return None
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return dict(payload.get("model_cfg", {})) if isinstance(payload, dict) else None
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 读 ckpt model_cfg 失败：{type(exc).__name__}: {exc}", flush=True)
        return None


# ------------------------------------------------------------- 子进程体
def _run_cond_main(cli, launcher_args) -> int:
    """--run-cond 子进程体：起 flat env -> adapter 闭环 -> 打一行 AXIS_COND <json>。"""
    cond = json.loads(cli.run_cond)
    hv = _import_heavy()
    td = _import_train_g1_decoder()
    torch_mod = hv.torch
    device = getattr(launcher_args, "device", "cuda:0")
    seed = int(cond["seed"])
    vx = float(cond["vx"])
    print(f"[d061b] child: cond={cond} pid={os.getpid()} device={device}", flush=True)
    env = None
    try:
        # SF-2：关节序硬编码副本的**权威源自检**（App 已起，可 import 权威源）。
        from gear_sonic.envs.manager_env.robots.g1 import (
            G1_MUJOCO_TO_ISAACLAB_DOF as _AUTH,
        )
        assert tuple(_AUTH) == tuple(G1_MUJOCO_TO_ISAACLAB_DOF), "关节序与权威源失配"
        print(
            "[d061b] joint perm 权威源核对 OK "
            f"(gear_sonic...robots/g1.py != 硬编码副本时本行不出现) "
            f"md5={joint_perm_digest()['md5']}",
            flush=True,
        )
        # flat 配方照抄 eval_g1_decoder._eval_one_main flat 分支：建 rough cfg 再换平面几何，
        # 保 obs=321（与 D064 双臂 ckpt 同 obs 维；height_scan 在平地上读 ≈0）。
        kw = dict(
            terrain="rough",
            num_envs=cli.num_envs,
            action="decoder",
            token_stats=cli.token_stats or "",
            token_alpha=cli.token_alpha if cli.token_alpha is not None else 1.0,
            token_bound=cli.token_bound or "tanh",
            seed=seed,
        )
        if cli.onnx_path:  # SF-1：未指定则走 make_env_cfg 工厂默认，勿传空串覆盖
            kw["onnx_path"] = cli.onnx_path
        cfg = hv.make_env_cfg(**kw)
        make_terrain_importer_cfg = _import_terrain_cfg()
        cfg.scene.terrain = make_terrain_importer_cfg("plane", seed=seed)
        cfg.curriculum.terrain_levels = None
        _gen = getattr(cfg.scene.terrain, "terrain_generator", None)
        if _gen is not None and hasattr(_gen, "curriculum"):
            _gen.curriculum = False
        cmd_cfg = cfg.commands.base_velocity
        cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)  # 命令钉死（批次内再戳）
        # 假设 A2：评测消费 policy obs 建 state，须关观测噪声（eval_g1_decoder 未显式关）
        cfg.observations.policy.enable_corruption = False
        # 落地 ep_steps 口径（env 继承官方 episode_length_s=20s）
        cfg.episode_length_s = cli.ep_steps / 50.0
        env = hv.ManagerBasedRLEnv(cfg=cfg)
        mel = getattr(env, "max_episode_length", None)
        if mel is not None and int(mel) != int(cli.ep_steps):
            print(
                f"AXIS_COND_FAIL AssertionError: max_episode_length={mel} != ep_steps={cli.ep_steps}"
                "（episode_length_s 口径未落地，判据失效）",
                flush=True,
            )
            return 1
        print(
            f"[d061b] episode_length_s={cfg.episode_length_s} max_episode_length={mel} "
            f"decimation={getattr(cfg, 'decimation', None)} sim_dt={getattr(cfg.sim, 'dt', None)}",
            flush=True,
        )

        # ---- action term：拿 token 统计 / 关节映射（R1：isinstance 失败类名兜底）
        term_name, term = td._find_decoder_term(env, hv.SonicDecoderActionTerm)
        if term is None:
            raise RuntimeError("未找到 SONIC decoder action term（action='decoder' 是否生效？）")
        joint_ids = list(term._joint_ids)
        if sorted(joint_ids) != list(range(N_JOINTS)):
            raise AssertionError(f"term._joint_ids 非 0..{N_JOINTS - 1} 排列：{joint_ids}")
        perm_identity = joint_ids == list(range(N_JOINTS))
        if not perm_identity:
            print(
                f"[WARN] term._joint_ids 非恒等 {joint_ids}：policy joint_pos（资产序）"
                "按该映射前向重排到 SONIC 序（obs 关节序假设的逆映射兜底）",
                flush=True,
            )
        joint_ids_t = torch_mod.as_tensor(joint_ids, dtype=torch_mod.long, device=device)

        token_mean = term.core.token_mean
        token_std = term.core.token_std
        token_alpha = float(term.cfg.token_alpha)
        token_bound = str(term.cfg.token_bound)
        default_isaac = torch_mod.as_tensor(sonic_default_isaac(), dtype=torch_mod.float32, device=device)

        # ---- author ckpt
        model, model_cfg = load_author_ckpt(cli.ckpt, device="cpu")
        model = model.to(device)
        adapter = AuthorPolicyAdapter(
            model,
            code_min=int(model_cfg["code_min"]),
            vocab=int(model_cfg["vocab"]),
            token_mean=token_mean,
            token_std=token_std,
            num_envs=cli.num_envs,
            device=device,
            token_alpha=token_alpha,
            token_bound=token_bound,
            token_scale=TOKEN_SCALE_DEFAULT,
        )
        # token_scale 反查（语料 meta 记 16；ckpt 无此字段 -> 用默认并显式声明，见假设 A6）
        ctx = ctx_from_command(vx, 0.0, 0.0, terrain="plane")

        robot = env.scene["robot"]
        n = cli.num_envs
        episodes: list[dict] = []
        obs_dim_policy = action_dim = None
        for _batch in range(cli.episodes):
            obs, _ = env.reset()
            egd._stamp_command(env, vx)
            adapter.reset()
            adapter.set_ctx(ctx)
            obs_p = obs["policy"]
            obs_dim_policy = int(obs_p.shape[-1])
            action_dim = int(env.action_manager.total_action_dim)
            if action_dim != TOKEN_DIM:
                raise AssertionError(f"action_dim={action_dim} != {TOKEN_DIM}")
            root0 = robot.data.root_pos_w[:, :2].clone()
            alive = torch_mod.ones(n, dtype=torch_mod.bool, device=device)
            fall_step = torch_mod.full((n,), -1, dtype=torch_mod.long, device=device)
            root_end = root0.clone()
            vx_sq = torch_mod.zeros(n, device=device)
            vx_sum = torch_mod.zeros(n, device=device)
            vx_cnt = torch_mod.zeros(n, device=device)
            for step in range(cli.ep_steps):
                alive_pre = alive.clone()
                pre_pos = robot.data.root_pos_w[:, :2].clone()
                pre_vel = robot.data.root_lin_vel_b[:, 0].clone()
                jp = slice_obs_term(obs_p, "joint_pos")[:, joint_ids_t]
                quat = robot.data.root_quat_w  # wxyz
                trans = robot.data.root_pos_w - env.scene.env_origins
                st = build_state(jp, quat, trans, jp_absolute=False, default_isaac=default_isaac)
                with torch_mod.no_grad():
                    action = adapter.step(st)
                if step == 0:
                    print(
                        f"[d061b][diag] step0 t={adapter.window_len()} "
                        f"state_jp_range=({float(st[:, :N_JOINTS].min()):.3f},"
                        f"{float(st[:, :N_JOINTS].max()):.3f}) "
                        f"act_range=({float(action.min()):.3f},{float(action.max()):.3f}) "
                        f"quat0={[round(float(x), 3) for x in quat[0]]}",
                        flush=True,
                    )
                    # note-4（A4 trans 诊断）：state 末 3 维 root trans 逐维 min/mean/max，
                    # 整局只在首批次首步打一次（不阻断）。与语料 trans_m 量级对照，
                    # 用于判定 env 局部系（root_pos_w - env_origins）与语料参考系是否同量级。
                    if _batch == 0:
                        tr_d = st[:, -3:]
                        print(
                            f"[d061b][diag] step0 trans(m) per-dim "
                            f"min={[round(float(x), 4) for x in tr_d.min(dim=0).values]} "
                            f"mean={[round(float(x), 4) for x in tr_d.mean(dim=0)]} "
                            f"max={[round(float(x), 4) for x in tr_d.max(dim=0).values]}",
                            flush=True,
                        )
                obs, _, terminated, truncated, _ = env.step(action)
                obs_p = obs["policy"]
                done = terminated | truncated
                newly_fallen = alive_pre & terminated
                fall_step[newly_fallen] = step
                root_end[newly_fallen] = pre_pos[newly_fallen]
                root_end[alive_pre & done & ~terminated] = pre_pos[alive_pre & done & ~terminated]
                keep = alive_pre & ~done
                root_end[keep] = robot.data.root_pos_w[:, :2][keep]
                if cli.command_rmse:
                    vx_sq[keep] += (pre_vel[keep] - vx) ** 2
                    vx_sum[keep] += pre_vel[keep]
                    vx_cnt[keep] += 1
                alive = keep
                if not bool(alive.any()):
                    break
            disp = (root_end - root0).norm(dim=1)
            disp_x = root_end[:, 0] - root0[:, 0]
            for i in range(n):
                episodes.append(
                    {
                        "survived": bool(fall_step[i].item() < 0),
                        "fall_step": int(fall_step[i].item()) if fall_step[i].item() >= 0 else None,
                        "disp": round(float(disp[i].item()), 3),
                        "disp_x": round(float(disp_x[i].item()), 3),
                        "vx_rmse": round(float((vx_sq[i] / vx_cnt[i]).sqrt().item()), 4)
                        if cli.command_rmse and vx_cnt[i].item() > 0
                        else None,
                    }
                )
        surv = sum(e["survived"] for e in episodes)
        agg = {
            "kind": "flat",
            "cond": cond,
            "n_episodes": len(episodes),
            "survival_rate": surv / len(episodes),
            "mean_disp_x": sum(e["disp_x"] for e in episodes) / len(episodes),
            "mean_vx_rmse": (
                sum(e["vx_rmse"] for e in episodes if e["vx_rmse"] is not None) / max(1, len(episodes))
                if cli.command_rmse
                else None
            ),
            "obs_dim_policy": obs_dim_policy,
            "action_dim": action_dim,
            "perm_identity": bool(perm_identity),
            "term_joint_ids": joint_ids,
            "token_bound": token_bound,
            "token_alpha": token_alpha,
            "token_scale": TOKEN_SCALE_DEFAULT,
            "onnx_path_requested": cli.onnx_path or "",
            "episodes": episodes,
        }
        cmd_term = env.command_manager.get_term("base_velocity")
        agg["diag"] = {
            "cmd_vx_mean_last_batch": float(cmd_term.vel_command_b[:, 0].mean().item()),
            "cmd_term_shape": list(cmd_term.vel_command_b.shape),
        }
        print(
            f"[d061b][diag] flat vx={vx} cmd_vx_mean(last batch)="
            f"{agg['diag']['cmd_vx_mean_last_batch']:.4f} survival={agg['survival_rate']:.3f}",
            flush=True,
        )
        print("AXIS_COND " + json.dumps(agg, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 —— 判读要可见：类型名必进 FAIL 行
        print(f"AXIS_COND_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


# ------------------------------------------------------------- 条件展开
def expand_conditions(cli) -> list[dict]:
    """battery x vx-cmds x seeds -> 条件列表（v0 仅 flat）。"""
    if cli.battery != "flat":
        raise ValueError(f"v0 仅支持 --battery flat（收到 {cli.battery!r}）")
    vxs = egd._parse_csv_float(cli.vx_cmds)
    seeds = egd._parse_csv_int(cli.seeds)
    return [{"kind": "flat", "vx": vx, "seed": s} for s in seeds for vx in vxs]


def _child_argv(cli, cond: dict) -> list[str]:
    """从父 cli 重建子进程 argv（num-envs/episodes/ep-steps 透传；--run-cond 注入条件）。"""
    argv = [sys.executable, os.path.abspath(__file__)]
    argv += ["--ckpt", cli.ckpt, "--battery", cli.battery]
    argv += ["--num-envs", str(cli.num_envs), "--vx-cmds", cli.vx_cmds, "--seeds", cli.seeds]
    argv += ["--episodes", str(cli.episodes), "--ep-steps", str(cli.ep_steps)]
    argv += ["--token-stats", cli.token_stats or ""]
    if cli.onnx_path:
        argv += ["--onnx-path", cli.onnx_path]
    if cli.token_alpha is not None:
        argv += ["--token-alpha", str(cli.token_alpha)]
    if cli.token_bound is not None:
        argv += ["--token-bound", cli.token_bound]
    if cli.command_rmse:
        argv.append("--command-rmse")
    if not cli.headless:
        argv.append("--no-headless")
    argv += ["--run-cond", json.dumps(cond, ensure_ascii=False)]
    return argv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="D061b Axis-B：D061a author ckpt 闭环评测（flat 电池，64 维动作 -> env）"
    )
    ap.add_argument("--ckpt", default="", help="D061a author ckpt（train_author_v0.save_ckpt 产物）")
    ap.add_argument("--battery", choices=("flat",), default="flat", help="v0 仅 flat")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--seeds", default="0", help="seed csv（每 seed 一组条件）")
    ap.add_argument("--vx-cmds", default=DEFAULT_VX, help="flat 固定 vx 命令 csv（默认 0.4,0.6）")
    ap.add_argument("--episodes", type=int, default=3, help="每条件批次数")
    ap.add_argument("--ep-steps", type=int, default=3000, help="每局步数=60s@50Hz")
    ap.add_argument("--command-rmse", action=argparse.BooleanOptionalAction, default=True,
                    help="是否统计 vx RMSE（诊断用；关掉省算力）")
    ap.add_argument("--out-json", default="outputs/d061b_axis_b.json", help="汇总 JSON 路径（相对执行根）")
    ap.add_argument("--token-stats", default="", help="官方 g1-mode token 统计 npz（decoder term 必填）")
    ap.add_argument("--onnx-path", default="", help="SONIC decoder ONNX（缺省跟工厂默认）")
    ap.add_argument("--token-alpha", type=float, default=None)
    ap.add_argument("--token-bound", choices=("none", "tanh"), default=None)
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--smoke", action="store_true", help="冒烟：episodes=1 + ep_steps=100")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划与身份信封（不 import isaac）")
    ap.add_argument("--selftest-adapter", action="store_true", help="纯 torch 自测（不 import isaac）")
    ap.add_argument("--run-cond", default=None, help=argparse.SUPPRESS)
    return ap


def _apply_smoke(cli) -> None:
    if cli.smoke:
        cli.episodes = 1
        cli.ep_steps = min(int(cli.ep_steps), 100)
        cli.num_envs = min(int(cli.num_envs), 8)


def _dry_run(cli) -> int:
    """--dry-run：不 import isaac，打印计划 + 身份信封（内含可用性核对）。"""
    _apply_smoke(cli)
    conds = expand_conditions(cli)
    model_cfg = _read_model_cfg(cli.ckpt)
    ident = build_identity(cli.ckpt, model_cfg, cli.onnx_path)
    if model_cfg:
        if int(model_cfg.get("state_dim", -1)) != STATE_DIM or int(model_cfg.get("token_dim", -1)) != TOKEN_DIM:
            print(f"[WARN] ckpt model_cfg 契约不符：{model_cfg}", flush=True)
    else:
        print("[WARN] ckpt 不可读/未指定：身份信封的 ckpt_md5/model_cfg 为空（正式跑必填）", flush=True)
    plan = {
        "mode": "dry-run",
        "battery": cli.battery,
        "n_conditions": len(conds),
        "conditions": conds,
        "num_envs": cli.num_envs,
        "episodes": cli.episodes,
        "ep_steps": cli.ep_steps,
        "smoke": bool(cli.smoke),
        "out_json": cli.out_json,
        "identity": ident,
        "reuse": {
            "import_heavy": "train_g1_decoder._import_heavy",
            "find_term": "train_g1_decoder._find_decoder_term",
            "stamp_command": "eval_g1_decoder._stamp_command",
            "md5_git": "eval_g1_decoder._file_md5 / _git_head",
        },
        "token_scale": TOKEN_SCALE_DEFAULT,
        "joint_perm": {
            "mujoco_to_isaac": list(G1_MUJOCO_TO_ISAACLAB_DOF),
            "isaac_to_mujoco": ISAAC_TO_MUJOCO_PERM.tolist(),
        },
        "ctx_plane_vx04": [round(float(x), 6) for x in ctx_from_command(0.4)],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    print("AXIS_B_DRYRUN_OK", flush=True)
    return 0


def main() -> None:
    cli = build_args().parse_args()

    # ---- 纯本机分支：绝不 import isaac ----
    if cli.selftest_adapter:
        run_selftest_adapter()
        print("AXIS_B_SELFTEST_PASS", flush=True)
        return
    if cli.dry_run:
        sys.exit(_dry_run(cli))

    if not cli.ckpt:
        build_args().error("--ckpt 必填（或改用 --dry-run / --selftest-adapter）")

    if cli.run_cond is not None:
        # ---- 子进程体：AppLauncher 链与 train/eval 同款 ----
        from isaaclab.app import AppLauncher

        lp = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(lp)
        launcher_args, _ = lp.parse_known_args()
        launcher_args.num_envs = cli.num_envs
        launcher_args.headless = cli.headless
        app_launcher = AppLauncher(launcher_args)
        simulation_app = app_launcher.app
        code = 1
        try:
            code = _run_cond_main(cli, launcher_args)
        finally:
            simulation_app.close()
        sys.exit(code)

    # ---- 父进程：条件展开 -> 逐条件子进程 -> 汇总 JSON ----
    _apply_smoke(cli)
    conds = expand_conditions(cli)
    if not conds:
        build_args().error("条件展开为空（检查 --seeds/--vx-cmds）")
    model_cfg = _read_model_cfg(cli.ckpt)
    identity = build_identity(cli.ckpt, model_cfg, cli.onnx_path)
    out_path = Path(cli.out_json)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for cond in conds:
        argv = _child_argv(cli, cond)
        print(f"[d061b] condition {cond} -> spawn", flush=True)
        try:
            # encoding 必须显式：C locale 下 text=True 默认 ASCII 解码遇 UTF-8 崩（train r5 实证）
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"cond": cond, "error": f"spawn-failed: {type(exc).__name__}: {exc}"})
            continue
        out = err = ""
        timed_out = False
        try:
            out, err = proc.communicate(timeout=COND_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            timed_out = True
        if out:
            print(out, flush=True)
        cond_res = None
        for line in out.splitlines():
            if line.startswith("AXIS_COND "):
                try:
                    cond_res = json.loads(line[len("AXIS_COND "):])
                except json.JSONDecodeError:
                    cond_res = None
        if cond_res is None:
            fail = next((ln for ln in out.splitlines() if ln.startswith("AXIS_COND_FAIL")), None)
            cond_res = {
                "cond": cond,
                "error": "timeout" if timed_out else "child-failed",
                "child_fail_line": fail,
                "returncode": proc.returncode,
                "stderr_tail": err[-2000:] if err else None,
            }
        results.append(cond_res)

    ok = [r for r in results if "error" not in r]
    obs_dims = sorted({r.get("obs_dim_policy") for r in ok if r.get("obs_dim_policy")})
    act_dims = sorted({r.get("action_dim") for r in ok if r.get("action_dim")})
    env_dim_check = {
        "obs_dim_policy_observed": obs_dims,
        "action_dim_observed": act_dims,
        "obs_dim_expected": POLICY_OBS_DIM_ROUGH,
        "action_dim_expected": TOKEN_DIM,
        # flat 臂建 rough cfg 换平面几何 -> obs 应为 321（eval_g1_decoder.py:283-322 同款）
        "ok": bool(obs_dims) and obs_dims == [POLICY_OBS_DIM_ROUGH] and act_dims == [TOKEN_DIM],
    }
    n_ep = sum(r.get("n_episodes", 0) for r in ok)
    summary = {
        "format": "d061b.axis_b.v0",
        "identity": identity,
        "env_dim_check": env_dim_check,
        "conditions": results,
        "aggregate": {
            "n_conditions": len(conds),
            "n_ok": len(ok),
            "n_failed": len(results) - len(ok),
            "n_episodes": n_ep,
            "survival_rate": (sum(r["survival_rate"] * r["n_episodes"] for r in ok) / n_ep) if n_ep else None,
            "mean_disp_x": (sum(r["mean_disp_x"] * r["n_episodes"] for r in ok) / n_ep) if n_ep else None,
        },
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[d061b] 汇总：{len(ok)}/{len(conds)} 条件成功，{n_ep} 局，"
        f"dim_check={env_dim_check['ok']} -> {out_path}",
        flush=True,
    )
    print("AXIS_B_DONE", flush=True)
    sys.exit(0 if len(ok) == len(conds) and env_dim_check["ok"] else 1)


if __name__ == "__main__":
    main()
