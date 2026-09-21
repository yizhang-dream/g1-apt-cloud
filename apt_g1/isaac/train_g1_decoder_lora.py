"""D065 C 臂训练入口 + 冒烟一体脚本（v2 in-graph；FORK，同源 `train_g1_decoder.py` 分叉不回并）。

D065 C 臂 v2 = **A 臂架构（29 维关节空间动作）+ μ 路径经 LoRA decoder**：
    obs → token head MLP(64) → E49/D055 token 仿射 → 组装 994（token 64 + 本体 930）
        → SonicLoRADecoder（7 个 Linear 各挂 rank-16 零初始化 adapter）→ μ(29)
动作分布 N(μ, σ)，σ 冻结；log_prob/熵对 29 维（= E44 的架构，但用低秩 adapter 替代全权重微调）。
预注册 = [refine-logs/ds/D065_DECODER_LORA_ARM.md](../../refine-logs/ds/D065_DECODER_LORA_ARM.md)。

**为什么不是 v1（decoder 挂 env 侧 ActionTerm）**：那样 PPO 的 log_prob/熵只依赖 policy 自身
输出，目标对 adapter 梯度恒为 0/None（v1 已实测：全冻结 policy ⇒ `loss.requires_grad=False`
⇒ `loss.backward()` 首个 update 即 RuntimeError），C 臂会退化成「逐位等于 B 臂」的假阴性。
ORCS/ViBe 的等价实现把 decoder 放进策略本体（`rsl_rl/models/sonic_base_model.py:316-340`），
E44 也是同形态（`refine-logs/tracker/E.md:159`「梯度经 log N(a;μ,σ) 直达解码器」）。

【相对父本 train_g1_decoder.py 的增量】
  H1 常量/身份串：`--arm lora_policy`（env 侧 = A 臂动作通路逐字 + policy 组末位 930 维本体观测）、
     D065 任务 id、默认输出目录 d065_c_<terrain>_s<seed>、默认 `--max-iterations 5000`
     （预注册 §3 的 5000 iters 预算；父本 CLI 默认 3000 是 D064 的早期值）。
  H2 CLI：`--rank 16` `--alpha 1.0` `--trainable {policy+adapter,adapter-only}`
     `--init-head-from <ckpt>`（adapter-only 必填）`--sigma 1.0` `--sigma-trainable`
     `--allow-partial-head`（adapter-only 下显式接受 head 部分加载；默认 False=完整初始化要求）。
  H3 policy 接线（v2 核心）：把 rsl_rl 的策略类换成 `SonicLoRAPolicy`（`sonic_lora_policy.py`），
     经 runner cfg 的 `policy` dict 透传 decoder/token 常量/obs 布局；并现场断言四条：
     ① policy 组末位 = `sonic_proprio`（930 维，official_obs_dim = A 臂 actor 输入维）；
     ② `SonicActionCore.sonic_default` == 资产 `default_joint_pos`（joint_pos_rel 口径前提）；
     ③ 动作项 = JointPositionActionCfg(scale=0.5, use_default_offset=True)（= A 臂逐字）；
     ④ μ 路径的「仿射 + 994 组装」与 ActionTerm 的 `SonicActionCore` 逐位一致。
     **量化必须走 STE**：裸 `torch.round` 的零导数会把 token head 的梯度整体抹零
     （实测 head grad 全 0），故 policy 侧 `decode_obs(..., ste=True)`（前向逐位不变、反向保留
     上游梯度，同 ORCS `fsq_quantize` 的 STE 写法）；adapter 在量化器下游，不受影响。
  H4 可训集合（`--trainable`）：优化器由 runner 按 `policy.parameters()` 建好后，本脚本把
     `requires_grad=False` 的参数**从 param_groups 里摘掉**（σ 冻结 = 「std_param 不进优化器」
     的落地；adapter-only 时 token head 也被摘掉）；adapter 参数在 policy 图内 ⇒ 天然在优化器里，
     且梯度可达（`LORA_GRAD` 行会报 `grad_tensors=14/14`，与 v1 的 0/14 对照）。
     **摘除先于 `runner.load`**（SF-3：ckpt 的 optimizer state 按已摘除的组存，先 load 再摘会
     ValueError）；adapter-only 档下 head 装载 skipped 非空即报错退出（除非 `--allow-partial-head`）。
  H5 日志诊断：启动身份信封行 + 每 `LORA_DIAG_STEPS` 控制步一行 `LORA_DIAG`
     （adapter 范数、adapter/base 输出比、梯度有无、σ 值）+ 训练后 base 冻结断言。
  H6 身份信封：run.json 增 `lora` 块（rank/alpha/σ 值/σ 来源/trainable/obs 布局/base md5/
     token 仿射资产：npz 文件 md5 + **内容 md5**（mean/std 字节流）+ token_alpha/token_bound）；
     信封另注入 ckpt 顶层 `lora_envelope` 键 + 侧车 `lora_adapter.pt`（含 adapter 权重，便于单独取用；
     **adapter 权重本身已在 ckpt 的 model_state_dict 里**——decoder 属 policy 子模块）。
  H7 冒烟：`--smoke` 增 v2 段（policy 类/obs 布局/等价断言/可训集合/σ 不进优化器/梯度可达）。
  H8 子进程透传：`--smoke-terrain-check` 子命令补 v2 全部新参。

【配方】lr/kl/clip/epochs/mini-batches/entropy/gamma/lam/max_grad_norm/envs(4096)/课程/奖励/终止
全部逐字继承父本；与 A/B 臂的唯一差异 = policy 网络（见上）+ σ 冻结（预注册口径；A/B 的 σ 可训）。

冒烟（D065-C0）: `python apt_g1/isaac/train_g1_decoder_lora.py --smoke`
    # num_envs 强制 8、iterations 强制 2；结束输出单行
    # "SMOKE PASS: ..." 或 "SMOKE FAIL: ..."（供盯守代理 grep）
    # (c) 段逐地形 height_scan 检查以独立子进程跑（隐藏子命令
    # --smoke-terrain-check <terrain>，每地形超时 480s；父进程内重建第二个
    # ManagerBasedRLEnv 在 Isaac Lab 已知会静默挂起——cvgl 冒烟 r3 根因）。
    # 子进程机器可读行（flush）："TERRAIN_CHECK <terrain> PASS|FAIL <msg>"

设计决定（与派发约定二选一处）：
- gym 任务集中在本脚本注册（try/except 防重复）。训练路径不走 gym.make，
  而是直接以 ManagerBasedRLEnv(cfg=...) 构造（entry_point 事实来源
  tmp/isaac_ref/g1_init.py:16 "isaaclab.envs:ManagerBasedRLEnv"），
  因此不依赖 sibling 文件是否注册；任务 id 仅作外部可发现性登记。
- agents 超参逐字段抄官方 G1 rough PPO cfg（tmp/isaac_ref/g1_agents_ppo.py:12-37，
  字段全集见 tmp/isaac_ref/rl_cfg.py）；obs/action 维度一律从 env 运行期取。
- asymmetric policy/critic： IsaacLab 侧 wrapper 在存在 "critic" 观测组时
  返回 dict obs {"policy","critic"}。环境实测 rsl-rl-lib 2.3.3（经典 API
  OnPolicyRunner/ActorCritic/PPO），是否真支持双 obs 组以**安装源码**探测
  为准（OnPolicyRunner/RolloutStorage 源码含 "critic" 键处理）；不支持则置
  cfg.observations.critic = None 降为单组并打 [WARN]（退化路径），不硬写
  新版 API。
- 环境配置一律经 sibling 的 make_env_cfg(terrain, action=...) 工厂获取；
  v2 的 C 臂用 action="lora_policy"（= Direct 臂动作通路逐字 + policy 组末位
  930 维本体观测），token/onnx 参数不再进 env，而是进 policy（见 H3）。
- 策略类经 `register_policy_class_in_rsl_rl` 注入 rsl_rl 的类名解析命名空间，
  再以 runner cfg 的 `policy.class_name` 指定；runner 建好后**断言** policy 实例
  类型（防静默回落到 ActorCritic 而把 C 臂训成 A 臂）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version as _pkg_version
from pathlib import Path
from types import SimpleNamespace

# 仓根（本文件位于 <repo>/apt_g1/isaac/）
REPO_ROOT = Path(__file__).resolve().parents[2]

ARMS = ("lora_policy",)  # D065 C 臂 v2：env 侧 = A 臂动作通路 + 930 维本体观测（decoder 在 policy 内）
TERRAINS = ("plane", "rough", "stairs", "stones", "discrete")
SMOKE_TERRAINS = ("plane", "rough", "stairs", "stones", "discrete")
SMOKE_NUM_ENVS = 8
SMOKE_ITERS = 2
# (c) 段每地形子进程超时（秒）：起 env + reset + 3 步 + 判定的上限（r3）
TERRAIN_CHECK_TIMEOUT_S = 480
# decoder 臂 raw action = 64 维 token；direct/lora_policy 臂 = 29 关节目标
DECODER_RAW_ACTION_DIM = 64
DIRECT_ACTION_DIM = 29
# SONIC token 仿射输出（FSQ 预量化坐标，无硬界；token_bound="tanh" 时才限幅
# 在 mean±alpha*std）的量级上限：仅拦截 1e6 级的明显坏值，不做窄限幅
TOKEN_ABS_MAX = 50.0
# plane 上 height_scan≈0 的判据：policy 组 height_scan 带 Unoise(±0.1)
# （tmp/isaac_ref/velocity_env_cfg.py:134-139），均值绝对值阈值放宽到 0.15
PLANE_MEAN_ABS_TOL = 0.15

# ---- D065 C 臂 v2 常量 ----
LORA_RANK = 16             # 预注册 §3：rank=16
LORA_ALPHA = 1.0           # 预注册 §3：alpha=1.0（scale=alpha/rank=1/16）
LORA_DECODER_LAYERS = 7    # 挂 decoder 全部 7 个 Linear
LORA_DIAG_STEPS = 1200     # 每 N 个控制步打一行 LORA_DIAG（1200 = 50 iters × 24 steps）
LORA_ADAPTER_FILENAME = "lora_adapter.pt"  # 侧车（信封 + adapter 权重副本）
PROPRIO_DIM = 930          # decoder 本体历史维（= sonic_lora_policy.PROPRIO_DIM）
# decoder 输入契约总维 = token 64 + 本体 930 = 994（= sonic_lora_policy.DECODER_OBS_DIM；
# 冒烟 b2 段的 `decoder.obs_dim` 断言用，原先漏定义会在该行 NameError）
DECODER_OBS_DIM = 64 + PROPRIO_DIM
POLICY_DEFAULT_ITERS = 5000  # 预注册 §3 的 5000 iters 预算（与 A/B 平行对照）
TRAINABLE_MODES = ("policy+adapter", "adapter-only")
# SONIC decoder ONNX 默认路径（相对服务器执行根；= g1_velocity_decoder_env.SONIC_DECODER_ONNX）
SONIC_DECODER_ONNX_DEFAULT = "gear_sonic_deploy/policy/release/model_decoder.onnx"

TASK_IDS = {
    "lora_policy": "Isaac-Velocity-LoRAPolicy-G1-D065-v0",
}


class _D064ArgumentParser(argparse.ArgumentParser):
    """带跨字段校验的 parser：decoder 臂缺 --token-stats 直接 argparse error。

    校验放在 parse_args 本体（而非 main）——任何调用方 parse_args 即得到
    SystemExit(2)，不依赖入口编排。
    """

    def parse_args(self, args=None, namespace=None):
        ns = super().parse_args(args, namespace)
        if not ns.token_stats:
            self.error(
                "--token-stats <npz> 必填（C 臂 v2 的 token 仿射在 policy 内，需要官方 g1-mode "
                "token 统计 mean/std；空串时 SonicLoRAPolicy 直接抛 ValueError）"
            )
        if int(ns.rank) < 1:
            self.error(f"--rank 须 >= 1（D065 §3 规格 rank=16），收到 {ns.rank}")
        if ns.trainable == "adapter-only" and not ns.init_head_from:
            self.error(
                "--trainable adapter-only 需要 --init-head-from <ckpt>（B 臂 ckpt 或 author v0）："
                "token head 冻结后必须由外部提供合理初始化，随机冻结 head = 废臂（预注册 C 臂规格）"
            )
        return ns


def build_args() -> argparse.ArgumentParser:
    """CLI 定义（风格照 apt_g1/isaac/train_apt_isaac.py 的 build_args 段）。"""
    ap = _D064ArgumentParser(
        description=(
            "D065 C 臂 v2: Isaac Lab manager-based velocity 栈 + rsl_rl 训练入口，"
            "decoder(挂 rank-16 零初始化 LoRA) 在策略本体（in-graph）"
            "（--smoke 同时是 D065-C0 冒烟验收）"
        )
    )
    ap.add_argument(
        "--arm",
        choices=ARMS,
        default="lora_policy",
        help="lora_policy=env 侧 A 臂动作通路（29 维关节目标）+ 930 维本体观测；"
        "decoder 与 LoRA 在 policy 本体内（v2 in-graph），故只有这一臂",
    )
    ap.add_argument(
        "--terrain",
        choices=TERRAINS,
        default="rough",
        help="地形变体（透传 sibling make_env_cfg 工厂）",
    )
    ap.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="并行环境数（官方 velocity 默认 4096；--smoke 强制 8）",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（官方 runner cfg 默认 42）",
    )
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=POLICY_DEFAULT_ITERS,
        help=f"PPO 迭代数（D065 预注册 §3 = {POLICY_DEFAULT_ITERS}，与 A/B 平行对照；--smoke 强制 2）",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="输出目录（默认 apt_g1/outputs/d065_c_<terrain>_s<seed>，锚定仓根）",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="D064-G0 冒烟：num_envs=8、iterations=2，逐项自检并输出 SMOKE PASS/FAIL 行",
    )
    ap.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="无头启动（默认开；--no-headless 开视口）",
    )
    ap.add_argument("--resume", action="store_true", help="从 --ckpt 恢复训练")
    ap.add_argument("--ckpt", default="", help="恢复用 ckpt 路径（--resume 时必填）")
    ap.add_argument(
        "--token-stats",
        default="",
        help="官方 g1-mode token 统计 npz（mean/std 键）；必填（token 仿射在 policy 内）",
    )
    ap.add_argument(
        "--onnx-path",
        default="",
        help="SONIC ONNX decoder 路径（policy 侧 LoRA decoder 的 base；缺省用工厂默认值）",
    )
    ap.add_argument(
        "--token-alpha",
        type=float,
        default=None,
        help="token 仿射系数（缺省跟工厂默认 1.0）",
    )
    ap.add_argument(
        "--token-bound",
        choices=("none", "tanh"),
        default=None,
        help="token 仿射限幅（缺省跟工厂默认 tanh）",
    )
    # ---- D065 v2 新增：LoRA 超参 / 可训集合 / σ / head 初始化 ----
    ap.add_argument(
        "--rank",
        type=int,
        default=LORA_RANK,
        help=f"LoRA rank（D065 §3 规格 {LORA_RANK}；rank>=min(in,out) 时按 ORCS 语义退化全秩）",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=LORA_ALPHA,
        help=f"LoRA alpha（D065 §3 规格 {LORA_ALPHA}；低秩分支 scale=alpha/rank）",
    )
    ap.add_argument(
        "--trainable",
        choices=TRAINABLE_MODES,
        default="policy+adapter",
        help="可训集合：policy+adapter（默认）=token head/critic/adapter 全可训（from-scratch 平行对照）；"
        "adapter-only=token head 冻结（须 --init-head-from），critic 与 adapter 仍可训。"
        "两档 σ 都冻结（--sigma-trainable 可放开）",
    )
    ap.add_argument(
        "--init-head-from",
        default="",
        help="token head 初始化 ckpt（B 臂 ckpt 形状全匹配；A 臂 ckpt 末层形状不符会被跳过并报告）",
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="冻结的动作噪声 σ（缺省 = init_noise_std = 1.0，与 A/B 同起值）；σ 不进优化器",
    )
    ap.add_argument(
        "--sigma-trainable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="放开 σ 可训（默认 False = 冻结，预注册口径；A/B 臂的 σ 是可训的）",
    )
    ap.add_argument(
        "--allow-partial-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="adapter-only 档下允许 --init-head-from 部分加载（skipped 非空不再报错退出；"
        "默认 False=要求 head 完整初始化——随机冻结头=废臂）",
    )
    ap.add_argument(
        "--smoke-terrain-check",
        choices=SMOKE_TERRAINS,
        default=None,
        metavar="TERRAIN",
        help=argparse.SUPPRESS,  # 隐藏子命令：仅供 --smoke (c) 段的子进程内部使用
    )
    return ap


def _default_out_dir(arm: str, terrain: str, seed: int) -> Path:
    return REPO_ROOT / "apt_g1" / "outputs" / f"d065_c_{terrain}_s{seed}"


def _git_head() -> str:
    """git rev-parse HEAD；失败返回 ""（同款 recipe 见 train_apt_isaac.py:563-573）。"""
    try:
        g = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return g.stdout.strip() if g.returncode == 0 else ""
    except Exception:
        return ""


def _pkg_ver(name: str) -> str:
    try:
        return _pkg_version(name)
    except Exception:
        return "unknown"


def _check(cond: bool, msg: str) -> None:
    """冒烟断言：不满足即 RuntimeError（带明确原因，避免裸 assert 被 -O 剥掉）。"""
    if not cond:
        raise RuntimeError(f"smoke check failed: {msg}")


def _rsl_rl_supports_dict_obs() -> tuple[bool, str]:
    """探测安装的 rsl_rl runner/storage 是否支持 dict obs（policy/critic 两组分流）。

    环境实测 rsl-rl-lib 2.3.3（经典 API，OnPolicyRunner/ActorCritic/PPO）。
    探测以**安装源码**为准：OnPolicyRunner + RolloutStorage 源码里含 "critic"
    键处理才认定双 obs 组可用；源码不可读时退回版本号 >= 2.3 粗判（dict obs
    支持理论上自 2.3.0 起）；两者都失败保守返回 True（与 IsaacLab 2.1 桥接
    配套一致），任何误判会在冒烟 (d) 步暴露。
    """
    ver = "unknown"
    for name in ("rsl-rl-lib", "rsl-rl"):
        try:
            ver = _pkg_version(name)
            break
        except Exception:
            continue
    try:
        import inspect

        try:
            from rsl_rl.storage.rollout_storage import RolloutStorage
        except ImportError:
            from rsl_rl.storage import RolloutStorage
        from rsl_rl.runners import OnPolicyRunner

        src = inspect.getsource(OnPolicyRunner) + inspect.getsource(RolloutStorage)
        has_critic_key = '"critic"' in src or "'critic'" in src
        return has_critic_key, ver
    except Exception:
        pass
    try:
        major, minor = (int(x) for x in ver.split(".")[:2])
        return (major, minor) >= (2, 3), ver
    except Exception:
        return True, ver


def _build_env_cfg(make_env_cfg, terrain: str, arm: str, cli, device: str, suffix: str):
    """经 sibling 工厂构建 env cfg 并做训练侧必需的运行期改写（v2）。

    v2 的 C 臂 env 侧 = `make_env_cfg(terrain, action="lora_policy")`：
    - action="lora_policy" ⇒ `G1SonicLoRAPolicyEnvCfg` = **Direct 臂（A 臂）动作通路逐字**
      （同一 `DirectActionsCfg` 类：JointPositionActionCfg scale=0.5/use_default_offset=True）
      + policy 组末位追加 930 维本体观测（decoder 输入契约，见 env 模块）。
    - token/onnx 参数**不进 env**（v2 里它们在 policy 内，见 `_build_policy_wiring`），
      故不再有 v1 的 token kwargs 透传/回退后赋路径（避免两条活路）。
    改写点（全部守卫 + 打点）：num_envs / seed / sim.device / 每实例唯一地形 prim path
    （顺序多 env 冒烟防 /World/ground 冲突，height_scanner 的 mesh_prim_paths 同步改，
    路径事实 velocity_env_cfg.py:45,72）。
    """
    cfg = make_env_cfg(terrain, action=arm)
    how = f"make_env_cfg(terrain={terrain}, action={arm})"

    cfg.scene.num_envs = cli.num_envs
    if hasattr(cfg, "seed"):
        cfg.seed = cli.seed
    if hasattr(cfg.sim, "device"):
        cfg.sim.device = device

    terrain_cfg = getattr(cfg.scene, "terrain", None)
    if terrain_cfg is not None and suffix != "main":
        old_path = getattr(terrain_cfg, "prim_path", None)
        new_path = f"/World/ground_d065_{suffix}"
        terrain_cfg.prim_path = new_path
        scanner = getattr(cfg.scene, "height_scanner", None)
        if scanner is not None and old_path:
            paths = list(getattr(scanner, "mesh_prim_paths", None) or [])
            scanner.mesh_prim_paths = [new_path if p == old_path else p for p in paths]
    return cfg, how


def _register_tasks(gym, make_env_cfg, cli) -> None:
    """集中注册 D064 任务 id（try/except 防重复；训练路径不走 gym.make，见模块 docstring）。

    env_cfg_entry_point 用 functools.partial 显式绑 action 与 token kwargs：
    make_env_cfg 的默认 action="decoder"，不绑会让 direct id 经 gym.make 拿到
    decoder 臂 cfg（语义错位）。
    """
    import functools

    for arm in ARMS:
        entry_kwargs = {"action": arm}
        try:
            gym.register(
                id=TASK_IDS[arm],
                entry_point="isaaclab.envs:ManagerBasedRLEnv",  # g1_init.py:16 事实
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": functools.partial(make_env_cfg, **entry_kwargs),
                },
            )
            print(f"[d065] gym.register: {TASK_IDS[arm]} (action={arm})", flush=True)
        except gym.error.Error:
            print(f"[WARN] 任务 {TASK_IDS[arm]} 已注册，跳过（防重复）", flush=True)


# ---------------------------------------------------------------------------
# D065 v2 段（H3 policy 接线 / H4 可训集合 / H5 诊断 / H6 身份信封与落盘）
# ---------------------------------------------------------------------------
def _obs_layout(env, torch) -> dict:
    """policy 组的项布局 + decoder 本体观测项定位（H3 断言①）。

    返回 {"terms": [(name, start, width)], "official_obs_dim", "proprio_dim", "proprio_term"}。
    要求 policy 组**最后一项**是 `sonic_proprio` 且宽度 = 930（dataclass 字段序保证末位，
    见 `g1_velocity_decoder_env.LoRAPolicyObservationsCfg`）——这样 policy 侧
    `obs[:, :official_obs_dim]` 恰为官方八项，token head 输入维与 A/B 臂 actor 逐字相同。
    """
    om = env.observation_manager
    names = list(om.active_terms["policy"])
    dims = list(om.group_obs_term_dim["policy"])
    if len(names) != len(dims):
        raise RuntimeError(f"policy 组 names/dims 错位：{len(names)} vs {len(dims)}")
    terms, start = [], 0
    for name, d in zip(names, dims):
        w = int(torch.tensor(list(d)).prod())
        terms.append((str(name), start, w))
        start += w
    last_name, last_start, last_w = terms[-1]
    if last_name != "sonic_proprio" or last_w != PROPRIO_DIM:
        raise RuntimeError(
            f"policy 组末位应为 sonic_proprio({PROPRIO_DIM})，实际 {last_name}({last_w})——"
            "env cfg 的观测项序被改动（LoRAPolicyObservationsCfg 契约），拒绝开训"
        )
    return {
        "terms": terms,
        "official_obs_dim": int(last_start),
        "proprio_dim": int(last_w),
        "proprio_term": last_name,
        "obs_dim": int(start),
    }


def _build_policy_wiring(env, cli, device, hv, torch) -> dict:
    """H3：v2 policy 接线的前置构造与现场断言（任一不符即 RuntimeError，不静默）。

    ① obs 布局：policy 组末位 = `sonic_proprio`(930) ⇒ official_obs_dim = A 臂 actor 输入维；
    ② 动作通路前提：JointPositionActionCfg `scale=0.5` + `use_default_offset=True`
       （= A 臂同一 cfg 类；`last_act=(q_des-default)/sonic_scale` 的口径依据）；
    ③ `SonicActionCore.sonic_default` == 资产 `default_joint_pos`（`joint_pos_rel = q - sonic_default` 前提）；
    ④ LoRA decoder（7 层、构造时零初始化 C==B 自检）+ token 统计 npz（std 下限截断同 ActionTerm）；
    ⑤ μ 路径「仿射 + 994 组装」与 ActionTerm 的 `SonicActionCore` **逐位一致**（H3 断言③）；
    ⑥ 注册策略类名（供 runner cfg 的 `policy.class_name` 使用）。
    """
    layout = _obs_layout(env, torch)

    # ② 动作通路前提（A 臂同一 cfg 类）
    joint_term_name, joint_cfg = None, None
    for name in list(env.action_manager.active_terms):
        cfg = getattr(env.action_manager.get_term(name), "cfg", None)
        if cfg is not None and hasattr(cfg, "scale") and hasattr(cfg, "use_default_offset"):
            joint_term_name, joint_cfg = str(name), cfg
            break
    if joint_cfg is None:
        raise RuntimeError("env 里找不到关节位置动作项（JointPositionActionCfg）——动作通路与 A 臂不符")
    if float(joint_cfg.scale) != 0.5 or not bool(joint_cfg.use_default_offset):
        raise RuntimeError(
            f"动作通路与 A 臂不一致：scale={joint_cfg.scale} use_default_offset={joint_cfg.use_default_offset}"
            "（期望 0.5 / True，DirectActionsCfg 逐字）——last_act 归一化口径会失真，拒绝开训"
        )

    # ③ sonic_default == 资产 default_joint_pos
    asset = env.scene["robot"]
    ids, identity = hv.resolve_sonic_joint_ids(asset.joint_names)
    ids_t = torch.as_tensor(ids, dtype=torch.long, device=env.device)
    sonic_default = torch.as_tensor(hv._sonic_default_isaac(), dtype=torch.float32, device=env.device)
    default_pos = asset.data.default_joint_pos[:, ids_t]
    default_maxdiff = float((default_pos - sonic_default).abs().max().item())
    if default_maxdiff != 0.0:
        raise RuntimeError(
            f"资产 default_joint_pos 与 SONIC 默认角不一致（maxdiff={default_maxdiff:.3e}）——"
            "`joint_pos_rel = q - sonic_default` 的前提不成立，decoder 输入口径会漂移，拒绝开训"
        )

    # ④ decoder + token 常量
    onnx_path = cli.onnx_path or SONIC_DECODER_ONNX_DEFAULT
    token_mean, token_std = hv.load_token_stats(cli.token_stats, device=device)
    token_alpha = float(cli.token_alpha) if cli.token_alpha is not None else 1.0
    token_bound = cli.token_bound or "tanh"
    decoder = hv.build_decoder_from_onnx(onnx_path, rank=cli.rank, alpha=cli.alpha, device=device)
    decoder.collect_stats = True  # H5：逐层 adapter/base 输出比
    ci = decoder.construction_identity or {}
    if not ci.get("bitwise"):
        raise RuntimeError(
            f"零初始化 LoRA 与 base decoder 前向不一致（构造时 C != B）：maxdiff={ci.get('maxdiff')}"
        )
    if not decoder.base_frozen():
        raise RuntimeError("LoRA base 未全部冻结（requires_grad 应为 False）")
    n_adapters = len([a for a in decoder.adapters if a is not None])
    if n_adapters != LORA_DECODER_LAYERS:
        raise RuntimeError(f"adapter 层数 {n_adapters} != {LORA_DECODER_LAYERS}")

    # ⑤ μ 路径 vs ActionTerm 的仿射+组装（同一份 SonicActionCore 代码）
    mu_path = hv.SonicLoRAMuPath(
        official_obs_dim=layout["official_obs_dim"],
        decoder=decoder,
        token_mean=token_mean,
        token_std=token_std,
        token_alpha=token_alpha,
        token_bound=token_bound,
    )
    core = hv.SonicActionCore(
        num_envs=4,
        device=device,
        token_mean=token_mean,
        token_std=token_std,
        sonic_default=hv._sonic_default_isaac(),
        sonic_scale=hv._sonic_scale_isaac(),
        decoder=None,
        token_alpha=token_alpha,
        token_bound=token_bound,
    )
    eq = hv.assert_matches_action_term(mu_path, core, torch_mod=torch)
    if not eq["bitwise_assemble"]:
        raise RuntimeError(
            f"μ 路径的仿射+组装与 ActionTerm 不一致（maxdiff={eq['maxdiff_assemble']:.3e}）——拒绝开训"
        )

    # ⑥ 策略类注册 + kwargs（经 runner cfg 的 policy dict 透传）
    class_name = hv.register_policy_class_in_rsl_rl()
    policy_kwargs = {
        "official_obs_dim": int(layout["official_obs_dim"]),
        "proprio_dim": int(layout["proprio_dim"]),
        "decoder": decoder,
        "token_mean": token_mean,
        "token_std": token_std,
        "token_alpha": token_alpha,
        "token_bound": token_bound,
        "sigma": float(cli.sigma) if cli.sigma is not None else None,
        "sigma_trainable": bool(cli.sigma_trainable),
        "trainable": str(cli.trainable),
    }
    return {
        "layout": layout,
        "joint_term_name": joint_term_name,
        "joint_cfg": {"scale": float(joint_cfg.scale), "use_default_offset": bool(joint_cfg.use_default_offset)},
        "joint_ids_identity": bool(identity),
        "default_pos_maxdiff": default_maxdiff,
        "decoder": decoder,
        "mu_path": mu_path,
        "policy_class": hv.SonicLoRAPolicy,
        "base_snapshot": decoder.base_state_snapshot(),
        "base_weight_md5": decoder.base_weight_md5(),
        "construction_identity": ci,
        "equivalence": eq,
        "class_name": class_name,
        "policy_kwargs": policy_kwargs,
        "onnx_path": onnx_path,
        "token_alpha": token_alpha,
        "token_bound": token_bound,
        "degenerate_layers": [i for i, a in enumerate(decoder.adapters) if a is not None and not a.low_rank],
        "layer_shapes": [[int(lin.in_features), int(lin.out_features)] for lin in decoder.base_linears],
    }


def _resolve_ppo_and_optimizer(runner) -> tuple[object, object, object]:
    """从 rsl_rl runner 里取 (alg, optimizer, policy)（属性名做兼容查找，找不到即显式失败）。

    实读结论（rsl-rl-lib 2.3.3 经典 API）：``OnPolicyRunner.alg`` = PPO，其
    ``__init__`` 里对 ``policy.parameters()`` 建好优化器。v2 的 decoder（含 adapter）
    是 policy 子模块 ⇒ adapter 参数天然在该优化器里；本脚本只负责把冻结参数摘掉
    （σ、以及 adapter-only 下的 token head）。属性名若因版本漂移而不符，宁可启动即报错。
    """
    alg = getattr(runner, "alg", None) or getattr(runner, "algorithm", None)
    if alg is None:
        raise RuntimeError(
            f"rsl_rl runner 上找不到 alg/algorithm 属性（{type(runner).__name__}；"
            f"现有属性 {sorted(vars(runner))[:20]}）"
        )
    optimizer = getattr(alg, "optimizer", None) or getattr(alg, "optim", None)
    if optimizer is None:
        raise RuntimeError(
            f"PPO（{type(alg).__name__}）上找不到 optimizer/optim 属性"
            f"（现有属性 {sorted(vars(alg))[:20]}）——无法核对可训集合，拒绝继续"
        )
    policy = getattr(alg, "policy", None)
    if policy is None:
        raise RuntimeError(
            f"PPO（{type(alg).__name__}）上找不到 policy 属性（现有属性 {sorted(vars(alg))[:20]}）"
        )
    return alg, optimizer, policy


def _assert_policy_class(runner, policy_cls):
    """断言 runner 里构造出的策略是本模块的 `SonicLoRAPolicy`（防静默回落）。

    `register_policy_class_in_rsl_rl` + `policy.class_name` 这条路依赖 rsl_rl 的类名解析方式
    （经典 API 为 `eval(class_name)`）。若版本漂移导致解析失败/回落到 ActorCritic，C 臂会被
    悄悄训成 A 臂——故此处**显式断言实例类型**，不符即报错（含实际类型与 class_name 提示）。
    """
    _alg, _opt, policy = _resolve_ppo_and_optimizer(runner)
    if not isinstance(policy, policy_cls):
        raise RuntimeError(
            f"runner 构造出的 policy 类型是 {type(policy).__name__}，期望 {policy_cls.__name__}——"
            "策略类注册/解析失败（class_name 未被 rsl_rl 采纳），拒绝把 C 臂训成 A 臂"
        )
    return policy


def _prune_frozen_params(optimizer) -> dict:
    """把 `requires_grad=False` 的参数从优化器 param_groups 里摘掉（H4）。

    这是「σ 冻结 = std_param 不进优化器」的落地机制：σ 是 Parameter（进 state_dict、
    也进 `policy.parameters()`），但 `requires_grad=False`；摘掉后 Adam 的 state 里不会
    再有它的动量槽，且 adaptive schedule 逐组写 lr 时不会波及它。adapter-only 模式下
    token head 同样被摘掉。被摘参数仍留在模块里（可 save/load、可诊断）。

    **幂等**：二次调用摘除 0 项（组内已无冻结参数）。**首摘报告缓存**在 optimizer 上
    （`_d065_prune_report`）——SF-3 要求摘除先于 `runner.load` 执行（resume 时 ckpt 内
    optimizer state 是按已摘除的 param_groups 存的，先 load 再摘会 ValueError），
    而 `_wire_trainable` 仍须报出首摘明细，故用缓存复用而不是再摘一次报 0。
    """
    removed, kept = [], 0
    for grp in optimizer.param_groups:
        keep = [p for p in grp["params"] if p.requires_grad]
        removed.extend([p for p in grp["params"] if not p.requires_grad])
        grp["params"] = keep
        kept += len(keep)
    report = {
        "n_removed_tensors": len(removed),
        "n_kept_tensors": int(kept),
        "removed_numel": int(sum(p.numel() for p in removed)),
    }
    if getattr(optimizer, "_d065_prune_report", None) is None:
        optimizer._d065_prune_report = report
    return report


def _wire_trainable(runner, cli, torch) -> dict:
    """H4：head 初始化（可选）+ 冻结参数摘除 + 可训集合报告。

    - `--init-head-from <ckpt>`：把 token head（actor.*）灌进 policy（adapter-only 的必需项；
      policy+adapter 下也可用作热启动）。形状不符的键跳过并报告。
    - `_prune_frozen_params`：σ（始终冻结）与 adapter-only 下的 token head 被摘出优化器。
      首摘已在 `_run` 里先于 `runner.load` 做过（SF-3），此处**复用缓存报告**；若调用方
      未预先摘除（如本机自测），这里做首摘。
    - **adapter-only 的 head 完整性守卫（SF-4）**：冻结的 head 必须被完整初始化——skipped
      非空（B 臂 ckpt 首层 321≠286 / A 臂 ckpt 末层 29≠64 等）即报错退出，除非显式
      `--allow-partial-head`（该档下随机头=废臂，不能静默放行）。
    - 报告里显式给出「σ 是否在优化器里」与「adapter 参数是否在优化器里」两个可核对项。
    """
    _alg, optimizer, policy = _resolve_ppo_and_optimizer(runner)
    init_report = None
    if cli.init_head_from:
        init_report = policy.load_head_from_ckpt(cli.init_head_from, torch_mod=torch)
        if init_report["skipped"]:
            print(
                f"[WARN] --init-head-from 有 {len(init_report['skipped'])} 个键形状不符被跳过："
                f"{init_report['skipped'][:6]}（B 臂 ckpt 的 actor 首层输入 321（obs 里 actions 项 = 64 维 "
                "raw token）≠ token head 的 286（actions 项 = 29 维关节目标）⇒ 首层被跳过；"
                "A 臂 ckpt 的 actor 末层 29 维 ≠ head 的 64 维 ⇒ 末层被跳过）",
                flush=True,
            )
        if init_report["skipped"] and str(cli.trainable) == "adapter-only" and not bool(
            getattr(cli, "allow_partial_head", False)
        ):
            raise RuntimeError(
                f"adapter-only 档的 head 初始化不完整（{len(init_report['skipped'])} 个键被跳过："
                f"{init_report['skipped'][:6]}）——冻结的 head 必须被完整初始化，否则是随机头=废臂；"
                "确认接受部分加载请显式加 --allow-partial-head"
            )
    prune = getattr(optimizer, "_d065_prune_report", None)
    if prune is None:
        prune = _prune_frozen_params(optimizer)
    in_opt = {id(p) for grp in optimizer.param_groups for p in grp["params"]}
    adapter_params = policy.mu_path.adapter_params()
    report = {
        "mode": str(cli.trainable),
        "optimizer_class": type(optimizer).__name__,
        "prune": prune,
        "n_param_groups": len(optimizer.param_groups),
        "n_param_tensors_in_optimizer": len(in_opt),
        "adapter_in_optimizer": sum(1 for p in adapter_params if id(p) in in_opt),
        "adapter_tensors": len(adapter_params),
        "sigma_in_optimizer": any(id(p) == id(policy.std) for grp in optimizer.param_groups for p in grp["params"]),
        "sigma_requires_grad": bool(policy.std.requires_grad),
        "head_requires_grad": sum(1 for p in policy.actor.parameters() if p.requires_grad),
        "critic_requires_grad": sum(1 for p in policy.critic.parameters() if p.requires_grad),
        "base_requires_grad": sum(1 for p in policy.mu_path.base_parameters() if p.requires_grad),
        "lr_group0": optimizer.param_groups[0].get("lr") if optimizer.param_groups else None,
        "init_head_from": init_report,
        "trainable_report": policy.trainable_report(),
    }
    if report["adapter_in_optimizer"] != report["adapter_tensors"]:
        raise RuntimeError(
            f"adapter 参数未全部进优化器（{report['adapter_in_optimizer']}/{report['adapter_tensors']}）——"
            "decoder 不在 policy 图内？拒绝开训"
        )
    if report["sigma_in_optimizer"]:
        raise RuntimeError("σ 仍在优化器里（冻结参数摘除失效）——拒绝开训")
    if report["base_requires_grad"] != 0:
        raise RuntimeError("decoder base 仍可训（应全冻结）——拒绝开训")
    if report["mode"] == "adapter-only":
        if report["head_requires_grad"] != 0:
            raise RuntimeError("adapter-only 模式下 token head 仍可训——拒绝开训")
        if not cli.init_head_from:
            raise RuntimeError("adapter-only 必须 --init-head-from（随机冻结 head = 废臂）")
        if report["critic_requires_grad"] == 0:
            raise RuntimeError("adapter-only 模式下 critic 也被冻结——PPO 的 loss 将无可导路径")
    return report


def _lora_diag_line(policy, tag: str = "") -> str:
    """H5：一行 LoRA 诊断（adapter 范数 / adapter:base 输出比 / σ / 梯度有无）。"""
    mu_path = policy.mu_path
    ds = mu_path.adapter_delta_stats()
    layers = [d for d in ds["layers"] if d.get("adapter") is not None]
    fro = [d["delta_fro_norm"] for d in layers]
    rel = [d["rel_fro"] for d in layers]
    ls = getattr(mu_path.decoder, "last_stats", None) or {}
    ratios = [v["ratio"] for k, v in ls.items() if k.startswith("layer") and isinstance(v, dict)]
    adapter_params = mu_path.adapter_params()
    n_grad = sum(1 for p in adapter_params if p.grad is not None)
    return (
        f"LORA_DIAG{(' ' + tag) if tag else ''} envelope={mu_path.lora_scale:.4f} "
        f"sigma={float(policy.std.detach().reshape(-1)[0].item()):.4f} "
        f"delta_fro_max={max(fro) if fro else 0.0:.6e} rel_fro_max={max(rel) if rel else 0.0:.6e} "
        f"out_ratio_mean={sum(ratios) / len(ratios) if ratios else 0.0:.6e} "
        f"out_ratio_max={max(ratios) if ratios else 0.0:.6e} "
        f"grad_tensors={n_grad}/{len(adapter_params)} base_frozen={mu_path.decoder.base_frozen()}"
    )


def _lora_grad_warn(policy) -> str | None:
    """adapter 无梯度时的显式告警行（v2 里不应出现；出现即接线坏了）。"""
    adapter_params = policy.mu_path.adapter_params()
    n_grad = sum(1 for p in adapter_params if p.grad is not None)
    if n_grad == 0:
        return (
            f"LORA_GRAD_WARN adapter 参数梯度全空（0/{len(adapter_params)}）：v2 架构下 decoder 在 policy "
            "图内、log_prob/熵对 29 维动作 ⇒ 梯度必须可达。全空说明接线有问题（decoder 不在 policy 里？"
            "act_inference 被误用？），本臂结果不可信。"
        )
    return None


def _lora_payload(policy, cli, git_commit: str, onnx_path: str | None, onnx_md5: str | None) -> dict:
    """H6：adapter 权重 + 身份信封（onnx md5 / rank / alpha / σ / 种子 / git commit / 自检）。"""
    envelope = policy.mu_path.identity_envelope(
        extra={
            "seed": int(cli.seed),
            "git_commit": git_commit,
            "trainable_mode": str(cli.trainable),
            "sigma": float(policy.std.detach().reshape(-1)[0].item()),
            "sigma_trainable": bool(cli.sigma_trainable),
            "official_obs_dim": int(policy.official_obs_dim),
            "policy_class": type(policy).__name__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "onnx_path": onnx_path,
            "onnx_md5": onnx_md5 if onnx_md5 else "absent",
            "entry": "apt_g1/isaac/train_g1_decoder_lora.py",
            "arch": "in_graph_v2（decoder 在 policy 内；env 侧 = A 臂动作通路 + 930 维本体观测）",
        }
    )
    return {
        "format": 2,
        "experiment": "D065",
        "arm": "C",
        "adapter_state_dict": policy.mu_path.decoder.adapter_state_dict(),
        "envelope": envelope,
    }


def _save_lora_sidecar(out_dir: Path, payload: dict, torch) -> Path:
    """信封 + adapter 权重副本落侧车 ``lora_adapter.pt``（权重本身也在 ckpt 的 model_state_dict 内）。"""
    path = out_dir / LORA_ADAPTER_FILENAME
    torch.save(payload, str(path))
    return path


def _inject_lora_into_ckpt(ckpt_path: str | Path, payload: dict, torch) -> bool:
    """把 LoRA **信封**塞进 rsl_rl 的 ckpt（顶层 `lora` 键）；失败只告警不中断。

    v2 不重复注入权重：decoder 是 policy 子模块，adapter 权重已在 `model_state_dict` 内。
    """
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            print(f"[WARN] ckpt {ckpt_path} 顶层不是 dict（{type(obj).__name__}），未注入 LoRA 信封", flush=True)
            return False
        obj["lora"] = {
            "envelope": payload.get("envelope", {}),
            "note": "adapter 权重在 model_state_dict 内（decoder 属 policy 子模块）",
        }
        torch.save(obj, str(ckpt_path))
        return True
    except Exception as exc:  # noqa: BLE001 —— 注入失败不能中断训练（侧车仍是完整产物）
        print(f"[WARN] 向 ckpt {ckpt_path} 注入 LoRA 信封失败：{type(exc).__name__}: {exc}", flush=True)
        return False


def _wrap_runner_save(runner, payload_fn, torch) -> None:
    """让 rsl_rl 的周期性 ``runner.save`` 也带上 LoRA 信封（侧车仍独立落盘）。"""

    orig_save = runner.save

    def _save_with_lora(path, *args, **kwargs):
        orig_save(path, *args, **kwargs)
        try:
            _inject_lora_into_ckpt(path, payload_fn(), torch)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] runner.save 包装注入 LoRA 失败：{type(exc).__name__}: {exc}", flush=True)

    runner.save = _save_with_lora


def _load_lora_payload(ckpt_path: str | None, sidecar_path: Path, torch) -> dict | None:
    """读 LoRA 载荷（评测/诊断用）：优先 ckpt 内 `lora` 键，其次侧车 ``lora_adapter.pt``。"""
    if ckpt_path and Path(ckpt_path).is_file():
        try:
            obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and isinstance(obj.get("lora"), dict):
                return obj["lora"]
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 读 ckpt 内 LoRA 信封失败：{type(exc).__name__}: {exc}", flush=True)
    if sidecar_path.is_file():
        try:
            return torch.load(str(sidecar_path), map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 读 LoRA 侧车失败：{type(exc).__name__}: {exc}", flush=True)
    return None


def _install_lora_step_diag(wrapped, policy, torch, every: int = LORA_DIAG_STEPS) -> dict:
    """H5：在 EnvWrapper.step 上挂钩，每 `every` 个控制步打一行 LORA_DIAG。

    不改变 step 语义（转调原方法）；异常只告警不中断训练。返回计数器 dict（可变，便于外部读）。
    """
    state = {"steps": 0, "grad_warned": False}
    orig_step = wrapped.step

    def _step_with_diag(actions, *args, **kwargs):
        out = orig_step(actions, *args, **kwargs)
        state["steps"] += 1
        if state["steps"] % int(every) == 0:
            try:
                print(_lora_diag_line(policy, tag=f"step={state['steps']}"), flush=True)
                warn = _lora_grad_warn(policy)
                if warn is not None and not state["grad_warned"]:
                    print(warn, flush=True)
                    state["grad_warned"] = True
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] LORA_DIAG 打印失败：{type(exc).__name__}: {exc}", flush=True)
        return out

    wrapped.step = _step_with_diag
    return state


def _group_total_dim(raw) -> int:
    """group_obs_dim 某组的总维度（兼容 int / torch.Size / 逐 term 维度元组）。"""
    import torch

    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, torch.Size):
        return int(raw[0]) if len(raw) == 1 else int(torch.tensor(list(raw)).prod())
    try:
        return int(sum(int(torch.tensor(list(d)).prod()) for d in raw))
    except Exception:
        return -1


def _height_scan_obs(env, policy_obs):
    """取 height_scan 观测：(tensor|None, 取法说明)。

    isaaclab 2.1.0 事实（cvgl 冒烟 r6 根因修正）：ObservationManager 无
    `_group_obs_term_obses` 属性；`active_terms[group]`（2.1.0 名，新版叫 group_obs_term_names）是按 term
    顺序的 list[str]，`group_obs_term_dim[group]` 是同序的
    list[tuple[int, ...]]（逐 term 维度 tuple，不是 term 名 -> 维度的 dict）。
    切片法：按名定位 height_scan 的 index i，前 i 个 term 的维度乘积累加得
    起点 start，第 i 个的乘积得宽度 w，从 (N, D) 的 policy obs 切
    [:, start:start+w]——不假设其在组内位置（velocity_env_cfg.py:134-139
    基准 cfg 在末位仅是特例）。组内无 "height_scan"（如 plane 移除该
    term）-> "term-absent"（plane 由 _terrain_verdict 既有分支判 ≈0）。
    任何异常必须带类型名进返回 msg，禁止裸 "unavailable"（r4-r6 连续被
    异常吞没掩蔽的教训）。
    """
    import torch

    om = env.observation_manager
    try:
        names = om.active_terms["policy"]  # 2.1.0 属性名（group_obs_term_names 是新版名，r7 实证不存在）
        dims = om.group_obs_term_dim["policy"]
        if "height_scan" not in names:
            return None, "term-absent"
        idx = names.index("height_scan")
        if idx >= len(dims):
            return None, f"IndexError: names/dims 错位 names={len(names)} dims={len(dims)}"
        start = 0
        for term_dims in dims[:idx]:
            start += int(torch.tensor(list(term_dims)).prod())
        width = int(torch.tensor(list(dims[idx])).prod())
        return policy_obs[:, start:start + width], f"slice@{idx}[{start}:{start + width}]"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _terrain_verdict(terrain: str, hs, how: str) -> tuple[bool, str]:
    """单地形 height_scan 判据。plane 断言 ≈0；其余断言非零且有方差。"""
    import torch

    if hs is None:
        if terrain == "plane" and how == "term-absent":
            return True, "height_scan term 不存在（plane cfg 移除，同 flat_env_cfg.py:22-23）→ 视为 ≈0"
        return False, f"height_scan obs 不可取（{how}）"
    finite = bool(torch.isfinite(hs).all())
    mean_abs = float(hs.abs().mean())
    std = float(hs.std())
    mx = float(hs.abs().max())
    stats = f"mean_abs={mean_abs:.4f} std={std:.4f} max={mx:.4f} finite={finite} ({how})"
    if not finite:
        return False, f"height_scan 含非有限值 {stats}"
    if terrain == "plane":
        ok = mean_abs < PLANE_MEAN_ABS_TOL
        return ok, ("≈0 " if ok else "非零(超阈值) ") + stats
    ok = mx > 0.02 and std > 1e-3
    return ok, ("非零有方差 " if ok else "疑似全零/无方差 ") + stats


def _token_stats_content_md5(path: str | None) -> dict:
    """token 仿射资产 npz 的**内容**指纹（N-3）。

    与文件字节 md5 的区别：npz 是 zip 容器，重新打包/换压缩等级会改文件 md5 但仿射常量
    不变；`load_token_stats` 实际消费的是 `mean`/`std` 两键 ⇒ 内容 md5 = 这两键数组
    **原始字节流**的 md5（float32 序，与 dtype/形状一并落账）。读不了就返回
    ``unreadable:<异常名>``（身份块缺一项不阻断训练，但值本身可 grep 判读）。
    """
    try:
        import numpy as _np

        z = _np.load(str(path))
        mean = _np.asarray(z["mean"], dtype=_np.float32)
        std = _np.asarray(z["std"], dtype=_np.float32)
        h = hashlib.md5(mean.tobytes() + std.tobytes()).hexdigest()
        return {
            "content_md5": h,
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "keys": sorted(str(k) for k in z.files),
        }
    except Exception as exc:  # noqa: BLE001 —— 身份块缺项不阻断训练
        return {"content_md5": f"unreadable:{type(exc).__name__}", "error": str(exc)[:200]}


def _write_run_json(out_dir: Path, cli, git_commit: str, onnx_path: str | None, onnx_md5,
                    env_info: dict, versions: dict, lora_envelope: dict | None = None) -> None:
    """身份信封（最小实现，照 train_apt_isaac.py 的 ckpt 身份做法落 run.json）。

    D065 H6：增 ``lora`` 块（rank/alpha/层表/base 权重 md5/两条遗忘检查/onnx md5/种子/
    git commit）——C 臂的身份必须能自证「挂的是哪一层、什么超参、base 是否原样」。
    """
    payload = {
        "entry": "apt_g1/isaac/train_g1_decoder_lora.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "format": 1,
        "git_commit": git_commit,
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cli).items()},
        "decoder_onnx": {
            "path": onnx_path,
            "md5": onnx_md5 if onnx_md5 else "absent",
        },
        "lora": lora_envelope if lora_envelope is not None else "absent",
        "env_cfg": env_info,
        "versions": versions,
    }
    path = out_dir / "run.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[d065] run.json -> {path}", flush=True)


def _import_heavy() -> SimpleNamespace:
    """App 启动后的全部 isaaclab / rsl_rl / sibling imports（app 起来之前禁止）。

    父进程（_run）与 terrain-check 子进程（_terrain_check_main）共用此唯一
    入口——双兼容 import 只写一份，保证两进程解析到同一模块身份。
    返回 SimpleNamespace：torch/np/gym/ManagerBasedRLEnv/RslRl*Cfg/EnvWrapper/
    wrapper_name/OnPolicyRunner/mdp/file_md5/make_env_cfg + v2 policy 接线组
    （SonicLoRAPolicy/register_policy_class_in_rsl_rl/build_decoder_from_onnx/
    load_token_stats/SonicLoRAMuPath/assert_matches_action_term/PROPRIO_DIM）
    + ActionTerm 侧常量与核心（SonicActionCore/_sonic_default_isaac/_sonic_scale_isaac/
    resolve_sonic_joint_ids，供 H3 的等价断言与前提核对用）。
    """
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg
    try:
        # 派发约定名（IsaacLab 2.1 实测镜像）；部分版本改名为 RslRlVecEnvWrapper
        from isaaclab_rl.rsl_rl import RlRlVecEnvWrapper as EnvWrapper
        wrapper_name = "RlRlVecEnvWrapper"
    except ImportError:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as EnvWrapper
        wrapper_name = "RslRlVecEnvWrapper"
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:  # 本地参考文件(tmp/isaac_ref/rsl_rl_wrapper.py)为 404 占位，此名运行期验证
        raise RuntimeError(
            "rsl_rl 不可用（IsaacLab 2.1 桥接 rsl-rl-lib 2.3.3 实测）；"
            "请在 isaaclab venv 内运行"
        ) from exc
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

    if str(REPO_ROOT) not in sys.path:
        # 以 `python apt_g1/isaac/train_g1_decoder_lora.py` 直跑时 sys.path[0] 是脚本
        # 目录而非仓根，补上保证 apt_g1.* 可导入（服务器 PYTHONPATH 亦含仓根）
        sys.path.insert(0, str(REPO_ROOT))
    # sibling 双兼容 import：与 g1_velocity_decoder_env.py:146-150 完全同构
    # （主支 isaac.* / except ImportError / 回退 apt_g1.isaac.*）。服务器
    # PYTHONPATH 同时含仓根与仓根/apt_g1，同一文件会被加载成两个模块身份；
    # 主支必须走 isaac.*，保证 policy 侧的 LoRA 模块与 env 侧的 obs 项
    # （其内部同为 isaac.*）解析到同一身份，否则 isinstance/常量比对会失真
    # （cvgl 冒烟第二轮根因）。
    try:
        from isaac.ckpt_identity import file_md5
        from isaac.g1_velocity_decoder_env import make_env_cfg
        from isaac.sonic_action_term import (
            SonicActionCore,
            _sonic_default_isaac,
            _sonic_scale_isaac,
            resolve_sonic_joint_ids,
        )
        from isaac.sonic_lora_policy import (
            PROPRIO_DIM as _PROPRIO_DIM,
            SonicLoRAMuPath,
            SonicLoRAPolicy,
            assert_matches_action_term,
            build_decoder_from_onnx,
            load_token_stats,
            register_policy_class_in_rsl_rl,
        )
    except ImportError:  # pragma: no cover（本机走此支，需 apt_g1 可导入）
        try:
            from apt_g1.isaac.ckpt_identity import file_md5
            from apt_g1.isaac.g1_velocity_decoder_env import make_env_cfg
            from apt_g1.isaac.sonic_action_term import (
                SonicActionCore,
                _sonic_default_isaac,
                _sonic_scale_isaac,
                resolve_sonic_joint_ids,
            )
            from apt_g1.isaac.sonic_lora_policy import (
                PROPRIO_DIM as _PROPRIO_DIM,
                SonicLoRAMuPath,
                SonicLoRAPolicy,
                assert_matches_action_term,
                build_decoder_from_onnx,
                load_token_stats,
                register_policy_class_in_rsl_rl,
            )
        except ImportError as exc:
            raise RuntimeError(
                "sibling 模块缺失（isaac.sonic_action_term / isaac.sonic_lora_policy 与 "
                "g1_velocity_decoder_env 均不可导入；应由 D064/D065 同批产出）；"
                f"原始错误: {exc}"
            ) from exc

    return SimpleNamespace(
        torch=torch,
        np=np,
        gym=gym,
        ManagerBasedRLEnv=ManagerBasedRLEnv,
        RslRlPpoActorCriticCfg=RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg=RslRlPpoAlgorithmCfg,
        RslRlOnPolicyRunnerCfg=RslRlOnPolicyRunnerCfg,
        EnvWrapper=EnvWrapper,
        wrapper_name=wrapper_name,
        OnPolicyRunner=OnPolicyRunner,
        mdp=mdp,
        file_md5=file_md5,
        make_env_cfg=make_env_cfg,
        # ---- v2 policy 接线组 ----
        SonicLoRAPolicy=SonicLoRAPolicy,
        SonicLoRAMuPath=SonicLoRAMuPath,
        register_policy_class_in_rsl_rl=register_policy_class_in_rsl_rl,
        build_decoder_from_onnx=build_decoder_from_onnx,
        load_token_stats=load_token_stats,
        assert_matches_action_term=assert_matches_action_term,
        PROPRIO_DIM=_PROPRIO_DIM,
        # ---- ActionTerm 侧常量/核心（H3 前提核对与等价断言用）----
        SonicActionCore=SonicActionCore,
        _sonic_default_isaac=_sonic_default_isaac,
        _sonic_scale_isaac=_sonic_scale_isaac,
        resolve_sonic_joint_ids=resolve_sonic_joint_ids,
    )


def _run(cli, out_dir: Path, launcher_args) -> None:
    """App 起来之后的全部逻辑（smoke 与正式训练共用）。"""
    hv = _import_heavy()
    # 解包为局部名：_run 既有引用零改动
    torch, np = hv.torch, hv.np
    gym = hv.gym
    ManagerBasedRLEnv = hv.ManagerBasedRLEnv
    RslRlPpoActorCriticCfg = hv.RslRlPpoActorCriticCfg
    RslRlPpoAlgorithmCfg = hv.RslRlPpoAlgorithmCfg
    RslRlOnPolicyRunnerCfg = hv.RslRlOnPolicyRunnerCfg
    EnvWrapper = hv.EnvWrapper
    wrapper_name = hv.wrapper_name
    OnPolicyRunner = hv.OnPolicyRunner
    mdp = hv.mdp
    file_md5 = hv.file_md5
    make_env_cfg = hv.make_env_cfg
    SonicLoRAPolicy = hv.SonicLoRAPolicy

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    device = getattr(launcher_args, "device", "cuda:0")
    print(
        f"[CFG] out={out_dir} arm={cli.arm} terrain={cli.terrain} "
        f"num_envs={cli.num_envs} iters={cli.max_iterations} seed={cli.seed} "
        f"smoke={cli.smoke} headless={cli.headless} device={device} "
        f"wrapper={wrapper_name} isaaclab={_pkg_ver('isaaclab')} rsl_rl={_pkg_ver('rsl-rl-lib')}",
        flush=True,
    )

    _register_tasks(gym, make_env_cfg, cli)

    # ---- 主 env（arm/terrain 按 CLI）----
    cfg, cfg_how = _build_env_cfg(make_env_cfg, cli.terrain, cli.arm, cli, device, suffix="main")
    print(f"[d065] env cfg via {cfg_how}", flush=True)

    # asymmetric 退化判定（须在 env 创建前决定是否丢弃 critic 组）
    has_critic_group = getattr(cfg.observations, "critic", None) is not None
    asym_ok, rsl_ver = _rsl_rl_supports_dict_obs()
    if has_critic_group and not asym_ok:
        cfg.observations.critic = None
        has_critic_group = False
        print(
            f"[WARN] rsl_rl {rsl_ver} 不支持 dict obs（policy/critic 两组），"
            "退化路径：丢弃 critic 观测组，ActorCritic 共用单组 obs", flush=True
        )
    else:
        print(f"[d065] asymmetric(python rsl_rl={rsl_ver}): critic 组存在={has_critic_group}", flush=True)

    env = ManagerBasedRLEnv(cfg=cfg)
    obs, _ = env.reset()
    if not (isinstance(obs, dict) and "policy" in obs):
        raise RuntimeError(f"env obs 应为含 'policy' 组的 dict，实际 {type(obs)} keys={list(obs) if isinstance(obs, dict) else '-'}")
    policy_obs = obs["policy"]
    critic_obs = obs.get("critic") if isinstance(obs, dict) else None

    obs_dim_policy = int(policy_obs.shape[-1])
    obs_dim_critic = int(critic_obs.shape[-1]) if critic_obs is not None else 0
    action_dim = int(env.action_manager.total_action_dim)
    expected_action = DIRECT_ACTION_DIM  # C 臂 v2 动作空间 = 29 维关节目标（同 A 臂）
    print(
        f"[d065] obs(policy)={obs_dim_policy} obs(critic)={obs_dim_critic or '-'} "
        f"action={action_dim}（期望 {expected_action}）",
        flush=True,
    )

    # ---- D065 v2 H3: policy 接线（decoder 在 policy 内；env 侧 = A 臂动作通路 + 930 维本体观测）----
    wiring = _build_policy_wiring(env, cli, device, hv, torch)
    onnx_path = wiring["onnx_path"]
    onnx_md5 = file_md5(onnx_path) if onnx_path else None
    layout = wiring["layout"]
    print(
        f"[d065] LORA wiring: arm={cli.arm} rank={cli.rank} alpha={cli.alpha} "
        f"trainable={cli.trainable} sigma={wiring['policy_kwargs']['sigma']} "
        f"layers={len(wiring['layer_shapes'])} shapes={wiring['layer_shapes']} "
        f"degenerate_layers={wiring['degenerate_layers']}",
        flush=True,
    )
    print(
        f"[d065] LORA obs 契约: obs(policy)={layout['obs_dim']} = official={layout['official_obs_dim']}"
        f"（A 臂 actor 输入维）+ {layout['proprio_term']}={layout['proprio_dim']}；"
        f"动作通路 scale={wiring['joint_cfg']['scale']} use_default_offset={wiring['joint_cfg']['use_default_offset']}"
        f"（= A 臂逐字）；资产 default_joint_pos vs SONIC 默认角 maxdiff={wiring['default_pos_maxdiff']}",
        flush=True,
    )
    print(
        f"[d065] LORA identity: base_md5={wiring['base_weight_md5']} onnx={onnx_path!r} "
        f"onnx_md5={onnx_md5 or 'absent'} construction_identity(bitwise)="
        f"{wiring['construction_identity'].get('bitwise')} "
        f"equivalence(bitwise_assemble)={wiring['equivalence'].get('bitwise_assemble')} "
        f"maxdiff_assemble={wiring['equivalence'].get('maxdiff_assemble')}",
        flush=True,
    )
    if wiring["degenerate_layers"]:
        print(
            f"[WARN] rank={cli.rank} 在层 {wiring['degenerate_layers']} 触发 ORCS 全秩退化分支"
            "（rank>=min(in,out)）——预注册 §3 要求 rank=16 不触发，请核对 --rank",
            flush=True,
        )

    _write_run_json(
        out_dir,
        cli,
        _git_head(),
        onnx_path,
        onnx_md5,
        {
            "arm": cli.arm,
            "terrain": cli.terrain,
            "num_envs": cli.num_envs,
            "seed": cli.seed,
            "cfg_built_via": cfg_how,
            "task_id": TASK_IDS[cli.arm],
            "action_dim": action_dim,
            "obs_policy_dim": obs_dim_policy,
            "obs_critic_dim": obs_dim_critic,
            "official_obs_dim": layout["official_obs_dim"],
            "proprio_dim": layout["proprio_dim"],
            "obs_terms": [list(t) for t in layout["terms"]],
            "action_term": wiring["joint_term_name"],
            "action_cfg": wiring["joint_cfg"],
            "default_pos_maxdiff": wiring["default_pos_maxdiff"],
            "has_critic_group": has_critic_group,
            "obs_groups": sorted(obs.keys()),
            "decimation": getattr(cfg, "decimation", None),
            "episode_length_s": getattr(cfg, "episode_length_s", None),
            "sim_dt": getattr(cfg.sim, "dt", None),
            "sim_device": device,
            "policy_class": wiring["class_name"],
        },
        {
            "isaaclab": _pkg_ver("isaaclab"),
            "isaacsim": _pkg_ver("isaacsim"),
            "rsl_rl": _pkg_ver("rsl-rl-lib") if _pkg_ver("rsl-rl-lib") != "unknown" else _pkg_ver("rsl-rl"),
            "torch": torch.__version__,
        },
        lora_envelope=wiring["decoder"].identity_envelope(
            extra={
                "seed": int(cli.seed),
                "git_commit": _git_head(),
                "trainable_mode": str(cli.trainable),
                # N-3：σ 值（生效值 = --sigma 或 init_noise_std 默认 1.0；run.json 早于
                # policy 构造，故写 cfg 口径而非 policy.std 实测值，来源显式标注）
                "sigma": float(cli.sigma) if cli.sigma is not None else 1.0,
                "sigma_source": "--sigma" if cli.sigma is not None else "init_noise_std=1.0（默认）",
                "sigma_trainable": bool(cli.sigma_trainable),
                # N-3：token 仿射资产（内容 md5 + 仿射系数/限幅——decoder 输入契约的另一半）
                "token_stats_path": str(cli.token_stats),
                "token_stats_file_md5": (file_md5(cli.token_stats) or "unreadable") if cli.token_stats else "absent",
                "token_stats_content": _token_stats_content_md5(cli.token_stats),
                "token_alpha": float(wiring["token_alpha"]),
                "token_bound": str(wiring["token_bound"]),
                "official_obs_dim": int(layout["official_obs_dim"]),
                "policy_class": wiring["class_name"],
                "onnx_path": onnx_path,
                "onnx_md5": onnx_md5 if onnx_md5 else "absent",
                "arch": "in_graph_v2",
            }
        ),
    )

    # ---- rsl_rl runner（超参逐字段=官方 G1 rough，g1_agents_ppo.py:12-37 / rl_cfg.py）----
    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.seed = cli.seed
    runner_cfg.device = device
    runner_cfg.num_steps_per_env = 24  # g1_agents_ppo.py:13
    runner_cfg.max_iterations = cli.max_iterations  # 覆盖 :14
    runner_cfg.save_interval = 50  # :15
    runner_cfg.experiment_name = out_dir.name  # :16（改为 D064 运行名）
    runner_cfg.empirical_normalization = False  # :17
    runner_cfg.policy = RslRlPpoActorCriticCfg(  # :18-23
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    runner_cfg.algorithm = RslRlPpoAlgorithmCfg(  # :24-37
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    runner_cfg.run_name = f"{cli.arm}_{cli.terrain}"

    # ---- D065 v2 H3：策略类 + v2 kwargs 注入 runner cfg 的 policy dict ----
    # rsl_rl 经典 API 的 runner 用 eval(class_name) 在其模块命名空间解析类名
    # （register_policy_class_in_rsl_rl 已把类挂进该命名空间），extra kwargs 经
    # **cfg["policy"] 透传给策略构造函数（decoder/token 常量/obs 布局/σ/trainable）。
    runner_cfg.policy.class_name = wiring["class_name"]
    runner_cfg_dict = runner_cfg.to_dict()
    runner_cfg_dict["policy"].update(wiring["policy_kwargs"])

    if cli.smoke:
        _smoke(
            cli=cli,
            out_dir=out_dir,
            env=env,
            obs=obs,
            policy_obs=policy_obs,
            critic_obs=critic_obs,
            obs_dim_policy=obs_dim_policy,
            obs_dim_critic=obs_dim_critic,
            action_dim=action_dim,
            expected_action=expected_action,
            has_critic_group=has_critic_group,
            runner_cfg=runner_cfg_dict,
            EnvWrapper=EnvWrapper,
            OnPolicyRunner=OnPolicyRunner,
            torch=torch,
            wiring=wiring,
        )
        return

    # ---- 正式训练 ----
    wrapped = EnvWrapper(env)
    runner = OnPolicyRunner(wrapped, runner_cfg_dict, log_dir=str(out_dir), device=runner_cfg.device)
    policy = _assert_policy_class(runner, SonicLoRAPolicy)
    # ---- SF-3：冻结参数摘除必须先于 runner.load ----
    # ckpt 里的 optimizer state 是按「已摘除 σ（及 adapter-only 下的 token head）」的
    # param_groups 存的；若先 load 再摘，torch 的 `Optimizer.load_state_dict` 会因
    # 「组内参数个数不符」抛 ValueError（本机 torch 2.12 实测：
    # "loaded state dict contains a parameter group that doesn't match the size of
    # optimizer's group"）⇒ resume 回归。摘除幂等，`_wire_trainable` 复用首摘报告。
    _alg_pre, optimizer_pre, _policy_pre = _resolve_ppo_and_optimizer(runner)
    pre_prune = _prune_frozen_params(optimizer_pre)
    print(
        f"[d065] LORA pre-load prune（先于 runner.load，保 PPO 状态可装载）: {pre_prune}",
        flush=True,
    )
    if cli.resume:
        if not cli.ckpt:
            raise RuntimeError("--resume 需要 --ckpt <path>")
        runner.load(str(Path(cli.ckpt).resolve()))
        print(f"[d065] resumed from {cli.ckpt}（adapter 权重在 model_state_dict 内，随 policy 一起恢复）", flush=True)

    # ---- D065 v2 H4: 可训集合（head 初始化 + 冻结参数摘除 + 报告）----
    opt_report = _wire_trainable(runner, cli, torch)
    print(
        f"[d065] LORA trainable: mode={opt_report['mode']} class={opt_report['optimizer_class']} "
        f"param_groups={opt_report['n_param_groups']} tensors_in_opt={opt_report['n_param_tensors_in_optimizer']} "
        f"adapter_in_opt={opt_report['adapter_in_optimizer']}/{opt_report['adapter_tensors']} "
        f"sigma_in_opt={opt_report['sigma_in_optimizer']} sigma={opt_report['trainable_report']['sigma']['value']} "
        f"head_trainable={opt_report['head_requires_grad']} critic_trainable={opt_report['critic_requires_grad']} "
        f"base_trainable={opt_report['base_requires_grad']} pruned={opt_report['prune']}",
        flush=True,
    )
    if opt_report["init_head_from"] is not None:
        print(f"[d065] LORA head init: {opt_report['init_head_from']}", flush=True)
    print(_lora_diag_line(policy, tag="init"), flush=True)

    # 周期性 ckpt 也带 LoRA 信封 + 每 LORA_DIAG_STEPS 控制步一行诊断
    _wrap_runner_save(
        runner,
        lambda: _lora_payload(policy, cli, _git_head(), onnx_path, onnx_md5),
        torch,
    )
    diag_state = _install_lora_step_diag(wrapped, policy, torch)

    runner.learn(num_learning_iterations=runner_cfg.max_iterations)
    final_ckpt = out_dir / "model_final.pt"
    runner.save(str(final_ckpt))
    print(f"[d065] done. final ckpt -> {final_ckpt}", flush=True)

    # ---- D065 v2 H5/H6: 训练后收尾（诊断 + base 冻结断言 + 梯度事实 + 侧车）----
    print(_lora_diag_line(policy, tag=f"final step={diag_state['steps']}"), flush=True)
    warn = _lora_grad_warn(policy)
    if warn is not None:
        print(warn, flush=True)
    adapter_grads = sum(1 for p in policy.mu_path.adapter_params() if p.grad is not None)
    print(
        f"[d065] LORA grad check: adapter 参数带梯度 {adapter_grads}/{len(policy.mu_path.adapter_params())}"
        "（v2 期望 14/14；v1 架构下为 0/14）",
        flush=True,
    )
    base_diff = policy.mu_path.base_state_maxdiff(wiring["base_snapshot"])
    print(
        f"[d065] LORA base frozen check: maxdiff={base_diff:.3e}（预注册 §3 要求 0.0；"
        f"base_md5={policy.mu_path.base_weight_md5()}）",
        flush=True,
    )
    sidecar = _save_lora_sidecar(
        out_dir, _lora_payload(policy, cli, _git_head(), onnx_path, onnx_md5), torch
    )
    print(f"[d065] LoRA adapter 侧车 -> {sidecar}", flush=True)
    if base_diff != 0.0:
        raise RuntimeError(
            f"训练后 base 权重与初始不一致（maxdiff={base_diff:.3e}）——"
            "「base 冻结、只适配 adapter」的冻结范围被破坏，C 臂无效（ckpt/侧车已落盘备查）"
        )
    env.close()


def _smoke(
    *,
    cli,
    out_dir: Path,
    env,
    obs,
    policy_obs,
    critic_obs,
    obs_dim_policy: int,
    obs_dim_critic: int,
    action_dim: int,
    expected_action: int,
    has_critic_group: bool,
    runner_cfg,
    EnvWrapper,
    OnPolicyRunner,
    torch,
    wiring: dict,
) -> None:
    """D065-C0 冒烟（--smoke 时 num_envs=8 / iters=2 已在 main 强制）。"""
    n = cli.num_envs
    layout = wiring["layout"]
    decoder = wiring["decoder"]
    dev = runner_cfg.get("device") if isinstance(runner_cfg, dict) else getattr(runner_cfg, "device", "cuda:0")
    _check(n == SMOKE_NUM_ENVS, f"smoke num_envs 应为 {SMOKE_NUM_ENVS}，实际 {n}")
    _check(cli.max_iterations == SMOKE_ITERS, f"smoke iterations 应为 {SMOKE_ITERS}")

    # (a) obs 形状断言 + v2 契约（critic 组 = policy 官方八项 ⇒ 维度必须相等）
    _check(policy_obs.ndim == 2, f"policy obs 应 2 维，实际 shape={tuple(policy_obs.shape)}")
    _check(policy_obs.shape[0] == n, f"policy obs batch={policy_obs.shape[0]} != num_envs={n}")
    _check(obs_dim_policy > 0, "policy obs 维度应 > 0")
    dim_from_mgr = _group_total_dim(env.observation_manager.group_obs_dim.get("policy"))
    if dim_from_mgr > 0:
        _check(dim_from_mgr == obs_dim_policy, f"policy obs 维度 {obs_dim_policy} != manager 报告 {dim_from_mgr}")
    _check(layout["obs_dim"] == obs_dim_policy,
           f"obs 布局总维 {layout['obs_dim']} != policy obs 维 {obs_dim_policy}")
    if has_critic_group:
        _check(critic_obs is not None, "cfg 声明有 critic 组但 env 未返回 critic obs")
        _check(critic_obs.ndim == 2 and critic_obs.shape[0] == n and int(critic_obs.shape[-1]) > 0,
               f"critic obs 形状异常 shape={tuple(critic_obs.shape)}")
        # critic 组 = policy 官方八项的原值版 ⇒ 其维度必须等于 head 输入维（= A/B 臂 actor 输入维）
        _check(int(critic_obs.shape[-1]) == layout["official_obs_dim"],
               f"critic obs 维 {int(critic_obs.shape[-1])} != official_obs_dim {layout['official_obs_dim']}"
               "（critic 应与 A/B 特权组逐字同构）")
    else:
        print("[WARN] 单 obs 组模式（env cfg 无 critic 组或已退化）", flush=True)
    print(
        f"[d065] smoke a) obs shapes PASS policy={obs_dim_policy}"
        f"（official={layout['official_obs_dim']}+{layout['proprio_term']}={layout['proprio_dim']}）"
        f" critic={obs_dim_critic or '-'}",
        flush=True,
    )

    # (b) 动作通路：29 维关节目标（= A 臂 DirectActionsCfg），step 正常
    _check(action_dim == DIRECT_ACTION_DIM, f"C 臂动作维度应 {DIRECT_ACTION_DIM}，实际 {action_dim}")
    _check(expected_action == DIRECT_ACTION_DIM, "expected_action 应为 29（A 臂同口径）")
    env.step(torch.zeros(n, action_dim, device=env.device))
    env.step(0.1 * torch.randn(n, action_dim, device=env.device))
    joint_term = env.action_manager.get_term(wiring["joint_term_name"])
    processed = getattr(joint_term, "processed_actions", None)
    _check(processed is not None, f"动作项 {wiring['joint_term_name']!r} 无 processed_actions")
    _check(tuple(processed.shape) == (n, DIRECT_ACTION_DIM),
           f"processed_actions 形状 {tuple(processed.shape)} != {(n, DIRECT_ACTION_DIM)}")
    _check(bool(torch.isfinite(processed).all()), "processed_actions 含非有限值")
    print(
        f"[d065] smoke b) 动作通路 PASS term={wiring['joint_term_name']!r} "
        f"shape={tuple(processed.shape)} max|.|={float(processed.abs().max()):.4f} "
        f"（scale={wiring['joint_cfg']['scale']} use_default_offset={wiring['joint_cfg']['use_default_offset']} = A 臂逐字）",
        flush=True,
    )

    # (b2) v2 LoRA 段：构造时 C==B / 置零回退 / 仿射+组装等价 / 参数集合 / 层结构
    ci = wiring["construction_identity"]
    _check(bool(ci.get("bitwise")),
           f"零初始化 LoRA 与 base decoder 未逐位一致（maxdiff={ci.get('maxdiff')}）")
    zs = decoder.verify_lora_scale_zero()
    _check(bool(zs["bitwise"]), f"lora_scale_zero_() 后与 base 未逐位一致（maxdiff={zs['maxdiff']}）")
    _check(float(decoder.lora_scale) == 1.0, f"置零检查后包络未还原：{decoder.lora_scale}")
    eq = wiring["equivalence"]
    _check(bool(eq.get("bitwise_assemble")),
           f"μ 路径的仿射+组装与 ActionTerm 不一致（maxdiff={eq.get('maxdiff_assemble')}）")
    rep = decoder.adapter_param_report()
    _check(rep["n_tensors"] == 2 * LORA_DECODER_LAYERS,
           f"adapter 张量数 {rep['n_tensors']} != {2 * LORA_DECODER_LAYERS}（7 层 × down/up）")
    _check(rep["n_requires_grad"] == rep["n_tensors"], "adapter 参数应全部 requires_grad=True")
    _check(decoder.base_frozen(), "decoder base 未全部冻结（requires_grad 应为 False）")
    _check(len(decoder.base_linears) == LORA_DECODER_LAYERS, f"base 层数 {len(decoder.base_linears)} != 7")
    _check(all(r == cli.rank for r in decoder.layer_ranks), f"layer_ranks={decoder.layer_ranks} != {cli.rank}")
    _check(decoder.obs_dim == DECODER_OBS_DIM, f"decoder obs 维 {decoder.obs_dim} != {DECODER_OBS_DIM}")
    _check(not wiring["degenerate_layers"],
           f"rank={cli.rank} 触发全秩退化分支的层 {wiring['degenerate_layers']}（预注册 §3 不允许）")
    _check(float(wiring["default_pos_maxdiff"]) == 0.0,
           f"资产 default_joint_pos 与 SONIC 默认角不一致（maxdiff={wiring['default_pos_maxdiff']}）")
    print(
        f"[d065] smoke b2) LoRA PASS rank={decoder.rank} alpha={decoder.alpha} "
        f"adapter={rep['n_tensors']}t/{rep['n_elements']}e "
        f"construction_identity maxdiff={ci.get('maxdiff')} scale_zero maxdiff={zs['maxdiff']} "
        f"assemble maxdiff={eq.get('maxdiff_assemble')} base_md5={wiring['base_weight_md5']}",
        flush=True,
    )

    # (d) runner.learn 两步 + ckpt 落盘（先于 (c) 跑，避免多 env 建场污染主 env）
    wrapped = EnvWrapper(env)
    runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=str(out_dir), device=dev)
    # (d1) v2 关键断言：策略实例类型（防静默回落 ActorCritic ⇒ C 臂被训成 A 臂）
    policy = _assert_policy_class(runner, wiring["policy_class"])
    tr = policy.trainable_report()
    _check(int(policy.official_obs_dim) == layout["official_obs_dim"],
           f"policy.official_obs_dim={policy.official_obs_dim} != {layout['official_obs_dim']}")
    _check(int(policy.proprio_dim) == PROPRIO_DIM, f"policy.proprio_dim={policy.proprio_dim} != {PROPRIO_DIM}")
    print(
        f"[d065] smoke d1) policy={type(policy).__name__} official_obs_dim={policy.official_obs_dim} "
        f"proprio_dim={policy.proprio_dim} num_actions={policy.num_actions} "
        f"sigma={tr['sigma']['value']}(frozen={tr['sigma']['frozen']})",
        flush=True,
    )
    # (d2) 可训集合（正式训练同一条代码路径）
    opt_report = _wire_trainable(runner, cli, torch)
    print(
        f"[d065] smoke d2) trainable mode={opt_report['mode']} "
        f"tensors_in_opt={opt_report['n_param_tensors_in_optimizer']} "
        f"adapter_in_opt={opt_report['adapter_in_optimizer']}/{opt_report['adapter_tensors']} "
        f"sigma_in_opt={opt_report['sigma_in_optimizer']} "
        f"head_trainable={opt_report['head_requires_grad']} "
        f"critic_trainable={opt_report['critic_requires_grad']} "
        f"base_trainable={opt_report['base_requires_grad']} pruned={opt_report['prune']}",
        flush=True,
    )
    _check(opt_report["adapter_in_optimizer"] == rep["n_tensors"], "adapter 参数未全部进优化器")
    _check(opt_report["base_requires_grad"] == 0, "base 参数仍可训（应为 0）")
    _check(not opt_report["sigma_in_optimizer"], "σ 仍在优化器里（冻结参数摘除失效）")
    if cli.trainable == "adapter-only":
        _check(opt_report["head_requires_grad"] == 0, "adapter-only 下 token head 仍可训")
        _check(opt_report["critic_requires_grad"] > 0, "adapter-only 下 critic 被冻结（PPO 无梯度路径）")
        _check(bool(cli.init_head_from), "adapter-only 必须 --init-head-from")
    else:
        _check(opt_report["head_requires_grad"] > 0, "policy+adapter 下 token head 应可训")

    _wrap_runner_save(runner, lambda: _lora_payload(policy, cli, _git_head(), None, None), torch)
    runner.learn(num_learning_iterations=SMOKE_ITERS)
    ckpt_path = out_dir / "model_smoke.pt"
    runner.save(str(ckpt_path))
    _check(ckpt_path.exists() and ckpt_path.stat().st_size > 0, f"ckpt 未落盘: {ckpt_path}")
    # (d3) 身份信封落盘 + 训练后冻结断言 + **梯度可达**（v2 的核心可检查项）
    smoke_payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    _check(isinstance(smoke_payload, dict) and isinstance(smoke_payload.get("lora"), dict),
           "ckpt 内未注入 lora 信封（H6 注入路径失效）")
    _check(bool(smoke_payload["lora"]["envelope"].get("base_weight_md5")),
           "ckpt 内 LoRA 信封缺 base_weight_md5")
    base_diff = decoder.base_state_maxdiff(wiring["base_snapshot"])
    _check(base_diff == 0.0, f"learn 后 base 权重变了（maxdiff={base_diff:.3e}）")
    n_adapter = len(policy.mu_path.adapter_params())
    n_grad = sum(1 for p in policy.mu_path.adapter_params() if p.grad is not None)
    _check(n_grad == n_adapter,
           f"adapter 梯度不可达（{n_grad}/{n_adapter}）——v2 in-graph 接线失败（v1 架构下恒为 0）")
    print(_lora_diag_line(policy, tag=f"smoke step={SMOKE_ITERS}"), flush=True)
    print(
        f"[d065] smoke d) learn x{SMOKE_ITERS} PASS, ckpt -> {ckpt_path} "
        f"base_frozen_maxdiff={base_diff} adapter_grad={n_grad}/{n_adapter}",
        flush=True,
    )
    env.close()

    # (c) 逐地形 height_scan 断言：每地形起独立子进程（--smoke-terrain-check）。
    # 父进程内重建第二个 ManagerBasedRLEnv 是 Isaac Lab 已知雷区（主 env.close()
    # 后场景/terrain 重建静默挂起，cvgl 冒烟 r3 根因），try/except 兜不住 hang。
    terrain_results: dict[str, tuple[bool, str]] = _smoke_terrain_checks(cli, dev)

    # (e) 汇总行（盯守代理 grep 该行）
    bad_terrains = [t for t, (ok, _msg) in terrain_results.items() if not ok]
    _check(not bad_terrains, f"地形 height_scan 断言失败: {bad_terrains}")
    summary = _smoke_final_line(cli, obs_dim_policy, obs_dim_critic, action_dim, terrain_results, ckpt_path.name)
    print(f"SMOKE PASS: {summary}", flush=True)


def _smoke_final_line(cli, obs_dim_policy: int, obs_dim_critic: int, action_dim: int,
                      terrain_results: dict, ckpt_name: str) -> str:
    """SMOKE PASS 汇总行文本（纯函数，便于本机结构自测；terrain_results 值为 (ok, msg)）。"""
    return (
        f"arm={cli.arm} obs(policy)={obs_dim_policy} "
        f"obs(critic)={obs_dim_critic or 'none'} action={action_dim} "
        f"terrains=" + ",".join(f"{t}:{'PASS' if terrain_results[t][0] else 'FAIL'}" for t in SMOKE_TERRAINS)
        + f" learn={SMOKE_ITERS}it ckpt={ckpt_name}"
    )


def _terrain_check_argv(cli, terrain: str, device: str) -> list[str]:
    """父进程 cli 命名空间 -> 子进程 --smoke-terrain-check argv。

    透传 arm/terrain/seed/token-*（token 常量在 policy 侧）与 v2 的 LoRA/σ/head 初始化参数，
    num_envs 强制 8，headless 仅在父进程显式关闭时传 --no-headless（默认开不传）。
    """
    argv = [sys.executable, str(Path(__file__).resolve()), "--smoke-terrain-check", terrain]
    argv += ["--arm", cli.arm, "--terrain", cli.terrain, "--seed", str(cli.seed)]
    argv += ["--num-envs", str(SMOKE_NUM_ENVS), "--max-iterations", str(SMOKE_ITERS)]
    argv += ["--token-stats", cli.token_stats or ""]
    if cli.onnx_path:
        argv += ["--onnx-path", cli.onnx_path]
    if cli.token_alpha is not None:
        argv += ["--token-alpha", str(cli.token_alpha)]
    if cli.token_bound is not None:
        argv += ["--token-bound", cli.token_bound]
    # D065 v2：LoRA 超参 / 可训集合 / σ / head 初始化必须透传，否则子进程 argparse 会因
    # adapter-only 缺 --init-head-from 直接 exit 2，且 cfg 语义会与父进程不一致
    argv += ["--rank", str(cli.rank), "--alpha", str(cli.alpha), "--trainable", cli.trainable]
    if cli.sigma is not None:
        argv += ["--sigma", str(cli.sigma)]
    if cli.sigma_trainable:
        argv += ["--sigma-trainable"]
    if cli.init_head_from:
        argv += ["--init-head-from", cli.init_head_from]
    if bool(getattr(cli, "allow_partial_head", False)):
        argv += ["--allow-partial-head"]
    if cli.output_dir:
        argv += ["--output-dir", str(cli.output_dir)]
    # 不传 --device：训练 argparse 无此参数（主进程 device 来自 AppLauncher parser，
    # 子进程 worker 同样经 launcher_args 自取）；传了会 argparse exit 2（r4 实证）。
    if not cli.headless:
        argv.append("--no-headless")
    return argv


def _smoke_terrain_checks(cli, device: str) -> dict[str, tuple[bool, str]]:
    """(c) 段：每地形独立子进程跑 --smoke-terrain-check 并收集结果。

    - 子进程 stdout/stderr 由父进程 capture 后原样转发（选 capture-and-forward：
      tee 仍能看到完整 Isaac 日志与 TERRAIN_CHECK 行，同时父进程可解析结果行）。
    - 超时 kill（TimeoutExpired 先 proc.kill() 再 communicate 收尸）；
      超时 / 非零退出 / 无 TERRAIN_CHECK 行 -> 该地形 FAIL 并注明原因
      （timeout>480s / exit N / no-line）。
    - 环境变量默认继承（PYTHONPATH 随之），无任何定制 env。
    """
    results: dict[str, tuple[bool, str]] = {}
    for terrain in SMOKE_TERRAINS:
        argv = _terrain_check_argv(cli, terrain, device)
        out = err = ""
        timed_out = False
        try:
            # encoding 必须显式：det 容器 C locale 下 text=True 默认 ASCII 解码，
            # 子进程输出含 UTF-8 汉字即 UnicodeDecodeError 且吞掉转发（r5 实证）。
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
        except Exception as exc:  # noqa: BLE001 —— 子进程都起不来
            results[terrain] = (False, f"spawn-failed: {type(exc).__name__}: {exc}")
            print(f"[d065] smoke c) terrain {terrain}: FAIL — {results[terrain][1]}", flush=True)
            continue
        try:
            out, err = proc.communicate(timeout=TERRAIN_CHECK_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()  # 显式 kill，防悬挂子进程残留
            out, err = proc.communicate()
        # capture 后转发（父 tee 仍可见完整子进程输出）
        if err:
            sys.stderr.write(err if err.endswith("\n") else err + "\n")
            sys.stderr.flush()
        if out:
            sys.stdout.write(out if out.endswith("\n") else out + "\n")
            sys.stdout.flush()
        line = next((ln for ln in (out or "").splitlines() if ln.startswith("TERRAIN_CHECK ")), None)
        if timed_out:
            ok, msg = False, f"timeout>{TERRAIN_CHECK_TIMEOUT_S}s"
        elif line is None:
            ok, msg = False, f"no-line (exit {proc.returncode})"
        else:
            parts = line.split(maxsplit=3)  # ["TERRAIN_CHECK", terrain, PASS|FAIL, msg...]
            child_ok = len(parts) >= 3 and parts[2] == "PASS"
            child_msg = parts[3] if len(parts) == 4 else ""
            if proc.returncode == 0 and child_ok:
                ok, msg = True, child_msg
            elif proc.returncode != 0:
                ok, msg = False, f"exit {proc.returncode}: {child_msg}".rstrip(": ")
            else:
                ok, msg = False, child_msg or "child FAIL"
        results[terrain] = (ok, msg)
        print(f"[d065] smoke c) terrain {terrain}: {'PASS' if ok else 'FAIL'} — {msg}", flush=True)
    return results


def _terrain_check_main(cli, launcher_args) -> int:
    """--smoke-terrain-check 子命令体（独立子进程运行）。

    与 --smoke 相同的 AppLauncher/env 构建链（num_envs 已强制 8）：
    起 env -> reset + 3 步 zero action -> _height_scan_obs/_terrain_verdict 判定
    -> 打一行机器可读 "TERRAIN_CHECK <terrain> PASS|FAIL <msg>"（flush）。
    exit 0/1。复用 make_env_cfg 工厂与既有判定函数，判定逻辑不复制。
    """
    hv = _import_heavy()
    torch = hv.torch
    device = getattr(launcher_args, "device", "cuda:0")
    terrain = cli.smoke_terrain_check
    n = cli.num_envs
    print(
        f"[d065] terrain-check child: terrain={terrain} arm={cli.arm} num_envs={n} "
        f"device={device} pid={os.getpid()}",
        flush=True,
    )
    env_t = None
    ok, msg = False, ""
    try:
        cfg_t, _ = _build_env_cfg(hv.make_env_cfg, terrain, cli.arm, cli, device, suffix=terrain)
        env_t = hv.ManagerBasedRLEnv(cfg=cfg_t)
        obs_t, _ = env_t.reset()
        adim_t = int(env_t.action_manager.total_action_dim)
        for _ in range(3):
            obs_t, *_ = env_t.step(torch.zeros(n, adim_t, device=env_t.device))
        hs, how = _height_scan_obs(env_t, obs_t["policy"])
        ok, msg = _terrain_verdict(terrain, hs, how)
    except Exception as exc:  # noqa: BLE001 —— 任何失败都落到 TERRAIN_CHECK FAIL 行
        ok, msg = False, f"{type(exc).__name__}: {exc}"
    finally:
        if env_t is not None:
            try:
                env_t.close()
            except Exception:
                pass
    print(f"TERRAIN_CHECK {terrain} {'PASS' if ok else 'FAIL'} {msg}".rstrip(), flush=True)
    return 0 if ok else 1


def main() -> None:
    # CLI 解析放最前：-h/--help 在任何 isaaclab import 之前完成（--help 不依赖 isaaclab）
    cli = build_args().parse_args()
    if cli.resume and not cli.ckpt:
        build_args().error("--resume 需要 --ckpt <path>")
    if cli.smoke or cli.smoke_terrain_check:
        cli.num_envs = SMOKE_NUM_ENVS
        cli.max_iterations = SMOKE_ITERS

    out_dir = Path(cli.output_dir) if cli.output_dir else _default_out_dir(cli.arm, cli.terrain, cli.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # AppLauncher 启动链（照 train_apt_isaac.py:280-299：app 起来之前不 import isaaclab 栈）
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = cli.num_envs
    launcher_args.headless = cli.headless
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    exit_code = 0
    try:
        if cli.smoke_terrain_check:
            # 单地形检查子进程体：打 TERRAIN_CHECK 行后以 exit 0/1 结束
            exit_code = _terrain_check_main(cli, launcher_args)
        else:
            _run(cli, out_dir, launcher_args)
    except Exception as exc:
        if cli.smoke:
            print(f"SMOKE FAIL: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        simulation_app.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
