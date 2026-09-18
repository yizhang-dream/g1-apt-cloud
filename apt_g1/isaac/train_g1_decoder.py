"""D064 训练入口 + D064-G0 冒烟一体脚本。

Isaac Lab manager-based velocity 栈（同批 sonic_action_term.py /
g1_velocity_decoder_env.py 提供 env cfg 家族与 SONIC decoder action term）
+ rsl_rl OnPolicyRunner 训练入口。

用法（服务器 cvgl，IsaacLab 2.1.0 / rsl-rl-lib 2.3.3 / torch 2.5.1，仓根 cwd）:
    python apt_g1/isaac/train_g1_decoder.py --arm decoder --terrain rough \
        --num-envs 4096 --max-iterations 3000 --token-stats <g1-mode token 统计 npz>
冒烟（D064-G0）:
    python apt_g1/isaac/train_g1_decoder.py --smoke
    # num_envs 强制 8、iterations 强制 2；结束输出单行
    # "SMOKE PASS: ..." 或 "SMOKE FAIL: ..."（供盯守代理 grep）

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
- 环境配置一律经 sibling 的 make_env_cfg(terrain, action=..., token_stats=...,
  onnx_path=..., token_alpha=..., token_bound=...) 工厂获取（真实签名见
  g1_velocity_decoder_env.py:484-494；decoder 臂 --token-stats 必由 CLI 提供）；
  若签名不符（TypeError）依次回退 make_env_cfg(terrain, action=...)（token
  kwargs 后赋到 decoder term cfg）、make_env_cfg(terrain)（direct 臂此时就地
  替换为官方 JointPositionActionCfg，tmp/isaac_ref/velocity_env_cfg.py:112，
  打 [WARN]）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from importlib.metadata import version as _pkg_version
from pathlib import Path

# 仓根（本文件位于 <repo>/apt_g1/isaac/）
REPO_ROOT = Path(__file__).resolve().parents[2]

ARMS = ("decoder", "direct")
TERRAINS = ("plane", "rough", "stairs", "stones", "discrete")
SMOKE_TERRAINS = ("plane", "rough", "stairs", "stones", "discrete")
SMOKE_NUM_ENVS = 8
SMOKE_ITERS = 2
# decoder 臂 raw action = 64 维 token；direct 臂 = 29 关节目标
DECODER_RAW_ACTION_DIM = 64
DIRECT_ACTION_DIM = 29
# SONIC token 仿射输出（FSQ 预量化坐标，无硬界；token_bound="tanh" 时才限幅
# 在 mean±alpha*std）的量级上限：仅拦截 1e6 级的明显坏值，不做窄限幅
TOKEN_ABS_MAX = 50.0
# plane 上 height_scan≈0 的判据：policy 组 height_scan 带 Unoise(±0.1)
# （tmp/isaac_ref/velocity_env_cfg.py:134-139），均值绝对值阈值放宽到 0.15
PLANE_MEAN_ABS_TOL = 0.15

TASK_IDS = {
    "decoder": "Isaac-Velocity-Decoder-G1-D064-v0",
    "direct": "Isaac-Velocity-Direct-G1-D064-v0",
}


class _D064ArgumentParser(argparse.ArgumentParser):
    """带跨字段校验的 parser：decoder 臂缺 --token-stats 直接 argparse error。

    校验放在 parse_args 本体（而非 main）——任何调用方 parse_args 即得到
    SystemExit(2)，不依赖入口编排。
    """

    def parse_args(self, args=None, namespace=None):
        ns = super().parse_args(args, namespace)
        if ns.arm == "decoder" and not ns.token_stats:
            self.error(
                "--arm decoder 需要 --token-stats <npz>（官方 g1-mode token 统计 mean/std；"
                "token_stats 为空串时 SonicDecoderActionTerm.__init__ 直接抛 ValueError）"
            )
        return ns


def build_args() -> argparse.ArgumentParser:
    """CLI 定义（风格照 apt_g1/isaac/train_apt_isaac.py 的 build_args 段）。"""
    ap = _D064ArgumentParser(
        description=(
            "D064: Isaac Lab manager-based velocity 栈 + rsl_rl 训练入口"
            "（--smoke 同时是 D064-G0 冒烟验收）"
        )
    )
    ap.add_argument(
        "--arm",
        choices=ARMS,
        default="decoder",
        help="decoder=64 维 token→冻结 SONIC ONNX decoder→29 关节目标；direct=29 维直接关节位置目标",
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
        default=3000,
        help="PPO 迭代数（G1RoughPPORunnerCfg 默认 3000；--smoke 强制 2）",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="输出目录（默认 apt_g1/outputs/d064_<arm>_<terrain>_s<seed>，锚定仓根）",
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
        help="官方 g1-mode token 统计 npz（mean/std 键）；--arm decoder 必填"
        "（SonicDecoderActionTermCfg.token_stats 为空在 term init 即抛 ValueError）",
    )
    ap.add_argument(
        "--onnx-path",
        default="",
        help="SONIC ONNX decoder 路径（decoder 臂；缺省用 make_env_cfg 工厂默认值）",
    )
    ap.add_argument(
        "--token-alpha",
        type=float,
        default=None,
        help="token 仿射系数（decoder 臂；缺省跟 make_env_cfg 工厂默认 1.0）",
    )
    ap.add_argument(
        "--token-bound",
        choices=("none", "tanh"),
        default=None,
        help="token 仿射限幅（decoder 臂；缺省跟 make_env_cfg 工厂默认 tanh）",
    )
    return ap


def _default_out_dir(arm: str, terrain: str, seed: int) -> Path:
    return REPO_ROOT / "apt_g1" / "outputs" / f"d064_{arm}_{terrain}_s{seed}"


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


def _token_factory_kwargs(cli) -> dict:
    """token 相关 CLI -> make_env_cfg 工厂 kwargs（未指定的键不传，跟工厂默认值）。"""
    kwargs: dict = {}
    if cli.token_stats:
        kwargs["token_stats"] = cli.token_stats
    if cli.onnx_path:
        kwargs["onnx_path"] = cli.onnx_path
    if cli.token_alpha is not None:
        kwargs["token_alpha"] = cli.token_alpha
    if cli.token_bound is not None:
        kwargs["token_bound"] = cli.token_bound
    return kwargs


def _apply_token_kwargs(action_cfg, cli) -> None:
    """回退路径：把 token CLI 逐键后赋到 SonicDecoderActionTermCfg 实例上。"""
    for key, val in _token_factory_kwargs(cli).items():
        setattr(action_cfg, key, val)


def _build_env_cfg(make_env_cfg, mdp, sonic_action_term_cfg_cls, terrain: str, arm: str, cli, device: str, suffix: str):
    """经 sibling 工厂构建 env cfg 并做训练侧必需的运行期改写。

    主路径对齐真实签名（g1_velocity_decoder_env.py:484-494）：
    make_env_cfg(terrain, num_envs, action, play, onnx_path, token_stats,
    token_alpha, token_bound, seed)。
    改写点（全部守卫 + 打点）：num_envs / seed / sim.device / 每实例唯一地形
    prim path（顺序多 env 冒烟防 /World/ground 冲突，height_scanner 的
    mesh_prim_paths 同步改，路径事实 velocity_env_cfg.py:45,72）。
    """
    token_kwargs = _token_factory_kwargs(cli) if arm == "decoder" else {}
    try:
        cfg = make_env_cfg(terrain, action=arm, **token_kwargs)
        how = "make_env_cfg(terrain, action=...)"
        if token_kwargs:
            how += "+token kwargs"
    except TypeError:
        try:
            cfg = make_env_cfg(terrain, action=arm)
            how = "make_env_cfg(terrain, action=...)"
        except TypeError:
            cfg = make_env_cfg(terrain)
            how = "make_env_cfg(terrain)"
            if arm == "direct":
                # 官方 ActionsCfg 写法（velocity_env_cfg.py:112）就地替换，
                # 并清掉 sibling 可能放进去的 decoder term
                for key in list(vars(cfg.actions).keys()):
                    setattr(cfg.actions, key, None)
                cfg.actions.joint_pos = mdp.JointPositionActionCfg(
                    asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
                )
                how += "+direct 臂就地替换为 JointPositionActionCfg"
                print(f"[WARN] make_env_cfg 不接受 action 参数：{how}", flush=True)
        # 回退签名缺 token kwargs：后赋到 decoder term cfg
        # （不补的话 SonicDecoderActionTerm.__init__ 会因 token_stats 空串抛 ValueError）
        applied = False
        for obj in vars(cfg.actions).values():
            if isinstance(obj, sonic_action_term_cfg_cls):
                _apply_token_kwargs(obj, cli)
                applied = True
                how += "+token kwargs 后赋"
                break
        if arm == "decoder" and token_kwargs and not applied:
            print(
                "[WARN] 回退路径未见 SonicDecoderActionTermCfg：token kwargs 未生效"
                "（decoder term init 会因 token_stats 为空报错）",
                flush=True,
            )

    cfg.scene.num_envs = cli.num_envs
    if hasattr(cfg, "seed"):
        cfg.seed = cli.seed
    if hasattr(cfg.sim, "device"):
        cfg.sim.device = device

    terrain_cfg = getattr(cfg.scene, "terrain", None)
    if terrain_cfg is not None and suffix != "main":
        old_path = getattr(terrain_cfg, "prim_path", None)
        new_path = f"/World/ground_d064_{suffix}"
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
        if arm == "decoder":
            entry_kwargs.update(_token_factory_kwargs(cli))
        try:
            gym.register(
                id=TASK_IDS[arm],
                entry_point="isaaclab.envs:ManagerBasedRLEnv",  # g1_init.py:16 事实
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": functools.partial(make_env_cfg, **entry_kwargs),
                },
            )
            print(f"[d064] gym.register: {TASK_IDS[arm]} (action={arm})", flush=True)
        except gym.error.Error:
            print(f"[WARN] 任务 {TASK_IDS[arm]} 已注册，跳过（防重复）", flush=True)


def _find_decoder_term(env, sonic_action_term_cls):
    """按 isinstance 找 SONIC decoder action term（不依赖 sibling 的 cfg 字段命名）。"""
    am = env.action_manager
    for name in list(am.active_terms):
        try:
            term = am.get_term(name)
        except Exception:
            continue
        if isinstance(term, sonic_action_term_cls):
            return name, term
    return None, None


def _discover_decoder_onnx(term) -> str | None:
    """从 action term 的 cfg 里发现 *.onnx 路径（用于 run.json md5 信封）。"""
    cfg = getattr(term, "cfg", None)
    if cfg is None:
        return None
    try:
        for value in vars(cfg).values():
            if isinstance(value, (str, Path)) and str(value).endswith(".onnx"):
                return str(value)
    except Exception:
        pass
    return None


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

    优先走 ObservationManager 的逐 term 缓存；否则用 group_obs_term_dim 的
    维度按“height_scan 是 policy 组最后一个 term”切片（velocity_env_cfg.py
    中 height_scan 确为 policy 组末位，:134-139；sibling 若改顺序则该回退失真，
    打印取法说明供判读）。都取不到返回 None。
    """
    import torch

    om = env.observation_manager
    cache = getattr(om, "_group_obs_term_obses", None)
    if isinstance(cache, dict):
        grp = cache.get("policy")
        if isinstance(grp, dict) and "height_scan" in grp:
            try:
                return grp["height_scan"], "per-term-cache"
            except Exception:
                pass
    try:
        term_dims = om.group_obs_term_dim["policy"]
        hs_dim_raw = term_dims.get("height_scan")
        if hs_dim_raw is None:
            return None, "term-absent"
        hs_dim = int(hs_dim_raw) if isinstance(hs_dim_raw, int) else int(torch.tensor(list(hs_dim_raw)).prod())
        return policy_obs[:, -hs_dim:], "last-slice"
    except Exception:
        return None, "unavailable"


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


def _write_run_json(out_dir: Path, cli, git_commit: str, onnx_path: str | None, onnx_md5, env_info: dict, versions: dict) -> None:
    """身份信封（最小实现，照 train_apt_isaac.py 的 ckpt 身份做法落 run.json）。"""
    payload = {
        "entry": "apt_g1/isaac/train_g1_decoder.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "format": 1,
        "git_commit": git_commit,
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cli).items()},
        "decoder_onnx": {
            "path": onnx_path,
            "md5": onnx_md5 if onnx_md5 else "absent",
        },
        "env_cfg": env_info,
        "versions": versions,
    }
    path = out_dir / "run.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[d064] run.json -> {path}", flush=True)


def _run(cli, out_dir: Path, launcher_args) -> None:
    """App 起来之后的全部逻辑（smoke 与正式训练共用）。"""
    import numpy as np
    import torch

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    # ---- isaaclab / rsl_rl / sibling imports（依据清单见模块 docstring 与汇报）----
    import gymnasium as gym
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

    import sys

    if str(REPO_ROOT) not in sys.path:
        # 以 `python apt_g1/isaac/train_g1_decoder.py` 直跑时 sys.path[0] 是脚本
        # 目录而非仓根，补上保证 apt_g1.* 可导入（服务器 PYTHONPATH 亦含仓根）
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from apt_g1.isaac.ckpt_identity import file_md5
        from apt_g1.isaac.g1_velocity_decoder_env import make_env_cfg
        from apt_g1.isaac.sonic_action_term import SonicDecoderActionTerm, SonicDecoderActionTermCfg
    except ImportError as exc:
        raise RuntimeError(
            "sibling 模块缺失（apt_g1/isaac/sonic_action_term.py / "
            "g1_velocity_decoder_env.py 应由 D064 同批产出）；"
            f"原始错误: {exc}"
        ) from exc

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
    cfg, cfg_how = _build_env_cfg(
        make_env_cfg, mdp, SonicDecoderActionTermCfg, cli.terrain, cli.arm, cli, device, suffix="main"
    )
    print(f"[d064] env cfg via {cfg_how}", flush=True)

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
        print(f"[d064] asymmetric(python rsl_rl={rsl_ver}): critic 组存在={has_critic_group}", flush=True)

    env = ManagerBasedRLEnv(cfg=cfg)
    obs, _ = env.reset()
    if not (isinstance(obs, dict) and "policy" in obs):
        raise RuntimeError(f"env obs 应为含 'policy' 组的 dict，实际 {type(obs)} keys={list(obs) if isinstance(obs, dict) else '-'}")
    policy_obs = obs["policy"]
    critic_obs = obs.get("critic") if isinstance(obs, dict) else None

    obs_dim_policy = int(policy_obs.shape[-1])
    obs_dim_critic = int(critic_obs.shape[-1]) if critic_obs is not None else 0
    action_dim = int(env.action_manager.total_action_dim)
    expected_action = DECODER_RAW_ACTION_DIM if cli.arm == "decoder" else DIRECT_ACTION_DIM
    print(
        f"[d064] obs(policy)={obs_dim_policy} obs(critic)={obs_dim_critic or '-'} "
        f"action={action_dim}（期望 {expected_action}）",
        flush=True,
    )

    # decoder onnx 身份（从 term cfg 发现路径）
    term_name, term = _find_decoder_term(env, SonicDecoderActionTerm)
    onnx_path = _discover_decoder_onnx(term) if term is not None else None
    onnx_md5 = file_md5(onnx_path) if onnx_path else None
    if cli.arm == "decoder":
        print(f"[d064] decoder term={term_name!r} onnx={onnx_path!r} md5={onnx_md5 or 'absent'}", flush=True)

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
            "has_critic_group": has_critic_group,
            "obs_groups": sorted(obs.keys()),
            "decimation": getattr(cfg, "decimation", None),
            "episode_length_s": getattr(cfg, "episode_length_s", None),
            "sim_dt": getattr(cfg.sim, "dt", None),
            "sim_device": device,
            "decoder_term_name": term_name,
        },
        {
            "isaaclab": _pkg_ver("isaaclab"),
            "isaacsim": _pkg_ver("isaacsim"),
            "rsl_rl": _pkg_ver("rsl-rl-lib") if _pkg_ver("rsl-rl-lib") != "unknown" else _pkg_ver("rsl-rl"),
            "torch": torch.__version__,
        },
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
            term=term,
            has_critic_group=has_critic_group,
            runner_cfg=runner_cfg,
            EnvWrapper=EnvWrapper,
            OnPolicyRunner=OnPolicyRunner,
            ManagerBasedRLEnv=ManagerBasedRLEnv,
            make_env_cfg=make_env_cfg,
            mdp=mdp,
            sonic_action_term_cfg_cls=SonicDecoderActionTermCfg,
            torch=torch,
        )
        return

    # ---- 正式训练 ----
    wrapped = EnvWrapper(env)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=str(out_dir), device=runner_cfg.device)
    if cli.resume:
        if not cli.ckpt:
            raise RuntimeError("--resume 需要 --ckpt <path>")
        runner.load(str(Path(cli.ckpt).resolve()))
        print(f"[d064] resumed from {cli.ckpt}", flush=True)
    runner.learn(num_learning_iterations=runner_cfg.max_iterations)
    final_ckpt = out_dir / "model_final.pt"
    runner.save(str(final_ckpt))
    print(f"[d064] done. final ckpt -> {final_ckpt}", flush=True)
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
    term,
    has_critic_group: bool,
    runner_cfg,
    EnvWrapper,
    OnPolicyRunner,
    ManagerBasedRLEnv,
    make_env_cfg,
    mdp,
    sonic_action_term_cfg_cls,
    torch,
) -> None:
    """D064-G0 冒烟（--smoke 时 num_envs=8 / iters=2 已在 main 强制）。"""
    n = cli.num_envs
    _check(n == SMOKE_NUM_ENVS, f"smoke num_envs 应为 {SMOKE_NUM_ENVS}，实际 {n}")
    _check(cli.max_iterations == SMOKE_ITERS, f"smoke iterations 应为 {SMOKE_ITERS}")

    # (a) obs 形状断言
    _check(policy_obs.ndim == 2, f"policy obs 应 2 维，实际 shape={tuple(policy_obs.shape)}")
    _check(policy_obs.shape[0] == n, f"policy obs batch={policy_obs.shape[0]} != num_envs={n}")
    _check(obs_dim_policy > 0, "policy obs 维度应 > 0")
    dim_from_mgr = _group_total_dim(env.observation_manager.group_obs_dim.get("policy"))
    if dim_from_mgr > 0:
        _check(dim_from_mgr == obs_dim_policy, f"policy obs 维度 {obs_dim_policy} != manager 报告 {dim_from_mgr}")
    if has_critic_group:
        _check(critic_obs is not None, "cfg 声明有 critic 组但 env 未返回 critic obs")
        _check(critic_obs.ndim == 2 and critic_obs.shape[0] == n and int(critic_obs.shape[-1]) > 0,
               f"critic obs 形状异常 shape={tuple(critic_obs.shape)}")
    else:
        print("[WARN] 单 obs 组模式（env cfg 无 critic 组或已退化）", flush=True)
    print(f"[d064] smoke a) obs shapes PASS policy={obs_dim_policy} critic={obs_dim_critic or '-'}", flush=True)

    # (b) decoder 臂：raw action 64 维 / q_des 非零有限 / token 仿射范围
    if cli.arm == "decoder":
        _check(action_dim == DECODER_RAW_ACTION_DIM, f"decoder 臂 action 维度应 {DECODER_RAW_ACTION_DIM}，实际 {action_dim}")
        _check(term is not None, "decoder 臂下未找到 SonicDecoderActionTerm 实例")
        env.step(torch.zeros(n, action_dim, device=env.device))
        env.step(0.1 * torch.randn(n, action_dim, device=env.device))
        raw = getattr(term, "raw_actions", None)
        _check(raw is not None, "decoder term 无 raw_actions 属性")
        _check(tuple(raw.shape) == (n, DECODER_RAW_ACTION_DIM), f"raw action 形状 {tuple(raw.shape)} != {(n, DECODER_RAW_ACTION_DIM)}")
        _check(bool(torch.isfinite(raw).all()), "raw action 含非有限值")
        q_des = getattr(term, "processed_actions", None)
        _check(q_des is not None, "decoder term 无 processed_actions(q_des) 属性")
        _check(tuple(q_des.shape) == (n, DIRECT_ACTION_DIM), f"q_des 形状 {tuple(q_des.shape)} != {(n, DIRECT_ACTION_DIM)}")
        _check(bool(torch.isfinite(q_des).all()), "q_des 含非有限值")
        _check(float(q_des.abs().max()) > 1e-6, "q_des 全零（零输入下 decoder 输出应非零）")
        # token 仿射输出：主路径 = term.core.map_token(raw)（sonic_action_term.py
        # 的 SonicActionCore.map_token，token 不在 term 上缓存）；回退 = term 上
        # 名含 token 的 (n,64) 张量属性
        tokens, token_via = None, ""
        core = getattr(term, "core", None)
        if core is not None and hasattr(core, "map_token"):
            tokens, token_via = core.map_token(raw), "core.map_token()"
        else:
            for attr_name, attr_val in vars(term).items():
                if "token" in attr_name.lower() and isinstance(attr_val, torch.Tensor) \
                        and attr_val.ndim == 2 and attr_val.shape[0] == n:
                    tokens, token_via = attr_val, f"term.{attr_name}"
                    break
        if tokens is not None:
            token_max = float(tokens.abs().max())
            _check(bool(torch.isfinite(tokens).all()), "token 仿射输出含非有限值")
            _check(token_max <= TOKEN_ABS_MAX, f"token 仿射输出量级异常: max|.|={token_max:.4f} > {TOKEN_ABS_MAX}")
            print(
                f"[d064] smoke b) decoder PASS raw={tuple(raw.shape)} "
                f"q_des max|.|={float(q_des.abs().max()):.4f} "
                f"token max|.|={token_max:.4f} via {token_via}",
                flush=True,
            )
        else:
            print(
                "[WARN] token 仿射输出不可取（term.core.map_token 与 token 缓存属性均无）——未验证",
                flush=True,
            )
    else:
        _check(action_dim == DIRECT_ACTION_DIM, f"direct 臂 action 维度应 {DIRECT_ACTION_DIM}，实际 {action_dim}")
        print("[d064] smoke b) direct 臂：decoder 检查不适用（跳过）", flush=True)

    # (d) runner.learn 两步 + ckpt 落盘（先于 (c) 跑，避免多 env 建场污染主 env）
    wrapped = EnvWrapper(env)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=str(out_dir), device=runner_cfg.device)
    runner.learn(num_learning_iterations=SMOKE_ITERS)
    ckpt_path = out_dir / "model_smoke.pt"
    runner.save(str(ckpt_path))
    _check(ckpt_path.exists() and ckpt_path.stat().st_size > 0, f"ckpt 未落盘: {ckpt_path}")
    print(f"[d064] smoke d) learn x{SMOKE_ITERS} PASS, ckpt -> {ckpt_path}", flush=True)
    env.close()

    # (c) 逐地形 height_scan 断言（每地形独立起 8-env env）
    terrain_results: dict[str, bool] = {}
    for terrain in SMOKE_TERRAINS:
        ok, msg = False, ""
        try:
            cfg_t, _ = _build_env_cfg(
                make_env_cfg, mdp, sonic_action_term_cfg_cls, terrain, cli.arm, cli, device, suffix=terrain
            )
            env_t = ManagerBasedRLEnv(cfg=cfg_t)
            obs_t, _ = env_t.reset()
            adim_t = int(env_t.action_manager.total_action_dim)
            for _ in range(3):
                obs_t, *_ = env_t.step(torch.zeros(n, adim_t, device=env_t.device))
            hs, how = _height_scan_obs(env_t, obs_t["policy"])
            ok, msg = _terrain_verdict(terrain, hs, how)
            env_t.close()
        except Exception as exc:  # noqa: BLE001 —— 单地形失败不吞掉整体冒烟
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        terrain_results[terrain] = ok
        print(f"[d064] smoke c) terrain {terrain}: {'PASS' if ok else 'FAIL'} — {msg}", flush=True)

    # (e) 汇总行（盯守代理 grep 该行）
    bad_terrains = [t for t, ok in terrain_results.items() if not ok]
    _check(not bad_terrains, f"地形 height_scan 断言失败: {bad_terrains}")
    summary = (
        f"arm={cli.arm} obs(policy)={obs_dim_policy} "
        f"obs(critic)={obs_dim_critic or 'none'} action={action_dim} "
        f"terrains=" + ",".join(f"{t}:{'PASS' if terrain_results[t] else 'FAIL'}" for t in SMOKE_TERRAINS)
        + f" learn={SMOKE_ITERS}it ckpt={ckpt_path.name}"
    )
    print(f"SMOKE PASS: {summary}", flush=True)


def main() -> None:
    # CLI 解析放最前：-h/--help 在任何 isaaclab import 之前完成（--help 不依赖 isaaclab）
    cli = build_args().parse_args()
    if cli.resume and not cli.ckpt:
        build_args().error("--resume 需要 --ckpt <path>")
    if cli.smoke:
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

    try:
        _run(cli, out_dir, launcher_args)
    except Exception as exc:
        if cli.smoke:
            print(f"SMOKE FAIL: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
