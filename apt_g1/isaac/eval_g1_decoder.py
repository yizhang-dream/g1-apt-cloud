"""D064 评测电池：双臂归因重跑的判读仪表（预注册=DS_CONTINUOUS_EXECUTION_PLAN §5u）。

三节电池（每条件独立子进程——同进程重建 env 会悬挂，train_g1_decoder.py r3 实证）：
  rough_levels : 官方 rough 课程地形按固定行出生（terrain_levels 直写），指标=存活/摔步/净位移；
                 「课程 terrain level 能力」以固定行扫描近似（训练日志未落 terrain_levels，评测期
                 出生行扫描是诚实工具）。
  rough_paper  : E 系连续性口径——terrain_cfg.rough_paper{noise} 固定噪声档（0.06/0.08），
                 直答「E 系 0.08 悬崖在新配方+大规模下是否仍存在」。
  flat         : 平地固定 vx 命令 {0.4,0.6}，指标=存活/vx RMSE（对命令跟踪，E 系 60s 口径）。

判据口径冻结（不得改）：ep_steps=3000（60s@50Hz）；episodes=每条件批次数×num_envs；
survived=整段未 terminated（truncated 到时不算摔）；vx_rmse=body 前向速度对命令的 RMSE
（仅 flat，摔步/到时步本身不计入样本）。策略=det 推理（actor 均值动作，无采样）。

命令钉死回填（2026-09-22，从 D065 fork `eval_g1_decoder_lora.py` 同构回并，只修测量链、
**不改任何评测口径默认值**）：cfg 侧 resampling_time_range=(1e9,1e9) 只挡 step 内周期重采样，
**reset 路径**（上游 command_manager.reset() 无条件 _resample）仍会把命令覆写成 U(0,1)，
heading_command=True 每步覆写 wz、rel_standing_envs=0.02 清零行亦属同源污染 ⇒ flat 条件
**每控制步重戳** `vel_command_b`；diag 改**逐步采样**（`cmd_vx_*_over_steps` + 策略实际消费
obs 槽 `obs_cmd_vx_*`）汇总进条件结果 JSON，旧循环外读法 `cmd_vx_mean_last_batch` 保留为
legacy 对照；子进程入口播 RNG 种子（`_seed_everything`）保可复现。

用法（cvgl）：
  python apt_g1/isaac/eval_g1_decoder.py --arm decoder \
      --ckpt .../d064_g1_decoder_rough_s0/model_final.pt \
      --token-stats $G/apt_g1/outputs/e49/token_stats_e49.npz \
      --onnx-path $G/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx

运行环境：IsaacLab 2.1.0 / rsl-rl-lib 2.3.3 / torch 2.5.1（本机无 isaaclab，仅 py_compile 验证）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_LEVELS = "1,3,5,7,9"
DEFAULT_NOISE = "0.06,0.08"
DEFAULT_VX = "0.4,0.6"
EVAL_NUM_ENVS = 8
COND_TIMEOUT_S = 1500  # 单条件子进程上限：env 启动 ~1min + 批次 rollout，宽裕
# 命令通道诊断容差（只作用于 diag 判读，不改动任何评测口径默认值；与 D065 fork 同名同值）
CMD_NOMINAL_TOL = 1e-6
NEAR_STATIC_VX = 0.05


def _parse_csv_float(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_int(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _file_md5(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


class _EvalArgumentParser(argparse.ArgumentParser):
    """跨字段守卫放 parse_args 本体：decoder 臂缺 token 资产即 SystemExit(2)（train 同款纪律）。"""

    def parse_args(self, args=None, namespace=None):
        cli = super().parse_args(args, namespace)
        if cli.arm == "decoder" and not cli.token_stats:
            self.error("--arm decoder 需要 --token-stats <npz>（token 仿射统计）")
        if not cli.ckpt:
            self.error("--ckpt <path> 必填（待评 ckpt）")
        return cli


def build_args() -> argparse.ArgumentParser:
    ap = _EvalArgumentParser(description="D064 评测电池（双臂判读仪表；判据=§5u 预注册，不得改）")
    ap.add_argument("--ckpt", default="", help="待评 ckpt（rsl_rl model_*.pt，含 model_state_dict）")
    ap.add_argument("--arm", choices=("decoder", "direct"), required=True)
    ap.add_argument(
        "--battery",
        choices=("rough_levels", "rough_paper", "flat", "all"),
        default="all",
        help="跑哪节电池（all=三节全跑）",
    )
    ap.add_argument("--levels", default=DEFAULT_LEVELS, help="rough_levels 出生行 csv（默认 1,3,5,7,9）")
    ap.add_argument("--noise", default=DEFAULT_NOISE, help="rough_paper 噪声档 csv（默认 0.06,0.08）")
    ap.add_argument("--vx-cmds", default=DEFAULT_VX, help="flat 固定 vx 命令 csv（默认 0.4,0.6）")
    ap.add_argument("--episodes", type=int, default=3, help="每条件批次数（每批 num_envs 局）")
    ap.add_argument("--ep-steps", type=int, default=3000, help="每局步数=60s@50Hz（判据冻结）")
    ap.add_argument("--num-envs", type=int, default=EVAL_NUM_ENVS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="", help="JSON 输出目录（默认 ckpt 同目录）")
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--token-stats", default="", help="decoder 臂必填（守卫见 _EvalArgumentParser）")
    ap.add_argument("--onnx-path", default="", help="SONIC ONNX decoder（decoder 臂；缺省跟工厂默认）")
    ap.add_argument("--token-alpha", type=float, default=None)
    ap.add_argument("--token-bound", choices=("none", "tanh"), default=None)
    ap.add_argument(
        "--eval-one",
        default=None,
        metavar="COND_JSON",
        help=argparse.SUPPRESS,  # 内部子命令：父进程按条件逐个 spawn，输出 EVAL_COND 行
    )
    return ap


def expand_conditions(cli) -> list[dict]:
    """按 battery 展开条件列表（顺序稳定，父进程逐个派子进程）。"""
    conds: list[dict] = []
    if cli.battery in ("rough_levels", "all"):
        for lv in _parse_csv_int(cli.levels):
            conds.append({"kind": "rough_levels", "level": lv})
    if cli.battery in ("rough_paper", "all"):
        for nz in _parse_csv_float(cli.noise):
            conds.append({"kind": "rough_paper", "noise": nz})
    if cli.battery in ("flat", "all"):
        for vx in _parse_csv_float(cli.vx_cmds):
            conds.append({"kind": "flat", "vx": vx})
    return conds


def _child_argv(cli, cond: dict) -> list[str]:
    """从父 cli 重建子进程 argv（num-envs/episodes/ep-steps 透传；--eval-one 注入条件）。"""
    argv = [sys.executable, os.path.abspath(__file__)]
    argv += ["--arm", cli.arm, "--ckpt", cli.ckpt, "--battery", cli.battery]
    argv += ["--levels", cli.levels, "--noise", cli.noise, "--vx-cmds", cli.vx_cmds]
    argv += ["--episodes", str(cli.episodes), "--ep-steps", str(cli.ep_steps)]
    argv += ["--num-envs", str(cli.num_envs), "--seed", str(cli.seed)]
    argv += ["--token-stats", cli.token_stats or ""]
    if cli.onnx_path:
        argv += ["--onnx-path", cli.onnx_path]
    if cli.token_alpha is not None:
        argv += ["--token-alpha", str(cli.token_alpha)]
    if cli.token_bound is not None:
        argv += ["--token-bound", cli.token_bound]
    if not cli.headless:
        argv.append("--no-headless")
    argv += ["--eval-one", json.dumps(cond, ensure_ascii=False)]
    return argv


def _load_actor_weights(hv, ckpt_path: str, obs_dim: int, action_dim: int, device):
    """rsl_rl ckpt → ActorCritic（仅 actor 权重；empirical_normalization=False 无归一化器）。"""
    import torch  # noqa: WPS433 —— 子进程内 torch（App 起来后）
    from rsl_rl.modules import ActorCritic

    ac = ActorCritic(
        num_actor_obs=obs_dim,  # rsl-rl-lib 2.3.3 签名= num_actor_obs（镜像实装源码核实；新版的 num_obs 会被 TypeError）
        num_critic_obs=obs_dim,  # 评测只用 actor；critic 形状不影响 det 推理
        num_actions=action_dim,
        actor_hidden_dims=[512, 256, 128],  # = train 侧 RslRlPpoActorCriticCfg（g1_agents_ppo.py:18-23）
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    sd = payload.get("model_state_dict", payload)
    actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    if not actor_sd:
        raise RuntimeError(f"ckpt 无 actor.* 键：{ckpt_path}（keys 示例 {list(sd)[:5]}）")
    missing = ac.actor.load_state_dict(actor_sd, strict=True)
    ac.eval()
    return ac


def _seed_everything(seed: int, hv=None) -> None:
    """子进程入口播全局 RNG 种子（torch + numpy + random）——条件可复现。

    条件子进程此前只把 seed 传给 env 配置，torch/numpy/random 全局 RNG 未播种：
    resampling / 噪声 / 复位等随机量在不同子进程间无固定重放。此处入口补齐（与
    `eval_author_v0_decoder.seed_everything` 同款）；在 env 构造与 reset 之前调用。
    """
    import random  # noqa: WPS433

    random.seed(seed)
    try:
        import numpy as np  # noqa: WPS433

        np.random.seed(seed)
    except ImportError:  # numpy 属重型栈，理论必在；缺失不阻断评测
        pass
    torch = getattr(hv, "torch", None)
    if torch is None:  # 主进程 early 路径理论上不调；保底自行 import
        import torch  # noqa: WPS433
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stamp_command(env, vx: float):
    """把命令钉死为 (vx, 0, 0)，返回本步命令的 vx 列（供逐步采样）。

    命令钉死第一步（cfg 侧）把 resampling_time_range 拉至 1e9，但那只挡 **step 内的周期
    重采样**；**reset 路径无条件 `_resample`**（上游 command_manager.reset() 每让一个 env
    复位就重采样一条 U(0,1) 命令，D064 flat 实测 cmd_vx_mean≈0.5 即此）——故批次内必须
    **每个控制步重戳**（D065 fork 已验证的黄金模式；通道正常时为幂等 no-op，被覆写时立刻
    纠正）。重戳写回整条 (vx,0,0) 同时对冲同源越权：rel_standing_envs=0.02 的 standing
    行清零、heading_command=True 每步对 wz 的覆写。
    """
    import torch  # noqa: WPS433

    term = env.command_manager.get_term("base_velocity")
    n = term.vel_command_b.shape[0]  # 冒烟核对：UniformVelocityCommand.vel_command_b（2.1.0）
    term.vel_command_b[:] = torch.tensor([vx, 0.0, 0.0], device=term.vel_command_b.device).repeat(n, 1)
    return term.vel_command_b[:, 0].clone()


def _policy_obs_cmd_slot(env, torch) -> dict:
    """定位 policy obs 组里 `velocity_commands` 的切片（命令通道核验的采样槽）。

    与 D065 fork `eval_g1_decoder_lora._policy_obs_cmd_slot` 同款（静态 + 运行期双保险）：
    - **静态**：policy 组 = `velocity_env_cfg.ObservationsCfg.PolicyCfg` 的 term 序
      （base_lin_vel, base_ang_vel, projected_gravity, **velocity_commands**,
      joint_pos, joint_vel, actions, height_scan）⇒ velocity_commands 预期 index=3、
      start=9、width=3。
    - **运行期**：term 序 = `observation_manager.active_terms[group]` 顺序（拼接序），逐 term
      维度乘积累加得起点——不硬编码索引；term 序若漂移，此处随之漂移并显式报错而非静默错位。

    采样链独立性：该槽取值 = obs 项 `mdp.generated_commands` 读命令项的 `command`，与
    `_stamp_command` 写的是同一个 `vel_command_b` ⇒ 采样验证的是「写入 → 命令项 → obs
    计算 → 策略输入」整段管线（含 resampling 覆写、obs 槽位错位、噪声项）；真正独立于命令
    通道的证据是 `actual_vx_*`（实测机体速度）。
    """
    om = env.observation_manager
    names = [str(n) for n in om.active_terms["policy"]]
    dims = list(om.group_obs_term_dim["policy"])
    if len(names) != len(dims):
        raise RuntimeError(f"policy 组 names/dims 错位：{len(names)} vs {len(dims)}")
    if "velocity_commands" not in names:
        raise RuntimeError(
            f"policy 组无 velocity_commands 项（现有 {names}）——命令通道核验无处采样，拒绝出数"
        )
    idx = names.index("velocity_commands")
    start = 0
    for term_dims in dims[:idx]:
        start += int(torch.tensor(list(term_dims)).prod())
    width = int(torch.tensor(list(dims[idx])).prod())
    if width < 1:
        raise RuntimeError(f"velocity_commands 槽宽 {width} 非法（dims={dims[idx]}）")
    return {"term": "velocity_commands", "index": int(idx), "start": int(start),
            "width": int(width), "policy_terms": names}


def _obs_cmd_vx(obs, slot):
    """从 policy obs 里切出命令槽的 vx 列（(N,)）——判据采样口径（不是刚写进去的值）。"""
    return obs[:, slot["start"]: slot["start"] + slot["width"]][:, 0]


def _run_batches(env, ac, cli, cond, hv) -> dict:
    """批次 rollout：每 env 记 首次 terminated 步/净位移/vx RMSE（首次 done 后不计）。

    D064 命令钉死缺陷回填：flat 条件**每控制步重戳**命令，核验采样走**策略实际消费的
    obs 命令槽**（`_policy_obs_cmd_slot` + `_obs_cmd_vx`；不是刚写进去的值），diag 增
    per-step `cmd_vx_*` 与 `obs_cmd_vx_*`。
    """
    torch = hv.torch
    device = env.device
    n = cli.num_envs
    episodes: list[dict] = []
    cmd_hist: list[float] = []  # 每控制步「重戳后」命令项 vx 均值（写通道自检）
    obs_cmd_hist: list[float] = []  # 策略实际消费的 obs 命令槽 vx 采样（判据口径）
    obs_cmd_step0: list[float] = []  # 第 0 步 obs 出自 env.reset()（重采样先于戳写、obs 未重算）
    slot = _policy_obs_cmd_slot(env, torch) if cond["kind"] == "flat" else None
    for _batch in range(cli.episodes):
        obs, _ = env.reset()
        if cond["kind"] == "flat":
            _stamp_command(env, cond["vx"])
        robot = env.scene["robot"].data
        root0 = robot.root_pos_w[:, :2].clone()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fall_step = torch.full((n,), -1, dtype=torch.long, device=device)
        # IsaacLab 的 step() 返回前就对 done env 完成自动复位（apt_flat_env.py:1640-1662 /
        # eval_apt_isaac.py:541-547 的 D048e 教训）——终末量绝不能读 step 之后的 robot.data。
        # 策略=「步前缓存」：done 者定格步前位置（滞后一个控制步 20ms，可接受）；未 done 者滚动更新。
        root_end = root0.clone()
        vx_sq_sum = torch.zeros(n, device=device)
        vx_cnt = torch.zeros(n, device=device)
        vx_sum = torch.zeros(n, device=device)  # 诊断：命令通道核验（D048h 纪律——指标身份先于机制归因）
        obs_p = obs["policy"]
        for step in range(cli.ep_steps):
            alive_pre = alive.clone()
            pre_pos = robot.root_pos_w[:, :2].clone()
            pre_vel = robot.root_lin_vel_b[:, 0].clone()
            if cond["kind"] == "flat":
                # 核验采样 = 策略本步实际消费的 obs（obs_p）命令槽。第 0 步 obs 出自
                # env.reset()（reset 内命令重采样发生在戳写之前、obs 未重算）⇒ 单独记录、
                # 不参与判据；step>=1 的 obs 含上一步戳写值。reset 路径的重采样只影响被复位
                # 的行（done/reset），故只对仍存活的行（alive_pre）采样。
                obs_cmd = _obs_cmd_vx(obs_p, slot)
                if step == 0:
                    obs_cmd_step0.append(float(obs_cmd.mean().item()))
                elif bool(alive_pre.any()):
                    obs_cmd_hist.extend(float(v) for v in obs_cmd[alive_pre].tolist())
                # 每控制步重戳（D064 教训：只在 reset 后戳一次会被 reset 路径的 resampling
                # 覆写；同时纠正 heading/standing 的同源覆写）
                cmd_hist.append(float(_stamp_command(env, cond["vx"]).mean().item()))
            with torch.no_grad():
                action = ac.act_inference(obs_p)  # det：actor 均值动作（冒烟核对：rsl_rl 2.3.3 API）
            obs, _, terminated, truncated, _ = env.step(action)
            obs_p = obs["policy"]
            done = terminated | truncated
            newly_fallen = alive_pre & terminated   # truncated=到时不算摔
            fall_step[newly_fallen] = step
            root_end[newly_fallen] = pre_pos[newly_fallen]             # 首摔定格（步前位置）
            root_end[alive_pre & done & ~terminated] = pre_pos[alive_pre & done & ~terminated]  # 到时定格
            keep = alive_pre & ~done
            root_end[keep] = robot.root_pos_w[:, :2][keep]             # 未复位者滚动更新
            if cond["kind"] == "flat":
                m = keep  # 摔步/到时步不计入 RMSE（摔步速度若计入=复位后≈0 的伪样本）
                vx_sq_sum[m] += (pre_vel[m] - cond["vx"]) ** 2
                vx_sum[m] += pre_vel[m]
                vx_cnt[m] += 1
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
                    "vx_rmse": round(float((vx_sq_sum[i] / vx_cnt[i]).sqrt().item()), 4)
                    if cond["kind"] == "flat" and vx_cnt[i].item() > 0
                    else None,
                }
            )
    surv = sum(e["survived"] for e in episodes)
    agg = {
        "kind": cond["kind"],
        "cond": cond,
        "n_episodes": len(episodes),
        "survival_rate": surv / len(episodes),
        "mean_disp_x": sum(e["disp_x"] for e in episodes) / len(episodes),
        "mean_vx_rmse": (
            sum(e["vx_rmse"] for e in episodes if e["vx_rmse"] is not None) / max(1, len(episodes))
            if cond["kind"] == "flat"
            else None
        ),
        "episodes": episodes,
    }
    if cond["kind"] == "flat":
        # 诊断（命令通道核验，D064 缺陷回填）：命令是否真的钉在 vx、实际速度是否≈0——区分
        # 「策略没学会走」与「评测命令通道失效」两种解释，判定前必读。
        # 判据口径 = **逐步采样**（旧法在循环外读 vel_command_b 只反映批次末一步，且 reset 路径
        # 重采样可把它覆盖成 U(0,1) 值 ⇒ 口径脏，仅作 legacy 对照保留）。
        cmd_term = env.command_manager.get_term("base_velocity")
        nominal = float(cond["vx"])
        cmd_mean = sum(cmd_hist) / len(cmd_hist) if cmd_hist else None
        cmd_var = (
            sum((v - cmd_mean) ** 2 for v in cmd_hist) / len(cmd_hist) if cmd_hist else None
        )
        obs_mean = sum(obs_cmd_hist) / len(obs_cmd_hist) if obs_cmd_hist else None
        obs_var = (
            sum((v - obs_mean) ** 2 for v in obs_cmd_hist) / len(obs_cmd_hist) if obs_cmd_hist else None
        )
        actual_per_env = [
            round(float((vx_sum[i] / vx_cnt[i]).item()), 4) if vx_cnt[i].item() > 0 else None
            for i in range(n)
        ]
        actual_valid = [x for x in actual_per_env if x is not None]
        actual_mean = sum(actual_valid) / len(actual_valid) if actual_valid else None
        agg["diag"] = {
            "cmd_vx_nominal": nominal,
            # legacy 对照：循环外读法（只反映批次末命令项，受 reset 路径重采样污染）
            "cmd_vx_mean_last_batch": float(cmd_term.vel_command_b[:, 0].mean().item()),
            # per-step 采样（判据口径）
            "cmd_vx_mean_over_steps": cmd_mean,
            "cmd_vx_min_over_steps": min(cmd_hist) if cmd_hist else None,
            "cmd_vx_max_over_steps": max(cmd_hist) if cmd_hist else None,
            "cmd_vx_std_over_steps": (cmd_var ** 0.5) if cmd_var is not None else None,
            "cmd_steps_sampled": len(cmd_hist),
            # ---- 判据采样 = 策略实际消费的 obs 命令槽 ----
            "obs_cmd_slot": {"term": slot["term"], "index": slot["index"], "start": slot["start"],
                             "width": slot["width"]},
            "obs_cmd_vx_n_samples": len(obs_cmd_hist),
            "obs_cmd_vx_mean": obs_mean,
            "obs_cmd_vx_min": min(obs_cmd_hist) if obs_cmd_hist else None,
            "obs_cmd_vx_max": max(obs_cmd_hist) if obs_cmd_hist else None,
            "obs_cmd_vx_std": (obs_var ** 0.5) if obs_var is not None else None,
            "obs_cmd_vx_step0_mean": (sum(obs_cmd_step0) / len(obs_cmd_step0)) if obs_cmd_step0 else None,
            "cmd_matches_nominal": bool(
                obs_cmd_hist and all(abs(v - nominal) <= CMD_NOMINAL_TOL for v in obs_cmd_hist)
            ),
            "actual_vx_mean_per_env": actual_per_env,
            "actual_vx_mean_all_envs": actual_mean,
            "actual_vx_near_static": (abs(actual_mean) < NEAR_STATIC_VX) if actual_mean is not None else None,
            "note": "obs_cmd_* = 策略消费的 obs 命令槽 per-step 采样（判据口径；第 0 步 obs 出自 "
            "env.reset() 不参与、done/reset 行排除）；cmd_* = 戳写路径 per-step 采样（写通道自检）；"
            "cmd_vx_mean_last_batch = 旧循环外读法，仅 legacy 对照；"
            "actual_* = 未 done 步的 root_lin_vel_b[:,0]",
        }
        print(
            f"[d064-eval][diag] flat vx={nominal} obs_cmd_mean={obs_mean} "
            f"obs_cmd[min,max]=[{agg['diag']['obs_cmd_vx_min']},{agg['diag']['obs_cmd_vx_max']}] "
            f"obs_cmd_n={agg['diag']['obs_cmd_vx_n_samples']} "
            f"matches_nominal={agg['diag']['cmd_matches_nominal']} "
            f"cmd_stamp_mean={cmd_mean} cmd_last_batch={agg['diag']['cmd_vx_mean_last_batch']:.4f} "
            f"actual_vx_mean={actual_mean} near_static={agg['diag']['actual_vx_near_static']} "
            f"actual_vx_mean(env0-2)={actual_per_env[:3]}",
            flush=True,
        )
    return agg


def _eval_one_main(cli, launcher_args) -> int:
    """--eval-one 子命令体：起 env → 批次 rollout → 打一行 EVAL_COND <json>。exit 0/1。"""
    cond = json.loads(cli.eval_one)
    hv = _import_heavy()  # 同款重型 import 入口（train_g1_decoder.py:496-560 的镜像，见下）
    torch = hv.torch
    device = getattr(launcher_args, "device", "cuda:0")
    # RNG 种子：条件子进程入口播全局 RNG（torch/numpy/random）——resampling/噪声/复位等随机量
    # 跨子进程可复现。在 env 构造与 reset 之前；不改任何评测口径默认值。
    _seed_everything(cli.seed, hv)
    print(f"[d064-eval] child: cond={cond} arm={cli.arm} pid={os.getpid()}", flush=True)
    env = None
    try:
        kw = dict(
            # flat 也建 rough cfg 再换平面几何：ckpt 训练于 rough（obs=321 含 height_scan），
            # plane cfg 的 obs=134 会让 actor 首层 size mismatch（12269 实证）；换几何后
            # height_scan 在平地上读 ≈0 = flat 语义，奖励项在评测中不被消费。
            terrain="rough",
            num_envs=cli.num_envs,
            action=cli.arm,
            token_stats=cli.token_stats,
            token_alpha=cli.token_alpha if cli.token_alpha is not None else 1.0,
            token_bound=cli.token_bound or "tanh",
            seed=cli.seed,
        )
        if cli.onnx_path:  # 未指定则走 make_env_cfg 工厂默认（SONIC_DECODER_ONNX），勿传空串覆盖
            kw["onnx_path"] = cli.onnx_path
        cfg = hv.make_env_cfg(**kw)
        if cond["kind"] == "rough_levels":
            # 出生行固定：关课程（多行假设不成立时直写 terrain_levels 语义更清晰）+ reset 前写行号
            cfg.curriculum.terrain_levels = None  # 冒烟核对：configclass 置 None 是否等效删 term
        if cond["kind"] == "rough_paper":
            from isaac.terrain_cfg import make_terrain_importer_cfg  # 双路径见 _import_heavy 同款处理

            cfg.scene.terrain = make_terrain_importer_cfg(
                "rough_paper", noise=cond["noise"], seed=cli.seed
            )
            cfg.curriculum.terrain_levels = None
            _gen = getattr(cfg.scene.terrain, "terrain_generator", None)
            if _gen is not None and hasattr(_gen, "curriculum"):
                _gen.curriculum = False  # 与工厂 stairs/stones/discrete 分支对齐（同构不漂移）
        if cond["kind"] == "flat":
            from isaac.terrain_cfg import make_terrain_importer_cfg  # 双路径见 _import_heavy 同款处理

            # flat=rough cfg + 平面几何（保 obs 321，见上 kw 注释）
            cfg.scene.terrain = make_terrain_importer_cfg("plane", seed=cli.seed)
            cfg.curriculum.terrain_levels = None
            _gen_f = getattr(cfg.scene.terrain, "terrain_generator", None)
            if _gen_f is not None and hasattr(_gen_f, "curriculum"):
                _gen_f.curriculum = False
            # 命令钉死第一步：重采样区间拉至无穷大（cfg 侧，只挡 step 内的周期重采样）
            # ⚠ 挡不住 reset 路径：上游 command_manager.reset() 对复位 env 无条件 _resample
            #   ⇒ 真正钉死靠 `_run_batches` 里**每控制步重戳**（见 _stamp_command docstring）
            cmd_cfg = cfg.commands.base_velocity
            cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)  # 冒烟核对：term cfg 字段名
        # B2：落地 60s 口径——env 继承官方 episode_length_s=20s，不覆写则 ~1000 步即全体 truncated
        cfg.episode_length_s = cli.ep_steps / 50.0
        env = hv.ManagerBasedRLEnv(cfg=cfg)
        mel = getattr(env, "max_episode_length", None)
        ctrl_dt = (getattr(cfg, "decimation", None), getattr(cfg.sim, "dt", None))
        print(
            f"[d064-eval] episode_length_s={cfg.episode_length_s} max_episode_length={mel} "
            f"(decimation,sim_dt)={ctrl_dt}（60s 口径=ep_steps/50Hz）",
            flush=True,
        )
        if mel is not None:
            assert int(mel) == cli.ep_steps, (
                f"max_episode_length={mel} != ep_steps={cli.ep_steps}（60s 口径未落地，判据失效，拒绝继续）"
            )
        if cond["kind"] == "rough_levels":
            lv = cond["level"]
            # 官方写法=env.scene.terrain（gear_sonic/envs/manager_env/mdp/curriculum.py:53-66）；
            # update_env_origins 是课程 ±1 行 API 不可借用，改用无参重算 origins（冒烟核对：2.1.0 名/元数）
            terrain_imp = env.scene.terrain
            terrain_imp.terrain_levels[:] = lv  # 冒烟核对：TerrainImporter.terrain_levels（2.1.0）
            terrain_imp.configure_env_origins()  # 冒烟核对：无参重算 env_origins（2.1.0 API 名/元数）
            # SF1 读回保护：若该 API 会按 max_init_terrain_level 重采样行号，这里立刻响（防静默错行失真主指标①）
            lv_back = terrain_imp.terrain_levels
            assert bool((lv_back == lv).all()), f"出生行未固定：期望 {lv}，实得 {lv_back.tolist()}"
            print(f"[d064-eval] rough_levels fixed row={lv} confirmed", flush=True)
        obs, _ = env.reset()
        obs_dim = int(obs["policy"].shape[-1])
        action_dim = int(env.action_manager.total_action_dim)
        ac = _load_actor_weights(hv, cli.ckpt, obs_dim, action_dim, device)
        agg = _run_batches(env, ac, cli, cond, hv)
        print("EVAL_COND " + json.dumps(agg, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 —— 判读要可见：类型名必进 FAIL 行
        print(f"EVAL_COND_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _import_heavy():
    """App 起来后的重型 import（复用 train_g1_decoder._import_heavy 的组装，避免双份漂移）。"""
    try:
        from isaac import train_g1_decoder as td
    except ImportError:  # 仓根包 import
        from apt_g1.isaac import train_g1_decoder as td
    return td._import_heavy()


def main() -> None:
    cli = build_args().parse_args()
    if cli.eval_one is not None:
        # 子进程体：AppLauncher 链与 train 同款
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
            code = _eval_one_main(cli, launcher_args)
        finally:
            simulation_app.close()
        sys.exit(code)

    # ---- 父进程：展开条件 → 逐条件子进程 → 聚合 JSON ----
    conds = expand_conditions(cli)
    if not conds:
        build_args().error("battery 展开为空（检查 --levels/--noise/--vx-cmds）")
    ckpt_md5 = _file_md5(cli.ckpt)
    decoder_md5 = _file_md5(cli.onnx_path) if cli.arm == "decoder" and cli.onnx_path else None
    out_dir = Path(cli.output_dir) if cli.output_dir else Path(cli.ckpt).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(cli.ckpt).stem
    out_json = out_dir / f"d064_eval_{cli.arm}_{ckpt_name}.json"

    results: list[dict] = []
    for cond in conds:
        argv = _child_argv(cli, cond)
        print(f"[d064-eval] condition {cond} -> spawn", flush=True)
        try:
            # encoding 必须显式：C locale 下 text=True 默认 ASCII 解码遇 UTF-8 崩（train r5 实证）
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"kind": cond["kind"], "cond": cond, "error": f"spawn-failed: {type(exc).__name__}: {exc}"})
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
            print(out, flush=True)  # capture-and-forward：tee 可见完整 Isaac 日志
        cond_res = None
        for line in out.splitlines():
            if line.startswith("EVAL_COND "):
                try:
                    cond_res = json.loads(line[len("EVAL_COND "):])
                except json.JSONDecodeError:
                    cond_res = None
        if cond_res is not None:
            results.append(cond_res)
        else:
            why = f"timeout>{COND_TIMEOUT_S}s" if timed_out else f"no-line (exit {proc.returncode})"
            for line in out.splitlines():
                if line.startswith("EVAL_COND_FAIL "):
                    why = line[len("EVAL_COND_FAIL "):]
                    break
            else:
                if err:  # 段错误类原生崩只有 exit code：stderr 尾部是唯一线索，转发助诊
                    print("[d064-eval] child stderr tail:\n" + "\n".join(err.splitlines()[-5:]), flush=True)
            results.append({"kind": cond["kind"], "cond": cond, "error": why})

    payload = {
        "identity": {
            "ckpt": str(Path(cli.ckpt).resolve()),
            "ckpt_md5": ckpt_md5,
            "decoder_onnx_md5": decoder_md5,
            "arm": cli.arm,
            "battery": cli.battery,
            "episodes": cli.episodes,
            "ep_steps": cli.ep_steps,
            "num_envs": cli.num_envs,
            "seed": cli.seed,
            "levels": cli.levels,
            "noise": cli.noise,
            "vx_cmds": cli.vx_cmds,
            "git_head": _git_head(),
        },
        "results": results,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = [r for r in results if "error" in r]
    if bad:
        print(f"EVAL FAIL: {len(bad)}/{len(results)} 条件失败 -> {out_json}", flush=True)
        sys.exit(1)
    print(f"EVAL DONE: {out_json}", flush=True)


if __name__ == "__main__":
    main()
