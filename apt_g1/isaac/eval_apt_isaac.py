"""Isaac Lab A/B/C/D eval for the APT aux policy (mirrors MuJoCo eval_apt_aux).

A. 60 s straight walk @ vx=0.8
B. disturbance impulses (500 N, 4 dirs, t=10 s and t=25 s) during 45 s walk
C. vx/vy command-switch marathon (68 s)
D. jump with explicit mode command (20 s)

Each test runs with aux=0 and with the trained policy aux (and optionally the
policy phase in phase_mode), 3 seeds, and reports the same metrics as the
MuJoCo harness (steps, completed, fall_step, h_min, vx, displacement).

D048e 评测契约 v2（JSON 顶层 eval_contract=2；旧 JSON 无此键 = v1 口径）：
  ① 逐局 seed 配对：--contract train 分支在 env.reset() 前重设全局随机源，
    同 seed 局在 aux/noaux 两遍之间初态逐位相同（--sample 下噪声序列也同源，
    唯一差异 = 残差开/关）；
  ② done 步统计截留：term/trunc 步的 robot.data 已被 step 内部 auto-reset
    覆盖，终末 xy/yaw/quat 改从 env._reset_idx 复位前截留的 _final_pos_w /
    _final_quat_w 取，摔倒局终点不再被新局出生点污染；
  ③ disp 统一为"局首快照→终末快照"的 2D 净位移（v1 起点是首步 step 后，
    终点在 term 局被污染）；
  ④ 新增航向系指标：fwd_signed / lat_signed（初始航向系带符号投影，米）、
    yaw_err（终末-yaw0 wrap 到 [-π,π]，rad）、upright_final（终末 upright）、
    ended（"term"/"trunc"/"full"）；JSON 顶层落盘 res_scale 防 0.4/0.15 类
    训练/评测失配再犯。

D048f 阶段0 评测资格（JSON 顶层 metrics_contract="d048f_stage0"）：
  ⑤ 任务成功指标：vx_rmse（机体系 vx 对本局命令的逐步 RMSE）、yaw_err_int
    （|wrap(yaw_t−yaw0)|·dt 时间积分）、lat_max / fwd_max（初始航向系逐步
    横偏/前进包络）、survived_budget（零摔且跑满时限，completed 同式显式
    别名）、task_success（0.4 m/s×20s 契约下按冻结门逐局判定；门外契约
    一律 null，门常量落盘 task_gate_flat04，不得事后改门）；
  ⑥ 逐局初态指纹 init（z0/yaw0/关节位 md5/首帧 obs md5+sum）——seed 配对
    核验从"调用了 manual_seed"升级为"产物可事后逐位比对"；
  ⑦ 每个 schedule entry 写命令后刷新首帧观测（命令变更后的首个动作看到新
    命令）。单 entry 测试（A/B/D）刷新时点与 v2 完全一致，行为逐位不变；
    多 entry（C 切换）此前只有首个 entry 刷新，后续段首动作仍消费旧命令 obs；
  ⑧ 显式失败纪律：checkpoint 缺失（AppLauncher 之前快速失败）/加载失败/
    零局产出 → 非零退出，绝不退回未训练 policy 当 aux=0 跑完（v1 缺文件
    只 WARNING，产出看似合法的 JSON）；顶层异常 os._exit(1)（kit 接管
    excepthook 后解释器可能仍以 0 退出）；B test 冲量幅值 --impulse-n
    可调（term 分支探针需要必摔局，全存活批次永远走不到该分支）；
  ⑨ D048h：ckpt 配置身份核验（ckpt_identity 信封）——身份块存在则逐键比对
    res_scale / res_clip / latent_residual / obs·action 维度 / vae_md5 /
    decoder_md5，任何失配在跑第一局之前 os._exit(6)（0.4-vs-0.15 类训练/
    评测配置错配被结构性拒绝）；旧格式纯 state_dict = legacy，WARNING 后
    继续（无核验）。匹配/legacy 状态与两侧关键值落盘 out["ckpt_identity"]。
    D048h 另增 --res-stats：逐局残差执行统计（饱和/分位/差分）落盘
    out[*]["res_diag"] + 顶层跨局聚合，metrics_contract 翻转为
    "d048h_resdiag"（默认路径保持 "d048f_stage0" 不变）。

D048l 扩展：--force-vbin 评测干预（eval_interventions 显式记录）+ yaw_err_deg/heading_gate_pass 单位防误读字段；eval_contract 语义不变。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import os

import numpy as np
import torch


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--checkpoint", default=None)
    # E49：未训练初始化对照——跳过 checkpoint 加载，直接用按旗标新构造的
    # policy 评测（确定性模式照旧；token 模式下确定性初始策略 ≈ 固定官方
    # 均值 token 回放）
    ap.add_argument(
        "--init-policy",
        action="store_true",
        help="未训练初始化对照：跳过 --checkpoint 加载，直接用按旗标新构造的 "
        "policy 评测（确定性模式照旧；token 模式下确定性初始策略 ≈ 固定官方"
        "均值 token 回放）",
    )
    # E49 诊断：采样 vs 确定性对照。--sample 时动作改用策略分布采样
    # (deterministic=False，与训练 rollout 同口径)；默认 False = 历史确定性
    # 评测逐位不变。JSON 顶层会标注 mode: "det" | "sample"。
    ap.add_argument(
        "--sample",
        action="store_true",
        help="sample actions from the policy distribution (deterministic="
        "False, same口径 as the training rollout) instead of the mean action",
    )
    # E49 诊断步骤②：训练契约评测。train = episode 上限 20s（250 控制步@50Hz，
    # 同训练）+ 初态 = env 原生 reset 分布（_reset_idx 的 z=0.76+U(±0.02)、
    # SONIC 默认关节精确值、零速度、随机 walk-clock 相位），跳过 eval 契约的
    # 高斯 jitter 覆盖；eval = 现状 120s 上限 + jitter 初态。
    ap.add_argument(
        "--contract",
        choices=["eval", "train"],
        default="eval",
        help="评测契约：eval=历史口径（120s 上限、高斯 jitter 初态）；"
        "train=训练契约（20s episode 上限、env 原生 reset 初态）",
    )
    # 每组 rollout 数（默认 3 与历史 seeds 对齐；token 模式 aux/noaux 为同
    # 分布独立重复，双组结构不变）
    ap.add_argument("--num-rollouts", type=int, default=3,
                    help="number of rollouts per (test, key); default 3")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base evaluation seed for policy sampling and env reset RNG; "
        "rollout i uses seed+i (default 0 preserves historical seed0..N-1)",
    )
    # E49 诊断：cmd 随机化对照（仅 A test 生效）。每局 rollout 前从
    # uniform(0, --a-cmd-vx) 抽本局 vx 写入命令，并记进 JSON 该 rollout 条目
    # 的 cmd_vx 字段；默认 False = 固定 --a-cmd-vx（与已有数据单变量可比）。
    ap.add_argument(
        "--cmd-sample",
        action="store_true",
        help="per-rollout vx ~ uniform(0, --a-cmd-vx) for test A (recorded "
        "per-rollout as cmd_vx in the JSON); default fixed a-cmd-vx",
    )
    ap.add_argument("--tests", default="A,B,C,D")
    ap.add_argument("--env", choices=["apt", "vanilla"], default="apt")
    ap.add_argument("--out", default="outputs/isaac_eval.json")
    ap.add_argument("--phase-mode", action="store_true")
    ap.add_argument("--phase-anchor", action="store_true")
    ap.add_argument("--aux-scale", type=float, default=0.2)
    ap.add_argument("--latent-mode", action="store_true")
    # E44: decoder fine-tuning policy (29-d joint-target actions). The
    # checkpoint contains the fine-tuned SONIC decoder; run A/B/C only (the
    # D jump test needs the router mode path, which decft bypasses).
    ap.add_argument("--decft", action="store_true")
    ap.add_argument(
        "--latent-vae-path",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/token_vae_e27/vae.pt",
    )
    # E28: command-conditioned gait cadence (must mirror training for consistency)
    ap.add_argument("--latent-cmd-phase-rate", action="store_true")
    ap.add_argument("--latent-phase-rate-ref", type=float, default=0.6)
    ap.add_argument("--latent-phase-rate-max", type=float, default=2.0)
    ap.add_argument("--stillness-vx-scale", type=float, default=0.05)
    # E31: speed-conditioned VAE decoder (must mirror training)
    ap.add_argument("--latent-speed-bins", action="store_true")
    # E35: direction+speed-conditioned VAE decoder (must mirror training)
    ap.add_argument("--latent-dir-bins", action="store_true")
    # E48: full-joint residual escape channel (must mirror training). aux head
    # of the checkpoint is 29-d; --aux-zero gives the residual-off ablation.
    ap.add_argument("--latent-residual", action="store_true")
    # E49: direct-token RL (no VAE). B arm = walk clock [sin, cos] in obs.
    ap.add_argument("--token-mode", action="store_true")
    ap.add_argument("--token-phase-obs", action="store_true")
    ap.add_argument("--token-alpha", type=float, default=1.0)
    ap.add_argument("--token-bound", choices=["none", "tanh"], default="none")
    ap.add_argument("--token-stats", default="",
                    help="npz with mean/std/rate from official g1-mode tokens")
    ap.add_argument("--res-scale", type=float, default=0.4)
    ap.add_argument("--res-clip", type=float, default=1.0)
    # D048h: per-rollout residual execution statistics (sat/near-sat/quantile
    # pools accumulated env-side, harvested as res_diag per rollout plus a
    # top-level aggregate; metrics_contract flips to "d048h_resdiag")
    ap.add_argument("--res-stats", action="store_true",
                    help="D048h: accumulate per-step residual execution stats "
                    "(sat_frac/near_sat/quantiles/diff) and store res_diag "
                    "per rollout + top-level aggregate")
    # D048f: B test impulse magnitude (N). Default 500 = historical value;
    # the stage-0 term-branch probe passes larger values to force falls, so
    # the term/trunc terminal-stats path is actually exercised (an
    # all-survive batch never reaches that branch).
    ap.add_argument("--impulse-n", type=float, default=500.0)
    ap.add_argument("--use-elevation", type=int, default=0)
    # E32: heading/yaw reward (rollout dynamics only; no effect on eval metrics)
    ap.add_argument("--yaw-scale", type=float, default=0.5)
    ap.add_argument("--heading-scale", type=float, default=0.0)
    # TO38: override test A's commanded vx (default 0.8 = E-battery standard;
    # low-speed band evals pass e.g. 0.277 / 0.2 / 0.35)
    ap.add_argument("--a-cmd-vx", type=float, default=0.8)
    # D048l: evaluation-only intervention -- force the VAE speed-bin condition
    # fed to decode (vb) instead of the natural bucketize(cmd) assignment.
    ap.add_argument("--force-vbin", type=int, default=-1,
                    help="D048l 评测干预：强制 VAE 速度档 0..2，-1=自然分档；记入 eval_interventions")
    # D048q: soft speed-bin intervention -- blend the speed embedding as
    # alpha*E[a]+(1-alpha)*E[b] (Q1 连续可调性画像)；与 --force-vbin 互斥。
    ap.add_argument("--force-vbin-soft", type=str, default="",
                    help="D048q 评测干预：软速度档 \"a,b,alpha\"（a/b∈{0,1,2} 档对、"
                         "alpha∈[0,1] 为 a 档权重，decode speed_embed 软混），"
                         "与 --force-vbin 互斥；记入 eval_interventions")
    ap.add_argument("--force-dbin", type=int, default=-1,
                    help="D048q 评测干预：强制 VAE 方向档 0..7（Q2 方位语义闭环"
                         "画像），-1=自然方位分桶（仅带 db 的 latent-dir-bins 分支"
                         "生效）；记入 eval_interventions")
    # TO38: reference obs injection (must match the trained policy's obs dim)
    ap.add_argument("--to-ref", action="store_true")
    ap.add_argument("--to-ref-npz", default="")
    ap.add_argument("--to-ref-obs-zero", action="store_true")
    ap.add_argument("--to-tau", action="store_true")
    ap.add_argument("--to-tau-w", type=float, default=1.0)
    # E33: open-loop yaw-bias compensation (rad/s). Cancels a systematic
    # turning bias (e.g. E31's ~-0.07 rad/s = -4 deg/s left drift).
    ap.add_argument("--yaw-bias-comp", type=float, default=0.0)
    ap.add_argument("--terrain", choices=["plane", "rough", "rough_paper", "rough_sym"], default="plane")
    ap.add_argument("--terrain-noise", type=float, default=0.04)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument(
        "--phase-zero",
        action="store_true",
        help="ablation: zero the policy phase output (pure router clock in "
        "anchored mode, no modulation)",
    )
    ap.add_argument(
        "--aux-zero",
        action="store_true",
        help="ablation: zero the policy aux output (isolate phase mechanism)",
    )
    ap.add_argument("--use-2hz-gate", type=int, default=1)
    ap.add_argument("--router-model-dir",
                    default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final")
    ap.add_argument(
        "--decoder-path",
        default=(
            "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
            "gear_sonic_deploy/policy/release/model_decoder.onnx"
        ),
    )
    return ap


def jitter_and_reset(env, seed: int, contract: str = "eval"):
    """Reset all envs (num_envs=1) and apply MuJoCo-parity reset jitter.

    contract="train" (E49 诊断): skip the jitter overlay entirely -- the
    initial state is exactly the env's native ``env.reset()`` distribution
    (``_reset_idx``: z=0.76+U(±0.02), exact SONIC default joints, zero
    velocities, random walk-clock phase), i.e. the same per-episode spawn
    distribution the policy saw during training. We consume the reset-returned
    obs and refresh ``_last_obs`` with it (same discipline as the E49 fix in
    the eval path below).
    """
    if contract == "train":
        # D048e 评测契约 v2 ①：逐局重设随机源后再 reset。env._reset_idx 的全部
        # 初态随机量（walk-clock 相位 _latent_phase、to_phase、z 抖动、cmd 采样）
        # 都消费全局 torch 流（torch.rand/uniform_ 无独立 generator），v1 只在
        # 进程启动设一次 seed，--sample 又持续消耗同一流 → aux/noaux 两遍的同名
        # seed 局实际初态/噪声序列错位。逐局 manual_seed 后：同 seed 局在两遍
        # 之间初态逐位相同；两遍 rollout 逐步消费同一条流 → 采样噪声同源配对，
        # 唯一差异 = 残差开/关。
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        obs_dict, _ = env.reset()
        env._last_obs = obs_dict["policy"]
        return
    rng = np.random.default_rng(1000 + seed)
    env.reset()
    device = env.device
    env_ids = torch.arange(env.num_envs, device=device)

    root = env.robot.data.default_root_state[env_ids].clone()
    root[:, 2] = env.scene.env_origins[env_ids, 2] + 0.76 + torch.tensor(
        rng.normal(0.0, 0.005, len(env_ids)), dtype=torch.float32, device=device
    )
    env.robot.write_root_state_to_sim(root, env_ids)

    jp = env.robot.data.default_joint_pos[env_ids].clone()
    sonic = torch.from_numpy(env._sonic_default).to(device)
    noise = torch.tensor(
        rng.normal(0.0, 0.01, (len(env_ids), 29)), dtype=torch.float32, device=device
    )
    jp[:, env._body_idx] = sonic + noise
    jv = torch.tensor(
        rng.normal(0.0, 0.02, (len(env_ids), env.robot.num_joints)),
        dtype=torch.float32,
        device=device,
    )
    env.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    # refill history from the (jittered) current state
    ang_vel = env._base_ang_vel()
    jpos_rel = env.robot.data.joint_pos[:, env._body_idx] - sonic
    jvel = env.robot.data.joint_vel[:, env._body_idx]
    gravity = env.robot.data.projected_gravity_b
    n = env.num_envs
    env._hist_ang_vel[:] = ang_vel[:, None, :].expand(n, 10, 3).clone()
    env._hist_joint_pos[:] = jpos_rel[:, None, :].expand(n, 10, 29).clone()
    env._hist_joint_vel[:] = jvel[:, None, :].expand(n, 10, 29).clone()
    env._hist_last_actions[:] = 0.0
    env._hist_gravity[:] = gravity[:, None, :].expand(n, 10, 3).clone()
    if env._router_state is not None:
        env._router_state = env._router.reset_state(env.num_envs)
    env._gate_mode[:] = 0
    env._gate_tick[:] = False
    env._gate_count[:] = 0
    env._q_des[:] = sonic
    env._last_phase[:] = 0.0
    env._last_aux[:] = 0.0
    # E49 fix (owner R4 #2): reset()'s return obs was discarded and the
    # jitter/history edits above left _last_obs stale -- every rollout's
    # first policy step acted on the PREVIOUS rollout's last obs (pre-existing
    # defect; for E49-B it also broke the phase-consistency requirement since
    # the cached phi did not match the post-reset walk clock). Reassemble and
    # cache fresh obs after all reset/jitter state edits.
    env._last_obs = env._get_observations()["policy"]


def rollout(
    env,
    policy,
    schedule,
    seed,
    use_aux,
    impulses=None,
    phase_policy=False,
    phase_zero=False,
    aux_zero=False,
    latent_policy=False,
    yaw_bias=0.0,  # E33: open-loop yaw command offset (rad/s) to cancel a
    # systematic turning bias (e.g. E31's -4 deg/s left drift)
    sample: bool = False,  # E49 诊断: True = act(deterministic=False) 采样动作
    contract: str = "eval",  # E49 诊断: "train" = 20s episode 上限（env 原生
    # reset 初态），rollout 在 max_episode_length 处按正常截断收尾（非 fall）
):
    """schedule: list of (vx, vy, seconds) or (Command, seconds) pairs."""
    from apt_g1.encoder import Command

    jitter_and_reset(env, seed, contract=contract)
    # E49 --sample: 单次 act() 按步记忆化。det 模式与旧的分散调用逐位等价
    # （均值头与调用次数无关）；sample 模式下保证 aux/latent/gate 分支消费
    # 同一次采样 draw（训练 rollout 就是每步一次 act）。obs 更新后清缓存。
    _act_cache: dict = {}

    def _act():
        if "out" not in _act_cache:
            with torch.no_grad():
                _act_cache["out"] = policy.act(
                    env._last_obs, deterministic=not sample
                )
        return _act_cache["out"]
    total_steps = int(
        sum(entry[2] if len(entry) == 3 else entry[1] for entry in schedule) * 50
    )
    if contract == "train":
        # 训练契约：episode 在 max_episode_length（20s@50Hz=1000 步）处被 env
        # 截断并自动 reset——schedule 只会跑到那里（60s 的 A test 实际 20s）。
        total_steps = min(total_steps, int(env.max_episode_length))
    imp = {s: f for s, f in (impulses or [])}
    heights, vxs, vys = [], [], []
    xys = []
    ugs = []  # D048c: per-step |projected gravity xy| for the upright score
    yaws = []  # D048f: per-step yaw for the heading-error time integral
    fall = None
    ep_done = False
    ended = "full"  # D048e: "term"（摔倒）/ "trunc"（episode 时限截断）/ "full"

    # D048e 评测契约 v2 ②③：航向系指标需要"局首快照（命令设定后、首个 act 前）
    # →终末快照"两端的 xy/yaw/quat。term/trunc 步的 robot.data 已被 step 内部
    # auto-reset 覆盖（DirectRLEnv.step 内部 _get_dones → _reset_idx →
    # _get_observations 之后才返回），终末物理量只能从 env._reset_idx 在
    # super() 之前截留的 _final_pos_w / _final_quat_w 取；存活步逐步把终末
    # 指针刷新为当前 root 状态。
    first_entry = True
    xy0 = None
    yaw0 = 0.0
    final_xy = None
    final_yaw = 0.0
    final_ug = 0.0  # 终末步 |projected gravity xy|（终末 quat 派生）
    init_fp = None  # D048f ⑥：逐局初态指纹

    def _yaw_of(q):
        # 标准 quat→yaw（Isaac root_quat_w 为 w-first），与环境 elevation 采样
        # 的内联公式一致（apt_flat_env._get_observations）
        w, x, y, z = q[0], q[1], q[2], q[3]
        return float(
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).item()
        )

    def _grav_xy_of(q):
        # 世界重力 [0,0,-1] 旋转到机体系（g_b = R(q)^T g_w）的 xy 模：
        # 2*sqrt((xz-wy)² + (yz+wx)²)；直立为 0，与环境 projected_gravity_b
        # 同口径（训练 upright 项 = exp(-g_xy²/0.1)）
        w, x, y, z = q[0], q[1], q[2], q[3]
        return 2.0 * math.sqrt((x * z - w * y) ** 2 + (y * z + w * x) ** 2)

    t = 0
    for entry in schedule:
        if len(entry) == 3:
            vx, vy, secs = entry
            item = (vx, vy)
        else:
            item, secs = entry
        if isinstance(item, Command):
            env.router_commands[0] = item
            env._commands[0] = torch.zeros(3, dtype=torch.float32, device=env.device)
        else:
            vx, vy = item
            env.router_commands[0] = None
            env._commands[0] = torch.tensor(
                [vx, vy, yaw_bias], dtype=torch.float32, device=env.device
            )
        if first_entry:
            # D048e 局首快照：第一个 entry 的命令写入后、首个 act 之前做一次
            # （C 多 entry schedule 也只快照这一回）。此刻 robot.data 仍是 reset
            # 初态；顺带刷新首帧观测，让首个动作看到本局命令（v1 首帧 obs 带
            # 的是 reset 期采样的旧命令）。
            xy0 = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy()
            yaw0 = _yaw_of(env.robot.data.root_quat_w[0].detach())
            final_xy = xy0
            final_yaw = yaw0
            # D048f ⑥：初态指纹与首帧 obs 取自同一次 _get_observations() 调用
            # （单 entry 行为与 v2 逐位一致）；md5 使同 seed 局在 aux/noaux 两遍
            # 之间的初态一致性可以事后从 JSON 直接核验，不依赖"调用了
            # manual_seed"这一静态事实。
            _obs_t = env._get_observations()["policy"]
            env._last_obs = _obs_t
            _jp = env.robot.data.joint_pos[0, env._body_idx].detach().cpu().numpy()
            _obs0 = _obs_t.detach().cpu().numpy()
            init_fp = {
                "z0": round(float(env.robot.data.root_pos_w[0, 2].item()), 5),
                "yaw0_deg": round(math.degrees(yaw0), 5),
                "joint_pos_md5": hashlib.md5(_jp.tobytes()).hexdigest()[:10],
                "obs_md5": hashlib.md5(_obs0.tobytes()).hexdigest()[:10],
                "obs_sum": round(float(_obs0.sum()), 6),
            }
            first_entry = False
        else:
            # D048f ⑦：后续 entry（命令变更）也刷新首帧观测——切换后的首个
            # 动作必须看到新命令。单 entry 测试不走此分支，与 v2 逐位一致。
            env._last_obs = env._get_observations()["policy"]
        for _ in range(int(secs * 50)):
            if t in imp:
                world_dir = torch.tensor(
                    imp[t], dtype=torch.float32, device=env.device
                ).reshape(1, 1, 3)
                body_dir = env._world_to_body(world_dir)
                forces = torch.zeros(1, 1, 3, dtype=torch.float32, device=env.device)
                # D048f 勘误：旧式 forces[0,0]=body_dir[0,0] 中 body_dir 形状
                # (1,3)、[0,0] 是标量 x 分量，广播后三轴全为 body_x——实际
                # 施力 = (body_x,body_x,body_x) 对角力而非设计方向：fwd/back
                # 变成斜向推、left/right（yaw≈0 时 body_x≈0）历来是零力无效
                # 扰动。修 = 整行赋真实机体系方向。
                forces[0, 0, :] = body_dir[0, :]
                env.robot.set_external_force_and_torque(
                    forces,
                    torch.zeros_like(forces),
                    body_ids=[env._root_body_idx[0]],
                )
            if use_aux:
                act, _, _, _, _ = _act()
                aux = act["aux"]
                if aux_zero:
                    aux = torch.zeros_like(aux)
            else:
                aux = torch.zeros(1, 12, dtype=torch.float32, device=env.device)
            if getattr(env, "_vanilla", False):
                act, _, _, _, _ = _act()
                action = act["aux"]
            elif getattr(env, "_decft", False):
                # E44: policy action IS the 29-d joint-target vector
                if not use_aux:  # should not happen (decft keys force use_aux)
                    act, _, _, _, _ = _act()
                action = act["aux"]
            elif getattr(env, "_gate_policy", False):
                act, _, _, _, _ = _act()
                action = torch.zeros(1, 13, dtype=torch.float32, device=env.device)
                action[:, :12] = aux
                action[:, 12] = act["gate"].float()
            elif latent_policy:
                act, _, _, _, _ = _act()
                action = act["latent"]
                if getattr(env, "_latent_residual", False):
                    # E48: append the full-joint residual (policy aux head).
                    # noaux / aux_zero rollouts zero it -> pure-prior ablation.
                    if use_aux and not aux_zero:
                        res = act["aux"]
                    else:
                        res = torch.zeros_like(act["aux"])
                    action = torch.cat([action, res], dim=1)
            else:
                action = torch.zeros(1, 14, dtype=torch.float32, device=env.device)
                action[:, 2:] = aux
                if phase_policy:
                    act, _, _, _, _ = _act()
                    if phase_zero:
                        act["phase"] = torch.zeros_like(act["phase"])
                    action[:, :2] = act["phase"]
            obs_dict, reward, term, trunc, _ = env.step(action)
            env._last_obs = obs_dict["policy"]
            _act_cache.clear()
            if t in imp:
                env.robot.set_external_force_and_torque(
                    torch.zeros(1, 1, 3, dtype=torch.float32, device=env.device),
                    torch.zeros(1, 1, 3, dtype=torch.float32, device=env.device),
                    body_ids=[env._root_body_idx[0]],
                )
            if term.any() or trunc.any():
                # D048e 评测契约 v2 ②：done 步不读 robot.data——step 内部
                # auto-reset 已把它覆盖为新局出生点/新局姿态（v1 term 局曾把
                # 新局初值混入 heights/xys 末位 → 终点≈出生点、disp 失真）。
                # 终末物理量取 _reset_idx 在 super() 之前截留的 _final_pos_w /
                # _final_quat_w（复位前值）；本步物理量不进逐步统计（v1 的
                # trunc-not-term 口径统一推广到 term，截留口径一致）。
                fq = env._final_quat_w[0].detach()
                final_xy = env._final_pos_w[0].detach().cpu().numpy()
                final_yaw = _yaw_of(fq)
                final_ug = _grav_xy_of(fq)
                ended = "term" if term.any() else "trunc"
                if term.any():
                    fall = t
                ep_done = True
                break
            h = float(env.robot.data.root_pos_w[0, 2].item())
            v = env._base_lin_vel()[0].detach().cpu().numpy()
            xy = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy()
            heights.append(h)
            vxs.append(float(v[0]))
            vys.append(float(v[1]))
            xys.append(xy)
            ugs.append(
                float(env.robot.data.projected_gravity_b[0, :2].norm().item())
            )
            # 存活步：终末指针刷新为当前 root 状态（schedule 自然跑满时即为
            # 局末值，与 term/trunc 截留口径对齐）
            final_xy = xy
            _yaw_t = _yaw_of(env.robot.data.root_quat_w[0].detach())
            yaws.append(_yaw_t)
            final_yaw = _yaw_t
            final_ug = ugs[-1]
            t += 1
        if fall is not None or ep_done:
            break
    heights = np.array(heights)
    vxs = np.array(vxs)
    vys = np.array(vys)
    xys = np.array(xys)
    h_min = float(heights.min()) if len(heights) else 0.0
    spd = np.sqrt(vxs**2 + vys**2)
    # D048e 评测契约 v2 ③：disp/drift_y 统一为"局首快照→终末快照"口径。
    # v1 的 xys[0] 是首步 step 后的位置、xys[-1] 在 term 局是 auto-reset 后的
    # 新局出生点（≈起点 → 摔倒局 disp≈0 失真）。快照对口径下 0 步存活局
    # （首步即摔）也有定义。disp/drift_y 保持 2D 净位移语义；fwd/lat 是同一位
    # 移在初始航向单位向量上的带符号投影——disp/vx（机体系前向速度）都不能
    # 单独当"直走进展"，航向系投影才是正确量纲。
    disp_vec = (final_xy - xy0) if final_xy is not None else np.zeros(2)
    displacement = float(np.linalg.norm(disp_vec))
    drift_y = float(abs(disp_vec[1]))
    # D048c: 漂移与姿态入报告。upright 与 train 奖励项同式
    # （exp(-g_xy²/0.1) 逐步算再平均）；upright_final 是终末快照的同式单点值。
    upright = (
        float(np.mean(np.exp(-(np.array(ugs) ** 2) / 0.1))) if ugs else 0.0
    )
    upright_final = float(np.exp(-(final_ug**2) / 0.1))
    fwd_signed = float(
        disp_vec[0] * math.cos(yaw0) + disp_vec[1] * math.sin(yaw0)
    )
    lat_signed = float(
        -disp_vec[0] * math.sin(yaw0) + disp_vec[1] * math.cos(yaw0)
    )
    yaw_err = float((final_yaw - yaw0 + math.pi) % (2.0 * math.pi) - math.pi)
    # D048f ⑤：任务成功指标。存活≠成功——终末 fwd/lat/yaw 是单点量，识别
    # 不了中途绕圈/超速回落，逐步量（vx RMSE、航向误差时间积分、最大横偏）
    # 与"存活预算"分开落盘；终末 yaw 不能识别中途绕圈，yaw_err_int 补这个洞。
    dt = 0.02  # 50 Hz 控制步
    yaws_arr = np.array(yaws)
    if len(xys):
        d = xys - xy0[None, :]
        fwd_t = d[:, 0] * math.cos(yaw0) + d[:, 1] * math.sin(yaw0)
        lat_t = -d[:, 0] * math.sin(yaw0) + d[:, 1] * math.cos(yaw0)
        lat_max = float(np.abs(lat_t).max())
        fwd_max = float(fwd_t.max())
    else:
        lat_max = 0.0
        fwd_max = 0.0
    if len(yaws_arr):
        yaw_err_t = (yaws_arr - yaw0 + math.pi) % (2.0 * math.pi) - math.pi
        yaw_err_int = float(np.abs(yaw_err_t).sum() * dt)
    else:
        yaw_err_int = 0.0
    cmd_vx_eff = schedule[0][0] if schedule and len(schedule[0]) == 3 else None
    vx_rmse = (
        float(np.sqrt(np.mean((vxs - cmd_vx_eff) ** 2)))
        if cmd_vx_eff is not None and len(vxs)
        else None
    )
    survived_budget = bool(fall is None and len(heights) >= total_steps - 1)
    # 冻结门（DS_CONTINUOUS_EXECUTION_PLAN §5：0.4 m/s×20s 平地直走，全部
    # 条件同时满足才计成功；门外契约不判定 → null）。门常量在 main() 顶层
    # 落盘 task_gate_flat04，不得依候选结果事后放宽。
    task_success = None
    if (
        contract == "train"
        and cmd_vx_eff is not None
        and abs(cmd_vx_eff - 0.4) < 1e-6
    ):
        task_success = bool(
            survived_budget
            and vx_rmse is not None
            and vx_rmse <= 0.10
            and 6.0 <= fwd_signed <= 10.0
            and lat_max <= 0.5
            and abs(yaw_err) <= math.radians(15.0)
            and upright >= 0.90
        )
    return {
        "steps": len(heights),
        "completed": fall is None and len(heights) >= total_steps - 1,
        "survived_budget": survived_budget,
        "fall_step": fall,
        "h_min": round(h_min, 3),
        "vx": round(float(vxs.mean()), 3) if len(vxs) else 0.0,
        "disp": round(displacement, 3),
        "v_speed": round(float(spd.mean()), 3) if len(spd) else 0.0,
        "drift_y": round(drift_y, 3),
        "upright": round(upright, 3),
        # D048e 评测契约 v2 新增键
        "fwd_signed": round(fwd_signed, 3),
        "lat_signed": round(lat_signed, 3),
        "yaw_err": round(yaw_err, 4),
        "yaw_err_deg": round(math.degrees(yaw_err), 1),   # D048l: 显式度数，防 rad 误读
        "heading_gate_pass": bool(abs(yaw_err) <= math.radians(15.0)),  # 门=task_gate_flat04 的 15°
        "upright_final": round(upright_final, 3),
        "ended": ended,
        # D048f 阶段0 新增键
        "vx_rmse": round(vx_rmse, 4) if vx_rmse is not None else None,
        "yaw_err_int": round(yaw_err_int, 3),
        "lat_max": round(lat_max, 3),
        "fwd_max": round(fwd_max, 3),
        "task_success": task_success,
        "init": init_fp,
    }


def _agg_res_diag(diags):
    """D048h: cross-rollout mean of per-rollout res_diag blocks.

    Scalars (and quantiles) are averaged across rollouts; res_joint_mean and
    res_joint_signed_mean are averaged element-wise (|r| vs signed caliber,
    kept separate); sub-blocks missing in every rollout stay None.
    """
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 6) if xs else None

    if not diags:
        return None
    agg = {"steps": _mean([d.get("steps") for d in diags])}
    for k in ("res_raw_abs", "res_dq_abs", "res_step_abs"):
        blocks = [b for b in (d.get(k) for d in diags) if isinstance(b, dict)]
        agg[k] = (
            {kk: _mean([b[kk] for b in blocks]) for kk in blocks[0]}
            if blocks
            else None
        )
    agg["sat_frac"] = _mean([d.get("sat_frac") for d in diags])
    agg["near_sat_frac"] = _mean([d.get("near_sat_frac") for d in diags])
    jm = [x for x in (d.get("res_joint_mean") for d in diags) if x]
    agg["res_joint_mean"] = (
        [round(sum(col) / len(col), 6) for col in zip(*jm)] if jm else None
    )
    sm = [x for x in (d.get("res_joint_signed_mean") for d in diags) if x]
    agg["res_joint_signed_mean"] = (
        [round(sum(col) / len(col), 6) for col in zip(*sm)] if sm else None
    )
    return agg


def main():
    cli = build_args().parse_args()

    # D048f ⑧：缺 checkpoint 在 AppLauncher 之前快速失败，不烧 GPU 启动。
    # v1 行为 = WARNING 后用未训练 policy 跑完全程并写出看似合法的 JSON
    # （"退回随机模型"），这是显式失败纪律要堵的坑。
    if not cli.init_policy:
        if cli.checkpoint is None:
            raise SystemExit("[eval] pass --checkpoint or --init-policy")
        if not os.path.isfile(cli.checkpoint):
            raise SystemExit(f"[eval] FATAL: checkpoint not found: {cli.checkpoint}")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = 1
    launcher_args.headless = cli.headless
    launcher_args.env_spacing = 4.0
    launcher_args.output_dir = str(Path(cli.out).parent)
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    from apt_g1.encoder import Command
    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.apt_flat_env_vanilla import (
        AptFlatG1VanillaEnv,
        AptFlatG1VanillaEnvCfg,
    )
    from apt_g1.isaac import ckpt_identity
    from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg
    from apt_g1.isaac.ppo_core import AptPPOPolicy

    # Seed before policy/env construction: AptFlatG1Env._reset_idx samples the
    # walk-clock phase from torch, and --sample consumes the same global RNG.
    # Previously the per-rollout seed only controlled numpy reset jitter.
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)

    if cli.env == "vanilla":
        cfg = AptFlatG1VanillaEnvCfg()
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space, aux_dim=29, use_phase=False
        ).to("cuda:0")
    elif cli.decft:
        from apt_g1.isaac.decft_policy import DecFtPolicy, OBS_DIM as DECFT_OBS_DIM

        cfg = AptFlatG1EnvCfg()
        cfg.observation_space = DECFT_OBS_DIM
        policy = DecFtPolicy(
            obs_dim=cfg.observation_space,
            vae_path=cli.latent_vae_path,
            decoder_path=cli.decoder_path,
            vx_max=0.8,
        ).to("cuda:0")
    else:
        cfg = AptFlatG1EnvCfg()
        if cli.latent_mode:
            cfg.observation_space += 14  # _last_phase 2 -> 16 in the observation
        if cli.token_mode:
            cfg.observation_space += 62  # _last_phase 2 -> 64 (raw action feedback)
            if cli.token_phase_obs:
                cfg.observation_space += 2  # [sin phi, cos phi] walk clock
        cfg.use_elevation = bool(cli.use_elevation)
        if cfg.use_elevation:
            cfg.observation_space += cfg.elev_grid * cfg.elev_grid
        if cli.latent_residual:
            cfg.observation_space += 29  # residual action feedback
        if cli.to_ref:
            cfg.observation_space += 12  # TO38 reference block
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space,
            aux_dim=29 if cli.latent_residual else 12,
            use_phase=not cli.latent_mode and not cli.token_mode,
            latent_dim=64 if cli.token_mode else (16 if cli.latent_mode else 0),
        ).to("cuda:0")
    cfg.scene.num_envs = 1
    cfg.terrain = make_terrain_importer_cfg(
        cli.terrain, cli.terrain_noise, seed=cli.terrain_seed
    )
    cfg.sonic_decoder_path = cli.decoder_path
    cfg.router_model_dir = cli.router_model_dir
    cfg.use_2hz_gate = bool(cli.use_2hz_gate)
    cfg.phase_mode = cli.phase_mode
    cfg.phase_anchor = cli.phase_anchor
    cfg.latent_mode = cli.latent_mode
    cfg.decft_mode = cli.decft
    cfg.latent_vae_path = cli.latent_vae_path
    cfg.latent_speed_bins = cli.latent_speed_bins
    cfg.latent_dir_bins = cli.latent_dir_bins
    cfg.latent_residual = cli.latent_residual
    cfg.token_mode = cli.token_mode
    cfg.token_phase_obs = cli.token_phase_obs
    cfg.token_alpha = cli.token_alpha
    cfg.token_bound = cli.token_bound
    cfg.token_stats = cli.token_stats
    cfg.res_scale = cli.res_scale
    cfg.res_clip = cli.res_clip
    cfg.res_stats = cli.res_stats  # D048h: env-side residual stat accumulation
    cfg.force_vbin = cli.force_vbin  # D048l: eval-only intervention; -1 = natural binning
    # D048q: parse "a,b,alpha" -> (a, b, alpha) tuple (soft speed-bin blend);
    # mutually exclusive with --force-vbin (hard bin vs soft blend).
    # Default "" -> () = no intervention (byte-identical legacy decode path).
    _force_vbin_soft = ()
    if cli.force_vbin_soft:
        _parts = [p.strip() for p in cli.force_vbin_soft.split(",")]
        if len(_parts) != 3:
            raise ValueError(f"--force-vbin-soft 需 3 个逗号分隔字段 a,b,alpha，"
                             f"收到 {cli.force_vbin_soft!r}")
        _a, _b, _alpha = int(_parts[0]), int(_parts[1]), float(_parts[2])
        if not (0 <= _a <= 2 and 0 <= _b <= 2):
            raise ValueError(f"--force-vbin-soft 档位 a/b 须 ∈{{0,1,2}}，"
                             f"收到 {cli.force_vbin_soft!r}")
        if not (0.0 <= _alpha <= 1.0):
            raise ValueError(f"--force-vbin-soft alpha 须 ∈[0,1]，"
                             f"收到 {cli.force_vbin_soft!r}")
        _force_vbin_soft = (_a, _b, _alpha)
    if _force_vbin_soft and cli.force_vbin >= 0:
        raise ValueError("--force-vbin 与 --force-vbin-soft 互斥"
                         "（硬档覆写 vs 软混覆写，D048q 口径）")
    cfg.force_vbin_soft = _force_vbin_soft
    cfg.force_dbin = cli.force_dbin  # D048q: eval-only db intervention; -1 = natural
    cfg.yaw_scale = cli.yaw_scale
    cfg.heading_scale = cli.heading_scale
    cfg.to_ref = cli.to_ref
    cfg.to_ref_npz = cli.to_ref_npz
    cfg.to_ref_obs_zero = cli.to_ref_obs_zero
    cfg.to_tau = cli.to_tau
    cfg.to_tau_w = cli.to_tau_w
    cfg.latent_cmd_phase_rate = cli.latent_cmd_phase_rate
    cfg.latent_phase_rate_ref = cli.latent_phase_rate_ref
    cfg.latent_phase_rate_max = cli.latent_phase_rate_max
    cfg.stillness_vx_scale = cli.stillness_vx_scale
    cfg.aux_scale = cli.aux_scale
    cfg.disturbance_prob = 0.0
    # E49 诊断 --contract：eval = 历史评测口径（120s 上限，实际最长 test 68s）；
    # train = 训练契约（20s 上限 = 250 控制步@50Hz，与训练 rollout 一致）
    cfg.episode_length_s = 20.0 if cli.contract == "train" else 120.0
    if cli.env == "vanilla":
        env = AptFlatG1VanillaEnv(cfg)
    else:
        env = AptFlatG1Env(cfg)
    # D048h: asset md5s computed ONCE here (identity expect + cfg 落盘共用)
    vae_md5 = ckpt_identity.file_md5(cli.latent_vae_path)
    decoder_md5 = ckpt_identity.file_md5(cli.decoder_path)
    # D048h: eval-side expect for the identity check. Only keys derivable
    # unambiguously from the eval flags are fed in. 2026-09-09 hardening: for
    # a format=1 ckpt EVERY VERIFY_KEY is required on this side too, so each
    # mode branch below must supply a derivable action_space (a missing key
    # would now be a mismatch, not a skip).
    ident_expect = {
        "res_scale": cli.res_scale,
        "res_clip": cli.res_clip,
        "latent_residual": cli.latent_residual,
        "obs_dim": int(cfg.observation_space),
        "vae_md5": vae_md5,
        "decoder_md5": decoder_md5,
    }
    if cli.latent_mode:
        # train side: action_space = 45 (z16+res29) with residual, else 16
        ident_expect["action_space"] = 45 if cli.latent_residual else 16
    elif cli.token_mode:
        ident_expect["action_space"] = 64
    elif cli.decft or cli.env == "vanilla":
        ident_expect["action_space"] = 29
    else:
        # default apt path: phase(2) + aux(12) = 14, the train default layout
        # (gate_sel/to42_sel have no eval flag). A standalone --latent-residual
        # widens the aux head to 29 -> 31, a combo no train ckpt can carry
        # (obs_dim would flag it first); kept consistent with the policy above.
        ident_expect["action_space"] = 31 if cli.latent_residual else 14

    ident_status = "legacy"  # "matched" | "legacy" (no identity block to check)
    ckpt_ident = None
    if cli.init_policy:
        # 未训练初始化对照：直接用按现有旗标新构造的 policy（确定性模式照旧）
        policy.eval()
        print("[eval] --init-policy: using freshly initialized policy "
              "(no checkpoint loaded)")
    else:
        try:
            sd, ckpt_ident = ckpt_identity.load_ckpt(
                cli.checkpoint, map_location="cuda:0"
            )
            policy.load_state_dict(sd)
            policy.eval()
        except Exception as exc:
            # 加载失败（文件损坏/权重结构不符/资产错配）= 显式失败，绝不
            # 退回随机模型继续跑。
            print(f"[eval] FATAL: checkpoint load failed: {exc!r}", flush=True)
            os._exit(3)
        print("[eval] loaded checkpoint", cli.checkpoint)
        if ckpt_ident is not None:
            # D048h：配置身份核验——训练/评测失配（res_scale 0.4 vs 0.15 类
            # 事故）在跑任何一局之前拒绝（exit 6）。
            mismatch = ckpt_identity.verify_ckpt_identity(ckpt_ident, ident_expect)
            if mismatch:
                for m in mismatch:
                    print(f"[eval] FATAL: ckpt identity mismatch: {m}", flush=True)
                os._exit(6)
            ident_status = "matched"
            print(
                "[eval] ckpt_identity matched: "
                f"entry={ckpt_ident.get('entry')} it={ckpt_ident.get('it')} "
                f"git_head={ckpt_ident.get('git_head')}",
                flush=True,
            )
        else:
            print("[eval] WARNING: ckpt_identity: legacy（无身份块，跳过配置核验）",
                  flush=True)

    # initial obs
    env._vanilla = cli.env == "vanilla"
    env._decft = cli.decft
    env._latent_residual = cli.latent_residual
    obs_dict, _ = env.reset()
    env._last_obs = obs_dict["policy"]

    # E49: action mode annotation ("det" = mean action [历史行为], "sample" =
    # policy-distribution sampling, training-rollout 口径)
    mode = "sample" if cli.sample else "det"
    print(f"[eval] act mode = {mode}", flush=True)
    print(f"[eval] contract = {cli.contract} "
          f"(episode_length_s={cfg.episode_length_s}, "
          f"num_rollouts={cli.num_rollouts})", flush=True)
    seeds = list(range(cli.seed, cli.seed + cli.num_rollouts))

    # D048h: --res-stats wrapper -- reset the env-side accumulators at rollout
    # start, harvest res_diag at rollout end (executed-residual口径; the
    # noaux/aux_zero zero-residual control arms accumulate all-zero stats,
    # which is exactly the paired diagnostic)
    res_stats_on = cli.res_stats and hasattr(env, "reset_res_stats")

    def do_rollout(*args, **kw):
        if res_stats_on:
            env.reset_res_stats()
        r = rollout(*args, **kw)
        if res_stats_on:
            r["res_diag"] = env.pop_res_stats()
        return r

    tests = set(cli.tests.split(","))
    out = {"A_walk60": {}, "B_disturbance": {}, "C_switch": {}, "D_jump": {}}
    pp = cli.phase_mode
    pz = cli.phase_zero
    az = cli.aux_zero
    lp = cli.latent_mode or cli.token_mode
    if cli.decft:
        # single policy key (no aux/noaux split; D needs the router mode path)
        key_list = ["aux"]
        if "D" in tests:
            tests.discard("D")
    else:
        key_list = ["aux", "noaux"] if not pp else ["phaseaux"]

    # ---- A. 60s walk @ 0.8 ----

    if "A" in tests:
        out["A_walk60"]["cmd_vx"] = cli.a_cmd_vx
        for key in key_list:
            out["A_walk60"][key] = {}
            use_aux = key != "noaux"
            for seed in seeds:
                sched = [(cli.a_cmd_vx, 0.0, 60)]
                if cli.cmd_sample:
                    # E49 --cmd-sample：每局独立抽 vx（按 seed 派生 RNG 可复现）；
                    # 只影响本局命令，cmd_vx 记进该 rollout 的 JSON 条目
                    vx_cmd = float(
                        np.random.default_rng(2000 + seed).uniform(0.0, cli.a_cmd_vx)
                    )
                    sched = [(vx_cmd, 0.0, 60)]
                r = do_rollout(env, policy, sched, seed, use_aux, phase_policy=pp, phase_zero=pz, aux_zero=az, latent_policy=lp, yaw_bias=cli.yaw_bias_comp, sample=cli.sample, contract=cli.contract)
                if cli.cmd_sample:
                    r["cmd_vx"] = sched[0][0]
                out["A_walk60"][key][f"seed{seed}"] = r
                print(f"A walk{sched[0][0]} {key} seed{seed} mode={mode} contract={cli.contract} done={r['completed']} h_min={r['h_min']} vx={r['vx']} disp={r['disp']}", flush=True)

    # ---- B. disturbance grid ----

    if "B" in tests:
        dirs = {
            "fwd": [cli.impulse_n, 0, 0],
            "back": [-cli.impulse_n, 0, 0],
            "left": [0, cli.impulse_n, 0],
            "right": [0, -cli.impulse_n, 0],
        }
        for key in key_list:
            out["B_disturbance"][key] = {}
            use_aux = key != "noaux"
            for dname, dvec in dirs.items():
                for seed in seeds:
                    imp = [(500, dvec), (1250, dvec)]
                    r = do_rollout(env, policy, [(0.8, 0.0, 45)], seed, use_aux, impulses=imp, phase_policy=pp, phase_zero=pz, aux_zero=az, latent_policy=lp, yaw_bias=cli.yaw_bias_comp, sample=cli.sample, contract=cli.contract)
                    out["B_disturbance"][key][f"{dname}_seed{seed}"] = r
                    print(f"B {dname} {key} seed{seed} mode={mode} done={r['completed']} h_min={r['h_min']}", flush=True)

    # ---- C. command-switch marathon ----

    if "C" in tests:
        sched = [
            (0.0, 0.0, 5), (0.8, 0.0, 8), (0.0, 0.0, 3), (-0.8, 0.0, 6),
            (0.0, 0.0, 3), (0.25, 0.0, 6), (0.0, 0.0, 3), (0.25, -0.43, 6),
            (0.0, 0.0, 3), (0.25, 0.43, 6), (0.0, 0.0, 3), (0.8, 0.0, 8),
        ]
        for key in key_list:
            out["C_switch"][key] = {}
            use_aux = key != "noaux"
            for seed in seeds:
                r = do_rollout(env, policy, [(vx, vy, s) for vx, vy, s in sched], seed, use_aux, phase_policy=pp, phase_zero=pz, aux_zero=az, latent_policy=lp, yaw_bias=cli.yaw_bias_comp, sample=cli.sample, contract=cli.contract)
                out["C_switch"][key][f"seed{seed}"] = r
                print(f"C switch {key} seed{seed} mode={mode} done={r['completed']} fall={r['fall_step']} h_min={r['h_min']}", flush=True)

    # ---- D. jump (explicit mode) ----

    if "D" in tests:
        jump_cmd = Command(
            mode=17, speed=-1.0,
            mdir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            fdir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        for key in key_list:
            out["D_jump"][key] = {}
            use_aux = key != "noaux"
            for seed in seeds:
                r = do_rollout(env, policy, [(jump_cmd, 20)], seed, use_aux, phase_policy=pp, phase_zero=pz, aux_zero=az, latent_policy=lp, yaw_bias=cli.yaw_bias_comp, sample=cli.sample, contract=cli.contract)
                out["D_jump"][key][f"seed{seed}"] = r
                print(f"D jump {key} seed{seed} mode={mode} done={r['completed']} h_min={r['h_min']} vx={r['vx']}", flush=True)

    if tests != {"A", "B", "C", "D"}:
        out = {k: v for k, v in out.items() if k.split("_")[0] in tests}
    # E49: 顶层动作模式标注（放在 tests 过滤之后，避免被滤掉）
    out["mode"] = mode
    # E49 诊断: 评测契约标注（"eval" = 历史口径 / "train" = 训练契约 20s）
    out["contract"] = cli.contract
    out["seed"] = cli.seed
    # D048e 评测契约 v2 标注（仓库先例 = train_log 的 rew_contract 三键；旧
    # JSON 无此键 = v1 口径）+ res_scale 落盘：训练/评测必须同值（训练 0.15，
    # CLI 默认 0.4 = 残差放大 2.67 倍，D048d 阶梯评测即栽在此），落盘防再犯。
    out["eval_contract"] = 2
    out["res_scale"] = cli.res_scale
    # D048l：单位防误读 + 评测干预显式记录（§5f 预注册）。分档边界与
    # apt_flat_env._compute_q_des 同式（linspace(0, vx_max, n_bins+1) 去首尾），
    # vx_max / n_bins 取 env cfg 现值（默认 0.8 / 3），避免字面量漂移。
    out["units"] = {"yaw_err": "rad", "yaw_err_deg": "deg", "yaw_err_int": "rad*s"}
    _fvb = cli.force_vbin if cli.force_vbin >= 0 else None
    _fdb = cli.force_dbin if cli.force_dbin >= 0 else None
    _edges = np.linspace(0.0, float(cfg.vx_max), int(cfg.latent_vae_n_bins) + 1)[1:-1]
    _nvb = min(
        max(int(np.searchsorted(_edges, cli.a_cmd_vx, side="left")), 0),
        int(cfg.latent_vae_n_bins) - 1,
    )
    # D048q 扩展：soft 介入记 soft_pair/alpha（forced_vbin 保持 null，两者互斥）；
    # dbin 介入加 forced_dbin。默认路径键集与旧版逐键一致（note="none"）。
    _iv = {
        "forced_vbin": _fvb,
        "natural_vb": _nvb,
        "cmd_vx": cli.a_cmd_vx,
    }
    if _force_vbin_soft:
        _iv["soft_pair"] = [int(_force_vbin_soft[0]), int(_force_vbin_soft[1])]
        _iv["alpha"] = float(_force_vbin_soft[2])
    if _fdb is not None:
        _iv["forced_dbin"] = _fdb
    if _fvb is not None:
        _iv["note"] = "D048l: forced_vbin 非 None 时为评测干预（覆写 VAE 速度档条件输入），非自然行为"
    elif _force_vbin_soft:
        _iv["note"] = "D048q: force_vbin_soft 评测干预（decode speed_embed 软混 alpha·E[a]+(1−alpha)·E[b]，覆写速度档条件），非自然行为"
    elif _fdb is not None:
        _iv["note"] = "D048q: forced_dbin 评测干预（覆写 VAE 方向档条件输入，仅带 db 的 latent-dir-bins 分支生效），非自然行为"
    else:
        _iv["note"] = "none"
    out["eval_interventions"] = _iv
    # D048f 阶段0：指标扩展标注 + 冻结任务门常量落盘（评测自描述，判读方
    # 不必回查文档；门值来自 DS_CONTINUOUS_EXECUTION_PLAN §5，先于候选
    # 结果冻结）。D048h --res-stats 开启时翻转为 resdiag 契约（默认路径
    # 保持 "d048f_stage0" 不变）。
    out["metrics_contract"] = "d048h_resdiag" if cli.res_stats else "d048f_stage0"
    out["task_gate_flat04"] = {
        "cmd_vx": 0.4,
        "window_s": 20,
        "vx_rmse_max": 0.10,
        "fwd_min": 6.0,
        "fwd_max": 10.0,
        "lat_abs_max": 0.5,
        "yaw_err_abs_max_deg": 15.0,
        "upright_min": 0.90,
    }
    # D048h：ckpt 身份核验结果落盘（matched = 新格式 ckpt 且 expect 全匹配；
    # legacy = 旧格式纯 state_dict / --init-policy，未做核验）。checks 左半 =
    # eval 侧期望值（与 ident_expect 同源），it/entry/git_head = ckpt 侧身份
    # （legacy 时 None）。失配根本走不到这里（已在加载后立即 exit 6）。
    out["ckpt_identity"] = {
        "status": ident_status,
        "checkpoint": cli.checkpoint,
        "checks": {
            "res_scale": cli.res_scale,
            "res_clip": cli.res_clip,
            "latent_residual": cli.latent_residual,
            "obs_dim": int(cfg.observation_space),
            "action_space": ident_expect.get("action_space"),
            "vae_md5": vae_md5,
            "decoder_md5": decoder_md5,
            "it": (ckpt_ident or {}).get("it"),
            "entry": (ckpt_ident or {}).get("entry"),
            "git_head": (ckpt_ident or {}).get("git_head"),
        },
    }
    if res_stats_on:
        # D048h：跨局聚合的残差执行统计（各标量对局取均值；分位取均值近似、
        # sat/near_sat_frac 均值、res_joint_mean / res_joint_signed_mean 逐
        # 关节均值，|r| 与带符号两口径分开聚合）
        _diags = [
            r["res_diag"]
            for grp in out.values()
            if isinstance(grp, dict)
            for cell in grp.values()
            if isinstance(cell, dict)
            for r in cell.values()
            if isinstance(r, dict) and isinstance(r.get("res_diag"), dict)
        ]
        out["res_diag"] = _agg_res_diag(_diags)
    # D048f ⑧：配置身份落盘（消费了什么就记什么；VAE/代码的 md5 让"资产
    # 身份一致"可以事后核验，不依赖启动脚本注释）。

    def _fmd5(path):
        try:
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return None

    code_md5 = {"eval_apt_isaac.py": _fmd5(__file__)}
    try:
        import apt_g1.isaac.apt_flat_env as _afe

        code_md5["apt_flat_env.py"] = _fmd5(_afe.__file__)
    except Exception:
        pass
    out["cfg"] = {
        "obs_dim": int(cfg.observation_space),
        "aux_dim": 29 if cli.latent_residual else 12,
        "contract": cli.contract,
        "episode_length_s": cfg.episode_length_s,
        "a_cmd_vx": cli.a_cmd_vx,
        "num_rollouts": cli.num_rollouts,
        "seeds": seeds,
        "sample": cli.sample,
        "cmd_sample": cli.cmd_sample,
        "aux_zero": cli.aux_zero,
        "init_policy": cli.init_policy,
        "checkpoint": cli.checkpoint,
        "res_scale": cli.res_scale,
        "res_clip": cli.res_clip,
        "res_stats": cli.res_stats,
        "latent_mode": cli.latent_mode,
        "latent_residual": cli.latent_residual,
        "latent_speed_bins": cli.latent_speed_bins,
        "latent_dir_bins": cli.latent_dir_bins,
        "latent_vae_path": cli.latent_vae_path,
        "vae_md5": vae_md5,  # D048h: computed once at the load block
        "decoder_path": cli.decoder_path,
        "decoder_md5": decoder_md5,  # D048h
        "terrain": cli.terrain,
        "impulse_n": cli.impulse_n,
        "code_md5": code_md5,
    }
    with open(cli.out, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", cli.out)
    # D048f ⑧：零局产出 = 显式失败（JSON 保留作证据，退出码非零）。
    n_eps = sum(
        1
        for grp in out.values()
        if isinstance(grp, dict)
        for cell in grp.values()
        if isinstance(cell, dict)
        for r in cell.values()
        if isinstance(r, dict) and "steps" in r
    )
    if n_eps == 0:
        print("[eval] FATAL: no rollouts produced", flush=True)
        os._exit(4)
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        # Isaac kit 接管 sys.excepthook 后，未捕获异常可能仍以 0 退出
        # （SERVER_GUIDE §7 det 判读口径"退出码不可信"）；此处显式拦截，
        # 保证失败可见。
        import traceback

        traceback.print_exc()
        os._exit(1)
