"""Train the APT (phase-router prior + aux) policy in Isaac Lab.

Controller line on Isaac: frozen phase-router token prior + PPO-trained aux,
with the paper's RL-stage mechanisms:
  - latent KL w.r.t. N(0, I) (coef 2.5e-6)
  - latent exploration bonus decaying to zero
  - 2 Hz gait-gate hold + feedback observation (cfg.use_2hz_gate)

Usage (from the repo root on the training server):
    PYTHONPATH=~/ros2_data/apt_g1:~/ros2_data/GR00T-WholeBodyControl \\
      python ~/ros2_data/apt_g1/isaac/train_apt_isaac.py \\
        --num-envs 64 --iters 500 --out outputs/isaac_apt_aux
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--rollout", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vx-max", type=float, default=0.8)
    # E34: randomize commanded yaw during training (domain randomization) so
    # the policy learns to steer toward the commanded heading. Default 0,0 =
    # E31 behavior (constant yaw=0, which caused the systematic drift).
    ap.add_argument("--yaw-min", type=float, default=0.0)
    ap.add_argument("--yaw-max", type=float, default=0.0)
    ap.add_argument("--disturbance-prob", type=float, default=0.0)
    ap.add_argument("--disturbance-ramp-iters", type=int, default=0)
    ap.add_argument("--use-2hz-gate", type=int, default=1)
    ap.add_argument("--phase-mode", action="store_true")
    ap.add_argument("--phase-anchor", action="store_true")
    ap.add_argument("--latent-mode", action="store_true")
    # E44: decoder fine-tuning. Action space = 29-d joint targets; the policy
    # owns a trainable SONIC decoder (E39 latent -> VAE -> token -> decoder).
    ap.add_argument("--decft", action="store_true")
    ap.add_argument("--decoder-reg", type=float, default=1.0)
    # E44: separate (smaller) LR for the fine-tuned decoder; None = same LR
    ap.add_argument("--decoder-lr", type=float, default=None)
    # E44v2: weight-space anchor + action noise scale (guards against the
    # decoder drifting off the official manifold, which broke E44 v1)
    ap.add_argument("--decoder-wreg", type=float, default=0.0)
    ap.add_argument("--decft-aux-std", type=float, default=-2.0)
    # E44 two-phase: phase 1 trains z against the FROZEN official decoder
    ap.add_argument("--freeze-decoder", action="store_true")
    # E44p1-fix: z-noise scale (phase_init_std); larger = real z exploration so
    # the z-head gets a usable score-function gradient through the decoder
    ap.add_argument("--decft-phase-std", type=float, default=-4.0)
    ap.add_argument(
        "--latent-vae-path",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/token_vae_e27/vae.pt",
    )
    ap.add_argument("--latent-warmstart-iters", type=int, default=0)
    ap.add_argument("--latent-kl", type=float, default=2.5e-6)
    ap.add_argument("--latent-expl", type=float, default=0.01)
    # E28: command-conditioned gait cadence + forward-speed reward shaping
    ap.add_argument("--latent-cmd-phase-rate", action="store_true")
    ap.add_argument("--latent-phase-rate-ref", type=float, default=0.6)
    ap.add_argument("--latent-phase-rate-max", type=float, default=2.0)
    ap.add_argument("--stillness-vx-scale", type=float, default=0.05)
    # E29: latent KL prior. "zero" = N(0,I) (E27); "walk" = N(z_walk, I) keeps z
    # on the SONIC walk manifold instead of pulling it toward the origin.
    ap.add_argument("--latent-kl-prior", choices=["zero", "walk"], default="zero")
    # E49: direct-token RL (no VAE). A arm = raw unbounded token coordinates
    # mapped onto official token stats; B arm adds the walk clock [sin, cos]
    # to the policy obs (attribution arm).
    ap.add_argument("--token-mode", action="store_true")
    ap.add_argument("--token-phase-obs", action="store_true")
    ap.add_argument("--token-alpha", type=float, default=1.0)
    ap.add_argument("--token-bound", choices=["none", "tanh"], default="none",
                    help="tanh = restricted-range ablation (anti-drift stabilizer)")
    ap.add_argument("--token-stats", type=str, default="",
                    help="npz with mean/std/rate from official g1-mode tokens")
    # E31: speed-conditioned VAE decoder (D(z, phase, v_bin) -> token)
    ap.add_argument("--latent-speed-bins", action="store_true")
    # E35: direction+speed-conditioned VAE decoder (D(z,phi,v_bin,psi_bin))
    ap.add_argument("--latent-dir-bins", action="store_true")
    # E48: full-joint residual escape channel (RuN/ReSkill-style). Action
    # becomes [z(16), res(29)]; q_des = q_decoder(z) + res_scale*clamp(res).
    ap.add_argument("--latent-residual", action="store_true")
    ap.add_argument("--res-scale", type=float, default=0.4)
    ap.add_argument("--res-clip", type=float, default=1.0)
    ap.add_argument("--res-l2", type=float, default=0.0)
    # E48c: freeze the residual (zeroed in the env) for the first N control
    # steps so the z-head first learns a working controller on terrain.
    ap.add_argument("--res-freeze-steps", type=int, default=0)
    # D049b-fix：连续 vb 软权重臂（DS_CONTINUOUS_EXECUTION_PLAN §5j 主臂）。
    # 动作末 3 维 = vb 连续动作（act 高斯采样自 vb_logits；softmax=w 进 decode
    # 的 vb_soft，det 模式=vb_logits 即原确定性 w）+ obs +3（w 反馈）。
    # 骑在 --latent-mode 上（需带 vb 条件的 --latent-speed-bins/--latent-dir-bins
    # 分支）；与 latent_residual **可组合**（§5j b 臂=D048i 配方逐字：
    # [z(16), res(29), vb(3)]=48，唯一自变量=vb 来源），plain latent 组合
    # = [z(16), aux(12), vb(3)]=31。与 token/decft/gate-sel/to42 互斥
    # （下方 assert + env __init__ 双层校验）
    ap.add_argument("--vb-from-policy", action="store_true",
                    help="D049b: policy 3-dim vb action (gaussian sample of "
                         "vb_logits; det=vb_logits) -> softmax soft weights "
                         "w consumed as decode vb_soft; obs +3 (w feedback); "
                         "action 48-dim with --latent-residual (D048i recipe), "
                         "31-dim plain latent")
    ap.add_argument("--vb-init-std", type=float, default=-4.0,
                    help="D049b-fix: vb_log_std 初始值（log 尺度）。默认 -4.0"
                         "=z/res 同款约定；消费者测试门 G2 在 -1.0 验证信用"
                         "分配（σ=e^-1 使 softmax 软权重有实质探索变异，"
                         "σ=e^-4 时 w 近确定、score-function 信号过弱），"
                         "复验 b 臂按门验证配置取 -1.0")
    # D051：BC 热启动 vb 头（DS_CONTINUOUS_EXECUTION_PLAN §5k 预注册）。
    # 主训练循环前先教 vb_logits「obs 里的 vx 命令 → 目标软权重 w*」映射
    #（只动 vb_logits 线性层，encoder/z/res/vb_log_std 全冻结），再验 PPO
    # 能否接住。默认 0 = 关（D049-R1 对照臂行为逐位不变）
    ap.add_argument("--vb-bc-warmup-steps", type=int, default=0,
                    help="D051: BC warmup steps for the vb head before PPO "
                         "(default 0 = off; requires --vb-from-policy)")
    ap.add_argument("--vb-bc-lr", type=float, default=1e-3,
                    help="D051: BC warmup Adam lr (vb_logits linear layer only)")
    ap.add_argument("--vb-bc-anchors", type=str, default="0.29,0.47,1.31",
                    help="D051: comma-separated speed anchors (m/s) of the BC "
                         "target w*(cmd)")
    # D053 obs-head 臂（DS_CONTINUOUS_EXECUTION_PLAN §5m）：obs 追加 2 维
    # [sin(yaw_rel), cos(yaw_rel)]（yaw_rel = wrap_pi(当前 yaw − episode
    # 初始 yaw)，与 eval yaw_err 同源提取）。默认 0 = 关（obs 维度/路径/
    # 日志逐字节不变）；开启时 obs 137→139（D051 b 臂配方）
    ap.add_argument("--obs-heading", type=int, default=0,
                    help="D053: append heading observability block "
                         "[sin(yaw_rel), cos(yaw_rel)] (+2 obs dims; default "
                         "0 = off, byte-identical legacy path)")
    # E32: heading/yaw reward strengthening (fights high-speed drift)
    ap.add_argument("--yaw-scale", type=float, default=0.5)
    ap.add_argument("--heading-scale", type=float, default=0.0)
    ap.add_argument("--to-ref", action="store_true",
                    help="TO38: append the 12-d TO reference obs block (zeros without --to-ref-npz)")
    ap.add_argument("--to-ref-npz", default="",
                    help="TO38: LUT from apt_g1/to38_export_ref.py (empty = zero block, paired control arm)")
    ap.add_argument("--to-ref-obs-zero", action="store_true",
                    help="TO38: control arm -- load LUT (clock/diagnostics) but zero the obs block")
    ap.add_argument("--to-ref-w", type=float, default=0.0, help="TO38: tracking reward weight")
    ap.add_argument("--to-ref-sigma2", type=float, default=0.1)
    ap.add_argument("--to-ref-gate-sigma2", type=float, default=0.0036)
    ap.add_argument("--to-tau", action="store_true",
                    help="TO40-C: cmd-gated torque feedforward from the LUT (no obs change)")
    ap.add_argument("--to-tau-w", type=float, default=1.0)
    ap.add_argument("--aux-scale", type=float, default=0.2)
    ap.add_argument("--aux-l2", type=float, default=0.0)
    ap.add_argument("--aux-rate", type=float, default=0.0)
    ap.add_argument("--yaw-sigma2", type=float, default=0.25)
    ap.add_argument("--vel-sigma2", type=float, default=0.25)
    ap.add_argument("--phase-warmstart-iters", type=int, default=0)
    ap.add_argument("--phase-warmstart-coef", type=float, default=10.0)
    ap.add_argument("--entropy", type=float, default=0.001)
    # D048b：res（aux）头熵奖励独立系数。latent_residual 的 aux=29d 执行残差，
    # 共享 entropy_coef 会持续推高残差探索噪声（E48 破坏模式种子之一）；
    # 0 = res 头不享受熵 bonus（log_prob/信任域不受影响）。默认 1.0 = 历史行为
    ap.add_argument("--res-ent-coef", type=float, default=1.0)
    # TO42 修订 v4（论文式大并行操作点）：2048 envs × 500it 配 minibatch 4096
    # （24×2048/4096 = 12 minibatch/epoch，整除）；默认 512 = 既有行为逐字不变
    ap.add_argument("--ppo-minibatch", type=int, default=512)
    # E49 修复：update() 改为真 epoch 循环后把 epoch 数暴露到 CLI；
    # 默认 1 = 历史 run 的实际行为（旧 update 从未循环，单遍）
    ap.add_argument("--ppo-epochs", type=int, default=1,
                    help="PPO epochs per update (historical behavior = single pass)")
    # E49-C：KL 信任域守卫（默认 None = 完全关闭 = 冻结版行为）。每 minibatch
    # step 后测新旧策略解析 KL，超阈回滚该步并缩小 lr；KL < 阈/2 时回升 lr
    # （上限钉在初始 lr）；连续回滚达 max-rolls 提前结束整个 update 循环
    ap.add_argument("--kl-guard", type=float, default=None,
                    help="E49-C: KL trust-region threshold (default None = off)")
    ap.add_argument("--kl-guard-shrink", type=float, default=0.8,
                    help="E49-C: lr multiplier on rollback (TRPO backtracking)")
    ap.add_argument("--kl-guard-grow", type=float, default=1.2,
                    help="E49-C: lr multiplier when KL < threshold/2 (1.0 = off)")
    ap.add_argument("--kl-guard-max-rolls", type=int, default=3,
                    help="E49-C: consecutive rollbacks before ending the update loop")
    # D048i：PPO 更新约束臂——步级解析 KL 看门 + 资格检查。默认关 = 完全保留
    # 历史行为；与 --kl-guard 互斥（同时开启 argparse 直接报错退出，见 main）
    ap.add_argument("--kl-step-guard", action="store_true",
                    help="D048i: per-step analytic-KL watchdog on PPO updates")
    ap.add_argument("--kl-step-target", type=float, default=0.05,
                    help="D048i: joint analytic KL(old||new) budget per optimizer step")
    ap.add_argument("--kl-backtracks", type=int, default=6,
                    help="D048i: lr-halving retries before a step is rejected")
    ap.add_argument("--qual-ratio-tol", type=float, default=1e-3,
                    help="D048i: max |exp(dlogp)-1| tolerance for the qualification check (exit 8)")
    ap.add_argument("--dead-rounds", type=int, default=5,
                    help="D048i: consecutive zero-accepted updates before fatal stop (exit 7)")
    # E49-C v2：短探针迭代上限。只截短训练循环（含末 iter 日志/ckpt 边界），
    # trainer.max_iters 仍 = --iters → expl_coef 等调度长度不被压缩（配方公平性）
    ap.add_argument("--probe-iters", type=int, default=None,
                    help="E49-C v2: cap training-loop iterations without shrinking schedules")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/isaac_apt_aux")
    ap.add_argument("--env", choices=["apt", "vanilla"], default="apt")
    ap.add_argument("--terrain", choices=["plane", "rough", "rough_paper", "rough_sym"], default="plane")
    ap.add_argument("--terrain-noise", type=float, default=0.04)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--use-elevation", type=int, default=0)
    ap.add_argument("--gate-sel", type=int, default=0)
    # TO42: learned regime selection on the frozen decoder substrate
    # (TO42_PLAN §3；骑在 --latent-mode 上；action 16→17，obs +2)
    ap.add_argument("--to42-sel", choices=["off", "lsel", "fbkt"], default="off")
    ap.add_argument("--to42-hold-steps", type=int, default=25)
    ap.add_argument("--progress-scale", type=float, default=0.0)
    # D048j: progress 项封顶上界从 1.0 改为当前命令（静态最优恰为 cmd，
    # 消除结构性超速偏置）；默认关 = 与旧公式逐位一致（REW_CONTRACT_VER 2；
    # 开启 = 3）
    ap.add_argument("--progress-cap-cmd", action="store_true")
    ap.add_argument("--anti-stop", type=float, default=0.0)
    ap.add_argument("--anti-stop-thresh", type=float, default=0.3)
    # E44v3: penalty on yaw-rate (omega_z^2) to suppress the spin gait
    ap.add_argument("--yaw-rate-penalty", type=float, default=0.0)
    ap.add_argument("--resume", default=None)
    # 热循环去同步：fused Adam（CUDA 融合实现）；默认 False 保持 E 系列历史 run 可比
    ap.add_argument("--fused-adam", action="store_true")
    # 每 N iter 打一行 [SPEED] 墙钟日志；0 = 关闭
    ap.add_argument("--speed-log-interval", type=int, default=50)
    # E49 诊断步骤③：窗口一致的奖励分项/位移诊断。env 侧 _last_rew_terms
    # 快照 + 训练侧逐控制步 GPU 累积、iter 末一次同步换算，统计窗口与
    # mean_rew 全同（修复旧口径 rew 全窗 vs fwd 末时刻的窗口错位）
    ap.add_argument("--diag-log", action="store_true",
                    help="E49: per-term reward / motion diagnostics (d_* hist keys)")
    ap.add_argument(
        "--router-model-dir",
        default="/home/cvgluser/ros2_data/apt_g1/outputs/distill_final",
    )
    ap.add_argument(
        "--decoder-path",
        default=(
            "/home/cvgluser/ros2_data/GR00T-WholeBodyControl/"
            "gear_sonic_deploy/policy/release/model_decoder.onnx"
        ),
    )
    return ap


def main():
    ap = build_args()

    # AppLauncher args (must be created before any isaaclab import)
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    cli = ap.parse_args()
    # D048i：两机制互斥——kl-guard（E49-C，曾锁死学习）与 kl-step-guard
    # （D048i）同时开启属配置错误，argparse 直接报错退出（exit 2）
    if cli.kl_step_guard and cli.kl_guard is not None:
        ap.error(
            "--kl-step-guard and --kl-guard are mutually exclusive "
            "(D048i step-level watchdog vs E49-C rollout-referenced guard)"
        )
    launcher_args.num_envs = cli.num_envs
    launcher_args.headless = cli.headless
    launcher_args.env_spacing = 4.0
    launcher_args.output_dir = cli.out
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    # safe to import isaaclab-dependent modules now
    from apt_g1.isaac.apt_flat_env import AptFlatG1Env, AptFlatG1EnvCfg
    from apt_g1.isaac.apt_flat_env_vanilla import (
        AptFlatG1VanillaEnv,
        AptFlatG1VanillaEnvCfg,
    )
    from apt_g1.isaac import ckpt_identity
    from apt_g1.isaac.ppo_core import AptPPOPolicy, PPOTrainer, vb_bc_loss
    from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    out_dir = Path(cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # E49-C：启动配置回显（此前版本无完整回显行；关键 PPO 参数 + 守卫四参数
    # + token-stats 路径，便于日志取证）。D048i：kl-step-guard 开启时追加五参数
    ksg_cfg = (
        f" kl_step_guard=1 kl_step_target={cli.kl_step_target} "
        f"kl_backtracks={cli.kl_backtracks} "
        f"qual_ratio_tol={cli.qual_ratio_tol} dead_rounds={cli.dead_rounds}"
        if cli.kl_step_guard
        else ""
    )
    # D048j：旗标关时回显逐字不变
    pcap_cfg = " progress_cap_cmd=1" if cli.progress_cap_cmd else ""
    # D049b：旗标关时回显逐字不变
    vb_cfg = " vb_from_policy=1" if cli.vb_from_policy else ""
    # D053：旗标关时回显逐字不变
    heading_cfg = " obs_heading=1" if cli.obs_heading else ""
    if cli.vb_bc_warmup_steps > 0:
        # D051：预热配置回显（发射链 smoke 检查 w 校准快照用；>0 蕴含
        # vb_from_policy=1，旗标关路径回显仍逐字不变）
        vb_cfg += (
            f" vb_bc_warmup_steps={cli.vb_bc_warmup_steps}"
            f" vb_bc_lr={cli.vb_bc_lr} vb_bc_anchors={cli.vb_bc_anchors!r}"
        )
    print(
        f"[CFG] out={cli.out} num_envs={cli.num_envs} iters={cli.iters} "
        f"probe_iters={cli.probe_iters} "
        f"lr={cli.lr} ppo_epochs={cli.ppo_epochs} minibatch={cli.ppo_minibatch} "
        f"token_stats={cli.token_stats!r} "
        f"kl_guard={cli.kl_guard} kl_shrink={cli.kl_guard_shrink} "
        f"kl_grow={cli.kl_guard_grow} kl_max_rolls={cli.kl_guard_max_rolls}"
        f"{ksg_cfg}{pcap_cfg}{vb_cfg}{heading_cfg}",
        flush=True,
    )

    if cli.env == "vanilla":
        cfg = AptFlatG1VanillaEnvCfg()
    else:
        cfg = AptFlatG1EnvCfg()
    cfg.scene.num_envs = cli.num_envs
    cfg.terrain = make_terrain_importer_cfg(
        cli.terrain, cli.terrain_noise, seed=cli.terrain_seed
    )
    cfg.sonic_decoder_path = cli.decoder_path
    cfg.router_model_dir = cli.router_model_dir
    cfg.vx_max = cli.vx_max
    cfg.yaw_min = cli.yaw_min
    cfg.yaw_max = cli.yaw_max
    cfg.disturbance_prob = 0.0 if cli.disturbance_ramp_iters > 0 else cli.disturbance_prob
    cfg.use_2hz_gate = bool(cli.use_2hz_gate)
    cfg.use_elevation = bool(cli.use_elevation)
    if cfg.use_elevation:
        cfg.observation_space += cfg.elev_grid * cfg.elev_grid
    cfg.use_gate_sel = bool(cli.gate_sel)
    cfg.progress_scale = cli.progress_scale
    cfg.progress_cap_cmd = cli.progress_cap_cmd
    cfg.anti_stop_scale = cli.anti_stop
    cfg.anti_stop_thresh = cli.anti_stop_thresh
    cfg.yaw_rate_penalty = cli.yaw_rate_penalty
    if cfg.use_gate_sel:
        cfg.action_space = 13  # aux(12) + gate(1)
    cfg.phase_mode = cli.phase_mode
    cfg.phase_anchor = cli.phase_anchor
    cfg.latent_mode = cli.latent_mode
    cfg.token_mode = cli.token_mode
    cfg.token_phase_obs = cli.token_phase_obs
    cfg.token_alpha = cli.token_alpha
    cfg.token_bound = cli.token_bound
    cfg.token_stats = cli.token_stats
    cfg.latent_vae_path = cli.latent_vae_path
    cfg.latent_speed_bins = cli.latent_speed_bins
    cfg.latent_dir_bins = cli.latent_dir_bins
    cfg.latent_residual = cli.latent_residual
    cfg.vb_from_policy = cli.vb_from_policy  # D049b hotfix：env 侧旗标接线（86c1057 漏传，b 臂 env 恒走自然桶致 obs 105!=cfg 108 断言，smoke Exp 10559 拦截；eval 侧 853 行本就有）
    cfg.obs_heading = bool(cli.obs_heading)  # D053 obs-head：env 侧旗标接线（默认 0=零变化）
    cfg.res_scale = cli.res_scale
    cfg.res_clip = cli.res_clip
    cfg.res_l2_scale = cli.res_l2
    cfg.res_freeze_steps = cli.res_freeze_steps
    cfg.yaw_scale = cli.yaw_scale
    cfg.heading_scale = cli.heading_scale
    cfg.to_ref = cli.to_ref
    cfg.to_ref_npz = cli.to_ref_npz
    cfg.to_ref_obs_zero = cli.to_ref_obs_zero
    cfg.to_ref_w = cli.to_ref_w
    cfg.to_ref_sigma2 = cli.to_ref_sigma2
    cfg.to_ref_gate2 = cli.to_ref_gate_sigma2
    cfg.to_tau = cli.to_tau
    cfg.to_tau_w = cli.to_tau_w
    if cfg.to_ref:
        # obs block: [sin psi, cos psi, q_ref6_rel, pitch, z, heel_x_rel, heel_z]
        cfg.observation_space += 12
    cfg.latent_cmd_phase_rate = cli.latent_cmd_phase_rate
    cfg.latent_phase_rate_ref = cli.latent_phase_rate_ref
    cfg.latent_phase_rate_max = cli.latent_phase_rate_max
    cfg.stillness_vx_scale = cli.stillness_vx_scale
    if cli.latent_mode:
        cfg.action_space = 16  # latent z only (no aux / gate)
        cfg.observation_space += 14  # _last_phase 2 -> 16 in the observation
        if cli.latent_residual:
            cfg.action_space = 16 + 29  # z(16) + full-joint residual(29)
            cfg.observation_space += 29  # residual action feedback
        if cli.vb_from_policy:
            # D049b-fix：动作末 3 维 vb 连续动作 + obs +3（当前软权重 w 反馈）。
            # 布局随模式（与 env _vb_action_slice 同步）：latent_residual
            # （D048i 配方）= [z(16), res(29), vb(3)] = 48；plain latent =
            # [z(16), aux(12), vb(3)] = 31。只在旗标开时 bump，且与 env 的
            # obs/动作组装同步（z_sweep hotfix3 教训：两侧必须同步，默认
            # 路径零变化）
            cfg.action_space = (16 + 29 + 3) if cli.latent_residual else (16 + 12 + 3)
            cfg.observation_space += 3
    if cli.token_mode:
        assert not cli.latent_mode and not cli.decft and not cli.gate_sel, (
            "--token-mode is exclusive with latent/decft/gate_sel")
        assert cli.token_stats, "--token-stats npz (mean/std/rate) is required"
        cfg.action_space = 64  # raw token coordinates (unbounded)
        cfg.observation_space += 62  # _last_phase 2 -> 64 (raw action feedback)
        if cli.token_phase_obs:
            cfg.observation_space += 2  # [sin phi, cos phi] walk clock (E49-B)
    if cli.to42_sel != "off":
        # TO42: selection rides on the latent decode path；两臂 obs/action
        # 布局完全一致，唯一差异 = 选择由策略学出还是由冻结 bucketize 产生
        assert not cli.gate_sel, "--to42-sel and --gate-sel are mutually exclusive"
        assert cli.latent_mode and not cli.decft and not cli.latent_residual, (
            "--to42-sel rides on plain --latent-mode")
        cfg.to42_sel = cli.to42_sel
        cfg.to42_hold_steps = cli.to42_hold_steps
        cfg.action_space = 17        # z(16) + sel bit(1)
        cfg.observation_space += 2   # [sel_state, gate_bool]
    to42_active = cli.to42_sel != "off"
    if cli.vb_from_policy:
        # D049b：前置组合校验（env __init__ 的 _check_vb_from_policy 是第二层）。
        # latent_residual 可组合（§5j b 臂=D048i 配方，动作 48 维）；训练侧
        # 与 force 干预不混用是自然成立的（train 无 force CLI，cfg 恒 -1）
        assert cli.latent_mode and not cli.token_mode \
            and not cli.decft and not cli.gate_sel and cli.to42_sel == "off", (
            "--vb-from-policy rides on --latent-mode, exclusive with "
            "token/decft/gate_sel/to42 (latent_residual allowed: D048i recipe)")
        assert cli.latent_speed_bins or cli.latent_dir_bins, (
            "--vb-from-policy needs a vb-conditioned decode branch "
            "(--latent-speed-bins or --latent-dir-bins)")
    if cli.decft:
        from apt_g1.isaac.decft_policy import OBS_DIM as DECFT_OBS_DIM

        cfg.decft_mode = True
        cfg.action_space = 29  # normalized joint targets (decoder output)
        cfg.observation_space = DECFT_OBS_DIM  # 91 base + 930 hist + 2 phase
    # D053 obs-head：obs +2 [sin(yaw_rel), cos(yaw_rel)]（§5m）。放在所有
    # 既有 bump 块之后（decft 分支整体覆写 observation_space，若允许组合也
    # 必须在其后补 +2）；与 env _get_observations 的追加（vb_w 反馈之后）
    # 严格同步（z_sweep hotfix3 教训：两侧必须同步，默认路径零变化）。
    # 旗标关 = 不进此分支，cfg.observation_space 表达式与改动前逐字节一致。
    _obs_dim_pre_heading = cfg.observation_space
    if cli.obs_heading:
        cfg.observation_space += 2
        # D053：身份断言按旗标分支——旗标开时 obs 恰为「无旗标同配方 +2」
        # （D051 b 臂配方 latent_residual+vb_from_policy 无 elevation：
        # 137→139）；旗标关路径零新增断言，既有 137/134/108 维度契约逐字
        # 不变（env 的 obs.shape[1]==cfg.observation_space 断言两种维度均
        # 通过，ckpt 身份块 obs_dim 同源自 cfg）。
        assert cfg.observation_space == _obs_dim_pre_heading + 2, (
            cfg.observation_space, _obs_dim_pre_heading)
        if (cli.latent_mode and cli.latent_residual and cli.vb_from_policy
                and not cli.use_elevation):
            assert cfg.observation_space == 139, cfg.observation_space
    cfg.aux_scale = cli.aux_scale
    cfg.aux_l2_scale = cli.aux_l2
    cfg.aux_rate_scale = cli.aux_rate
    cfg.yaw_sigma2 = cli.yaw_sigma2
    cfg.vel_sigma2 = cli.vel_sigma2
    cfg.episode_length_s = 20.0
    # pin the global numpy RNG to the terrain seed before env creation (see
    # eval_fast.py note: HfRandomUniformTerrainCfg uses the global np.random).
    np.random.seed(cli.terrain_seed)

    if cli.env == "vanilla":
        env = AptFlatG1VanillaEnv(cfg)
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space,
            aux_dim=29,
            gate_k=0,
            hidden_dim=256,
            use_phase=False,
        ).to("cuda:0")
    elif cli.decft:
        from apt_g1.isaac.decft_policy import DecFtPolicy

        env = AptFlatG1Env(cfg)
        policy = DecFtPolicy(
            obs_dim=cfg.observation_space,
            vae_path=cli.latent_vae_path,
            decoder_path=cli.decoder_path,
            vx_max=cli.vx_max,
            aux_init_std=cli.decft_aux_std,
            phase_init_std=cli.decft_phase_std,
            freeze_decoder=cli.freeze_decoder,
            device="cuda:0",
        ).to("cuda:0")
    else:
        env = AptFlatG1Env(cfg)
        policy = AptPPOPolicy(
            obs_dim=cfg.observation_space,
            aux_dim=29 if cli.latent_residual else 12,
            gate_k=(2 if to42_active else (3 if cfg.use_gate_sel else 0)),
            vb_head=cli.vb_from_policy,  # D049b
            vb_init_std=cli.vb_init_std,  # D049b-fix: 默认 -4.0 与 z/res 同款；门验证配置 -1.0
            hidden_dim=256,
            use_phase=(not cfg.use_gate_sel and not cfg.latent_mode
                       and not cli.token_mode),
            latent_dim=64 if cli.token_mode else (16 if cfg.latent_mode else 0),
        ).to("cuda:0")
        if (cli.latent_mode and not cli.latent_residual) or cli.token_mode or to42_active:
            # E49：这些模式 action = act["phase"]（aux 采样后丢弃），aux 头
            # 不进 PPO log_prob/entropy —— 与 update() 的重算同一约定。
            # 例外：latent_residual 的 aux 头就是 29d 残差执行动作
            # （action = [z, res]，两颗头都执行），必须保持 True 让残差头
            # 收到策略梯度
            policy.aux_executed = False

    # D048h: checkpoint identity (embedded in every ckpt via save_ckpt; the
    # eval side rejects a train/eval misconfiguration with exit 6 BEFORE the
    # first rollout). Asset md5s computed once here (the env has loaded the
    # vae/decoder by now); git_head / env_sha256 / rew_contract use the same
    # recipe as the train_log keys below and are SHARED with them (single
    # source, the two records can never diverge).
    rew_contract = getattr(env, "REW_CONTRACT_VER", 1)
    try:
        env_sha256 = hashlib.sha256(
            Path(inspect.getfile(type(env))).read_bytes()
        ).hexdigest()
    except Exception:
        env_sha256 = ""
    try:
        _g = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_head_v = _g.stdout.strip() if _g.returncode == 0 else ""
    except Exception:
        git_head_v = ""
    ident_base = ckpt_identity.build_identity(
        entry=out_dir.name,
        it=0,
        res_scale=cfg.res_scale,
        res_clip=cfg.res_clip,
        res_l2=cfg.res_l2_scale,
        res_freeze_steps=cfg.res_freeze_steps,
        latent_mode=cfg.latent_mode,
        latent_residual=cfg.latent_residual,
        vb_head=cli.vb_from_policy,  # D049b：结构信息键（非 VERIFY_KEY）
        action_space=cfg.action_space,
        obs_dim=cfg.observation_space,
        rew_contract=rew_contract,
        env_sha256=env_sha256,
        vae_md5=ckpt_identity.file_md5(cli.latent_vae_path),
        decoder_md5=ckpt_identity.file_md5(cli.decoder_path),
        git_head=git_head_v,
    )

    def _ident(it: int) -> dict:
        # per-save identity = shared config block with the iteration pinned
        d = dict(ident_base)
        d["it"] = int(it)
        return d

    latent_prior_mean = None
    if cli.latent_mode and cli.latent_kl_prior == "walk":
        zw = np.load(str(Path(cli.latent_vae_path).parent / "z_walk.npy"))
        latent_prior_mean = torch.from_numpy(zw).float().to("cuda:0")
    if cli.decft and cli.latent_kl_prior == "walk":
        # E44: keep z near the E39 walk manifold while the decoder adapts
        zw = np.load(str(Path(cli.latent_vae_path).parent / "z_walk.npy"))
        latent_prior_mean = torch.from_numpy(zw).float().to("cuda:0")
    trainer = PPOTrainer(
        policy,
        lr=cli.lr,
        entropy_coef=cli.entropy,
        res_ent_coef=cli.res_ent_coef,
        latent_kl_coef=cli.latent_kl,
        latent_expl_coef=cli.latent_expl,
        latent_prior_mean=latent_prior_mean,
        max_iters=cli.iters,
        device="cuda:0",
        decoder_reg_coef=cli.decoder_reg if cli.decft else 0.0,
        decoder_lr=cli.decoder_lr if cli.decft else None,
        decoder_wreg_coef=cli.decoder_wreg if cli.decft else 0.0,
        minibatch_size=cli.ppo_minibatch,
        num_epochs=cli.ppo_epochs,
        fused=cli.fused_adam,
        kl_guard=cli.kl_guard,
        kl_guard_shrink=cli.kl_guard_shrink,
        kl_guard_grow=cli.kl_guard_grow,
        kl_guard_max_rolls=cli.kl_guard_max_rolls,
        kl_step_guard=cli.kl_step_guard,
        kl_step_target=cli.kl_step_target,
        kl_backtracks=cli.kl_backtracks,
    )
    start_it = 0
    if cli.resume:
        # D048h: ckpts carry the identity envelope -- unwrap it here (legacy
        # bare state_dicts also load, identity side is ignored on resume)
        sd, _resume_ident = ckpt_identity.load_ckpt(
            cli.resume, map_location="cuda:0"
        )
        if cli.decft:
            # partial warm start: keep encoder/z-head/critic from the E39
            # checkpoint; decoder stays at official init, aux heads are dropped
            cur = policy.state_dict()
            sd = {
                k: v
                for k, v in sd.items()
                if k in cur and tuple(v.shape) == tuple(cur[k].shape)
            }
            policy.load_state_dict(sd, strict=False)
        else:
            policy.load_state_dict(sd)
        start_it = int(Path(cli.resume).stem.split("_")[-1])
        if start_it >= cli.iters:
            raise SystemExit(
                f"[train] resume iteration {start_it} >= --iters {cli.iters}; "
                "pass a cumulative --iters larger than the checkpoint iteration"
            )

    # E49：未训练初始化对照用的 iter-0 初始权重快照（resume 时 start_it>0，
    # 原始 init 已不存在，不重复落盘）
    if start_it == 0:
        ckpt_identity.save_ckpt(
            out_dir / "policy_it_0.pt", policy.state_dict(), _ident(0)
        )

    rollout = cli.rollout
    T, N, D = rollout, env.num_envs, cfg.observation_space
    aux_dim = 29 if (cli.env == "vanilla" or cli.decft or cli.latent_residual) else 12
    phase_labels_buf = None
    if cli.env == "vanilla":
        buf_phase_none = True
    else:
        buf_phase_none = False
    if (cli.phase_mode or cli.latent_mode) and (
        cli.phase_warmstart_iters > 0 or cli.latent_warmstart_iters > 0
    ):
        phase_labels_buf = torch.zeros(
            T, N, 16 if cli.latent_mode else 2, device="cuda:0"
        )
        if cli.latent_mode:
            zw = np.load(
                str(Path(cli.latent_vae_path).parent / "z_walk.npy")
            )
            phase_labels_buf[:] = torch.from_numpy(zw).to("cuda:0")

    buf = {
        "obs": torch.zeros(T, N, D, device="cuda:0"),
        "phase": torch.zeros(
            T, N,
            64 if cli.token_mode else (16 if (cli.latent_mode or cli.decft) else 2),
            device="cuda:0",
        ),
        "aux": torch.zeros(T, N, aux_dim, device="cuda:0"),
        "logp": torch.zeros(T, N, device="cuda:0"),
        "value": torch.zeros(T, N, device="cuda:0"),
        "reward": torch.zeros(T, N, device="cuda:0"),
        "done": torch.zeros(T, N, dtype=torch.bool, device="cuda:0"),
        "trunc": torch.zeros(T, N, dtype=torch.bool, device="cuda:0"),
        # E49: 复位前终末状态价值（超时步自举）；非 trunc 位被 ×trunc 清零
        "trunc_value": torch.zeros(T, N, device="cuda:0"),
    }
    if cli.kl_guard is not None:
        # E49-C v2：KL 守卫 old 侧参照 = rollout 采样时的分布参数。仅
        # --kl-guard 开启时分配/填充（关闭 = 零额外显存、零行为差异）。log_std
        # 是策略参数的 expand 视图，detach 不解除共享存储，必须 clone 后再存
        # （v1 守卫恒报 0 的别名根因）；均值是 Linear 新输出，无别名问题。
        # gate_sel/vanilla 无 phase 头（update 侧按 rollout["phase"] 是否为
        # None 决定是否取用）、aux 在 latent/token 模式不被执行（update 侧按
        # aux_scored 决定）——存储无害，照常分配。
        phase_d = 64 if cli.token_mode else (16 if (cli.latent_mode or cli.decft) else 2)
        buf["phase_mean_old"] = torch.zeros(T, N, phase_d, device="cuda:0")
        buf["phase_log_std_old"] = torch.zeros(T, N, phase_d, device="cuda:0")
        buf["aux_mean_old"] = torch.zeros(T, N, aux_dim, device="cuda:0")
        buf["aux_log_std_old"] = torch.zeros(T, N, aux_dim, device="cuda:0")
    if cli.gate_sel:
        buf["gate"] = torch.zeros(T, N, dtype=torch.long, device="cuda:0")
        buf["phase"] = None
    if to42_active:
        # phase z 保留 + 增加 gate(sel) 槽位；PPOTrainer.update 的 gate 分支
        # （Categorical log_prob/entropy 重算）原样复用
        buf["gate"] = torch.zeros(T, N, dtype=torch.long, device="cuda:0")
    if cli.vb_from_policy:
        # D049b-fix：vb 连续动作槽位（T,N,3 float）——act() 采样得到的
        # vb_action 同值贯穿：本 buffer → 送 env 的动作末 3 维（env 侧
        # softmax 得软权重 w）→ update() 的 log_prob/熵/资格检查重算
        buf["vb_action"] = torch.zeros(T, N, 3, device="cuda:0")

    if buf_phase_none:
        buf["phase"] = None
    hist = {
        "rewards": [],
        "vx": [],
        "vx_fwd": [],
        "fall_rate": [],
        # D048c：超时率单列（fall_rate=buf["done"] 只统计终止；零摔倒≠零
        # 复位，D048b 200it 两臂每 ~41.7 it 超时复位一次即第 42/84/125/167 轮）
        "timeout_rate": [],
        "approx_kl": [],
        "clip_frac": [],
        "act_std": [],
        "act_aux_std": [],
        "ent_aux": [],
        "ent_z": [],
        "post_update_kl": [],
        # E49-C：vloss / expl_var 无条件记录；KL 守卫四键仅 --kl-guard 开启时记录
        "vloss": [],
        "expl_var": [],
    }
    if cli.kl_guard is not None:
        hist["kl_mb"] = []
        hist["kl_mb_all"] = []
        hist["kl_rolls"] = []
        hist["lr_now"] = []
    # D048i：步级看门 + 资格检查新键序列（仅 --kl-step-guard 开启时初始化；
    # guard 关闭时 train_log.json 键集合与旧运行完全一致，不新增 null 列）
    KSG_HIST_KEYS = (
        "qual_logp_maxdev",
        "qual_ratio_maxdev",
        "qual_kl_self",
        "kl_steps_total",
        "kl_steps_accepted",
        "kl_steps_rejected",
        "kl_lr_scale_min",
        "kl_lr_scale_mean",
        "kl_analytic_joint_mean",
        "kl_analytic_joint_max",
        "kl_analytic_z_mean",
        "kl_analytic_res_mean",
        "kl_gate_mean",
        # D049b-fix：kl_vb 已进 ksg joint 目标（joint=z+res+vb），单列均值照记
        "kl_vb_mean",
        "param_rel_move",
    )
    if cli.kl_step_guard:
        for _k in KSG_HIST_KEYS:
            hist[_k] = []
    # D049b：策略软权重 w 的窗口统计（仅旗标开时初始化，train_log.json 键集合
    # 与旧 run 保持一致）。vb_w_mean[3] = 本 iter 全部 T*N 控制步 w 的均值；
    # vb_argmax_hist[3] = argmax 档位占比（w 最重的档位的步数占比）
    if cli.vb_from_policy:
        hist["vb_w_mean"] = []
        hist["vb_argmax_hist"] = []
        # D051：advantage×vb_action 逐维 Pearson 相关（伴生插桩，update 侧
        # vb 头关闭时 stats 记 None → hist 记 null；键仍只在旗标开时新增，
        # 旧 run 的 train_log.json 键集合不变）
        hist["adv_vb_corr"] = []
    # D053 obs-head：yaw_rel 监控（§5m spot check：首步≈(0,1)/训练中偏移量；
    # 仅旗标开时初始化，train_log.json 键集合与旧 run 保持一致）。
    # yaw_rel_mean[2] = 本 iter 全部 T*N 控制步 obs 尾 2 维 [sin, cos] 的均值
    if cli.obs_heading:
        hist["yaw_rel_mean"] = []
    # E49 诊断步骤③：分项诊断（--diag-log 开启时启用）。vanilla env 无
    # _last_rew_terms 快照，不支持。d_* 序列随 hist 整体序列化进 train_log.json。
    DIAG_KEYS = ("track_xy", "track_yaw", "upright", "height", "stillness")
    DIAG_HIST_KEYS = (
        "d_track_xy", "d_track_yaw", "d_upright", "d_height", "d_stillness",
        "d_vx_err", "d_cmd_vx", "d_stand_frac", "d_fwd_rate", "d_drift_rate",
    )
    diag = cli.diag_log and cli.env != "vanilla"
    if diag:
        for k in DIAG_HIST_KEYS:
            hist[k] = []
    # D048c：版本身份三键（rew_contract / env 源 sha256 / git HEAD）。
    # 旧日志无这三键 = v1 身份（heading 双旋转 + 位移窗口差分），不回刷。
    # sha256 为跨端锚（CVGL 执行目录非 git 仓时 git_head 记空串）。
    # D048h：取值逻辑上移到 ckpt 身份块构造处，这里只复用同一份值。
    hist["rew_contract"] = rew_contract
    hist["env_sha256"] = env_sha256
    hist["git_head"] = git_head_v
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    # E49-C v2：probe 模式只截短训练循环（末 iter 的日志/ckpt 边界一并随上限，
    # 保证短探针也产出末行日志与最终 ckpt）；trainer.max_iters 不变（=--iters，
    # 调度长度不压缩）
    iters_run = cli.probe_iters or cli.iters
    dead_streak = 0  # D048i：连续死轮计数（accepted==0 的更新轮）

    # ------------------------------------------------------------------
    # D051：BC 热启动 vb 头（DS_CONTINUOUS_EXECUTION_PLAN §5k 预注册）。
    # 单变量纪律：对照 = D049-R1 b-fix，唯一新增 = 发射前 BC 预热——只在
    # 线 rollout（act 正常采样、命令正常重采样）中对「当前 obs + 同期 cmd」
    # 做 CE 监督，独立 Adam 只更新 vb_logits 线性层（encoder/z/res/
    # vb_log_std 全冻结，不污染共享特征分布）。steps=0（默认）整块跳过，
    # 行为与 D049-R1 逐位一致
    # ------------------------------------------------------------------
    if cli.vb_bc_warmup_steps > 0:
        assert cli.vb_from_policy, (
            "--vb-bc-warmup-steps requires --vb-from-policy "
            "(D051 BC warmup only applies to the vb-head arm)"
        )
        assert policy.vb_head and hasattr(policy, "vb_logits"), (
            "vb_from_policy run must carry an AptPPOPolicy vb_head"
        )
        assert obs.shape[1] > 67, (
            f"obs dim {obs.shape[1]} has no cmd slot at index 67 (D051 "
            "calibration grid rewrites obs[:, 67])"
        )
        bc_anchors = tuple(
            float(s) for s in cli.vb_bc_anchors.split(",") if s.strip()
        )
        assert len(bc_anchors) == 3, (
            f"--vb-bc-anchors needs 3 comma-separated floats, got "
            f"{cli.vb_bc_anchors!r}"
        )
        bc_opt = torch.optim.Adam(
            policy.vb_logits.parameters(), lr=cli.vb_bc_lr
        )
        print(
            f"[BC-WARMUP] start steps={cli.vb_bc_warmup_steps} "
            f"lr={cli.vb_bc_lr} anchors={bc_anchors} "
            f"(only vb_logits trains; encoder/z/res/vb_log_std frozen)",
            flush=True,
        )
        bc_last_loss = 0.0
        for bc_t in range(cli.vb_bc_warmup_steps):
            # 采样在 no_grad（BC 前向要 grad：vb_bc_loss 内部自开——这里
            # 在默认 enable_grad 上下文，act 分支单独关掉即可）
            with torch.no_grad():
                bc_act, _bc_lp, _bc_ent, _bc_val, _bc_fwd = policy.act(
                    obs, deterministic=False
                )
            # 动作布局与主 rollout 的 vb 臂同款（residual 48 / plain 31，
            # 末 3 维 = 采样的 vb_action，env 侧 softmax 得带噪软权重）
            bc_action = torch.cat(
                [bc_act["phase"], bc_act["aux"], bc_act["vb_action"]], dim=1
            )
            # BC 目标用 step 之前的 obs（act 的输入）与同期 cmd 快照
            #（env._commands = (num_envs,3) 的 vx,vy,yaw，取 vx 列）
            bc_obs = obs
            bc_cmd = env._commands[:, 0].detach().clone()
            bc_obs_dict, _bc_rew, _bc_term, _bc_trunc, _ = env.step(bc_action)
            bc_loss = vb_bc_loss(policy, bc_obs, bc_cmd)
            bc_opt.zero_grad()
            bc_loss.backward()
            bc_opt.step()
            bc_last_loss = float(bc_loss.detach())
            obs = bc_obs_dict["policy"]
            if (bc_t + 1) % 100 == 0 or bc_t + 1 == cli.vb_bc_warmup_steps:
                with torch.no_grad():
                    _bc_w = torch.softmax(
                        policy.forward_actor(obs)["vb_logits"], dim=-1
                    ).mean(dim=0)
                print(
                    f"[BC-WARMUP] step={bc_t + 1}/{cli.vb_bc_warmup_steps} "
                    f"loss={bc_last_loss:.5f} "
                    f"w_mean=[{', '.join(f'{v:.3f}' for v in _bc_w.tolist())}]",
                    flush=True,
                )
        # 预热末校准快照：当前 obs 一份，obs[:, 67]（cmd 槽位，env obs 布局
        # 中 _commands 块起始索引）逐档改写为命令网格读确定性响应（it0 判读
        # 锚）。每档用全部 env 行（9×num_envs），前向 softmax 后逐档取均值
        # → 9×3 校准表
        bc_grid = tuple(i * 0.1 for i in range(9))
        bc_n = env.num_envs
        obs_grid = obs.detach().clone().repeat(len(bc_grid), 1)
        for _gi, _gv in enumerate(bc_grid):
            obs_grid[_gi * bc_n : (_gi + 1) * bc_n, 67] = _gv
        with torch.no_grad():
            _calib_logits = policy.forward_actor(obs_grid)["vb_logits"]
            bc_calib = torch.softmax(_calib_logits, dim=-1).reshape(
                len(bc_grid), bc_n, 3
            ).mean(dim=1)
        bc_calib_rows = [
            [round(v, 5) for v in row] for row in bc_calib.tolist()
        ]
        hist["vb_bc_calib"] = bc_calib_rows  # 9×3 校准表（行序 = bc_grid）
        hist["vb_bc_info"] = {
            "steps": int(cli.vb_bc_warmup_steps),
            "final_loss": bc_last_loss,
            "lr": float(cli.vb_bc_lr),
            "anchors": list(bc_anchors),
            "grid": list(bc_grid),
        }
        print("[BC-WARMUP] calibration w(cmd) (cmd -> [w0, w1, w2]):", flush=True)
        for _gv, _row in zip(bc_grid, bc_calib_rows):
            print(
                f"    cmd={_gv:.1f} -> [{', '.join(f'{v:.4f}' for v in _row)}]",
                flush=True,
            )
        print("BC warmup done, entering PPO", flush=True)

    for it in range(start_it, iters_run):
        t0 = time.time()
        if cli.disturbance_ramp_iters > 0:
            env.cfg.disturbance_prob = min(
                cli.disturbance_prob,
                cli.disturbance_prob * (it + 1) / cli.disturbance_ramp_iters,
            )
        # 奖励统计改为 GPU 张量累积（原先每控制步 rew.mean().item() 同步一次）；
        # 统计窗口不变：仍是本 iter 全部 T*N 个 reward 的算术均值，只在 iter 末
        # 的日志边界换算打印
        rew_sum = torch.zeros((), device="cuda:0")
        if cli.vb_from_policy:
            # D049b：窗口统计 GPU 累积器（与 rew_sum 同窗：本 iter 全部 T*N
            # 控制步；iter 末一次 .tolist() 同步，热循环零同步）
            vb_w_sum = torch.zeros(3, device="cuda:0")
            vb_am_sum = torch.zeros(3, device="cuda:0")
        if cli.obs_heading:
            # D053：obs 尾 2 维 [sin(yaw_rel), cos(yaw_rel)] 的窗口累积器
            # （同窗口径，iter 末一次同步）
            yaw_rel_sum = torch.zeros(2, device="cuda:0")
        if diag:
            # E49 诊断：GPU 标量累积器（每 iter 重建；iter 末一次 .cpu() 换算，
            # 热循环零同步）。
            diag_term_sum = {k: torch.zeros((), device="cuda:0") for k in DIAG_KEYS}
            diag_abs_err = torch.zeros((), device="cuda:0")
            diag_cmd_sum = torch.zeros((), device="cuda:0")
            diag_stand = torch.zeros((), device="cuda:0")
            diag_cnt = torch.zeros((), device="cuda:0")
            # D048c：位移改为逐步累计。旧口径 = iter 首末窗口差分
            # （diag_pos0），超时复位的传送跳变直接进差分——D048b 200it
            # "两臂净前进转负"即此伪影，已撤回（tracker D048c 行）。
            diag_prev_pos = env.robot.data.root_pos_w[:, :2].detach().clone()
            diag_disp_acc = torch.zeros_like(diag_prev_pos)
        for t in range(T):
            act, logp, ent, val, p_fwd = policy.act(obs)
            buf["obs"][t] = obs
            if cli.decft:
                buf["phase"][t] = act["phase"].detach()
                buf["aux"][t] = act["aux"].detach()
                action = act["aux"]
            elif cli.env == "vanilla":
                buf["aux"][t] = act["aux"].detach()
                action = act["aux"]
            else:
                buf["aux"][t] = act["aux"].detach()
                if cli.gate_sel:
                    buf["gate"][t] = act["gate"].detach()
                    action = torch.cat(
                        [act["aux"], act["gate"].float().unsqueeze(-1)], dim=1
                    )
                elif to42_active:
                    buf["phase"][t] = act["phase"].detach()
                    buf["gate"][t] = act["gate"].detach()
                    action = torch.cat(
                        [act["phase"], act["gate"].float().unsqueeze(-1)], dim=1
                    )
                else:
                    buf["phase"][t] = act["phase"].detach()
                    if cfg.latent_mode and cfg.latent_residual:
                        # E48/D049b：[z(16), res(29)]——aux 头即 29d 残差执行
                        # 动作（aux_executed=True，进 ratio/log_prob 既有逻辑
                        # 不动）；vb_from_policy 时再接末 3 维 vb_action=48 维
                        # （切片位置与 env _vb_action_slice 同步：45:48）
                        action = torch.cat([act["phase"], act["aux"]], dim=1)
                        if cli.vb_from_policy:
                            # D049b-fix：act 采样 / buffer / 送 env 同一张量
                            # （连续 vb_action，env softmax 得软权重 w）
                            buf["vb_action"][t] = act["vb_action"].detach()
                            action = torch.cat(
                                [action, act["vb_action"]], dim=1
                            )
                    elif cfg.latent_mode:
                        if cli.vb_from_policy:
                            # D049b-fix：[z(16), aux(12), vb_action(3)] = 31 维。
                            # 中段 12 维 = 采样后丢弃的 aux 头槽位（aux_executed
                            # =False 约定不变，env 忽略）；vb 段 = act 采样的
                            # 连续 vb_action（det 模式 =vb_logits）——env 侧
                            # softmax 得软权重 w，同一张量存 buffer 并供
                            # update() 对 Normal(vb_logits, vb_log_std) 重算
                            # log_prob（动作-概率契约闭合）
                            buf["vb_action"][t] = act["vb_action"].detach()
                            action = torch.cat(
                                [act["phase"], act["aux"], act["vb_action"]],
                                dim=1,
                            )
                        else:
                            action = act["phase"]
                    elif cfg.token_mode:
                        # E49: raw 64-d token coordinates (aux head sampled
                        # but discarded, same convention as latent mode)
                        action = act["phase"]
                    else:
                        action = torch.cat([act["phase"], act["aux"]], dim=1)
            buf["logp"][t] = logp.detach()
            if cli.kl_guard is not None:
                # E49-C v2：旧分布快照。log_std 必须 detach().clone()——它是
                # 策略参数的 expand 视图，detach 不解除共享存储（见 buf 分配处
                # 注释）；vanilla/gate_sel 无 phase 头，按键存在性跳过
                if "phase_mean" in p_fwd:
                    buf["phase_mean_old"][t] = p_fwd["phase_mean"].detach()
                    buf["phase_log_std_old"][t] = (
                        p_fwd["phase_log_std"].detach().clone()
                    )
                buf["aux_mean_old"][t] = p_fwd["aux_mean"].detach()
                buf["aux_log_std_old"][t] = p_fwd["aux_log_std"].detach().clone()
            buf["value"][t] = val.detach()
            obs_dict, rew, term, trunc, _ = env.step(action)
            buf["reward"][t] = rew
            buf["done"][t] = term
            buf["trunc"][t] = trunc
            if diag:
                # D048c：位移逐步累计。本步 done 的 env 已在 step 内复位、
                # root_pos 已回原点，终末位置用 _reset_idx 截留的
                # _final_pos_w 补上（term/trunc 都走 _reset_idx，均覆盖；
                # 非 done 位该值为陈旧数据，被 where 掩掉）。
                _pos_now = env.robot.data.root_pos_w[:, :2].detach()
                # D048g 勘误：done/trunc 位为 [N]，torch.where 条件按 [1,N]
                # 与 [N,2] 操作数对齐 → dim1 上 128 vs 2 崩（D048c 引入的
                # 训练侧位移代码此前从未真实训练执行过，py_compile/纯数值
                # 核验盖不住形状广播）；升维 [N,1] 与 [N,2] 广播。
                _end_pos = torch.where(
                    (buf["done"][t] | buf["trunc"][t]).unsqueeze(-1),
                    env._final_pos_w,
                    _pos_now,
                )
                diag_disp_acc += _end_pos - diag_prev_pos
                diag_prev_pos = _pos_now.clone()
            # E49: 超时步自举价值 = 复位前终末状态的价值。Isaac step 返回前
            # 已把 obs 换成复位后新局观测，不能用它算；_final_obs 由 env 的
            # _reset_idx 在 super() 之前截留。×trunc 把非 trunc 位清零防陈旧值。
            final_obs = getattr(env, "_final_obs", None)
            if final_obs is not None:
                buf["trunc_value"][t] = (
                    policy.get_value(final_obs).detach().squeeze(-1) * trunc
                )
            obs = obs_dict["policy"]
            rew_sum += rew.sum()
            if cli.obs_heading:
                # D053：obs 尾 2 维 = [sin(yaw_rel), cos(yaw_rel)]（§5m b 臂
                # 配方下 env heading 块即 parts 末块；obs 尾 2 维假设由上方
                # b 臂组合断言钉住，勿在 heading 之后再追加其他 obs 块）
                yaw_rel_sum += obs[:, -2:].sum(0)
            if cli.vb_from_policy:
                # D049b：本控制步的策略软权重 w 并入窗口累积（_pre_physics_step
                # 内已随动作更新；sum/one_hot 均 GPU 上完成，无同步）
                _w = env._last_vb_w
                vb_w_sum += _w.sum(0)
                vb_am_sum += torch.nn.functional.one_hot(
                    _w.argmax(-1), 3
                ).sum(0).float()
            if diag:
                # E49 诊断：本控制步的分项快照并入累积（与 rew_sum 同窗同方式；
                # 快照缺失时跳过，不影响训练）
                terms = getattr(env, "_last_rew_terms", None)
                if terms is not None:
                    for k in DIAG_KEYS:
                        diag_term_sum[k] += terms[k].sum()
                    diag_abs_err += terms["vx_err"].abs().sum()
                    diag_cmd_sum += env._commands[:, 0].sum()
                    diag_stand += (
                        env.robot.data.root_lin_vel_b[:, 0].abs() < 0.05
                    ).sum()
                    diag_cnt += N
            if (
                phase_labels_buf is not None
                and it < (
                    cli.latent_warmstart_iters
                    if cli.latent_mode
                    else cli.phase_warmstart_iters
                )
            ):
                if not cli.latent_mode:  # latent: z_walk labels pre-filled
                    cmds = env._build_commands_list()
                    proprio = env._proprio_np()
                    if cfg.phase_anchor:
                        # anchored mode: the policy phase head is a bounded offset,
                        # warmstart it toward zero offset (pure router clock).
                        phase_labels_buf[t] = torch.zeros(
                            env.num_envs, 2, dtype=torch.float32, device="cuda:0"
                        )
                    else:
                        sc, _ = env._router.phase_raw_batch(proprio, cmds)
                        phase_labels_buf[t] = torch.from_numpy(sc).to("cuda:0")
        last_val = policy.get_value(obs)
        # E49: T-1 步若有超时（trunc&~done）位，last_value（复位后 obs 的价值）
        # 对这些位无意义，用复位前终末状态价值覆盖
        m_last = buf["trunc"][T - 1] & ~buf["done"][T - 1]
        last_val = torch.where(m_last, buf["trunc_value"][T - 1], last_val.detach())
        buf["last_value"] = last_val

        warm_coef = 0.0
        warm_iters = (
            cli.latent_warmstart_iters if cli.latent_mode else cli.phase_warmstart_iters
        )
        if phase_labels_buf is not None and it < warm_iters:
            warm_coef = cli.phase_warmstart_coef * (
                1.0 - it / max(1, warm_iters)
            )
        stats = trainer.update(buf, phase_labels=phase_labels_buf, phase_warm_coef=warm_coef)
        if cli.kl_step_guard:
            # D048i B6：资格检查违约 → 逐条打印偏差详情后 exit 8（与 eval 身份
            # exit 6、死轮 exit 7 区分）。先落盘 train_log 再退，防诊断数据丢失
            if stats.get("qual_ratio_maxdev", 0.0) > cli.qual_ratio_tol:
                print(
                    f"[QUAL-VIOLATION] iter={it} qual_ratio_maxdev="
                    f"{stats.get('qual_ratio_maxdev', 0.0):.6e} > "
                    f"tol={cli.qual_ratio_tol:.3e}",
                    flush=True,
                )
                print(
                    f"    qual_logp_maxdev={stats.get('qual_logp_maxdev', 0.0):.6e} "
                    f"qual_kl_self={stats.get('qual_kl_self', 0.0):.3e} "
                    f"kl_steps_total={stats.get('kl_steps_total', 0)} "
                    f"param_rel_move={stats.get('param_rel_move', 0.0):.3e} "
                    f"(存储 logp 与本轮开始策略重算 logp 失配 -> exit 8)",
                    flush=True,
                )
                with open(out_dir / "train_log.json", "w") as f:
                    json.dump(hist, f)
                os._exit(8)
            # D048i B5：死轮停臂——accepted==0 记死轮，连续 --dead-rounds 轮
            # 死 → 诊断 ckpt（文件名带 _deadround 后缀）+ FATAL 摘要 + exit 7
            if stats.get("kl_steps_accepted", 0) == 0:
                dead_streak += 1
                if dead_streak >= cli.dead_rounds:
                    dead_ckpt = out_dir / f"policy_it_{it + 1}_deadround.pt"
                    ckpt_identity.save_ckpt(
                        dead_ckpt, policy.state_dict(), _ident(it + 1)
                    )
                    print(
                        f"[FATAL][kl-step-guard] iter={it} 连续 {dead_streak} 轮"
                        f"零接受步（>= --dead-rounds {cli.dead_rounds}）",
                        flush=True,
                    )
                    print(
                        f"    steps total={stats.get('kl_steps_total', 0)} "
                        f"rejected={stats.get('kl_steps_rejected', 0)} "
                        f"lr_scale_min={stats.get('kl_lr_scale_min', 0.0):.4g} "
                        f"lr_scale_mean={stats.get('kl_lr_scale_mean', 0.0):.4g} "
                        f"qual_logp_maxdev={stats.get('qual_logp_maxdev', 0.0):.3e} "
                        f"qual_ratio_maxdev={stats.get('qual_ratio_maxdev', 0.0):.3e} "
                        f"param_rel_move={stats.get('param_rel_move', 0.0):.3e} "
                        f"ckpt={dead_ckpt} -> exit 7",
                        flush=True,
                    )
                    with open(out_dir / "train_log.json", "w") as f:
                        json.dump(hist, f)
                    os._exit(7)
            else:
                dead_streak = 0
        it_time = time.time() - t0
        # iter 末（日志边界）一次 .item()；数值含义 = 本 iter 全部 reward 的均值
        mean_rew = (rew_sum / (T * N)).item()
        fall_rate = float(buf["done"].float().mean().item())
        to_rate = float(buf["trunc"].float().mean().item())
        hist["rewards"].append(mean_rew)
        hist["fall_rate"].append(fall_rate)
        hist["timeout_rate"].append(to_rate)
        # E49 修正 vx 口径：fwd = 机体系前后向速度（带符号，+x = 前进）；
        # 原模长口径保留为 spd（hist["vx"] 键语义不变，旧工具兼容）
        fwd = float(env.robot.data.root_lin_vel_b[:, 0].mean().detach())
        spd = float(
            torch.mean(
                torch.norm(env.robot.data.root_lin_vel_w[:, :2], dim=1).detach().cpu()
            )
        )
        hist["vx"].append(spd)
        hist["vx_fwd"].append(fwd)
        hist["approx_kl"].append(stats["approx_kl"])
        hist["clip_frac"].append(stats["clip_frac"])
        hist["act_std"].append(stats["act_std"])
        # D048b：res 头可观测三件套（.get 防御，旧 ckpt/路径无此键时不记录；
        # losses 侧键名 = aux_std）
        hist["act_aux_std"].append(stats.get("aux_std"))
        hist["ent_aux"].append(stats.get("ent_aux"))
        hist["ent_z"].append(stats.get("ent_z"))
        hist["post_update_kl"].append(stats["post_update_kl"])
        # E49-C：新键一律 .get 防御（键缺失时不记录，不抛错）
        hist["vloss"].append(stats.get("vloss"))
        hist["expl_var"].append(stats.get("expl_var"))
        if cli.vb_from_policy:
            # D049b：窗口均值落账（.tolist() 每 iter 一次同步）
            hist["vb_w_mean"].append(
                [round(v, 5) for v in (vb_w_sum / (T * N)).tolist()]
            )
            hist["vb_argmax_hist"].append(
                [round(v, 5) for v in (vb_am_sum / (T * N)).tolist()]
            )
            # D051：伴生插桩落账（update 侧 adv_vb_corr = 3 维 Pearson 列表
            # 或 None；.get 防御与 kl/守卫键同风格）
            hist["adv_vb_corr"].append(stats.get("adv_vb_corr"))
        if cli.obs_heading:
            # D053：yaw_rel 窗口均值落账 [sin, cos]（首 it 局首步≈(0,1)）
            hist["yaw_rel_mean"].append(
                [round(v, 5) for v in (yaw_rel_sum / (T * N)).tolist()]
            )
        if cli.kl_guard is not None:
            hist["kl_mb"].append(stats.get("kl_mb"))
            hist["kl_mb_all"].append(stats.get("kl_mb_all"))
            hist["kl_rolls"].append(stats.get("kl_rolls"))
            hist["lr_now"].append(stats.get("lr_now"))
        if cli.kl_step_guard:
            # D048i：新键 .get 防御（键缺失时记 None 不抛错）
            for _k in KSG_HIST_KEYS:
                hist[_k].append(stats.get(_k))
        dvals = None
        if diag:
            # E49 诊断口径（裁决 fix-s0 退化用的分项材料）：
            # · 分项均值 / |vx-cmd| / cmd 均值 / 站立占比 = 本 iter 全部 T*N 步
            #   的均值，与 mean_rew 同窗（窗口一致性是本诊断的核心）；
            # · D048c 位移口径 = 逐步累计净位移 / 控制时长（T * sim.dt *
            #   decimation），done 步用复位前终末位置补齐——超时/摔倒复位
            #   跳变不再进差分。与 0e11cb6 及之前的 d_fwd_rate/d_drift_rate
            #   （iter 首末窗口差分）不可直接比较。
            dur = T * env.cfg.sim.dt * env.cfg.decimation
            vals = torch.stack([
                *[diag_term_sum[k] for k in DIAG_KEYS],
                diag_abs_err, diag_cmd_sum, diag_stand,
            ]) / diag_cnt
            vals = torch.cat([
                vals,
                (diag_disp_acc[:, 0].mean() / dur).unsqueeze(0),
                (diag_disp_acc[:, 1].abs().mean() / dur).unsqueeze(0),
            ])
            vals = vals.cpu()  # iter 末唯一一次 GPU->CPU 同步
            dvals = dict(zip(
                ("track_xy", "track_yaw", "upright", "height", "stillness",
                 "vx_err", "cmd_vx", "stand_frac", "fwd_rate", "drift_rate"),
                vals.tolist(),
            ))
            for key, v in zip(DIAG_HIST_KEYS, vals.tolist()):
                hist[key].append(v)
        if cli.speed_log_interval > 0 and (
            it % cli.speed_log_interval == 0 or it == iters_run - 1
        ):
            # 墙钟速度日志（单 iter 瞬时口径，与下方 dt 一致）
            print(
                f"[SPEED] iter={it} elapsed={it_time:.1f}s "
                f"it_per_s={1.0 / max(it_time, 1e-9):.3f}",
                flush=True,
            )
        if it % 10 == 0 or it == iters_run - 1:
            dec_dw = 0.0
            if cli.decft:
                # E44: total weight drift of the fine-tuned decoder vs official
                dec_dw = float(
                    sum(
                        (p.detach() - rp.detach()).pow(2).sum().item()
                        for p, rp in zip(
                            policy.decoder.net.parameters(),
                            policy.decoder_ref.net.parameters(),
                        )
                    )
                    ** 0.5
                )
            # E49-C：vloss / 解释方差无条件追加；守卫开启时再追加 KL 守卫量
            # （stats.get 键存在性防御；roll = 本次 update 的回滚总数）
            ev_line = (
                f" vloss={stats.get('vloss', 0.0):.4f}"
                f" ev={stats.get('expl_var', 0.0):.3f}"
            )
            if cli.kl_guard is not None:
                ev_line += (
                    f" kl_g={stats.get('kl_mb', 0.0):.4g}"
                    f" roll={stats.get('kl_rolls', 0)}"
                    f" lr={stats.get('lr_now', 0.0):.2g}"
                )
            if cli.kl_step_guard:
                # D048i：步级看门监控尾巴（接受/总数、lr 下探、资格偏差、参数位移）
                ev_line += (
                    f" ksg={stats.get('kl_steps_accepted', 0)}/"
                    f"{stats.get('kl_steps_total', 0)}"
                    f" lrmin={stats.get('kl_lr_scale_min', 0.0):.3g}"
                    f" qdev={stats.get('qual_ratio_maxdev', 0.0):.2e}"
                    f" prm={stats.get('param_rel_move', 0.0):.2e}"
                )
            if cli.obs_heading:
                # D053：yaw_rel 均值尾巴（§5m spot check：episode 首步
                # (sin,cos)≈(0,1)；训练中读偏移量，与 hist["yaw_rel_mean"] 同源）
                _yr = (yaw_rel_sum / (T * N)).tolist()
                ev_line += f" yaw_sin={_yr[0]:+.3f} yaw_cos={_yr[1]:.3f}"
            print(
                f"[{it}/{cli.iters}] rew={mean_rew:.3f} fall={fall_rate:.3f} "
                f"to={to_rate:.3f} "
                f"fwd={fwd:.3f} spd={spd:.3f} loss={stats['loss']:.4f} "
                f"ploss={stats['ploss']:.4f} ent={stats['ent']:.4f} "
                f"klp={stats['kl_prior']:.6f} expl={stats['expl']:.5f} "
                f"dreg={stats['dreg']:.5f} dec_dw={dec_dw:.4f} "
                f"dt={it_time:.1f}s akl={stats['approx_kl']:.5f} "
                f"pkl={stats['post_update_kl']:.5f} "
                f"clip={stats['clip_frac']:.3f} std={stats['act_std']:.4f}"
                f"{ev_line}",
                flush=True,
            )
            if diag:
                # E49 诊断：紧凑一行（键与 hist d_* 对应）
                print(
                    f"    diag: d_xy={dvals['track_xy']:.3f} "
                    f"d_yaw={dvals['track_yaw']:.3f} d_up={dvals['upright']:.3f} "
                    f"d_h={dvals['height']:.3f} d_st={dvals['stillness']:.4f} "
                    f"verr={dvals['vx_err']:.3f} cmd={dvals['cmd_vx']:.3f} "
                    f"fwd_rate={dvals['fwd_rate']:.3f} "
                    f"drift={dvals['drift_rate']:.3f} stand={dvals['stand_frac']:.3f}",
                    flush=True,
                )
        if (it + 1) % 50 == 0 or it == iters_run - 1:
            ckpt = out_dir / f"policy_it_{it + 1}.pt"
            ckpt_identity.save_ckpt(ckpt, policy.state_dict(), _ident(it + 1))
            with open(out_dir / "train_log.json", "w") as f:
                json.dump(hist, f)

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(hist, f)
    ckpt_identity.save_ckpt(
        out_dir / "policy_final.pt", policy.state_dict(), _ident(iters_run)
    )
    print("saved", out_dir)
    os._exit(0)


if __name__ == "__main__":
    main()
