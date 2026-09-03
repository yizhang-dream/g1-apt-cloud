"""TO42 formal eval driver：单 (arm × v × train_seed) cell 的正式评测。

角色（SCRIPT_MAP 登记）：**state-changing execution code**（TO42_PLAN §8 步 4；
云 wave 驱动 per-cell 子进程调用，receipt 落盘后 os._exit 硬退出——Isaac 退出
挂死 gotcha 防御，与 eval_cell 同款进程模型）。

harness 语义逐字继承 TO41（launch_sanity/eval_cell）：cfg 镜像 build_cell_cfg
的 ctrl 臂形状（latent+speed/dir bins+to_ref 零块+to_tau 关）、jitter_and_reset
（rng(1000+seed)：root z ±5mm / 29 joint pos ±0.01rad / joint vel ±0.02rad/s）、
每 step 重申恒定 cmd、确定性策略（deterministic=True）、episode_length_s=120
（60s eval 内无 auto-reset）。TO42 增量 = cfg.to42_sel（臂身份）+ receipt 记录
selection 时间线（sel_state uint8 序列 + gate 脉冲步列表 + 策略选择头 p(vb1)
均值）——G0/G2 的审计原料。只产 record，不产 verdict（verdict 只出自
to42_checker.py；先审计后分析，checker 不读行为指标）。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from apt_g1.rung1.mode_a_runtime import (
    REPO_ROOT,
    _platform,
    _platform_python,
    _utcnow,
    sha256_file,
    state_dict_sha256,
)

RECEIPT_SCHEMA = "to42-eval-receipt/v1"
ENV_SOURCE = REPO_ROOT / "apt_g1/isaac/apt_flat_env.py"
GATE_SOURCE = REPO_ROOT / "apt_g1/isaac/to42_gate.py"
PPO_SOURCE = REPO_ROOT / "apt_g1/isaac/ppo_core.py"
TRAIN_SOURCE = REPO_ROOT / "apt_g1/isaac/train_apt_isaac.py"
SELF_SOURCE = Path(__file__).resolve()

GRID7 = (0.200, 0.225, 0.250, 0.275, 0.277, 0.300, 0.325)  # 冻结 TO41 网格


def runtime_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _set_commands(env, v: float) -> None:
    import torch

    env.router_commands[0] = None
    env._commands[0] = torch.tensor(
        [v, 0.0, 0.0], dtype=torch.float32, device=env.device)


def build_cell_cfg(vae_path: str, ref_npz: str, arm: str, hold_steps: int):
    """TO41 ctrl 臂配方形状 + TO42 selection（TO42_PLAN §3/§4 逐字）。"""
    from apt_g1.isaac.apt_flat_env import AptFlatG1EnvCfg
    from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg

    cfg = AptFlatG1EnvCfg()
    cfg.observation_space += 14          # latent_mode（_last_phase 2→16）
    cfg.observation_space += 12          # to_ref 零块（中性脚手架）
    cfg.observation_space += 2           # TO42 [sel_state, gate_bool]
    cfg.action_space = 17                # z(16) + sel bit(1)
    cfg.scene.num_envs = 1
    cfg.terrain = make_terrain_importer_cfg("plane", 0.04, seed=0)
    cfg.latent_mode = True
    cfg.latent_speed_bins = True
    cfg.latent_dir_bins = True
    cfg.latent_residual = False
    cfg.latent_cmd_phase_rate = False
    cfg.latent_vae_path = str(vae_path)
    cfg.to_ref = True
    cfg.to_ref_npz = str(ref_npz)
    cfg.to_ref_obs_zero = True
    cfg.to_ref_w = 0.0
    cfg.to_tau = False
    cfg.disturbance_prob = 0.0
    cfg.episode_length_s = 120.0
    cfg.to42_sel = arm
    cfg.to42_hold_steps = hold_steps
    return cfg


def rollout_eval(env, policy, cmd_vx: float, eval_seed: int, steps: int,
                 hold_steps: int) -> dict:
    """单 eval-seed episode：jitter + 恒定 cmd + 确定性策略 + selection 时间线。"""
    import torch

    from apt_g1.isaac.eval_apt_isaac import jitter_and_reset
    from apt_g1.isaac.to42_gate import natural_vb

    jitter_and_reset(env, seed=eval_seed)
    _set_commands(env, cmd_vx)

    nat = int(natural_vb(torch.tensor([cmd_vx])).clamp(0, 1)[0])
    prev_eplen = int(env.episode_length_buf[0].item())
    heights, vxs, vys, xys = [], [], [], []
    sel_timeline = bytearray()
    gate_steps = []
    p1_sum, p1_n = 0.0, 0
    fall = None
    n_auto = 0
    done = 0
    for t in range(1, steps + 1):
        _set_commands(env, cmd_vx)
        with torch.no_grad():
            act, _, _, _, p = policy.act(env._last_obs, deterministic=True)
        probs = torch.softmax(p["gate_logits"], dim=-1)[0]
        p1_sum += float(probs[1].item())
        p1_n += 1
        action = torch.cat([act["latent"], act["gate"].float().unsqueeze(-1)], dim=1)
        obs_dict, _, term, _, _ = env.step(action)
        env._last_obs = obs_dict["policy"]
        done += 1
        sel_timeline.append(int(env._to42.state[0].item()))
        if bool(env._to42.gate[0].item()):
            gate_steps.append(t)
        eplen = int(env.episode_length_buf[0].item())
        if eplen < prev_eplen:
            n_auto += 1
        prev_eplen = eplen
        heights.append(float(env.robot.data.root_pos_w[0, 2].item()))
        v = env._base_lin_vel()[0].detach().cpu().numpy()
        vxs.append(float(v[0]))
        vys.append(float(v[1]))
        xys.append(env.robot.data.root_pos_w[0, :2].detach().cpu().numpy())
        if term.any():
            fall = t
            break

    heights_arr = np.array(heights)
    vxs_arr = np.array(vxs)
    vys_arr = np.array(vys)
    xys_arr = np.array(xys)
    spd = np.sqrt(vxs_arr**2 + vys_arr**2)
    disp = (float(np.linalg.norm(xys_arr[-1] - xys_arr[0]))
            if len(xys_arr) > 1 else 0.0)
    sel_arr = np.frombuffer(bytes(sel_timeline), dtype=np.uint8)
    switch_steps = []
    prev = -1
    for i, s in enumerate(sel_arr.tolist(), start=1):
        if prev != -1 and s != prev:
            switch_steps.append(i)
        prev = s
    return {
        "eval_seed": eval_seed,
        "steps_requested": steps,
        "steps_done": done,
        "completed": bool(fall is None and done >= steps - 1),
        "fall_step": fall,
        "h_min": round(float(heights_arr.min()), 3) if len(heights_arr) else 0.0,
        "vx_mean": round(float(vxs_arr.mean()), 4) if len(vxs_arr) else 0.0,
        "err60s": round(float(np.abs(vxs_arr - cmd_vx).mean()), 4) if len(vxs_arr) else 1.0,
        "v_speed_mean": round(float(spd.mean()), 4) if len(spd) else 0.0,
        "disp": round(disp, 3),
        "reset_count": 1 + n_auto,
        "n_auto_resets": n_auto,
        "natural_bin": nat,
        "sel_timeline_b64": base64.b64encode(bytes(sel_timeline)).decode("ascii"),
        "sel_switch_steps": switch_steps,
        "n_switches": len(switch_steps),
        "sel_head_p1_mean": round(p1_sum / max(1, p1_n), 5),
        "sel_state_final": int(sel_arr[-1]) if len(sel_arr) else -1,
    }


def execute_eval_cell(arm: str, v: float, train_seed: int, ckpt_path: Path,
                      vae_path: Path, ref_npz: Path, steps: int,
                      eval_seeds: list, hold_steps: int, smoke: bool,
                      worker_tag: str = "w0") -> dict:
    import torch

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env
    from apt_g1.isaac.ppo_core import AptPPOPolicy

    t0 = _utcnow()
    wall0 = time.monotonic()
    if not ckpt_path.exists():
        raise SystemExit(f"FAIL: checkpoint 不存在: {ckpt_path}")
    ckpt_sha = sha256_file(ckpt_path)

    cfg = build_cell_cfg(str(vae_path), str(ref_npz), arm, hold_steps)
    env = AptFlatG1Env(cfg)
    policy = AptPPOPolicy(
        obs_dim=cfg.observation_space, aux_dim=12, use_phase=False,
        latent_dim=16, gate_k=2,
    ).to("cuda:0")
    policy.load_state_dict(torch.load(ckpt_path, map_location="cuda:0"))
    policy.eval()
    policy_sd_sha = state_dict_sha256(policy.state_dict())

    sd_before = state_dict_sha256(env._vae.state_dict())
    obs_dict, _ = env.reset()
    env._last_obs = obs_dict["policy"]

    episodes = [
        rollout_eval(env, policy, float(v), es, steps, hold_steps)
        for es in eval_seeds
    ]
    sd_after = state_dict_sha256(env._vae.state_dict())

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "cell_id": f"to42-{arm}-v{v:.3f}__s{train_seed}",
        "arm": arm,
        "train_seed": train_seed,
        "target_speed": float(v),
        "smoke": bool(smoke),
        "eval_seeds": list(eval_seeds),
        "eval_seed_note": "预注册 harness 清单 [0,1,2]；operational identity = "
                          "jitter rng(1000+seed) + sim dynamics（TO41 同款）",
        "checkpoint": {
            "ckpt_path": str(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "state_dict_sha256": policy_sd_sha,
            "policy_class": type(policy).__name__,
            "gate_k": 2,
            "obs_dim": int(cfg.observation_space),
            "action_dim": int(cfg.action_space),
        },
        "env_identity": {
            "env_class": type(env).__name__,
            "env_source_file_sha256": sha256_file(ENV_SOURCE),
            "to42_gate_source_file_sha256": sha256_file(GATE_SOURCE),
            "ppo_core_source_file_sha256": sha256_file(PPO_SOURCE),
            "train_source_file_sha256": sha256_file(TRAIN_SOURCE),
            "eval_source_file_sha256": sha256_file(SELF_SOURCE),
            "runtime_commit": runtime_commit(),
            "num_envs": int(env.num_envs),
            "device": str(env.device),
            "sim_dt": float(cfg.sim.dt),
            "decimation": int(cfg.decimation),
            "action_space": int(cfg.action_space),
            "observation_space": int(cfg.observation_space),
            "episode_length_s": float(cfg.episode_length_s),
        },
        "to42_cfg": {
            "to42_sel": cfg.to42_sel,
            "to42_hold_steps": int(cfg.to42_hold_steps),
            "to42_n_sel": int(cfg.to42_n_sel),
            "vx_max": float(cfg.vx_max),
            "latent_vae_n_bins": int(cfg.latent_vae_n_bins),
        },
        "neutral_scaffold": {
            "to_ref": True,
            "to_ref_npz": str(ref_npz),
            "to_ref_npz_sha256": sha256_file(ref_npz),
            "to_ref_obs_zero": True,
            "to_ref_w": 0.0,
            "to_tau": False,
            "disturbance_prob": 0.0,
        },
        "decoder_identity": {
            "vae_path": str(vae_path),
            "checkpoint_sha256": sha256_file(vae_path),
            "state_dict_sha256_before": sd_before,
            "state_dict_sha256_after": sd_after,
            "architecture": {
                "class": type(env._vae).__name__,
                "n_vbins": env._vae.n_vbins,
                "n_dbins": env._vae.n_dbins,
            },
        },
        "episodes": episodes,
        "execution": {
            "status": "completed",
            "steps_requested_per_seed": steps,
            "steps_done_total": sum(e["steps_done"] for e in episodes),
            "torch_version": torch.__version__,
            "python_version": _platform_python(),
            "platform": _platform(),
            "worker_tag": worker_tag,
            "started_utc": t0,
            "finished_utc": _utcnow(),
            "wall_seconds": round(time.monotonic() - wall0, 1),
        },
    }
    del env, policy
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(
        description="TO42 formal eval driver（单 cell 进程；receipt 落盘后硬退出）")
    ap.add_argument("--mode", choices=["static", "execute"], required=True)
    ap.add_argument("--arm", choices=["lsel", "fbkt"], default=None)
    ap.add_argument("--v", type=float, default=None)
    ap.add_argument("--train-seed", type=int, default=None)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--vae-path", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/token_vae_e39/vae.pt")
    ap.add_argument("--ref-npz", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/sync/to38_ref.npz")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/to42/eval")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--hold-steps", type=int, default=25)
    ap.add_argument("--worker-tag", default="w0",
                    help="并发 worker 位次（TO42_PLAN §9 执行身份修订 v2：入 receipt）")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--env-tag", choices=["lab-ts", "cloud"], default="cloud")
    args = ap.parse_args()

    import platform as _pf

    if _pf.system().lower().startswith("windows") and args.mode == "execute":
        raise SystemExit("FAIL: 本机（Windows, 无 venv）禁止 execute（Isaac 栈）")

    if args.mode == "static":
        rows = [{"arm": a, "v": v, "train_seed": s,
                 "cell_id": f"to42-{a}-v{v:.3f}__s{s}"}
                for a in ("lsel", "fbkt") for s in (0, 1) for v in GRID7]
        payload = {
            "artifact": "to42-eval-static-coverage/v1",
            "generated_utc": _utcnow(),
            "expected_receipts": len(rows),
            "expected_episodes": len(rows) * 3,
            "grid7": list(GRID7),
            "rows": rows,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / "static_coverage.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"OK to42 static coverage -> {out}")
        return 0

    if args.arm is None or args.v is None or args.train_seed is None \
            or args.ckpt is None:
        raise SystemExit("FAIL: execute 必须 --arm/--v/--train-seed/--ckpt")
    if args.arm not in ("lsel", "fbkt") or args.train_seed not in (0, 1):
        raise SystemExit("FAIL: --arm ∈ {lsel,fbkt}，--train-seed ∈ {0,1}")
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]

    # preflight（Isaac 启动前——失败保留真实退出码；Isaac atexit 会把 rc 吞成 0）
    if not Path(args.ckpt).exists():
        raise SystemExit(f"FAIL: checkpoint 不存在: {args.ckpt}")
    if not Path(args.vae_path).exists():
        raise SystemExit(f"FAIL: vae 不存在: {args.vae_path}")
    if not Path(args.ref_npz).exists():
        raise SystemExit(f"FAIL: ref npz 不存在: {args.ref_npz}")

    base_dir = args.out_dir / "smoke" if args.smoke else args.out_dir
    receipts_dir = base_dir / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"receipt_to42-{args.arm}-v{args.v:.3f}__s{args.train_seed}.json"

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = 1
    launcher_args.headless = True
    launcher_args.output_dir = str(base_dir)
    AppLauncher(launcher_args)

    print(f"[to42-eval] {args.arm} v={args.v} seed={args.train_seed} "
          f"× {len(eval_seeds)} eval seeds × {args.steps} steps", flush=True)
    try:
        receipt = execute_eval_cell(
            args.arm, float(args.v), args.train_seed, args.ckpt, args.vae_path,
            args.ref_npz, args.steps, eval_seeds, args.hold_steps, args.smoke,
            args.worker_tag)
    except SystemExit:
        raise
    except Exception:
        # 绕过 Isaac atexit 的 rc=0 吞没，保留失败退出码
        import traceback

        traceback.print_exc()
        print("[to42-eval] FAILED（见上方 traceback）", flush=True)
        os._exit(1)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 数据保全（owner 指令 2026-09-03：云环境不稳，数据必须可下载保存）——
    # receipt 全文 gz+b64 进任务日志：平台 ckpt 上传不可靠时，日志流即完整
    # 数据通道（单 receipt gz+b64 ~10KB，28 cells ~300KB 日志增量，可接受）
    import gzip
    _b64 = base64.b64encode(gzip.compress(receipt_path.read_bytes(), 9)).decode("ascii")
    print(f"TO42_RECEIPT_B64:{receipt_path.name}:{_b64}", flush=True)
    eps = receipt["episodes"]
    print(f"[to42-done] {receipt['cell_id']} "
          f"completed={[e['completed'] for e in eps]} "
          f"err60s={[e['err60s'] for e in eps]} "
          f"vx={[e['vx_mean'] for e in eps]} "
          f"switches={[e['n_switches'] for e in eps]} "
          f"wall={receipt['execution']['wall_seconds']}s "
          f"-> {receipt_path.name}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
