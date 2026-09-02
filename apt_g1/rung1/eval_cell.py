"""TO41 Rung 1 formal eval driver：28-cell × 训练 seed 的逐 cell 正式评测。

角色（SCRIPT_MAP 登记）：**state-changing execution code**。execution freeze
（三十八轮，76954a0）后的 Rung 1 compute eval 侧实现；把冻结 treatment
specification（TO40C_PLAN §10 + D/L 协议继承的 Mode A runtime）变成
正式评测可执行代码。**只实现，不重新解释**——协议侧争议唯一出口 =
genuine incompatibility → owner reopen。

架构（三十九轮 owner 裁定落实；与 D/L 阶段完全同构）：

    (v, C) → C_id     = mode_a_runtime.mapping_lookup   （lookup 1，condition 轴）
    v     → τ(v)      = mode_a_runtime.material_lookup   （lookup 2，material 轴）
    (v,C,z) → (C_id, τ(v), z) = resolve_cell 组合（Mode A：τ(v,C1)=τ(v,C2)
                          由两 lookup 的机械分离保证，非 dry-run 特例）

每 cell × 每 train_seed 一个独立进程（与 launch_sanity 同款 per-cell 进程
模型；AppLauncher 逐 cell 启动，receipt 落盘后 os._exit 硬退出）：

    cfg = launch_sanity.build_cell_cfg（原样复用 → 两臂 cfg diff == {to_tau}
          与 L3 conformance 直接继承；τ 材料 = L0 derived LUT 经
          cfg.to_ref_npz 进入冻结加载路径）
    env = AptFlatG1Env(cfg)（零 env 文件改动；冻结锚 preflight fail-fast）
    policy = AptPPOPolicy(latent_dim=16) ← selection manifest 固定的单一
          checkpoint（50-iter 窗口最优；同一 (arm, seed) 的全部 14 cells
          共用同一 ckpt——checkpoint selection 与 evaluation condition 隔离，
          禁止 C1/C2 或 ON/OFF 各选各的）
    wiring = env_wiring.ConditionOverrideHandle + TauConsumptionProbe
          （原样复用：override 位于 decode 输入边界，每 call 记录
          natural/applied 双记录 → 60s 正式 rollout 的 reset 后 persistence
          与 300-step probe 同一机制）

评测语义（预注册 harness 原样继承，TO38 §3 骨架 / TO40C §4 / eval_apt_isaac）：
每 eval seed 一次 jitter_and_reset（MuJoCo-parity）+ 60 s（3000 控制步）
恒定 cmd_vx 每 step 重申（reset 重采样 U(vx_min,vx_max) 的确定性压制，沿
sanity 语义）+ 确定性策略动作（deterministic=True）+ termination 即停
（fall_step 记录）。eval seed list = {0,1,2}（既有协议固定清单，原样使用
不新选；eval 随机性的 operational identity = jitter rng(1000+seed) + sim
动力学，与 training seed 的 policy init/rollout sampling 是不同 artifact，
整数重叠 {0,1} 不构成 replay）。

receipt = record（rung1-eval-receipt/v1）：outcome 聚合字段在本层允许出现
（本层就是正式 eval），但 verdict / PASS 仍只出自 eval_checker（§9 纪律）；
target_speed 恒为冻结 grid 值，v_realized / abs_err 只作 material diagnostic
照录（§6/§8：不进分析模型、不做筛选器、不得改写 cell 身份）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from apt_g1.rung1 import env_wiring
from apt_g1.rung1.env_wiring import canonical_array_sha256, tau_buffer_snapshot
from apt_g1.rung1.launch_sanity import (
    LUT_ARRAY_FIELDS,
    _cfg_snapshot,
    _preflight,
    _set_commands,
    build_cell_cfg,
    load_lut_manifest,
)
from apt_g1.rung1.mode_a_runtime import (
    REPO_ROOT,
    _platform,
    _platform_python,
    _utcnow,
    enumerate_cells,
    find_material,
    load_availability,
    load_mapping,
    load_registry,
    resolve_cell,
    sha256_file,
    state_dict_sha256,
)

RECEIPT_SCHEMA = "rung1-eval-receipt/v1"
DEFAULT_OUT = REPO_ROOT / "apt_g1/outputs/sync/to41_eval"
DEFAULT_SELECTION = DEFAULT_OUT / "ckpt_selection.json"
# τ_ff 臂 → 训练臂（预注册主结果：训练/评测 τ 状态一致；交叉注入属 TO40C
# 诊断语义，不在 Rung 1 主矩阵，本 driver 不实现）
POLICY_ARMS = {"on": "t10", "off": "ctrl"}
ENV_SOURCE = REPO_ROOT / "apt_g1/isaac/apt_flat_env.py"
ARCH_SOURCE = REPO_ROOT / "apt_g1/train_token_vae_e39.py"
TOKEN_VAE_SOURCE = REPO_ROOT / "apt_g1/isaac/token_window_vae.py"


def runtime_commit() -> str:
    """执行身份数据：sync clone HEAD（短 sha）；非 git 环境如实记 unknown。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def verify_lut_identity(lut_path: Path, lut_entry: dict) -> None:
    """消费文件 vs 冻结 manifest 的数组级身份核对（机械 conformance，fail-fast）。

    这是 driver 侧 preflight（N1 负例防线）：被消费 LUT 的 5 个数组字段
    canonical sha 必须逐一等于 LUT manifest 冻结值。判定仍以 eval_checker
    的独立重算为准（本检查属执行契约，非 verdict）。
    """
    with np.load(lut_path) as z:
        for field in LUT_ARRAY_FIELDS:
            if field not in z.files:
                raise SystemExit(f"FAIL lut identity: {lut_path} 缺字段 {field}")
            got = canonical_array_sha256(np.asarray(z[field]))
            expect = lut_entry["lut_array_sha256"][field]
            if got != expect:
                raise SystemExit(
                    f"FAIL lut identity: {lut_path} 字段 {field} 数组 sha != 冻结 "
                    f"manifest（got {got[:16]}… expect {expect[:16]}…）——材料消费"
                    f"身份被破坏，禁止执行（若为实现 bug 修 implementation 重跑，"
                    f"禁止改材料）")


def load_selection(path: Path) -> dict:
    """checkpoint selection manifest（select_checkpoint.py 产物）。

    身份 = 预注册 50-iter 窗口最优规则机械化产物；driver 只消费不选择。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("artifact") != "rung1-eval-ckpt-selection/v1":
        raise SystemExit(f"FAIL: {path} 不是 rung1-eval-ckpt-selection/v1")
    runs = raw.get("runs", {})
    for arm in ("ctrl", "t10"):
        for seed in ("0", "1"):
            if arm not in runs or seed not in runs[arm]:
                raise SystemExit(f"FAIL: selection manifest 缺 runs.{arm}.{seed}")
            entry = runs[arm][seed]
            for key in ("ckpt_file", "ckpt_sha256", "window_iters", "window_mean_rew"):
                if key not in entry:
                    raise SystemExit(f"FAIL: selection runs.{arm}.{seed} 缺 {key}")
    return raw


def rollout_eval(env, policy, cmd_vx: float, eval_seed: int, steps: int,
                 cond: "env_wiring.ConditionOverrideHandle",
                 tau: "env_wiring.TauConsumptionProbe") -> dict:
    """单 eval-seed episode：jitter_and_reset + 恒定 cmd + 策略驱动 60s。

    语义逐字继承 eval_apt_isaac.rollout（TO38/TO40C 评测 harness）+ sanity
    的 per-step cmd 重申；_last_obs 缓存语义与既有 harness 一致（jitter 后
    不额外刷新——首步沿用上轮末 obs，为 TO38/39/40C 数字可比性的既有实现）。
    只产 record，不产 verdict。
    """
    import torch

    from apt_g1.isaac.eval_apt_isaac import jitter_and_reset

    cond_calls_before = cond.n_calls
    tau_calls_before = tau.n_calls
    jitter_and_reset(env, seed=eval_seed)
    _set_commands(env, cmd_vx)
    boundaries = [{"step": 0, "type": "episode_start",
                   "cond_calls_before": cond_calls_before,
                   "tau_calls_before": tau_calls_before}]

    prev_eplen = int(env.episode_length_buf[0].item())
    heights, vxs, vys, xys = [], [], [], []
    fall = None
    n_auto = 0
    n_reassert = 1
    done = 0
    for t in range(1, steps + 1):
        _set_commands(env, cmd_vx)
        n_reassert += 1
        with torch.no_grad():
            act, _, _, _, _ = policy.act(env._last_obs, deterministic=True)
        obs_dict, reward, term, trunc, _ = env.step(act["latent"])
        env._last_obs = obs_dict["policy"]
        done += 1
        eplen = int(env.episode_length_buf[0].item())
        if eplen < prev_eplen:
            n_auto += 1
            boundaries.append({"step": t, "type": "auto_reset",
                               "cond_calls_after": cond.n_calls,
                               "tau_calls_after": tau.n_calls})
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
    disp = float(np.linalg.norm(xys_arr[-1] - xys_arr[0])) if len(xys_arr) > 1 else 0.0
    return {
        "eval_seed": eval_seed,
        "steps_requested": steps,
        "steps_done": done,
        "completed": bool(fall is None and done >= steps - 1),
        "fall_step": fall,
        "h_min": round(float(heights_arr.min()), 3) if len(heights_arr) else 0.0,
        "vx_mean": round(float(vxs_arr.mean()), 4) if len(vxs_arr) else 0.0,
        "v_speed_mean": round(float(spd.mean()), 4) if len(spd) else 0.0,
        "disp": round(disp, 3),
        "reset_count": 1 + n_auto,
        "n_auto_resets": n_auto,
        "n_cmd_reassertions": n_reassert,
        "boundaries": boundaries,
    }


def execute_eval_cell(cell: dict, cell_index: int, train_seed: int,
                      mapping: dict, availability: dict, registry: dict,
                      lut_manifest: dict, vae_path: Path,
                      materials_roots: list[Path], selection: dict,
                      steps: int, eval_seeds: list[int], smoke: bool) -> dict:
    """单 (cell × train_seed)：新建 env + policy → 接线 → rollout → receipt。

    材料双层身份与 launch_sanity 完全同构（canonical 冻结 npz = D 链锚；
    derived LUT = env 实际消费形式，数组级身份锚）。
    """
    import torch

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env
    from apt_g1.isaac.ppo_core import AptPPOPolicy

    t0 = _utcnow()
    wall0 = time.monotonic()
    assignment = resolve_cell(cell, mapping, availability)
    mat = assignment["tau_material"]
    mat_path, mat_root = find_material(mat["artifact"], materials_roots)
    if mat_path is None:
        raise FileNotFoundError(mat["artifact"])
    v_key = round(float(cell["target_speed"]), 3)
    lut_entry = lut_manifest["entries"][v_key]
    lut_path = lut_manifest["manifest_path"].parent / lut_entry["lut_file"]
    verify_lut_identity(lut_path, lut_entry)

    arm = POLICY_ARMS[cell["tau_ff"]]
    sel = selection["runs"][arm][str(train_seed)]
    ckpt_path = Path(sel["ckpt_file"])
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"FAIL: selected checkpoint 不存在: {ckpt_path}")
    ckpt_sha = sha256_file(ckpt_path)
    if ckpt_sha != sel["ckpt_sha256"]:
        raise SystemExit(
            f"FAIL: checkpoint sha256 != selection manifest（got {ckpt_sha[:16]}… "
            f"expect {sel['ckpt_sha256'][:16]}…）——selection 与消费文件不一致")

    cfg = build_cell_cfg(cell, lut_path, vae_path)
    env = AptFlatG1Env(cfg)

    policy = AptPPOPolicy(
        obs_dim=cfg.observation_space, aux_dim=12, use_phase=False, latent_dim=16,
    ).to("cuda:0")
    policy.load_state_dict(torch.load(ckpt_path, map_location="cuda:0"))
    policy.eval()
    policy_sd_sha = state_dict_sha256(policy.state_dict())

    sd_before = state_dict_sha256(env._vae.state_dict())
    buf_pre = tau_buffer_snapshot(env)
    cond = env_wiring.ConditionOverrideHandle(env, assignment["speed_bin"],
                                              assignment["dir_bin"])
    tau = env_wiring.TauConsumptionProbe(env)

    obs_dict, _ = env.reset()
    env._last_obs = obs_dict["policy"]

    episodes = [
        rollout_eval(env, policy, float(cell["target_speed"]), es, steps, cond, tau)
        for es in eval_seeds
    ]

    buf_post = tau_buffer_snapshot(env)
    sd_after = state_dict_sha256(env._vae.state_dict())
    mat_file_sha = sha256_file(mat_path)
    lut_file_sha = sha256_file(lut_path)
    reg_id = next((k for k, s in registry["sources"].items()
                   if s["path"] == mat["artifact"]), None)
    reg_entry = registry["sources"].get(reg_id) if reg_id else None
    nat_bin = int(np.searchsorted(
        np.linspace(0.0, cfg.vx_max, cfg.latent_vae_n_bins + 1)[1:-1],
        float(cell["target_speed"]), side="left"))
    nat_cond = next((cid for cid, c in mapping["conditions"].items()
                     if c["speed_bin"] == nat_bin and c["dir_bin"] == assignment["dir_bin"]),
                    None)

    steps_done_total = sum(e["steps_done"] for e in episodes)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "cell_id": cell["cell_id"],
        "cell_index": cell_index,
        "train_seed": train_seed,
        "policy_arm": arm,
        "smoke": bool(smoke),
        "target_speed": float(cell["target_speed"]),
        "condition_arm": cell["condition_arm"],
        "tau_ff": cell["tau_ff"],
        "eval_seeds": list(eval_seeds),
        "eval_seed_note": "pre-registered harness list (TO38 §3 / TO40C §4)；"
                          "operational identity = jitter rng(1000+seed) + sim dynamics",
        "assignment": {
            "decoder_condition_id": assignment["decoder_condition_id"],
            "speed_bin": assignment["speed_bin"],
            "dir_bin": assignment["dir_bin"],
            "natural_condition_id": nat_cond,
            "natural_speed_bin": nat_bin,
            "selection_source": "frozen_mapping_v2_lookup",
        },
        "checkpoint": {
            "policy_arm": arm,
            "train_seed": train_seed,
            "ckpt_path": str(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "selection_window_iters": sel["window_iters"],
            "selection_window_mean_rew": sel["window_mean_rew"],
            "selection_rule": selection.get("rule"),
            "state_dict_sha256": policy_sd_sha,
            "policy_class": type(policy).__name__,
            "obs_dim": int(cfg.observation_space),
            "action_dim": int(cfg.action_space),
            "shared_across_all_cells_of_arm_seed": True,
        },
        "env_identity": {
            "env_class": type(env).__name__,
            "env_source_file_sha256": sha256_file(ENV_SOURCE),
            "arch_source_file_sha256": sha256_file(ARCH_SOURCE),
            "token_window_vae_source_sha256": sha256_file(TOKEN_VAE_SOURCE),
            "runtime_commit": runtime_commit(),
            "num_envs": int(env.num_envs),
            "device": str(env.device),
            "sim_dt": float(cfg.sim.dt),
            "decimation": int(cfg.decimation),
            "action_space": int(cfg.action_space),
            "observation_space": int(cfg.observation_space),
            "env_instance_fresh_per_cell": True,
            "process_model": "one-fresh-process-per-(cell,train_seed) (AppLauncher; "
                             "receipt 落盘后 os._exit)",
        },
        "cfg_snapshot": _cfg_snapshot(cfg),
        "tau_material": {
            "lut_manifest_sha256": lut_manifest["manifest_sha256"],
            "frozen_material": {
                "artifact": mat["artifact"],
                "materials_root_used": str(mat_root),
                "npz_path": str(mat_path),
                "file_sha256": mat_file_sha,
                "sha256_16": mat_file_sha[:16],
                "source_lineage": mat["source"],
                "v_realized": mat["v_realized"],
                "abs_err": mat["abs_err"],
                "registry_id": reg_id,
                "registry_sha256_16": reg_entry["sha256_16"] if reg_entry else None,
            },
            "derived_lut": {
                "file": str(lut_path),
                "file_sha256": lut_file_sha,
                "array_sha256": {k: lut_entry["lut_array_sha256"][k] for k in LUT_ARRAY_FIELDS},
                "source_sha256_manifest": lut_entry["source_sha256"],
                "source_artifact_manifest": lut_entry["source_artifact"],
                "m_per_phase": lut_entry["m_per_phase"],
                "T": lut_entry["T"],
                "v_avg": lut_entry["v_avg"],
                "wrap_gap_q": lut_entry["wrap_gap_q"],
                "applied_to_env": True,
                "cfg_to_ref_npz": str(lut_path),
                "identity_verified_preflight": True,
            },
            "buffer_shape": buf_pre["shape"],
            "buffer_dtype": buf_pre["dtype"],
            "buffer_sha256_pre": buf_pre["buffer_sha256"],
            "buffer_sha256_post": buf_post["buffer_sha256"],
            "to_vavg": float(env._to_vavg),
            "to_m": int(env._to_m),
            "to_rate": float(env._to_rate),
            "to_kp_sagittal6": [round(float(x), 6) for x in env._to_kp.tolist()],
        },
        "condition_override": cond.record_block(),
        "tau_consumption": tau.record_block(),
        "decoder_identity": {
            "vae_path": str(vae_path),
            "checkpoint_sha256": sha256_file(vae_path),
            "state_dict_sha256_before": sd_before,
            "state_dict_sha256_after": sd_after,
            "state_dict_key_shapes": {
                k: list(v.shape) for k, v in sorted(env._vae.state_dict().items())},
            "architecture": {
                "class": type(env._vae).__name__,
                "token_dim": env._vae.token_dim,
                "window": env._vae.window,
                "latent_dim": env._vae.latent_dim,
                "n_vbins": env._vae.n_vbins,
                "n_dbins": env._vae.n_dbins,
            },
            "mode_layout": {
                "latent_mode": True,
                "latent_dir_bins": True,
                "decode_call_form": "decode(z, phase_sc(sin,cos), v_bin, d_bin)",
            },
        },
        "episodes": episodes,
        "execution": {
            "status": "completed",
            "steps_requested_per_seed": steps,
            "steps_done_total": steps_done_total,
            "expected_decode_calls": steps_done_total,
            "expected_tau_calls_on": steps_done_total * int(cfg.decimation),
            "torch_version": torch.__version__,
            "python_version": _platform_python(),
            "platform": _platform(),
            "started_utc": t0,
            "finished_utc": _utcnow(),
            "wall_seconds": round(time.monotonic() - wall0, 1),
        },
    }
    # 不调 env.close()（sim.clear_instance / 挂死 gotcha）；receipt 含全部快照，
    # 调用方落盘后 os._exit 终止进程
    del env, policy
    return receipt


def static_coverage(out_dir: Path, selection_path: Path) -> Path:
    """本机可跑的评测矩阵静态覆盖核对（无 torch/isaac；非正式 artifact）。

    28 cells × train seeds {0,1} → 预期 (condition, LUT, policy arm/ckpt 槽)；
    selection manifest 未产出时 ckpt 槽记 TBD（driver execute 时仍会硬性
    要求 manifest 存在）。
    """
    mapping = load_mapping()
    availability = load_availability()
    registry = load_registry()
    lut_manifest = load_lut_manifest()
    selection = None
    if selection_path.exists():
        selection = load_selection(selection_path)
    rows = []
    for i, cell in enumerate(enumerate_cells(mapping)):
        a = resolve_cell(cell, mapping, availability)
        v_key = round(float(cell["target_speed"]), 3)
        lut = lut_manifest["entries"][v_key]
        arm = POLICY_ARMS[cell["tau_ff"]]
        for seed in (0, 1):
            ckpt = None
            if selection is not None:
                ckpt = selection["runs"][arm][str(seed)]["ckpt_file"]
            rows.append({
                "cell_id": cell["cell_id"],
                "cell_index": i,
                "train_seed": seed,
                "target_speed": cell["target_speed"],
                "condition_arm": cell["condition_arm"],
                "tau_ff": cell["tau_ff"],
                "expected_condition": a["decoder_condition_id"],
                "expected_lut": lut["lut_file"],
                "expected_material": a["tau_material"]["artifact"],
                "policy_arm": arm,
                "selected_ckpt": ckpt,
            })
    payload = {
        "artifact": "rung1-eval-static-coverage/v1",
        "environment_tag": "local",
        "generated_utc": _utcnow(),
        "expected_receipts": 56,
        "resolved_rows": len(rows),
        "note": "28 cells × 2 train seeds；正式 receipt 仅 lab-ts execute 产出",
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "eval_static_coverage.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rung 1 formal eval driver（28-cell × train_seed，per-cell 进程）")
    ap.add_argument("--mode", choices=["static", "execute"], required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION)
    ap.add_argument("--vae-path", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/token_vae_e39/vae.pt")
    ap.add_argument("--materials-root", type=Path, nargs="+",
                    default=[REPO_ROOT / "apt_g1/outputs"],
                    help="material npz 搜索根（可多个，按序首中）")
    ap.add_argument("--env-tag", choices=["lab-ts", "local"], required=True,
                    help="lab-ts = frozen execution environment；本机必须标 local")
    ap.add_argument("--cell-index", type=int, default=None,
                    help="execute 必填：0..27 冻结枚举序号（per-cell 进程模型）")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="execute 必填：0|1（selection manifest 的训练 seed 槽）")
    ap.add_argument("--steps", type=int, default=3000,
                    help="每 eval seed 控制步数（50 Hz；默认 3000 = 60 s）")
    ap.add_argument("--eval-seeds", default="0,1,2",
                    help="预注册 eval seed 清单（默认 0,1,2 = 既有协议固定值，勿新选）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：receipt 写入 out/smoke/ 并标记 smoke=true"
                         "（不进正式 coverage，checker 不审计）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.env_tag == "lab-ts" and _platform().lower().startswith("windows"):
        raise SystemExit("FAIL: 本机（Windows, 无 venv）禁止以 lab-ts 身份执行")

    if args.mode == "static":
        if args.env_tag == "lab-ts":
            raise SystemExit("FAIL: static 为环境无关配置层核对，恒为 local 口径")
        out = static_coverage(args.out, args.selection_manifest)
        print(f"OK eval static coverage -> {out}")
        return 0

    import os

    if args.cell_index is None or args.train_seed is None:
        raise SystemExit("FAIL: execute 模式必须 --cell-index 0..27 且 --train-seed 0|1")
    if args.train_seed not in (0, 1):
        raise SystemExit("FAIL: --train-seed 仅允许 0|1（预注册 2 训练 seed）")

    mapping = load_mapping()
    lut_manifest = load_lut_manifest()
    cells = enumerate_cells(mapping)
    if not (0 <= args.cell_index < len(cells)):
        raise SystemExit(f"FAIL: --cell-index {args.cell_index} 越界（0..{len(cells) - 1}）")
    cell = cells[args.cell_index]

    _preflight(mapping, list(args.materials_root), args.vae_path, lut_manifest)
    availability = load_availability()
    registry = load_registry()
    selection = load_selection(args.selection_manifest)

    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]
    if eval_seeds != [0, 1, 2]:
        raise SystemExit(
            f"FAIL: eval seeds {eval_seeds} != 预注册清单 [0,1,2]（既有协议固定值，"
            "改清单属协议变更，须 owner reopen）")

    base_dir = args.out / "smoke" if args.smoke else args.out
    receipts_dir = base_dir / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"receipt_{cell['cell_id']}__s{args.train_seed}.json"
    if receipt_path.exists() and not args.force:
        raise SystemExit(f"FAIL: {receipt_path} 已存在（--force 覆盖该 cell）")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.num_envs = 1
    launcher_args.headless = True
    launcher_args.output_dir = str(base_dir)
    AppLauncher(launcher_args)

    print(f"[eval] cell {args.cell_index} seed{args.train_seed}: {cell['cell_id']} × "
          f"{len(eval_seeds)} eval seeds × {args.steps} steps，per-cell 进程，"
          f"仅记录不判定", flush=True)
    receipt = execute_eval_cell(
        cell, args.cell_index, args.train_seed, mapping, availability, registry,
        lut_manifest, args.vae_path, list(args.materials_root), selection,
        args.steps, eval_seeds, args.smoke)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    eps = receipt["episodes"]
    print(f"[done] {cell['cell_id']}__s{args.train_seed} "
          f"cond={receipt['assignment']['decoder_condition_id']} "
          f"ckpt={Path(receipt['checkpoint']['ckpt_path']).name} "
          f"steps={receipt['execution']['steps_done_total']} "
          f"completed={[e['completed'] for e in eps]} "
          f"vx={[e['vx_mean'] for e in eps]} "
          f"wall={receipt['execution']['wall_seconds']}s "
          f"-> {receipt_path.name}", flush=True)
    # Isaac 退出挂死 gotcha：receipt 已落盘，硬退出防挂死
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
