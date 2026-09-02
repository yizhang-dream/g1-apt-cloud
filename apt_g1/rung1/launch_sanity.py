"""TO41 Rung 1 launch sanity：28-cell 真实 env 接线验证（state-changing execution code）。

角色（SCRIPT_MAP 登记）：**state-changing execution code**。三十七轮 owner
裁定的纯执行 gate（不改 §10 科学设计、不是新 protocol round）：把冻结
Mode A runtime（mapping_lookup + material_lookup）接入**未改动的**真实
`apt_flat_env.py` 的 τ 注入/控制路径，完成 7 v × {C1,C2} × {τ ON,OFF} =
28 cell 的 L1–L4 接线验证。只测接线，不测性能（receipt 无任何
performance / locomotion 字段；verdict 只出自 l_checker）。

链位（三十七轮裁定后）：

    D1/D2/D3 decode conformance → L1–L4 env launch sanity
        → execution wiring PASS → owner execution freeze → Rung 1 compute

cell 执行结构（镜像 Rung 1 compute 的 per-cell 进程语义）：

    每 cell 新建真实 AptFlatG1Env（cfg = TO40C ctrl/t10 配方 × 冻结 τ(v)
    材料 × latent-dir-bins + e39 vae），jitter_and_reset（eval 同款）+
    恒定 cmd_vx 每 step 重申（reset 重采样 U(vx_min,vx_max) 的确定性压制），
    中段强制 episode boundary，零策略动作（z=0）驱动真实 step 循环；
    decode/τ 消费经由 env_wiring 的实例级探针全量记录。完成后 env.close()。

判据唯一事实源 = refine-logs/TO41_LAUNCH_SANITY.md；本文件只把 gate 变成
可执行代码。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from apt_g1.rung1 import env_wiring
from apt_g1.rung1.env_wiring import canonical_array_sha256, tau_buffer_snapshot
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
    mapping_lookup,
    resolve_cell,
    sha256_file,
    state_dict_sha256,
)

RECEIPT_SCHEMA = "rung1-launch-sanity-receipt/v1"
ENV_SOURCE = REPO_ROOT / "apt_g1/isaac/apt_flat_env.py"
ARCH_SOURCE = REPO_ROOT / "apt_g1/train_token_vae_e39.py"
TOKEN_VAE_SOURCE = REPO_ROOT / "apt_g1/isaac/token_window_vae.py"

# cfg 快照闭集（receipt cfg_snapshot 的全部键；两臂 diff 判定 L3 的证据面）
CFG_SNAPSHOT_FIELDS = [
    "scene.num_envs", "terrain_seed", "sim.dt", "decimation",
    "episode_length_s", "action_space", "observation_space", "disturbance_prob",
    "latent_mode", "latent_speed_bins", "latent_dir_bins", "latent_residual",
    "latent_vae_n_bins", "latent_vae_n_dbins", "latent_cmd_phase_rate",
    "latent_vae_path", "vx_max", "vx_min",
    "use_sonic_prior", "sonic_decoder_path", "router_model_dir", "use_2hz_gate",
    "to_ref", "to_ref_npz", "to_ref_obs_zero", "to_ref_w",
    "to_tau", "to_tau_w", "to_ref_gate2", "to_ref_sigma2",
]


def _cfg_snapshot(cfg) -> dict:
    out = {}
    for f in CFG_SNAPSHOT_FIELDS:
        v = cfg
        for part in f.split("."):
            v = getattr(v, part)
        if f == "to_ref_npz":
            v = str(v)
        out[f] = v
    out["terrain"] = "plane_importer(seed=0,noise=0.04)"
    return out


def build_cell_cfg(cell: dict, mat_path: Path, vae_path: Path):
    """TO40C ctrl/t10 配方 → env cfg（镜像 eval_apt_isaac.main 的 cli→cfg
    映射；训练侧 action_space=16 语义 = train_apt_isaac.py:226）。

    两臂唯一差异 = to_tau（OFF/ON）；to_ref_npz 两臂同为该 cell 的冻结
    τ(v) 材料——OFF 臂 LUT 照载但 obs 块置零 + reward 权 0 + to_tau 关，
    材料对动力学可证中性（与 TO40C ctrl 的 to38_ref.npz 同构，仅换材料
    身份；Mode A same-τ fingerprint 因此在 env 层对 4 cell/v 成立）。
    """
    from apt_g1.isaac.apt_flat_env import AptFlatG1EnvCfg
    from apt_g1.isaac.terrain_cfg import make_terrain_importer_cfg

    cfg = AptFlatG1EnvCfg()
    cfg.observation_space += 14          # latent_mode（_last_phase 2→16）
    cfg.observation_space += 12          # to_ref 参考块（两臂同开，obs 置零）
    cfg.action_space = 16                # latent z only（无 aux/gate/residual）
    cfg.scene.num_envs = 1
    cfg.terrain = make_terrain_importer_cfg("plane", 0.04, seed=0)
    cfg.latent_mode = True
    cfg.latent_speed_bins = True         # TO38a 配方形状（dir_bins 分支优先）
    cfg.latent_dir_bins = True
    cfg.latent_residual = False
    cfg.latent_cmd_phase_rate = False    # E27 固定标量 cadence
    cfg.latent_vae_path = str(vae_path)
    cfg.to_ref = True
    cfg.to_ref_npz = str(mat_path)
    cfg.to_ref_obs_zero = True
    cfg.to_ref_w = 0.0
    cfg.to_tau = cell["tau_ff"] == "on"
    cfg.to_tau_w = 1.0
    cfg.disturbance_prob = 0.0
    cfg.episode_length_s = 120.0
    return cfg


def static_coverage(out_dir: Path) -> Path:
    """本机可跑的 28-cell 配置层覆盖核对（无 torch/isaac；非 L artifact）。"""
    mapping = load_mapping()
    availability = load_availability()
    registry = load_registry()
    rows = []
    for i, cell in enumerate(enumerate_cells(mapping)):
        a = resolve_cell(cell, mapping, availability)
        nat_bin = _natural_bin(cell["target_speed"])
        nat_id = next((cid for cid, c in mapping["conditions"].items()
                       if c["speed_bin"] == nat_bin and c["dir_bin"] == a["dir_bin"]), None)
        rows.append({
            "cell_id": cell["cell_id"],
            "cell_index": i,
            "target_speed": cell["target_speed"],
            "condition_arm": cell["condition_arm"],
            "tau_ff": cell["tau_ff"],
            "expected_condition": a["decoder_condition_id"],
            "expected_speed_bin": a["speed_bin"],
            "expected_dir_bin": a["dir_bin"],
            "natural_condition": nat_id,
            "natural_speed_bin": nat_bin,
            "expected_material": a["tau_material"]["artifact"],
            "expected_to_tau": cell["tau_ff"] == "on",
            "expected_to_ref_npz": a["tau_material"]["artifact"],
            "registry_id": next((k for k, s in registry["sources"].items()
                                 if s["path"] == a["tau_material"]["artifact"]), None),
        })
    payload = {
        "artifact": "rung1-launch-sanity-static-coverage/v1",
        "environment_tag": "local",
        "generated_utc": _utcnow(),
        "expected_cells": 28,
        "resolved_cells": len(rows),
        "note": "静态配置层核对；非 L artifact（真实 env receipt 仅 lab-ts）",
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sanity_static_coverage.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def _natural_bin(v: float, vx_max: float = 0.8, n: int = 3) -> int:
    """冻结 bucketize（right=False）的 numpy 等价：first index with edge >= v。
    网格点均不触界（edges 0.2667/0.5333，最近点 0.275 相距 8.3e-3），无 ULP 敏感。"""
    edges = np.linspace(0.0, vx_max, n + 1)[1:-1]
    return int(np.searchsorted(edges, v, side="left"))


def _preflight(mapping: dict, materials_roots: list[Path], vae_path: Path) -> None:
    """冻结锚 preflight（Isaac 启动前 fail-fast）：三源哈希 + 7 材料 + vae。"""
    checks = [
        (ENV_SOURCE, mapping["hashes"]["preprocessing"], "apt_flat_env.py"),
        (ENV_SOURCE, mapping["hashes"]["normalization"], "apt_flat_env.py(normalization)"),
        (ARCH_SOURCE, mapping["hashes"]["decoder_architecture"], "train_token_vae_e39.py"),
    ]
    for path, expect, label in checks:
        got = sha256_file(path)
        if got != expect:
            raise SystemExit(
                f"FAIL preflight: {label} sha256 != 冻结锚（got {got[:16]}… expect {expect[:16]}…）"
                "——冻结 env 源被改动即 genuine incompatibility，禁止带病执行")
    if not vae_path.exists():
        raise SystemExit(f"FAIL preflight: vae.pt 不存在: {vae_path}")
    got = sha256_file(vae_path)
    if got != mapping["hashes"]["decoder_checkpoint"]:
        raise SystemExit(f"FAIL preflight: vae.pt sha256 != mapping decoder_checkpoint_hash")
    availability = load_availability()
    for v, mat in sorted(availability.items()):
        if find_material(mat["artifact"], materials_roots) is None:
            raise SystemExit(f"FAIL preflight: material 不可达: {mat['artifact']}")
    print(f"[preflight] 冻结锚三源 OK；vae.pt {got[:16]}…；7/7 material 可达")


def _set_commands(env, v: float) -> None:
    import torch

    env.router_commands[0] = None
    env._commands[0] = torch.tensor([v, 0.0, 0.0], dtype=torch.float32, device=env.device)


def execute_cell(cell: dict, cell_index: int, mapping: dict, availability: dict,
                 registry: dict, vae_path: Path, materials_roots: list[Path],
                 steps: int, boundary_step: int) -> dict:
    """单 cell：新建真实 env → 接线 → 真实 step 循环 → 快照 → close → receipt。"""
    import torch

    from apt_g1.isaac.apt_flat_env import AptFlatG1Env
    from apt_g1.isaac.eval_apt_isaac import jitter_and_reset

    t0 = _utcnow()
    wall0 = time.monotonic()
    assignment = resolve_cell(cell, mapping, availability)
    mat = assignment["tau_material"]
    mat_path, mat_root = find_material(mat["artifact"], materials_roots)
    if mat_path is None:
        raise FileNotFoundError(mat["artifact"])

    cfg = build_cell_cfg(cell, mat_path, vae_path)
    env = AptFlatG1Env(cfg)

    sd_before = state_dict_sha256(env._vae.state_dict())
    buf_pre = tau_buffer_snapshot(env)
    cond = env_wiring.ConditionOverrideHandle(env, assignment["speed_bin"], assignment["dir_bin"])
    tau = env_wiring.TauConsumptionProbe(env)

    v = cell["target_speed"]
    n_reassert = 0

    def _reset_and_cmd(seed: int) -> None:
        nonlocal n_reassert
        jitter_and_reset(env, seed=seed)
        _set_commands(env, v)
        n_reassert += 1

    boundaries = [{"step": 0, "type": "episode_start",
                   "cond_calls_after": 0, "tau_calls_after": 0}]
    _reset_and_cmd(1000 + cell_index)
    prev_eplen = int(env.episode_length_buf[0].item())
    action = torch.zeros(1, cfg.action_space, device=env.device)
    n_auto = 0
    done = 0
    for step in range(1, steps + 1):
        if step == boundary_step:
            cb, tb = cond.n_calls, tau.n_calls
            _reset_and_cmd(10_000 + cell_index)
            boundaries.append({"step": step, "type": "forced",
                               "cond_calls_before": cb,
                               "cond_calls_after": cond.calls_after(cb),
                               "tau_calls_before": tb,
                               "tau_calls_after": tau.calls_after(tb)})
        _set_commands(env, v)
        n_reassert += 1
        env.step(action)
        done += 1
        eplen = int(env.episode_length_buf[0].item())
        if eplen < prev_eplen:
            n_auto += 1
            boundaries.append({"step": step, "type": "auto_reset",
                               "cond_calls_after": cond.n_calls,
                               "tau_calls_after": tau.n_calls})
        prev_eplen = eplen

    buf_post = tau_buffer_snapshot(env)
    sd_after = state_dict_sha256(env._vae.state_dict())
    mat_file_sha = sha256_file(mat_path)
    reg_id = next((k for k, s in registry["sources"].items()
                   if s["path"] == mat["artifact"]), None)
    reg_entry = registry["sources"].get(reg_id) if reg_id else None
    nat_cond = next((cid for cid, c in mapping["conditions"].items()
                     if c["speed_bin"] == _natural_bin(v)
                     and c["dir_bin"] == assignment["dir_bin"]), None)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "cell_id": cell["cell_id"],
        "cell_index": cell_index,
        "target_speed": v,
        "condition_arm": cell["condition_arm"],
        "tau_ff": cell["tau_ff"],
        "assignment": {
            "decoder_condition_id": assignment["decoder_condition_id"],
            "speed_bin": assignment["speed_bin"],
            "dir_bin": assignment["dir_bin"],
            "natural_condition_id": nat_cond,
            "natural_speed_bin": _natural_bin(v),
            "selection_source": "frozen_mapping_v2_lookup",
        },
        "env_identity": {
            "env_class": type(env).__name__,
            "env_source_file_sha256": sha256_file(ENV_SOURCE),
            "arch_source_file_sha256": sha256_file(ARCH_SOURCE),
            "token_window_vae_source_sha256": sha256_file(TOKEN_VAE_SOURCE),
            "num_envs": int(env.num_envs),
            "device": str(env.device),
            "sim_dt": float(cfg.sim.dt),
            "decimation": int(cfg.decimation),
            "action_space": int(cfg.action_space),
            "observation_space": int(cfg.observation_space),
            "env_instance_fresh_per_cell": True,
        },
        "cfg_snapshot": _cfg_snapshot(cfg),
        "tau_material": {
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
            "applied_to_env": True,
            "cfg_to_ref_npz": str(mat_path),
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
            "state_dict_sha256_before_episode": sd_before,
            "state_dict_sha256_after_episode": sd_after,
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
        "execution": {
            "status": "completed",
            "steps_requested": steps,
            "steps_done": done,
            "boundary_step": boundary_step,
            "n_cmd_reassertions": n_reassert,
            "n_auto_resets": n_auto,
            "boundaries": boundaries,
            "torch_version": torch.__version__,
            "python_version": _platform_python(),
            "platform": _platform(),
            "started_utc": t0,
            "finished_utc": _utcnow(),
            "wall_seconds": round(time.monotonic() - wall0, 1),
        },
    }
    env.close()
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mode A → 真实 env 接线 launch sanity（28-cell，只测接线）")
    ap.add_argument("--mode", choices=["static", "execute"], required=True)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/sync/to41_sanity")
    ap.add_argument("--vae-path", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/token_vae_e39/vae.pt")
    ap.add_argument("--materials-root", type=Path, nargs="+",
                    default=[REPO_ROOT / "apt_g1/outputs"],
                    help="material npz 搜索根（可多个，按序首中；canonical 唯一性由 checker 独立重算保证）")
    ap.add_argument("--env-tag", choices=["lab-ts", "local"], required=True,
                    help="lab-ts = frozen execution environment；本机必须标 local")
    ap.add_argument("--steps", type=int, default=300,
                    help="每 cell 控制步数（50 Hz，默认 300 ≈ 6 s）")
    ap.add_argument("--boundary-step", type=int, default=150,
                    help="强制 episode boundary 的步号（默认 150）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.env_tag == "lab-ts" and _platform().lower().startswith("windows"):
        raise SystemExit("FAIL: 本机（Windows, 无 venv）禁止以 lab-ts 身份执行")

    if args.mode == "static":
        if args.env_tag == "lab-ts":
            raise SystemExit("FAIL: static 为环境无关配置层核对，恒为 local 口径")
        out = static_coverage(args.out)
        print(f"OK static coverage -> {out}")
        return 0

    mapping = load_mapping()
    _preflight(mapping, list(args.materials_root), args.vae_path)
    availability = load_availability()
    registry = load_registry()

    receipts_dir = args.out / "receipts"
    if receipts_dir.exists() and any(receipts_dir.iterdir()) and not args.force:
        raise SystemExit(f"FAIL: {receipts_dir} 非空（--force 覆盖）")
    receipts_dir.mkdir(parents=True, exist_ok=True)

    print(f"[launch-sanity] execute：28 cell × {args.steps} steps "
          f"(boundary@{args.boundary_step})，zero-action 驱动，仅记录不判定")
    for i, cell in enumerate(enumerate_cells(mapping)):
        receipt = execute_cell(cell, i, mapping, availability, registry,
                               args.vae_path, list(args.materials_root),
                               args.steps, args.boundary_step)
        rp = receipts_dir / f"receipt_{cell['cell_id']}.json"
        rp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
        print(f"[{i + 1:02d}/28] {cell['cell_id']} "
              f"cond={receipt['assignment']['decoder_condition_id']} "
              f"calls={receipt['condition_override']['n_decode_calls']} "
              f"tau_calls={receipt['tau_consumption']['n_tau_calls']} "
              f"auto_resets={receipt['execution']['n_auto_resets']} "
              f"wall={receipt['execution']['wall_seconds']}s", flush=True)

    print(f"OK: 28 receipts -> {receipts_dir}")
    print("next: python -m apt_g1.rung1.l_checker --receipts-dir ... （independent verdict）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
