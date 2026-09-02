"""TO41 Rung 1 Mode A conditioning runtime（state-changing execution code）。

角色（协议 §10.2 / SCRIPT_MAP 登记）：**state-changing execution code**。
frozen specification（TO41_D_DRYRUN_PROTOCOL.md，FROZEN）的可执行化：
实现且仅实现 Mode A 契约::

    tau_runtime(v, C) = tau_frozen(v)          # tau material 与 C 无关
    C_runtime(v, C)   = T_mapping(v, C)        # 唯一自由度 = condition selection

结构纪律（owner implementation 裁定）：

1. runtime 是执行器不是决策器 —— condition 与 material 都来自 frozen
   artifact lookup，禁止任何启发式（nearest / threshold / if-else 选择）。
2. mapping lookup（(v, arm) -> condition_id）与 material lookup（v ->
   tau material）是**两个独立函数**，机械保证两条 treatment 轴不合并。
3. 每 cell 先落 immutable execution record（receipt），再执行 decode；
   receipt 只有 record 字段（hash/shape/lineage/execution status），
   **禁止任何 verdict / PASS / ok 类字段**（协议 §9：verdict 只出自 checker）。

D 判据唯一事实源 = refine-logs/TO41_D_DRYRUN_PROTOCOL.md；本文件只是把
frozen specification 变成可执行代码，不得重新解释任何冻结字段。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING_YAML = REPO_ROOT / "apt_g1/configs/rung1_tau_dec_mapping.yaml"
REGISTRY_YAML = REPO_ROOT / "apt_g1/configs/rung1_source_registry.yaml"
GDOWN_SPEC_MD = REPO_ROOT / "refine-logs/TO41_G_DOWN_SPEC.md"

RECEIPT_SCHEMA = "rung1-d-receipt/v1"
CONDITION_ARMS = ("C1", "C2")  # assignment 槽位（mapping YAML decoder_conditions）
TAU_FF_ARMS = ("on", "off")
CONDITION_ID_RE = re.compile(r"^vb(\d+)_db(\d+)$")


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# frozen artifact loaders（runtime 侧；checker 另有独立实现，二者不共享代码）
# ---------------------------------------------------------------------------

def load_mapping(path: Path = MAPPING_YAML) -> dict:
    """解析冻结 mapping v2 YAML（14 rows 全交叉）。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("artifact_id") != "rung1-tau-dec-mapping" or raw.get("mapping_rule_version") != 2:
        raise SystemExit(f"FAIL: {path} 不是 mapping v2 artifact")
    conditions = {}
    for cid, spec in raw["decoder_conditions"].items():
        m = CONDITION_ID_RE.match(cid)
        if not m:
            raise SystemExit(f"FAIL: condition id 无法解析: {cid}")
        conditions[cid] = {
            "speed_bin": int(m.group(1)),
            "dir_bin": int(m.group(2)),
            "arm": spec["arm"],
        }
    rows = {}
    grid = []
    for row in raw["mappings"]:
        v = round(float(row["target_speed"]), 3)
        arm = row["condition_arm"]
        if arm not in CONDITION_ARMS:
            raise SystemExit(f"FAIL: 未知 condition_arm: {arm}")
        cid = row["decoder_condition_id"]
        if cid not in conditions:
            raise SystemExit(f"FAIL: mapping 行引用未注册 condition: {cid}")
        if conditions[cid]["arm"] != arm:
            raise SystemExit(f"FAIL: condition {cid} 的 arm 与 mapping 行不一致")
        rows[(v, arm)] = cid
        if v not in grid:
            grid.append(v)
    if len(rows) != 14 or len(grid) != 7:
        raise SystemExit(f"FAIL: mapping rows={len(rows)} grid={len(grid)} != 14/7")
    grid.sort()
    return {
        "artifact_id": raw["artifact_id"],
        "freeze_status": raw.get("freeze_status"),
        "grid": grid,
        "conditions": conditions,
        "rows": rows,
        "hashes": {
            "decoder_checkpoint": raw["decoder_checkpoint_hash"],
            "decoder_architecture": raw["decoder_architecture_hash"],
            "preprocessing": raw["preprocessing_hash"],
            "normalization": raw["normalization_hash"],
        },
    }


def load_registry(path: Path = REGISTRY_YAML) -> dict:
    """解析冻结 source registry（r_valid 源 + sha256_16 + mode/knots）。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("artifact_id") != "rung1-source-registry":
        raise SystemExit(f"FAIL: {path} 不是 source registry artifact")
    sources = {}
    for s in raw["sources"]:
        sources[s["id"]] = {
            "path": s["path"],
            "v_dump": s.get("v_dump"),
            "mode": s.get("mode"),
            "knots": s.get("knots"),
            "sha256_16": s.get("sha256_16"),
            "r_valid": s.get("r_valid"),
        }
    return {"sources": sources}


def load_availability(path: Path = GDOWN_SPEC_MD) -> dict:
    """解析 TO41_G_DOWN_SPEC.md §9 material availability map（冻结 md 表）。"""
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^## 9\. .*?$", text, re.M)
    if not m:
        raise SystemExit(f"FAIL: {path} 找不到 §9 availability map")
    section = text[m.start():]
    nxt = re.search(r"^## 10\. ", section, re.M)
    if nxt:
        section = section[: nxt.start()]
    table = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] in ("target", "---:") or set(cells[0]) <= {"-", ":"}:
            continue
        try:
            v = round(float(cells[0]), 3)
            v_real = float(cells[2])
            abs_err = float(cells[3])
        except ValueError:
            continue  # 表头/分隔行
        table[v] = {
            "artifact": cells[1],
            "v_realized": v_real,
            "abs_err": abs_err,
            "source": cells[4],
            "determinism": cells[5],
        }
    if len(table) != 7:
        raise SystemExit(f"FAIL: availability map 解析出 {len(table)} 行 != 7")
    return table


# ---------------------------------------------------------------------------
# 两个独立 lookup（Mode A 契约的机械保证）
# ---------------------------------------------------------------------------

def mapping_lookup(v: float, arm: str, mapping: dict) -> str:
    """Lookup 1（condition 轴）：(target_speed, arm) -> decoder_condition_id。

    纯查表；任何未命中都是 hard fail，不做 nearest / threshold 推断。
    """
    key = (round(float(v), 3), arm)
    if key not in mapping["rows"]:
        raise KeyError(f"mapping v2 无此 (target_speed, arm): {key}")
    return mapping["rows"][key]


def material_lookup(v: float, availability: dict) -> dict:
    """Lookup 2（material 轴）：target_speed -> 冻结 tau material（与 arm 无关）。

    Mode A：material 只依赖 v。任何未命中都是 hard fail。
    """
    key = round(float(v), 3)
    if key not in availability:
        raise KeyError(f"availability map 无此 target_speed: {key}")
    return dict(availability[key])


def enumerate_cells(mapping: dict) -> list[dict]:
    """28 cells：7 v × {C1, C2} × {τ on, τ off}，冻结枚举顺序。"""
    cells = []
    for v in mapping["grid"]:
        for arm in CONDITION_ARMS:
            for ff in TAU_FF_ARMS:
                cells.append({
                    "cell_id": f"v{v * 1000:04.0f}_{arm}_{ff}",
                    "target_speed": v,
                    "condition_arm": arm,
                    "tau_ff": ff,
                })
    return cells


def resolve_cell(cell: dict, mapping: dict, availability: dict) -> dict:
    """(v, C, tau_ff) -> 完整 assignment record（两 lookup 结果组合，无执行）。"""
    condition_id = mapping_lookup(cell["target_speed"], cell["condition_arm"], mapping)
    cond = mapping["conditions"][condition_id]
    material = material_lookup(cell["target_speed"], availability)
    return {
        "decoder_condition_id": condition_id,
        "speed_bin": cond["speed_bin"],
        "dir_bin": cond["dir_bin"],
        "tau_material": material,
    }


# ---------------------------------------------------------------------------
# decode 执行（torch 路径；仅 lab-ts frozen env 调用）
# ---------------------------------------------------------------------------

# 与 apt_flat_env.py _compute_q_des 冻结调用一致的 decode probe 输入
PROBE_WALK_PHASE_RAD = 1.234
PROBE_SEED = 20260902


def _torch():
    import torch  # 延迟导入：静态覆盖核对/本机自测不需要 torch

    return torch


def load_decoder(vae_path: Path, device: str = "cpu") -> dict:
    """与 apt_flat_env.py latent_dir_bins 分支逐字同源的加载路径。

    vae = DirSpeedPhaseTokenVAE(n_vbins=3, n_dbins=8); load_state_dict(ckpt,
    strict=False)（checkpoint 另含 encoder，仅 decoder 需要）; vae.eval()。
    返回 module + 加载身份（missing/unexpected keys）。
    """
    from apt_g1.isaac.token_window_vae import DirSpeedPhaseTokenVAE

    torch = _torch()
    vae = DirSpeedPhaseTokenVAE(n_vbins=3, n_dbins=8).to(device)
    ckpt = torch.load(vae_path, map_location=device)
    missing, unexpected = vae.load_state_dict(ckpt, strict=False)
    vae.eval()
    return {
        "module": vae,
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "checkpoint_keys": sorted(ckpt.keys()),
    }


def state_dict_sha256(sd: dict) -> str:
    """规范化 state_dict 哈希：sorted key + shape + dtype + 原始字节。"""
    torch = _torch()
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def tensor_sha256(t) -> str:
    torch = _torch()
    tt = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(tt.shape)).encode())
    h.update(str(tt.dtype).encode())
    h.update(tt.numpy().tobytes())
    return h.hexdigest()


def find_material(artifact: str, materials_roots: list[Path]) -> tuple[Path, Path] | None:
    """按序搜索 materials_roots，返回 (material 文件, 命中根)；canonical 唯一性
    由 checker 的独立 sha256 重算保证（多根仅是搜索路径，不引入选择自由度）。"""
    for root in materials_roots:
        p = root / artifact
        if p.exists():
            return p, root
    return None


def execute_cell(cell: dict, mapping: dict, availability: dict, registry: dict,
                 decoder: dict, vae_path: Path, materials_roots: list[Path],
                 device: str = "cpu") -> dict:
    """执行一个 cell：先固定 immutable execution record，再执行 decode probe。

    receipt = record（无 verdict）；decode 与 env 冻结路径同构
    （no_grad + eval + decode(z, phase_sc, v_bin, d_bin)）。
    """
    torch = _torch()
    import numpy as np

    t0 = _utcnow()
    assignment = resolve_cell(cell, mapping, availability)

    # ── immutable record 先于 decode 固定 ──
    receipt: dict = {
        "schema": RECEIPT_SCHEMA,
        "cell_id": cell["cell_id"],
        "target_speed": cell["target_speed"],
        "condition_arm": cell["condition_arm"],
        "tau_ff": cell["tau_ff"],
        "runtime_assignment": {
            "decoder_condition_id": assignment["decoder_condition_id"],
            "speed_bin": assignment["speed_bin"],
            "dir_bin": assignment["dir_bin"],
            "selection_source": "frozen_mapping_v2_lookup",
        },
        "tau_material": {},
        "decoder_identity": {},
        "mode_layout_shapes": {},
        "execution": {},
    }

    # material 轴 record（与 condition 轴独立；OFF cell 同样记录身份，
    # D2 的 Mode A fingerprint 覆盖全部 4 cell）
    mat = assignment["tau_material"]
    found = find_material(mat["artifact"], materials_roots)
    if found is None:
        raise FileNotFoundError(
            f"material npz 在所有 roots 均不存在: {mat['artifact']} in {[str(r) for r in materials_roots]}")
    mat_path, mat_root = found
    mat_sha = sha256_file(mat_path)
    reg_id = next((k for k, s in registry["sources"].items()
                   if s["path"] == mat["artifact"]), None)
    reg_entry = registry["sources"].get(reg_id) if reg_id else None
    with np.load(mat_path) as z:
        npz_keys = {k: list(z[k].shape) for k in sorted(z.files)}
    receipt["tau_material"] = {
        "artifact": mat["artifact"],
        "sha256": mat_sha,
        "sha256_16": mat_sha[:16],
        "source_lineage": mat["source"],
        "v_realized": mat["v_realized"],
        "abs_err": mat["abs_err"],
        "registry_id": reg_id,
        "registry_sha256_16": reg_entry["sha256_16"] if reg_entry else None,
        "materials_root_used": str(mat_root),
        "mode": reg_entry["mode"] if reg_entry else "foot(gdown-manifest-fixed-params)",
        "knots": reg_entry["knots"] if reg_entry else None,
        "npz_keys_shapes": npz_keys,
        "applied_to_env": False,  # decode-only dry-run；env 注入属 Rung 1 launch sanity
    }

    # decoder 身份 record（D1 七字段 + runtime identity）
    sd_before = decoder["module"].state_dict()
    receipt["decoder_identity"] = {
        "checkpoint_path": str(vae_path),
        "checkpoint_sha256": sha256_file(vae_path),
        "state_dict_sha256_before": state_dict_sha256(sd_before),
        "architecture": {
            "class": type(decoder["module"]).__name__,
            "token_dim": decoder["module"].token_dim,
            "window": decoder["module"].window,
            "latent_dim": decoder["module"].latent_dim,
            "hidden_dim": 256,
            "phase_dim": decoder["module"].phase_dim,
            "n_vbins": decoder["module"].n_vbins,
            "n_dbins": decoder["module"].n_dbins,
        },
        "state_dict_key_shapes": {k: list(v.shape) for k, v in sorted(sd_before.items())},
        "load_missing_keys": decoder["missing_keys"],
        "load_unexpected_keys": decoder["unexpected_keys"],
        "arch_source_file_sha256": sha256_file(REPO_ROOT / "apt_g1/train_token_vae_e39.py"),
        "env_source_file_sha256": sha256_file(REPO_ROOT / "apt_g1/isaac/apt_flat_env.py"),
        "token_window_vae_source_sha256": sha256_file(
            REPO_ROOT / "apt_g1/isaac/token_window_vae.py"),
    }

    # decode probe（确定性输入；condition 进入 decode = 唯一 treatment 自由度）
    g = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
    zb = torch.randn(1, decoder["module"].latent_dim, generator=g, dtype=torch.float32)
    phi = torch.tensor([[math.sin(PROBE_WALK_PHASE_RAD), math.cos(PROBE_WALK_PHASE_RAD)]],
                       dtype=torch.float32)
    vb = torch.tensor([assignment["speed_bin"]], dtype=torch.long)
    db = torch.tensor([assignment["dir_bin"]], dtype=torch.long)
    with torch.no_grad():
        out = decoder["module"].decode(zb, phi, vb, db)
    sd_after = decoder["module"].state_dict()

    receipt["decoder_identity"]["state_dict_sha256_after"] = state_dict_sha256(sd_after)
    receipt["mode_layout_shapes"] = {
        "mode": receipt["tau_material"]["mode"],
        "decode_arg_layout": ["z", "phase_sc(sin,cos)", "v_bin", "d_bin"],
        "input_shapes": {"z": list(zb.shape), "phase_sc": list(phi.shape),
                         "v_bin": list(vb.shape), "d_bin": list(db.shape)},
        "input_dtypes": {"z": str(zb.dtype), "phase_sc": str(phi.dtype),
                         "v_bin": str(vb.dtype), "d_bin": str(db.dtype)},
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "output_min": float(out.min()),
        "output_max": float(out.max()),
        "output_sha256": tensor_sha256(out),
    }
    receipt["execution"] = {
        "status": "completed",
        "device": device,
        "torch_version": torch.__version__,
        "python_version": _platform_python(),
        "platform": _platform(),
        "started_utc": t0,
        "finished_utc": _utcnow(),
    }
    return receipt


def _platform_python() -> str:
    import platform
    return platform.python_version()


def _platform() -> str:
    import platform
    return platform.platform()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Mode A runtime：static 覆盖核对 / 28-cell decode-only dry-run")
    ap.add_argument("--mode", choices=["static", "execute"], required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "apt_g1/outputs/sync/to41_d")
    ap.add_argument("--vae-path", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/token_vae_e39/vae.pt")
    ap.add_argument("--materials-root", type=Path, nargs="+",
                    default=[REPO_ROOT / "apt_g1/outputs"],
                    help="material npz 搜索根（可多个，按序首中为准；canonical 唯一性由 checker 独立重算保证）")
    ap.add_argument("--env-tag", choices=["lab-ts", "local"], required=True,
                    help="lab-ts = frozen execution environment（协议 §10.1）；本机必须标 local")
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的 receipts 目录")
    args = ap.parse_args()

    if args.env_tag == "lab-ts" and _platform().lower().startswith("windows"):
        raise SystemExit("FAIL: 本机（Windows, 无 venv）禁止以 lab-ts 身份执行（协议 §10.1）")

    mapping = load_mapping()
    availability = load_availability()
    registry = load_registry()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "static":
        rows = []
        for cell in enumerate_cells(mapping):
            a = resolve_cell(cell, mapping, availability)
            rows.append({
                "cell_id": cell["cell_id"],
                "target_speed": cell["target_speed"],
                "condition_arm": cell["condition_arm"],
                "tau_ff": cell["tau_ff"],
                "expected_condition": a["decoder_condition_id"],
                "expected_material": a["tau_material"]["artifact"],
            })
        payload = {
            "artifact": "rung1-d-static-coverage/v1",
            "environment_tag": args.env_tag,
            "generated_utc": _utcnow(),
            "expected_cells": 28,
            "resolved_cells": len(rows),
            "rows": rows,
        }
        out = args.out / "static_coverage.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"OK static coverage {len(rows)}/28 -> {out}")
        return 0

    # execute 模式：仅 lab-ts frozen env（torch + 材料 + vae.pt 都在服务器）
    if not args.vae_path.exists():
        raise SystemExit(f"FAIL: vae.pt 不存在: {args.vae_path}")
    receipts_dir = args.out / "receipts"
    if receipts_dir.exists() and any(receipts_dir.iterdir()) and not args.force:
        raise SystemExit(f"FAIL: {receipts_dir} 非空（--force 覆盖）")
    receipts_dir.mkdir(parents=True, exist_ok=True)

    decoder = load_decoder(args.vae_path)
    n_ok = 0
    for cell in enumerate_cells(mapping):
        receipt = execute_cell(cell, mapping, availability, registry, decoder,
                               args.vae_path, list(args.materials_root))
        rp = receipts_dir / f"receipt_{cell['cell_id']}.json"
        rp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
        n_ok += 1
        print(f"[{n_ok:02d}/28] {cell['cell_id']} condition={receipt['runtime_assignment']['decoder_condition_id']} "
              f"material={receipt['tau_material']['artifact']}")
    print(f"OK: 28 receipts -> {receipts_dir}")
    print("next: python -m apt_g1.rung1.d_checker --receipts-dir ... （independent verdict）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
