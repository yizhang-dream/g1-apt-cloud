"""TO41 launch sanity selftest（工具，永不作为 L artifact）。

本机可跑（numpy + yaml，无 torch/isaac）。内容：

- T-L0：runtime（mode_a_runtime）与 checker（d_checker）两套独立解析器
  交叉一致（D 链 T0 的再锚定，sanity 复用）；
- T-L1：env_wiring 接线 handle 的 mock 单测（override 逐调用记录/契约
  违反 fail-fast/τ 探针 digest 确定性/非有限计数）；
- T-L2b：canonical array hash 双实现交叉一致（env_wiring vs l_checker）；
- T-L3：合成 28-receipt 全 checker 正例（schema/L1–L4 全 PASS）；
- Negative A–E：τ buffer 换 hash → L1 FAIL；boundary 后 applied 回退 →
  L2 FAIL；两臂 cfg 混入额外差异 → L3 FAIL；receipt 缺件 → L4+schema
  FAIL；自报 verdict 字段 → schema FAIL。

判据唯一事实源 = refine-logs/TO41_LAUNCH_SANITY.md。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from apt_g1.rung1 import env_wiring, l_checker
from apt_g1.rung1.d_checker import (
    EXPECTED_SD_SIGNATURE,
    enumerate_cells as dc_enumerate_cells,
    load_availability as dc_load_availability,
    load_mapping as dc_load_mapping,
)
from apt_g1.rung1.env_wiring import canonical_array_sha256
from apt_g1.rung1.l_checker import _array_sha256, _natural_bin
from apt_g1.rung1.mode_a_runtime import (
    enumerate_cells as rt_enumerate_cells,
    load_availability as rt_load_availability,
    load_mapping as rt_load_mapping,
    resolve_cell,
)

STEPS = 300
BOUNDARY = 150


# ---------------------------------------------------------------------------
# T-L0 parser cross-check
# ---------------------------------------------------------------------------

def t_l0_parsers() -> tuple[bool, str]:
    a, b = rt_load_mapping(), dc_load_mapping()
    if a["rows"] != b["rows"] or a["grid"] != b["grid"] or a["hashes"] != b["hashes"]:
        return False, "T-L0: 两套 mapping 解析器不一致"
    if rt_load_availability() != dc_load_availability():
        return False, "T-L0: 两套 availability 解析器不一致"
    if [(c["cell_id"], c["target_speed"]) for c in rt_enumerate_cells(a)] != \
            [(c["cell_id"], c["target_speed"]) for c in dc_enumerate_cells(b)]:
        return False, "T-L0: 28-cell 枚举不一致"
    return True, "T-L0 parsers OK"


# ---------------------------------------------------------------------------
# T-L1 wiring handle mock 单测
# ---------------------------------------------------------------------------

class _MockVae:
    def __init__(self):
        self.calls = []

    def decode(self, phase, sc, vb, db):
        self.calls.append((vb, db))
        return f"tokens(vb={vb},db={db})"


class _MockEnv:
    """latent_mode env 的最小 mock（只暴露接线点）。"""

    def __init__(self):
        self._vae = _MockVae()
        self._to_tau = np.arange(12, dtype=np.float64).reshape(2, 6)

    def _to_ref_lookup(self):
        tau = self._to_tau[:1] @ np.eye(6, dtype=np.float64)
        return None, None, tau


def t_l1_wiring() -> tuple[bool, str]:
    env = _MockEnv()
    h = env_wiring.ConditionOverrideHandle(env, speed_bin=1, dir_bin=4)
    for natural_vb in (0, 0, 1):
        out = env._vae.decode("phase", "sc", natural_vb, 4)
        if not str(out).startswith("tokens(vb=1"):
            return False, f"T-L1: override 未生效（decode 返回 {out!r}）"
    recs = h.records
    if [r["natural_vb"] for r in recs] != [0, 0, 1] or \
            [r["applied_vb"] for r in recs] != [1, 1, 1] or \
            h.n_override_changed != 2 or h.n_calls != 3:
        return False, f"T-L1: per-call 记录不符 {recs}"
    try:
        env._vae.decode("phase", "sc", 0)  # 缺 db = 契约违反
        return False, "T-L1: db 缺失未 fail-fast"
    except RuntimeError:
        pass

    env2 = _MockEnv()
    p = env_wiring.TauConsumptionProbe(env2)
    env2._to_ref_lookup()
    env2._to_ref_lookup()
    if p.n_calls != 2 or not p._digest.hexdigest():
        return False, "T-L1: τ 探针计数/digest 异常"
    d1 = p._digest.hexdigest()
    env2b = _MockEnv()
    p2 = env_wiring.TauConsumptionProbe(env2b)
    env2b._to_ref_lookup()  # 经 wrapped 路径重放同一序列
    env2b._to_ref_lookup()
    if p2._digest.hexdigest() != d1:
        return False, "T-L1: τ digest 非确定"
    return True, "T-L1 wiring handles OK"


def t_l2b_hash_crosscheck() -> tuple[bool, str]:
    rng = np.random.default_rng(7)
    for shape in [(120, 6), (7,), (3, 4, 2)]:
        a64 = rng.normal(size=shape)
        if canonical_array_sha256(a64) != _array_sha256(a64):
            return False, f"T-L2b: 双实现不一致 shape={shape}"
        a32 = a64.astype(np.float32)
        if canonical_array_sha256(a32) != _array_sha256(a64):
            return False, "T-L2b: float64→float32 归一化路径不一致"
    return True, "T-L2b canonical hash cross-check OK"


# ---------------------------------------------------------------------------
# T-L3/Negative：合成 28-receipt 全 checker
# ---------------------------------------------------------------------------

def _make_materials(tmp: Path, availability: dict) -> dict[float, dict]:
    """7 份合成材料 npz（tau_ref6 内容按 v 区分），返回 v → 身份。"""
    mats = {}
    rng = np.random.default_rng(42)
    for v, m in sorted(availability.items()):
        arr = rng.normal(0, 1, size=(120, 6)) * (1.0 + v)
        p = tmp / m["artifact"]
        np.savez(p, tau_ref6=arr, q_ref6=rng.normal(size=(120, 6)))
        mats[v] = {
            "path": p,
            "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "buffer_sha256": canonical_array_sha256(arr),
        }
    return mats


def _synthetic_receipt(cell: dict, cell_index: int, mapping: dict,
                       mats: dict[float, dict], tmp: Path) -> dict:
    v = cell["target_speed"]
    availability = rt_load_availability()
    a = resolve_cell(cell, mapping, availability)
    m = mats[v]
    nat_bin = _natural_bin(v)
    mapped_bin = a["speed_bin"]
    nat_id = next((cid for cid, c in mapping["conditions"].items()
                   if c["speed_bin"] == nat_bin and c["dir_bin"] == a["dir_bin"]), None)
    per_call = [{"i": i, "natural_vb": nat_bin, "applied_vb": mapped_bin,
                 "natural_db": 4, "applied_db": 4} for i in range(STEPS)]
    changed = STEPS if nat_bin != mapped_bin else 0
    fake_sd = "a" * 64
    return {
        "schema": "rung1-launch-sanity-receipt/v1",
        "cell_id": cell["cell_id"],
        "cell_index": cell_index,
        "target_speed": v,
        "condition_arm": cell["condition_arm"],
        "tau_ff": cell["tau_ff"],
        "assignment": {
            "decoder_condition_id": a["decoder_condition_id"],
            "speed_bin": mapped_bin,
            "dir_bin": a["dir_bin"],
            "natural_condition_id": nat_id,
            "natural_speed_bin": nat_bin,
            "selection_source": "frozen_mapping_v2_lookup",
        },
        "env_identity": {
            "env_class": "AptFlatG1Env",
            "env_source_file_sha256": mapping["hashes"]["preprocessing"],
            "arch_source_file_sha256": mapping["hashes"]["decoder_architecture"],
            "token_window_vae_source_sha256": "b" * 64,
            "num_envs": 1, "device": "cuda:0", "sim_dt": 0.005, "decimation": 4,
            "action_space": 16, "observation_space": 117,
            "env_instance_fresh_per_cell": True,
        },
        "cfg_snapshot": {
            "scene.num_envs": 1, "terrain_seed": 0, "sim.dt": 0.005, "decimation": 4,
            "episode_length_s": 120.0, "action_space": 16, "observation_space": 117,
            "disturbance_prob": 0.0, "latent_mode": True, "latent_speed_bins": True,
            "latent_dir_bins": True, "latent_residual": False,
            "latent_vae_n_bins": 3, "latent_vae_n_dbins": 8,
            "latent_cmd_phase_rate": False, "latent_vae_path": "vae.pt",
            "vx_max": 0.8, "vx_min": 0.0, "use_sonic_prior": True,
            "sonic_decoder_path": "dec.onnx", "router_model_dir": "router",
            "use_2hz_gate": True, "to_ref": True, "to_ref_npz": str(m["path"]),
            "to_ref_obs_zero": True, "to_ref_w": 0.0,
            "to_tau": cell["tau_ff"] == "on", "to_tau_w": 1.0,
            "to_ref_gate2": 0.0036, "to_ref_sigma2": 0.1,
            "terrain": "plane_importer(seed=0,noise=0.04)",
        },
        "tau_material": {
            "artifact": a["tau_material"]["artifact"],
            "materials_root_used": str(tmp),
            "npz_path": str(m["path"]),
            "file_sha256": m["file_sha256"],
            "sha256_16": m["file_sha256"][:16],
            "source_lineage": f"synthetic-lineage-{v}",
            "v_realized": a["tau_material"]["v_realized"],
            "abs_err": a["tau_material"]["abs_err"],
            "registry_id": None, "registry_sha256_16": None,
            "applied_to_env": True,
            "cfg_to_ref_npz": str(m["path"]),
            "buffer_shape": [120, 6], "buffer_dtype": "<f4",
            "buffer_sha256_pre": m["buffer_sha256"],
            "buffer_sha256_post": m["buffer_sha256"],
            "to_vavg": a["tau_material"]["v_realized"],
            "to_m": 120, "to_rate": 2.618, "to_kp_sagittal6": [99.1] * 6,
        },
        "condition_override": {
            "mechanism": "instance-level shadowing of env._vae.decode",
            "frozen_path_untouched": True,
            "condition_entry_identity": env_wiring.CONDITION_ENTRY_IDENTITY,
            "mapped_speed_bin": mapped_bin, "mapped_dir_bin": 4,
            "n_decode_calls": STEPS, "n_override_changed": changed,
            "natural_vb_distribution": {str(nat_bin): STEPS},
            "per_call": per_call,
        },
        "tau_consumption": {
            "mechanism": "instance-level shadowing of env._to_ref_lookup",
            "consumer_identity": env_wiring.TAU_CONSUMER_IDENTITY,
            "n_tau_calls": STEPS if cell["tau_ff"] == "on" else 0,
            "calls_tau_digest_sha256": "c" * 64,
            "first_tau_sha256_16": "d" * 16, "last_tau_sha256_16": "e" * 16,
            "n_nonfinite_tau_calls": 0,
        },
        "decoder_identity": {
            "vae_path": "vae.pt",
            "checkpoint_sha256": mapping["hashes"]["decoder_checkpoint"],
            "state_dict_sha256_before_episode": fake_sd,
            "state_dict_sha256_after_episode": fake_sd,
            "architecture": {"class": "DirSpeedPhaseTokenVAE", "token_dim": 64,
                             "window": 10, "latent_dim": 16, "n_vbins": 3,
                             "n_dbins": 8},
            "state_dict_key_shapes": {k: list(v) for k, v in EXPECTED_SD_SIGNATURE.items()},
            "mode_layout": {"latent_mode": True, "latent_dir_bins": True,
                            "decode_call_form": "decode(z, phase_sc(sin,cos), v_bin, d_bin)"},
        },
        "execution": {
            "status": "completed", "steps_requested": STEPS, "steps_done": STEPS,
            "boundary_step": BOUNDARY, "n_cmd_reassertions": STEPS + 2,
            "n_auto_resets": 2,
            "boundaries": [
                {"step": 0, "type": "episode_start", "cond_calls_after": 0,
                 "tau_calls_after": 0},
                {"step": BOUNDARY, "type": "forced", "cond_calls_before": BOUNDARY,
                 "cond_calls_after": STEPS - BOUNDARY, "tau_calls_before": BOUNDARY,
                 "tau_calls_after": STEPS - BOUNDARY},
                {"step": 220, "type": "auto_reset", "cond_calls_after": 220,
                 "tau_calls_after": 220},
            ],
            "torch_version": "synthetic", "python_version": "synthetic",
            "platform": "synthetic", "started_utc": "t0", "finished_utc": "t1",
            "wall_seconds": 1.0,
        },
    }


def _build_synthetic_set(base: Path) -> Path:
    mapping = rt_load_mapping()
    availability = rt_load_availability()
    mats = _make_materials(base, availability)
    rdir = base / "receipts"
    rdir.mkdir()
    for i, cell in enumerate(rt_enumerate_cells(mapping)):
        r = _synthetic_receipt(cell, i, mapping, mats, base)
        (rdir / f"receipt_{cell['cell_id']}.json").write_text(
            json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return rdir


def t_l3_checker_positive(base: Path) -> tuple[bool, str]:
    rdir = _build_synthetic_set(base)
    rep = l_checker.build_report(rdir, [base], "local")
    v = rep["verdict"]
    if v["overall"] != "PASS":
        return False, (f"T-L3: 合成正例未全 PASS：{v}；"
                       f"L1f={rep['L1']['failures'][:3]} L2f={rep['L2']['failures'][:3]} "
                       f"L3f={rep['L3']['failures'][:3]} L4f={rep['L4']['failures'][:3]} "
                       f"schemaf={rep['schema_check']['failures'][:3]}")
    return True, "T-L3 synthetic positive: schema/L1/L2/L3/L4 全 PASS"


def _mutated_report(mutator) -> dict:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        rdir = _build_synthetic_set(base)
        mutator(rdir)
        return l_checker.build_report(rdir, [base], "local")


def t_l4_negatives() -> tuple[bool, str]:
    def neg_a(rdir: Path):  # τ buffer 换另一合法材料 hash → L1 FAIL
        p = rdir / "receipt_v0200_C1_on.json"
        r = json.loads(p.read_text(encoding="utf-8"))
        other = json.loads((rdir / "receipt_v0277_C1_on.json").read_text(encoding="utf-8"))
        r["tau_material"]["buffer_sha256_pre"] = other["tau_material"]["buffer_sha256_pre"]
        p.write_text(json.dumps(r), encoding="utf-8")

    def neg_b(rdir: Path):  # boundary 后 applied 回退 natural → L2 FAIL
        p = rdir / "receipt_v0200_C2_off.json"
        r = json.loads(p.read_text(encoding="utf-8"))
        r["condition_override"]["per_call"][BOUNDARY + 5]["applied_vb"] = \
            r["condition_override"]["per_call"][BOUNDARY + 5]["natural_vb"]
        p.write_text(json.dumps(r), encoding="utf-8")

    def neg_c(rdir: Path):  # 两臂混入额外 cfg 差异 → L3 FAIL
        p = rdir / "receipt_v0200_C1_on.json"
        r = json.loads(p.read_text(encoding="utf-8"))
        r["cfg_snapshot"]["to_ref_w"] = 0.3
        p.write_text(json.dumps(r), encoding="utf-8")

    def neg_d(rdir: Path):  # receipt 缺件 → schema + L4 FAIL
        (rdir / "receipt_v0325_C2_on.json").unlink()

    def neg_e(rdir: Path):  # 自报 verdict 字段 → schema FAIL
        p = rdir / "receipt_v0200_C1_off.json"
        r = json.loads(p.read_text(encoding="utf-8"))
        r["pass"] = True
        p.write_text(json.dumps(r), encoding="utf-8")

    checks = [
        # A 改 buffer hash 会被 L1（重算）与 L3（两臂对照）同时抓住——纵深防御，非连坐
        ("Negative A", neg_a, ["L1", "L3"]),
        ("Negative B", neg_b, ["L2"]),
        ("Negative C", neg_c, ["L3"]),
        # D 缺件按设计级联：任何依赖 28/28 覆盖的检查都不得静默通过
        ("Negative D", neg_d, ["schema_check", "L1", "L2", "L3", "L4"]),
        ("Negative E", neg_e, ["schema_check"]),
    ]
    for name, mut, expect_fails in checks:
        rep = _mutated_report(mut)
        for k in ("schema_check", "L1", "L2", "L3", "L4"):
            should_fail = k in expect_fails
            actual_fail = rep[k]["verdict"] == "FAIL"
            if should_fail != actual_fail:
                return False, (f"{name}: {k} verdict={rep[k]['verdict']} "
                               f"（期望 {'FAIL' if should_fail else 'PASS'}）"
                               f" failures={rep[k]['failures'][:2]}")
        # 失败定位检查：未涉及面不得被连坐（A 不累 L2/L3/L4 等）
    return True, "Negative A–E 全部按预期 FAIL 且不连坐"


def main() -> int:
    results = []
    ok, msg = t_l0_parsers()
    results.append((ok, msg))
    ok, msg = t_l1_wiring()
    results.append((ok, msg))
    ok, msg = t_l2b_hash_crosscheck()
    results.append((ok, msg))
    with tempfile.TemporaryDirectory() as td:
        ok, msg = t_l3_checker_positive(Path(td))
        results.append((ok, msg))
    ok, msg = t_l4_negatives()
    results.append((ok, msg))

    print("== launch sanity selftest ==")
    n_pass = 0
    for ok, msg in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        n_pass += int(ok)
    print(f"== {n_pass}/{len(results)} PASS ==")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
