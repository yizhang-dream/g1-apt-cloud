"""TO41 Rung 1 自测：lookup 单元测试 + checker negative tests A–D（协议 §9/owner 裁定）。

必须在正式 28-cell dry-run 之前全绿。本机（无 torch）可跑：所有用例基于
synthetic receipt + synthetic material 文件，不触发 decode、不写任何 D verdict。
negative 用例故意构造损坏，逐一验证 checker 报 FAIL 且失败定位正确：

- Negative A  某 cell τ hash 换成另一合法 material 的 hash  → D2 FAIL + D3B FAIL
- Negative B  某 cell runtime condition 改成错 arm 的 condition → D3A FAIL（D2 assignment 组件同 FAIL）
- Negative C  decoder mode/shape/超参被改               → D1 FAIL
- Negative D  runtime 自报 PASS flag 但 assignment 实际不一致 → checker 仍 FAIL（flag 不被消费）

非循环性：synthetic receipt 的 ground truth（state_dict 签名/超参/冻结 hash）
在本文件独立陈述（trace 注释），不 import d_checker 的期望常量——若 checker
期望写错，positive control 会先炸。

用法：python -m apt_g1.rung1.rung1_selftest --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
from pathlib import Path

from apt_g1.rung1 import d_checker
from apt_g1.rung1 import mode_a_runtime as rt

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── synthetic ground truth（独立于 d_checker 常量陈述）──
# trace：latent 16 + phase 2 + (speed_embed 8 + dir_embed 8) = decoder.0 in-features 34；
# hidden 256；token 64；nvbins 3 / ndbins 8。来源同 checker EXPECTED_SD_SIGNATURE 注释。
SYN_SD_SHAPES = {
    "speed_embed.weight": [3, 8],
    "dir_embed.weight": [8, 8],
    "decoder.0.weight": [256, 34],
    "decoder.0.bias": [256],
    "decoder.2.weight": [256, 256],
    "decoder.2.bias": [256],
    "decoder.4.weight": [64, 256],
    "decoder.4.bias": [64],
}
SYN_SD_HASH = hashlib.sha256(b"synthetic-state-dict").hexdigest()
SYN_TWV_HASH = hashlib.sha256(b"synthetic-token-window-vae").hexdigest()


def _b64(hexstr: str) -> str:
    return hexstr[:16]


def build_synthetic(tmp: Path) -> tuple[dict[str, dict], Path]:
    """构造 28 份合法 synthetic receipt + 7 个 synthetic material 文件。"""
    mapping = rt.load_mapping()
    availability = rt.load_availability()

    mat_dir = tmp / "materials"
    mat_dir.mkdir(parents=True)
    mat_sha = {}
    for v, spec in availability.items():
        p = mat_dir / spec["artifact"]
        p.write_bytes(f"SYNTHETIC MATERIAL {spec['artifact']}\n".encode())
        mat_sha[spec["artifact"]] = rt.sha256_file(p)

    # registry 3 件材料的冻结 sha256_16 锚是真文件哈希，synthetic 字节不可复现——
    # 本机自测改喂 synthetic registry 副本（锚=合成文件前缀），真锚验证留给 lab-ts
    # （协议 §10.1 本机/服务器分工）。
    import yaml as _yaml
    reg_raw = _yaml.safe_load(rt.REGISTRY_YAML.read_text(encoding="utf-8"))
    for s in reg_raw["sources"]:
        if s.get("r_valid") and s["path"] in mat_sha:
            s["sha256_16"] = mat_sha[s["path"]][:16]
    reg_syn = tmp / "registry_synthetic.yaml"
    reg_syn.write_text(
        _yaml.safe_dump(reg_raw, allow_unicode=True, sort_keys=False), encoding="utf-8")

    registry = rt.load_registry(reg_syn)
    reg_by_path = {s["path"]: (i, s) for i, s in registry["sources"].items()}

    receipts: dict[str, dict] = {}
    for cell in rt.enumerate_cells(mapping):
        v, arm, ff = cell["target_speed"], cell["condition_arm"], cell["tau_ff"]
        a = rt.resolve_cell(cell, mapping, availability)
        mat = a["tau_material"]
        reg_id, reg = reg_by_path.get(mat["artifact"], (None, None))
        sd = {k: list(sh) for k, sh in SYN_SD_SHAPES.items()}
        receipts[cell["cell_id"]] = {
            "schema": rt.RECEIPT_SCHEMA,
            "cell_id": cell["cell_id"],
            "target_speed": v,
            "condition_arm": arm,
            "tau_ff": ff,
            "runtime_assignment": {
                "decoder_condition_id": a["decoder_condition_id"],
                "speed_bin": a["speed_bin"],
                "dir_bin": a["dir_bin"],
                "selection_source": "frozen_mapping_v2_lookup",
            },
            "tau_material": {
                "artifact": mat["artifact"],
                "sha256": mat_sha[mat["artifact"]],
                "sha256_16": _b64(mat_sha[mat["artifact"]]),
                "source_lineage": mat["source"],
                "v_realized": mat["v_realized"],
                "abs_err": mat["abs_err"],
                "registry_id": reg_id,
                "registry_sha256_16": reg["sha256_16"] if reg else None,
                "mode": reg["mode"] if reg else "foot(gdown-manifest-fixed-params)",
                "knots": reg["knots"] if reg else None,
                "npz_keys_shapes": {"X": [40, 29], "v_aux": []},
                "applied_to_env": False,
            },
            "decoder_identity": {
                "checkpoint_path": "synthetic://vae.pt",
                "checkpoint_sha256": mapping["hashes"]["decoder_checkpoint"],
                "state_dict_sha256_before": SYN_SD_HASH,
                "state_dict_sha256_after": SYN_SD_HASH,
                "architecture": {
                    "class": "DirSpeedPhaseTokenVAE", "token_dim": 64, "window": 10,
                    "latent_dim": 16, "hidden_dim": 256, "phase_dim": 2,
                    "n_vbins": 3, "n_dbins": 8,
                },
                "state_dict_key_shapes": sd,
                "load_missing_keys": [],
                "load_unexpected_keys": ["encoder.0.weight", "mu.weight", "logvar.weight"],
                "arch_source_file_sha256": mapping["hashes"]["decoder_architecture"],
                "env_source_file_sha256": mapping["hashes"]["preprocessing"],
                "token_window_vae_source_sha256": SYN_TWV_HASH,
            },
            "mode_layout_shapes": {
                "mode": reg["mode"] if reg else "foot(gdown-manifest-fixed-params)",
                "decode_arg_layout": ["z", "phase_sc(sin,cos)", "v_bin", "d_bin"],
                "input_shapes": {"z": [1, 16], "phase_sc": [1, 2], "v_bin": [1], "d_bin": [1]},
                "input_dtypes": {"z": "torch.float32", "phase_sc": "torch.float32",
                                 "v_bin": "torch.int64", "d_bin": "torch.int64"},
                "output_shape": [1, 64],
                "output_dtype": "torch.float32",
                "output_min": -0.6,
                "output_max": 0.7,
                # decode 只依赖 (probe, condition)：同 (v, arm) 的 ON/OFF 必然相同
                "output_sha256": hashlib.sha256(f"probe:{v:.3f}:{arm}".encode()).hexdigest(),
            },
            "execution": {
                "status": "completed", "device": "cpu",
                "torch_version": "synthetic", "python_version": "synthetic",
                "platform": "synthetic",
                "started_utc": "1970-01-01T00:00:00Z",
                "finished_utc": "1970-01-01T00:00:00Z",
            },
        }
    return receipts, mat_dir, reg_syn


def write_receipts(receipts: dict[str, dict], d: Path) -> Path:
    rd = d / "receipts"
    rd.mkdir(parents=True, exist_ok=True)
    for cid, r in receipts.items():
        (rd / f"receipt_{cid}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rd


def run_checker(receipts_dir: Path, mat_dir: Path, reg_syn: Path) -> dict:
    return d_checker.build_report(receipts_dir, [mat_dir], "local", selftest=True,
                                  registry_path=reg_syn)


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "apt_g1/outputs/sync/to41_d")
    args = ap.parse_args()
    import tempfile
    results: list[dict] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append({"test": name, "pass": ok, "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    with tempfile.TemporaryDirectory(prefix="rung1-selftest-") as td:
        tmp = Path(td)
        receipts, mat_dir, reg_syn = build_synthetic(tmp)

        # ── T0 跨解析器一致性：runtime vs checker 两套独立解析 ──
        m_rt, m_ck = rt.load_mapping(), d_checker.load_mapping()
        a_rt, a_ck = rt.load_availability(), d_checker.load_availability()
        g_rt, g_ck = rt.load_registry(), d_checker.load_registry()
        ok = (m_rt["rows"] == m_ck["rows"] and m_rt["grid"] == m_ck["grid"]
              and m_rt["hashes"] == m_ck["hashes"]
              and m_rt["conditions"] == m_ck["conditions"]
              and a_rt == a_ck
              and all(g_rt["sources"][i]["sha256_16"] ==
                      g_ck["by_path"][s["path"]]["sha256_16"]
                      for i, s in g_rt["sources"].items())
              and all(g_ck["by_path"][s["path"]]["r_valid"] == s["r_valid"]
                      for i, s in g_rt["sources"].items()))
        record("T0 cross-parser agreement", ok)

        # ── T1 lookup 单元测试（Mode A 契约在 lookup 层成立）──
        ok, detail = True, ""
        try:
            for (v, arm), cid in m_rt["rows"].items():
                assert rt.mapping_lookup(v, arm, m_rt) == cid
            for v in m_rt["grid"]:
                c1 = rt.resolve_cell({"target_speed": v, "condition_arm": "C1",
                                      "tau_ff": "on", "cell_id": "x"}, m_rt, a_rt)
                c2 = rt.resolve_cell({"target_speed": v, "condition_arm": "C2",
                                      "tau_ff": "off", "cell_id": "x"}, m_rt, a_rt)
                assert c1["tau_material"] == c2["tau_material"], f"v={v} material 随 arm 变化"
            try:
                rt.mapping_lookup(0.999, "C1", m_rt)
                ok, detail = False, "未命中应 hard fail"
            except KeyError:
                pass
        except AssertionError as e:
            ok, detail = False, str(e)
        record("T1 lookup unit (Mode A identity + hard-fail)", ok, detail)

        # ── T2 静态覆盖：runtime 枚举 == checker 枚举，28/28 ──
        cov_ck = d_checker.static_coverage(m_ck, a_ck)
        cov_rt = [{"cell": c["cell_id"], "target_speed": c["target_speed"],
                   "condition_arm": c["condition_arm"], "tau_ff": c["tau_ff"],
                   "expected_condition": rt.resolve_cell(c, m_rt, a_rt)["decoder_condition_id"],
                   "expected_material": rt.resolve_cell(c, m_rt, a_rt)["tau_material"]["artifact"]}
                  for c in rt.enumerate_cells(m_rt)]
        record("T2 static coverage 28/28 + cross-impl identical",
               cov_rt == cov_ck and len(cov_ck) == 28,
               f"n={len(cov_ck)}")

        # ── T3 positive control：未损坏 synthetic set 全 PASS ──
        rd = write_receipts(receipts, tmp / "valid")
        rep = run_checker(rd, mat_dir, reg_syn)
        record("T3 positive control (all PASS)",
               rep["overall"] == "PASS"
               and all(rep[k]["verdict"] == "PASS"
                       for k in ("schema_check", "D1", "D2", "D3A", "D3B")),
               f"overall={rep['overall']}")

        def corrupted(name: str, mutate) -> tuple[Path, dict[str, dict]]:
            rs = copy.deepcopy(receipts)
            mutate(rs)
            d = tmp / name
            return write_receipts(rs, d), rs

        # ── Negative A：τ hash 换成另一合法 material ──
        def mut_a(rs):
            other = rs["v0275_C1_on"]["tau_material"]["sha256"]
            rs["v0250_C1_on"]["tau_material"]["sha256"] = other
        rd_a, _ = corrupted("negA", mut_a)
        rep_a = run_checker(rd_a, mat_dir, reg_syn)
        d2_hit = any("0.25" in f"{r['target_speed']}" for r in rep_a["D2"]["table"]
                     if r["verdict"] == "FAIL")
        d3b_hit = any("v0250_C1_on" in f for f in rep_a["D3B"]["failures"])
        record("Negative A: tau swap -> D2+D3B FAIL",
               rep_a["overall"] == "FAIL" and rep_a["D2"]["verdict"] == "FAIL"
               and rep_a["D3B"]["verdict"] == "FAIL" and rep_a["D1"]["verdict"] == "PASS"
               and d2_hit and d3b_hit,
               f"overall={rep_a['overall']} D1={rep_a['D1']['verdict']} "
               f"D2={rep_a['D2']['verdict']} D3B={rep_a['D3B']['verdict']}")

        # ── Negative B：condition 改成错 arm ──
        def mut_b(rs):
            rs["v0300_C2_off"]["runtime_assignment"]["decoder_condition_id"] = "vb0_db4"
            rs["v0300_C2_off"]["runtime_assignment"]["speed_bin"] = 0
        rd_b, _ = corrupted("negB", mut_b)
        rep_b = run_checker(rd_b, mat_dir, reg_syn)
        d3a_hit = any("v0300_C2_off" in f for f in rep_b["D3A"]["failures"])
        record("Negative B: wrong condition -> D3A FAIL",
               rep_b["overall"] == "FAIL" and rep_b["D3A"]["verdict"] == "FAIL"
               and rep_b["D1"]["verdict"] == "PASS" and d3a_hit,
               f"overall={rep_b['overall']} D3A={rep_b['D3A']['verdict']}")

        # ── Negative C：decoder 超参/shape 被改 → D1 FAIL ──
        def mut_c1(rs):
            rs["v0200_C1_on"]["decoder_identity"]["architecture"]["n_vbins"] = 4
        rd_c1, _ = corrupted("negC1", mut_c1)
        rep_c1 = run_checker(rd_c1, mat_dir, reg_syn)
        c1_hit = any("n_vbins" in f for f in rep_c1["D1"]["failures"])

        def mut_c2(rs):
            rs["v0325_C2_on"]["decoder_identity"]["state_dict_key_shapes"]["decoder.4.weight"] = [64, 512]
        rd_c2, _ = corrupted("negC2", mut_c2)
        rep_c2 = run_checker(rd_c2, mat_dir, reg_syn)
        c2_hit = any("decoder.4.weight" in f for f in rep_c2["D1"]["failures"])
        record("Negative C: decoder arch/shape tamper -> D1 FAIL",
               rep_c1["D1"]["verdict"] == "FAIL" and rep_c2["D1"]["verdict"] == "FAIL"
               and c1_hit and c2_hit and rep_c1["overall"] == "FAIL",
               f"C1 n_vbins→4: D1={rep_c1['D1']['verdict']}; "
               f"C2 shape: D1={rep_c2['D1']['verdict']}")

        # ── Negative D：自报 PASS flag + assignment 实际不一致 → 仍 FAIL ──
        def mut_d(rs):
            mut_b(rs)
            rs["v0300_C2_off"]["assignment_ok"] = True
            rs["v0300_C2_off"]["runtime_assignment"]["tau_ok"] = True
            rs["v0300_C2_off"]["runtime_assignment"]["d3_pass"] = True
        rd_d, _ = corrupted("negD", mut_d)
        rep_d = run_checker(rd_d, mat_dir, reg_syn)
        d_schema = any("assignment_ok" in f or "tau_ok" in f or "d3_pass" in f
                       for f in rep_d["schema_check"]["failures"])
        d3a_still = rep_d["D3A"]["verdict"] == "FAIL" and any(
            "v0300_C2_off" in f for f in rep_d["D3A"]["failures"])
        record("Negative D: self-reported PASS ignored -> still FAIL",
               rep_d["overall"] == "FAIL" and rep_d["schema_check"]["verdict"] == "FAIL"
               and d_schema and d3a_still,
               f"schema={rep_d['schema_check']['verdict']} D3A={rep_d['D3A']['verdict']}")

        # ── T4 receipt 封闭 schema：缺字段/多字段均 FAIL ──
        def mut_e(rs):
            del rs["v0225_C1_on"]["tau_material"]["source_lineage"]
        rd_e, _ = corrupted("negE", mut_e)
        rep_e = run_checker(rd_e, mat_dir, reg_syn)
        lineage_hit = any("source_lineage" in f for f in rep_e["schema_check"]["failures"])
        record("Negative E: lineage 字段缺失 -> schema FAIL（D2 亦不 PASS）",
               rep_e["overall"] == "FAIL" and lineage_hit
               and rep_e["D2"]["verdict"] == "FAIL",
               f"schema={rep_e['schema_check']['verdict']} D2={rep_e['D2']['verdict']}")

    n_pass = sum(1 for r in results if r["pass"])
    report = {
        "artifact": "rung1-selftest-report/v1",
        "selftest": True,
        "note": "checker 逻辑自测（synthetic receipt）；永不作为 D artifact（协议 §10.1）",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": __import__("platform").platform(),
        "results": results,
        "n_pass": n_pass,
        "n_total": len(results),
        "all_pass": n_pass == len(results),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selftest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nselftest: {n_pass}/{len(results)} PASS -> "
          f"{args.out / 'selftest_report.json'}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
