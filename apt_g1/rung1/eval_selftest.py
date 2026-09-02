"""TO41 Rung 1 eval 栈 selftest：正例 + 故意失败负例（本机可跑，无 torch/isaac）。

角色（SCRIPT_MAP 登记）：**read-only test**。三十九轮 owner 裁定的负例先行
纪律：正式 28-cell eval 之前，证明 evaluator 本身"不仅会跑，还会拦"。

场景 = 合成 receipt（28 cells × train_seed 0，v 网格全枚举），消费仓内真实
derived LUT 文件副本；审计侧数组哈希与冻结 manifest 交叉一致。用例：

    T1   正例：28 receipt 合成场景 → checker G1–G10 全 PASS
    N1   错 τ：LUT 文件 tau_ref6 换成另一 v 的内容 → G3 FAIL；
         driver 侧 verify_lut_identity 同步 hard fail（preflight 防线）
    N2   override 覆写：单 call applied != mapped → G5 FAIL
    N3   未授权 config：OFF 臂 cfg 改 to_ref_w → G6 FAIL
    N4   coverage 缺口：删一个 receipt → G1 FAIL（连带该 v 的 G4）
    N5   OFF 臂 τ 泄漏：n_tau_calls != 0 → G5 FAIL
    N6   checkpoint 混用：同 (arm,seed) 出现第二 ckpt sha → G2 FAIL
    N7   probe 记录不完整：per_call 数 != n_decode_calls → G5 FAIL

场景构建只用合成数据 + 仓内冻结 LUT + 审计侧独立解析；不触任何训练/实验
状态，不 import 被测 driver 的执行路径（仅 N1-driver 用 verify_lut_identity
本身作为被测对象）。
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from apt_g1.rung1.eval_checker import audit, load_mapping_independent
from apt_g1.rung1.l_checker import _natural_bin, load_lut_manifest

CKPT_CTRL = "a" * 64
CKPT_T10 = "b" * 64
DECIMATION = 4
STEPS_PER_EPISODE = 8
N_EPISODES = 3
TRAIN_SEED = 0


def _base_cfg(to_tau: bool, lut_path: Path) -> dict:
    return {
        "scene.num_envs": 1, "sim.dt": 0.005, "decimation": DECIMATION,
        "episode_length_s": 120.0, "action_space": 16, "observation_space": 213,
        "disturbance_prob": 0.0, "latent_mode": True, "latent_speed_bins": True,
        "latent_dir_bins": True, "latent_residual": False,
        "latent_vae_n_bins": 3, "latent_vae_n_dbins": 8,
        "latent_cmd_phase_rate": False, "latent_vae_path": "/x/vae.pt",
        "vx_max": 0.8, "vx_min": 0.0, "use_sonic_prior": True,
        "sonic_decoder_path": "/x/model_decoder.onnx",
        "router_model_dir": "/x/distill_final", "use_2hz_gate": 1,
        "to_ref": True, "to_ref_npz": str(lut_path), "to_ref_obs_zero": True,
        "to_ref_w": 0.0, "to_tau": to_tau, "to_tau_w": 1.0,
        "to_ref_gate2": 0.0036, "to_ref_sigma2": 0.06,
        "terrain": "plane_importer(seed=0,noise=0.04)",
    }


def make_receipt(cell_id: str, v: float, cond_arm: str, tau_ff: str,
                 mapped_vb: int, mapped_db: int, lut_path: Path,
                 entry: dict) -> dict:
    natural_vb = _natural_bin(v)
    steps_total = STEPS_PER_EPISODE * N_EPISODES
    arm = "t10" if tau_ff == "on" else "ctrl"
    per_call = [
        {"i": i, "natural_vb": natural_vb, "applied_vb": mapped_vb,
         "natural_db": 4, "applied_db": mapped_db}
        for i in range(steps_total)
    ]
    tau_sha16 = entry["lut_array_sha256"]["tau_ref6"][:16]
    return {
        "schema": "rung1-eval-receipt/v1",
        "cell_id": cell_id,
        "cell_index": 0,
        "train_seed": TRAIN_SEED,
        "policy_arm": arm,
        "smoke": False,
        "target_speed": v,
        "condition_arm": cond_arm,
        "tau_ff": tau_ff,
        "eval_seeds": [0, 1, 2],
        "eval_seed_note": "pre-registered harness list (test scenario)",
        "assignment": {
            "decoder_condition_id": f"vb{mapped_vb}_db{mapped_db}",
            "speed_bin": mapped_vb,
            "dir_bin": mapped_db,
            "natural_condition_id": f"vb{natural_vb}_db4",
            "natural_speed_bin": natural_vb,
            "selection_source": "frozen_mapping_v2_lookup",
        },
        "checkpoint": {
            "policy_arm": arm, "train_seed": TRAIN_SEED,
            "ckpt_path": f"/x/to41r1-{arm}-s{TRAIN_SEED}/policy_it_50.pt",
            "ckpt_sha256": CKPT_T10 if tau_ff == "on" else CKPT_CTRL,
            "selection_window_iters": [1, 50],
            "selection_window_mean_rew": 2.4,
            "state_dict_sha256": "e" * 64,
            "policy_class": "AptPPOPolicy",
            "obs_dim": 213, "action_dim": 16,
            "shared_across_all_cells_of_arm_seed": True,
        },
        "env_identity": {"env_class": "AptFlatG1Env", "runtime_commit": "test"},
        "cfg_snapshot": _base_cfg(tau_ff == "on", lut_path),
        "tau_material": {
            "lut_manifest_sha256": "0" * 64,
            "frozen_material": {
                "artifact": entry["source_artifact"],
                "v_realized": v, "abs_err": 6.7e-09,
                "source_lineage": "test",
            },
            "derived_lut": {
                "file": str(lut_path),
                "array_sha256": dict(entry["lut_array_sha256"]),
                "source_artifact_manifest": entry["source_artifact"],
                "applied_to_env": True,
                "cfg_to_ref_npz": str(lut_path),
                "identity_verified_preflight": True,
            },
            "buffer_shape": [entry["T"], 6], "buffer_dtype": "<f4",
            "buffer_sha256_pre": entry["lut_array_sha256"]["tau_ref6"],
            "buffer_sha256_post": entry["lut_array_sha256"]["tau_ref6"],
        },
        "condition_override": {
            "mapped_speed_bin": mapped_vb, "mapped_dir_bin": mapped_db,
            "n_decode_calls": steps_total,
            "n_override_changed": sum(1 for c in per_call if c["natural_vb"] != c["applied_vb"]),
            "natural_vb_distribution": {str(natural_vb): steps_total},
            "per_call": per_call,
        },
        "tau_consumption": {
            "n_tau_calls": steps_total * DECIMATION if tau_ff == "on" else 0,
            "calls_tau_digest_sha256": "c" * 64,
            "first_tau_sha256_16": tau_sha16,
            "last_tau_sha256_16": tau_sha16,
            "n_nonfinite_tau_calls": 0,
        },
        "decoder_identity": {"vae_path": "/x/vae.pt", "checkpoint_sha256": "f" * 64},
        "episodes": [
            {
                "eval_seed": es, "steps_requested": STEPS_PER_EPISODE,
                "steps_done": STEPS_PER_EPISODE, "completed": True, "fall_step": None,
                "h_min": 0.72, "vx_mean": 0.19, "v_speed_mean": 0.2, "disp": 1.6,
                "reset_count": 1, "n_auto_resets": 0, "n_cmd_reassertions": 9,
                "boundaries": [{"step": 0, "type": "episode_start",
                                "cond_calls_before": es * steps_total,
                                "tau_calls_before": es * steps_total * DECIMATION}],
            }
            for es in range(N_EPISODES)
        ],
        "execution": {
            "status": "completed",
            "steps_done_total": steps_total,
            "expected_decode_calls": steps_total,
            "expected_tau_calls_on": steps_total * DECIMATION,
        },
    }


def build_scenario(mutate=None) -> Path:
    """合成场景目录：28 receipt（× seed 0）+ 每 v 一份真实 LUT 副本。"""
    mapping = load_mapping_independent()
    manifest = load_lut_manifest()
    scenario = Path(tempfile.mkdtemp(prefix="rung1_eval_selftest_"))
    receipts_dir = scenario / "receipts"
    receipts_dir.mkdir(parents=True)
    receipts: dict[str, dict] = {}
    for v in mapping["grid"]:
        entry = manifest["entries"][round(float(v), 3)]
        lut_path = scenario / f"lut_{round(float(v) * 1000):04d}.npz"
        shutil.copy(manifest["manifest_path"].parent / entry["lut_file"], lut_path)
        for cond_arm in ("C1", "C2"):
            cid = mapping["rows"][(round(float(v), 3), cond_arm)]
            mapped_vb, mapped_db = mapping["conds"][cid]
            for tau_ff in ("on", "off"):
                cell_id = f"v{round(float(v) * 1000):04d}_{cond_arm}_{tau_ff}"
                r = make_receipt(cell_id, round(float(v), 3), cond_arm, tau_ff,
                                 mapped_vb, mapped_db, lut_path, entry)
                receipts[f"{cell_id}__s{TRAIN_SEED}"] = r
    if mutate is not None:
        mutate(receipts, scenario, manifest)
    for key, r in receipts.items():
        (receipts_dir / f"receipt_{key}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    sel = scenario / "ckpt_selection.json"
    sel.write_text(json.dumps({
        "artifact": "rung1-eval-ckpt-selection/v1",
        "rule": "selftest synthetic",
        "runs": {
            "ctrl": {"0": {"ckpt_sha256": CKPT_CTRL}},
            "t10": {"0": {"ckpt_sha256": CKPT_T10}},
        },
    }, indent=2), encoding="utf-8")
    return scenario


def run_case(name: str, expect_fail_in: set[str], mutate=None) -> bool:
    scenario = build_scenario(mutate)
    try:
        a = audit(scenario / "receipts", [scenario], scenario / "ckpt_selection.json",
                  [TRAIN_SEED], [0, 1, 2])
        failed = {gid for gid, c in a.checks.items() if c["verdict"] == "FAIL"}
        ok = failed == expect_fail_in
        print(f"{'PASS' if ok else 'UNEXPECTED'} {name}: failed={sorted(failed) or '∅'} "
              f"(expect {sorted(expect_fail_in) or '∅'})")
        if not ok:
            for m in a.failures[:8]:
                print(f"    {m}")
        return ok
    finally:
        shutil.rmtree(scenario, ignore_errors=True)


def _n1(receipts, scenario, manifest):
    """tau_ref6 换成另一 v（0.225）的内容 → 消费文件与冻结 manifest 不符。"""
    entry_other = manifest["entries"][0.225]
    lut_other = manifest["manifest_path"].parent / entry_other["lut_file"]
    with np.load(lut_other) as z:
        tau_bad = z["tau_ref6"]
    lut_used = scenario / "lut_0200.npz"
    with np.load(lut_used) as z:
        data = {k: z[k] for k in z.files}
    data["tau_ref6"] = tau_bad
    np.savez(lut_used, **data)


def _n2(receipts, scenario, manifest):
    receipts["v0200_C2_on__s0"]["condition_override"]["per_call"][3]["applied_vb"] = 0


def _n3(receipts, scenario, manifest):
    receipts["v0200_C2_off__s0"]["cfg_snapshot"]["to_ref_w"] = 0.3


def _n4(receipts, scenario, manifest):
    del receipts["v0200_C1_off__s0"]


def _n5(receipts, scenario, manifest):
    receipts["v0200_C2_off__s0"]["tau_consumption"]["n_tau_calls"] = 96


def _n6(receipts, scenario, manifest):
    receipts["v0200_C1_off__s0"]["checkpoint"]["ckpt_sha256"] = "9" * 64


def _n7(receipts, scenario, manifest):
    receipts["v0200_C1_on__s0"]["condition_override"]["per_call"].pop(0)


def _n1_driver() -> bool:
    """driver 侧 preflight 防线：verify_lut_identity 必须对错 τ 硬失败。"""
    from apt_g1.rung1.eval_cell import verify_lut_identity

    manifest = load_lut_manifest()
    entry = manifest["entries"][round(SCENARIO_V, 3)]
    scenario = Path(tempfile.mkdtemp(prefix="rung1_eval_selftest_n1d_"))
    try:
        lut_used = scenario / "lut_0200.npz"
        shutil.copy(manifest["manifest_path"].parent / entry["lut_file"], lut_used)
        verify_lut_identity(lut_used, entry)  # 原样 LUT → 通过
        _n1({}, scenario, manifest)           # 换 tau_ref6
        try:
            verify_lut_identity(lut_used, entry)
        except SystemExit:
            print("PASS N1-driver: verify_lut_identity 对错 τ hard fail")
            return True
        print("UNEXPECTED N1-driver: 错 τ 未被 preflight 拦截")
        return False
    finally:
        shutil.rmtree(scenario, ignore_errors=True)


SCENARIO_V = 0.2


def main() -> int:
    results = [
        run_case("T1 正例全 PASS", set()),
        run_case("N1 错 τ → G3", {"G3"}, _n1),
        _n1_driver(),
        run_case("N2 override 覆写 → G5", {"G5"}, _n2),
        run_case("N3 未授权 cfg → G6", {"G6"}, _n3),
        run_case("N4 coverage 缺口 → G1+G4", {"G1", "G4"}, _n4),
        run_case("N5 OFF 臂 τ 泄漏 → G5", {"G5"}, _n5),
        run_case("N6 ckpt 混用 → G2", {"G2"}, _n6),
        run_case("N7 probe 记录不完整 → G5", {"G5"}, _n7),
    ]
    n_ok = sum(results)
    print(f"selftest: {n_ok}/{len(results)} cases OK")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
