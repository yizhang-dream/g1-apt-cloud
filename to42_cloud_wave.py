"""TO42 云端 wave 驱动（flux task 入口，gm-run 执行）。

角色（SCRIPT_MAP 登记）：**state-changing execution orchestration**（TO42_PLAN
§8 执行序 2–5 的云上串行编排；单 A10 pod 内 4 runs 同环境串行 = 同机配对纪律）。

阶段（--stages 可选子集，默认全链，fail-fast）：
  selftest : G0 纯 torch 自检（to42_selftest，负例先行）
  smoke    : 双臂冒烟训练（128 envs × 30 it）+ Isaac 级 G0 冒烟 eval
             （300 步；fbkt 时间线逐位 == 自然 bin、lsel 切换 ⊆ 边界的行内断言）
  train    : 4 runs 全训（to42r1-{lsel,fbkt}-s{0,1}，E47 配方 + ctrl 旗标 +
             τ 恒 OFF，2000 it）
  select   : ckpt 机械选择（50-iter 窗口 argmax → manifest）
  eval     : 28 receipts（2 臂 × 2 seeds × 7 v × 3 eval seeds × 3000 步）
  check    : to42_checker 全 PASS 才进 report（先审计后分析）
  report   : err60s 效应表 + selection 描述统计（中性命名，不判机制）
  bundle   : 全部 receipts/manifest/audit/train_logs/effect 打包
             output/to42/to42_artifacts.pt（平台 ckpt 发现通道取回）+ stdout
             摘要（TO42_RESULT_JSON 行）。

pybind 教训（TO38）：Path 对象禁入 sys.path，一律 str() 包装（本文件头两行）。
"""

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from apt_g1.isaac.to42_gate import natural_vb

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
os.chdir(str(_REPO))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("PRIVACY_CONSENT", "Y")
os.environ["PYTHONPATH"] = str(_REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

GRID7 = (0.200, 0.225, 0.250, 0.275, 0.277, 0.300, 0.325)
MID_BAND = (0.275, 0.277, 0.300, 0.325)
ARMS = ("lsel", "fbkt")
SEEDS = (0, 1)
RUNS_ROOT = _REPO / "output/to42"
EVAL_DIR = _REPO / "apt_g1/outputs/to42/eval"
VAE = "apt_g1/outputs/token_vae_e39/vae.pt"
REF = "apt_g1/outputs/sync/to38_ref.npz"
DECODER = "gear_sonic_deploy/policy/release/model_decoder.onnx"
ROUTER = "apt_g1/outputs/distill_final"

BASE_ARGS = [
    "--latent-mode",
    "--latent-vae-path", VAE,
    "--latent-speed-bins", "--latent-dir-bins",
    "--latent-kl-prior", "zero",
    "--progress-scale", "1.0", "--heading-scale", "0.4",
    "--to-ref", "--to-ref-npz", REF, "--to-ref-obs-zero", "--to-ref-w", "0",
    "--num-envs", "128", "--iters", "2000",
]


def _run(tag: str, cmd: list, timeout: float | None = None) -> None:
    print(f"\n=== [{time.strftime('%F %T')}] LAUNCH {tag} ===", flush=True)
    print(" ".join(str(c) for c in cmd), flush=True)
    t0 = time.monotonic()
    rc = subprocess.call([str(c) for c in cmd], cwd=str(_REPO),
                         env=dict(os.environ), timeout=timeout)
    dt = time.monotonic() - t0
    print(f"=== [{time.strftime('%F %T')}] DONE {tag} rc={rc} "
          f"({dt/60:.1f} min) ===", flush=True)
    if rc != 0:
        raise SystemExit(f"WAVE ABORT: {tag} rc={rc}")


def run_py(tag: str, module: str, args: list, timeout: float | None = None) -> None:
    _run(tag, [sys.executable, "-m", module, *args], timeout=timeout)


def _receipt_path(arm: str, v: float, seed: int, smoke: bool) -> Path:
    base = EVAL_DIR / ("smoke" if smoke else "")
    return base / "receipts" / f"receipt_to42-{arm}-v{v:.3f}__s{seed}.json"


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def stage_selftest() -> None:
    run_py("g0-selftest", "apt_g1.isaac.to42_selftest", [], timeout=600)


def stage_smoke() -> None:
    for arm in ARMS:
        run_py(f"smoke-train-{arm}", "apt_g1.isaac.train_apt_isaac",
               BASE_ARGS[:-2] + ["--iters", "30", "--seed", "0",
                                 "--to42-sel", arm,
                                 "--out", f"output/to42/smoke-{arm}"],
               timeout=3600)
    # Isaac 级 G0：300 步冒烟 eval + 行内断言（协议 §5 G0 的 wiring 部分）
    for arm in ARMS:
        run_py(f"smoke-eval-{arm}", "apt_g1.rung1.to42_eval",
               ["--mode", "execute", "--arm", arm, "--v", "0.277",
                "--train-seed", "0",
                "--ckpt", f"output/to42/smoke-{arm}/policy_final.pt",
                "--steps", "300", "--eval-seeds", "0", "--smoke",
                "--env-tag", "cloud"], timeout=3600)
        rp = _receipt_path(arm, 0.277, 0, smoke=True)
        rc = _load_json(rp)
        ep = rc["episodes"][0]
        sel = list(base64.b64decode(ep["sel_timeline_b64"]))
        nat = int(natural_vb(torch.tensor([0.277])).clamp(0, 1)[0])
        assert len(sel) == ep["steps_done"] == 300, \
            f"G0 smoke timeline length broken: {len(sel)} vs {ep['steps_done']}"
        if arm == "fbkt":
            assert set(sel) == {nat} and ep["sel_switch_steps"] == [], \
                f"G0 smoke fbkt violated: sel set={set(sel)} expect={{{nat}}}"
        else:
            bad = [t for t in ep["sel_switch_steps"] if t % 25 != 0]
            assert not bad and set(sel) <= {0, 1}, \
                f"G0 smoke lsel violated: off-boundary={bad[:5]}"
        print(f"[g0-smoke] {arm}: timeline OK (natural={nat}, "
              f"switches={ep['sel_switch_steps']})", flush=True)


def stage_train() -> None:
    for seed in SEEDS:
        for arm in ARMS:
            run_py(f"train-{arm}-s{seed}", "apt_g1.isaac.train_apt_isaac",
                   BASE_ARGS + ["--seed", str(seed), "--to42-sel", arm,
                                "--out", f"output/to42/to42r1-{arm}-s{seed}"],
                   timeout=6 * 3600)


def stage_select() -> None:
    run_py("select", "apt_g1.rung1.to42_select",
           ["--runs-root", str(RUNS_ROOT),
            "--out", str(RUNS_ROOT / "ckpt_selection.json")], timeout=600)


def stage_eval() -> None:
    manifest = json.loads((RUNS_ROOT / "ckpt_selection.json").read_text("utf-8"))
    for seed in SEEDS:
        for arm in ARMS:
            ckpt = manifest["runs"][arm][str(seed)]["ckpt_file"]
            for v in GRID7:
                run_py(f"eval-{arm}-v{v:.3f}-s{seed}", "apt_g1.rung1.to42_eval",
                       ["--mode", "execute", "--arm", arm, "--v", f"{v:.3f}",
                        "--train-seed", str(seed), "--ckpt", ckpt,
                        "--steps", "3000", "--eval-seeds", "0,1,2",
                        "--out-dir", str(EVAL_DIR), "--env-tag", "cloud"],
                       timeout=3600)


def stage_check() -> None:
    run_py("checker", "apt_g1.rung1.to42_checker",
           ["--eval-dir", str(EVAL_DIR),
            "--selection-manifest", str(RUNS_ROOT / "ckpt_selection.json"),
            "--out", str(_REPO / "apt_g1/outputs/to42/to42_eval_audit.json")],
           timeout=1200)


def stage_report() -> dict:
    manifest = json.loads((RUNS_ROOT / "ckpt_selection.json").read_text("utf-8"))
    table = {}
    for arm in ARMS:
        for seed in SEEDS:
            for v in GRID7:
                rc = _load_json(_receipt_path(arm, v, seed, smoke=False))
                eps = rc["episodes"]
                table[f"{arm}|{seed}|{v:.3f}"] = {
                    "err60s": round(sum(e["err60s"] for e in eps) / len(eps), 4),
                    "vx_mean": round(sum(e["vx_mean"] for e in eps) / len(eps), 4),
                    "disp": round(sum(e["disp"] for e in eps) / len(eps), 2),
                    "h_min": min(e["h_min"] for e in eps),
                    "n_switches": round(sum(e["n_switches"] for e in eps) / len(eps), 1),
                    "sel_head_p1_mean": round(
                        sum(e["sel_head_p1_mean"] for e in eps) / len(eps), 4),
                    "completed": all(e["completed"] for e in eps),
                }
    contrasts = {}
    for seed in SEEDS:
        for v in GRID7:
            d = (table[f"lsel|{seed}|{v:.3f}"]["err60s"]
                 - table[f"fbkt|{seed}|{v:.3f}"]["err60s"])
            contrasts[f"{seed}|{v:.3f}"] = round(d, 4)
    mid = [v for v in GRID7 if v in MID_BAND]
    summary = {
        "err60s_table": table,
        "selection_interface_contrast(lsel-fbkt)_by_seed_v": contrasts,
        "mid_band_mean_contrast_by_seed": {
            str(s): round(sum(contrasts[f"{s}|{v:.3f}"] for v in mid) / len(mid), 4)
            for s in SEEDS},
        "note": "descriptive aggregation only（checker PASS 后）；机制级判读"
                "与 TO41 前沿对照在本地分析完成",
    }
    print("TO42_REPORT_JSON:" + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def stage_bundle(summary: dict) -> None:
    import hashlib

    import torch

    payload = {
        "artifact": "to42-artifacts/v1",
        "meta": {
            "grid7": list(GRID7),
            "mid_band": list(MID_BAND),
            "base_args": BASE_ARGS,
            "vae": VAE, "ref_npz": REF, "decoder": DECODER, "router": ROUTER,
        },
        "ckpt_selection": json.loads(
            (RUNS_ROOT / "ckpt_selection.json").read_text("utf-8")),
        "train_logs": {
            f"to42r1-{a}-s{s}": json.loads(
                (RUNS_ROOT / f"to42r1-{a}-s{s}" / "train_log.json").read_text("utf-8"))
            for a in ARMS for s in SEEDS},
        "receipts": {
            p.name: json.loads(p.read_text("utf-8"))
            for p in sorted((EVAL_DIR / "receipts").glob("receipt_*.json"))},
        "audit": json.loads(
            (_REPO / "apt_g1/outputs/to42/to42_eval_audit.json").read_text("utf-8")),
        "report": summary,
    }
    out = RUNS_ROOT / "to42_artifacts.pt"
    torch.save(payload, out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"[bundle] {out} sha256={h} size={out.stat().st_size}", flush=True)
    result = {
        "verdict_checker": payload["audit"]["verdict"],
        "ckpt_window": {f"{a}-s{s}": payload["ckpt_selection"]["runs"][a][str(s)]["window_iters"]
                        for a in ARMS for s in SEEDS},
        "mid_band_mean_contrast_by_seed": summary["mid_band_mean_contrast_by_seed"],
        "artifacts_sha256": h,
    }
    print("TO42_RESULT_JSON:" + json.dumps(result, ensure_ascii=False), flush=True)


def main() -> int:
    stages = (sys.argv[sys.argv.index("--stages") + 1].split(",")
              if "--stages" in sys.argv
              else ["selftest", "smoke", "train", "select", "eval", "check",
                    "report", "bundle"])
    _head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=str(_REPO), capture_output=True, text=True)
    commit = _head.stdout.strip() if _head.returncode == 0 else "unknown"
    print(f"=== TO42 wave start {time.strftime('%F %T')} "
          f"commit={commit} stages={stages} ===", flush=True)
    t0 = time.monotonic()
    for st in stages:
        if st == "selftest":
            stage_selftest()
        elif st == "smoke":
            stage_smoke()
        elif st == "train":
            stage_train()
        elif st == "select":
            stage_select()
        elif st == "eval":
            stage_eval()
        elif st == "check":
            stage_check()
        elif st == "report":
            globals()["_SUMMARY"] = stage_report()
        elif st == "bundle":
            stage_bundle(globals().get("_SUMMARY") or stage_report())
        else:
            raise SystemExit(f"unknown stage {st}")
    print(f"=== TO42 wave ALL DONE ({(time.monotonic()-t0)/3600:.2f} h) ===",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
