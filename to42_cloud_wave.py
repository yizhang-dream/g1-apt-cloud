"""TO42 云端 wave 驱动 v2（flux task 入口，gm-run 执行；worker-pool 并发版）。

角色（SCRIPT_MAP 登记）：**state-changing execution orchestration**（TO42_PLAN
§8 执行序 2–5 的云上编排；执行身份修订 v2 = owner 2026-09-03 深夜授权
「worker-pool 最省时间」：训练 2 并发（同 seed (lsel, fbkt) 成对并发，对内
共享争用状态、配对对称性保持），评测 3 并发（每 cell 单 env 确定性策略，
零科学足迹）且 mid-band 优先出队（中途死亡的 salvage 价值最大化）。并发位
次入 receipt（execution.worker_tag）与 bundle meta.concurrency。预计全程
4.4h → ~2.8–3.2h。）

阶段（--stages 可选子集，默认全链，fail-fast）：
  selftest : G0 纯 torch 自检（to42_selftest，负例先行）
  smoke    : 双臂冒烟训练（30it，2 并发）+ Isaac 级 G0 冒烟 eval（行内断言）
  train    : 4 runs（E47 配方 + ctrl 旗标 + τ 恒 OFF，2 并发同 seed 配对）
  select   : ckpt 机械选择（50-iter 窗口 argmax → manifest）
  eval     : 28 receipts（3 并发；mid-band 16 cells 先跑）
  check    : to42_checker 全 PASS 才进 report（先审计后分析）
  report   : err60s 效应表 + selection 描述统计（descriptive only）
  bundle   : 终版打包 output/to42/to42_artifacts.pt + TO42_RESULT_JSON

并发安全设计：worker 池只在**主线程**调 on_done（_save_bundle / 存在性断言
/ G0 断言），bundle 写永不并发；任一 job rc≠0 或 on_done 失败 → 终止其余
worker → ABORT。原子 bundle（tmp→fsync→rename+目录 fsync）防半写。

pybind 教训（TO38）：Path 禁入 sys.path，一律 str() 包装。
**不要覆写 PYTHONPATH**——gm-run 原环境（isaacsim/isaaclab 解析）必须原样
传给子进程；仓根由 `python -m` 的 cwd 语义自动进入子进程 sys.path。
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
os.environ.setdefault("MUJOCO_GL", "egl")

GRID7 = (0.200, 0.225, 0.250, 0.275, 0.277, 0.300, 0.325)
MID_BAND = (0.275, 0.277, 0.300, 0.325)
ARMS = ("lsel", "fbkt")
SEEDS = (0, 1)
TRAIN_WORKERS = 2   # 同 seed (lsel, fbkt) 成对并发；24G 显存约束下的上限
EVAL_WORKERS = 3    # 每 cell 单 env，GPU/CPU 占用低，owner 授权 3 并发
SMOKE_WORKERS = 2
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
    # 显式仓相对路径——train 入口的 --decoder-path/--router-model-dir 默认值是
    # lab-ts 绝对路径（云 pod 上不存在；2026-09-03 v2 首跑即挂此）
    "--router-model-dir", ROUTER,
    "--decoder-path", DECODER,
    "--num-envs", "128",
]


def _pool(jobs: list, workers: int) -> None:
    """并发池。jobs = [(tag, cmd, timeout_s, on_done)]；on_done 在主线程串行
    执行（bundle 写/断言因此无并发）。rc≠0 或 on_done 失败 → 终止其余 → ABORT。
    timeout 到点 terminate 并记失败（防单 worker 挂死拖死全链）。"""
    queue = list(jobs)
    running: dict = {}
    failures = []
    while queue or running:
        while queue and len(running) < workers:
            tag, cmd, timeout, on_done = queue.pop(0)
            print(f"\n=== [{time.strftime('%F %T')}] LAUNCH {tag} "
                  f"({len(running) + 1}/{workers} slots) ===", flush=True)
            print(" ".join(str(c) for c in cmd), flush=True)
            proc = subprocess.Popen([str(c) for c in cmd], cwd=str(_REPO),
                                    env=dict(os.environ))
            running[proc] = (tag, on_done, time.monotonic(), timeout)
        time.sleep(2.0)
        for proc in list(running):
            rc = proc.poll()
            if rc is None:
                tag, on_done, t0, timeout = running[proc]
                if timeout and time.monotonic() - t0 > timeout:
                    proc.terminate()
                    running.pop(proc)
                    failures.append((tag, "TIMEOUT"))
                    print(f"=== [{time.strftime('%F %T')}] DONE {tag} "
                          f"rc=TIMEOUT ===", flush=True)
                continue
            tag, on_done, t0, _ = running.pop(proc)
            dt = (time.monotonic() - t0) / 60
            print(f"=== [{time.strftime('%F %T')}] DONE {tag} rc={rc} "
                  f"({dt:.1f} min) ===", flush=True)
            if rc != 0:
                failures.append((tag, f"rc={rc}"))
                continue
            try:
                if on_done:
                    on_done()
            except SystemExit as e:
                failures.append((tag, f"post-check: {e}"))
            except Exception as e:  # noqa: BLE001
                failures.append((tag, f"post-check: {type(e).__name__}: {e}"))
        if failures:
            for p in running:
                p.terminate()
            raise SystemExit(f"WAVE ABORT: {failures}")


def _run_py(tag: str, module: str, args: list) -> tuple:
    return (tag, [sys.executable, "-m", module, *args], None, None)


def _receipt_path(arm: str, v: float, seed: int, smoke: bool = False) -> Path:
    base = EVAL_DIR / ("smoke" if smoke else "")
    return base / "receipts" / f"receipt_to42-{arm}-v{v:.3f}__s{seed}.json"


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _require_train_outputs(run_dir: Path) -> None:
    """训练产物存在性断言——Isaac 崩溃会把退出码吞成 0（atexit 硬退出），
    rc 不可信，产物在才算数。train_log 全文 gz+b64 进日志 + 立即增量打包
    （数据保全：平台 ckpt 上传不可靠，日志流 + 分阶段 bundle 是双保险）。"""
    for f in ("policy_final.pt", "train_log.json"):
        p = run_dir / f
        if not p.exists():
            raise SystemExit(f"训练产物缺失 {p}（训练子进程实际失败，"
                             "rc 被 Isaac atexit 吞掉）")
    import gzip

    _b64 = base64.b64encode(
        gzip.compress((run_dir / "train_log.json").read_bytes(), 9)).decode("ascii")
    print(f"TO42_GZ_B64:trainlog:{run_dir.name}:{_b64}", flush=True)
    _save_bundle(None)


def _g0_smoke_assert(arm: str) -> None:
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


def _save_bundle(summary: dict | None) -> Path:
    """增量打包当前已有产物 → output/to42/to42_artifacts.pt（任意阶段重写，
    pod 死掉最多丢最后一批 cell；summary 非空 = 终版）。原子写：write tmp →
    flush → fsync → os.replace + 目录 fsync（防半写撞坏上一版好 bundle）。"""
    import hashlib

    payload = {
        "artifact": "to42-artifacts/v1",
        "partial": summary is None,
        "meta": {
            "grid7": list(GRID7),
            "mid_band": list(MID_BAND),
            "base_args": BASE_ARGS,
            "vae": VAE, "ref_npz": REF, "decoder": DECODER, "router": ROUTER,
            "concurrency": {
                "train_workers": TRAIN_WORKERS,
                "train_pairing": [["to42r1-lsel-s0", "to42r1-fbkt-s0"],
                                  ["to42r1-lsel-s1", "to42r1-fbkt-s1"]],
                "eval_workers": EVAL_WORKERS,
                "eval_order": "mid-band-first",
                "authorized": "owner 2026-09-03 深夜 worker-pool 授权；"
                              "TO42_PLAN §9 执行身份修订 v2",
            },
        },
    }
    sel_path = RUNS_ROOT / "ckpt_selection.json"
    if sel_path.exists():
        payload["ckpt_selection"] = json.loads(sel_path.read_text("utf-8"))
    logs = {}
    for a in ARMS:
        for s in SEEDS:
            p = RUNS_ROOT / f"to42r1-{a}-s{s}" / "train_log.json"
            if p.exists():
                logs[f"to42r1-{a}-s{s}"] = json.loads(p.read_text("utf-8"))
    payload["train_logs"] = logs
    rdir = EVAL_DIR / "receipts"
    if rdir.exists():
        payload["receipts"] = {p.name: json.loads(p.read_text("utf-8"))
                               for p in sorted(rdir.glob("receipt_*.json"))}
    audit_path = _REPO / "apt_g1/outputs/to42/to42_eval_audit.json"
    if audit_path.exists():
        payload["audit"] = json.loads(audit_path.read_text("utf-8"))
    if summary is not None:
        payload["report"] = summary
    out = RUNS_ROOT / "to42_artifacts.pt"
    tmp = out.with_suffix(".pt.tmp")
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)  # POSIX 原子
    _dirfd = os.open(str(out.parent), os.O_DIRECTORY)
    try:
        os.fsync(_dirfd)
    finally:
        os.close(_dirfd)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"[bundle] partial={payload['partial']} "
          f"receipts={len(payload.get('receipts', {}))} runs={len(logs)} "
          f"sha256={h[:16]}… size={out.stat().st_size}", flush=True)
    return out


def stage_selftest() -> None:
    _pool([("g0-selftest", [sys.executable, "-m", "apt_g1.isaac.to42_selftest"],
            600, None)], 1)


def stage_smoke() -> None:
    train_jobs, eval_jobs = [], []
    for i, arm in enumerate(ARMS):
        train_jobs.append((
            f"smoke-train-{arm}",
            [sys.executable, "-m", "apt_g1.isaac.train_apt_isaac", *BASE_ARGS,
             "--iters", "30", "--seed", "0", "--to42-sel", arm,
             "--out", f"output/to42/smoke-{arm}"],
            3600,
            (lambda arm=arm: _require_train_outputs(RUNS_ROOT / f"smoke-{arm}"))))
        eval_jobs.append((
            f"smoke-eval-{arm}",
            [sys.executable, "-m", "apt_g1.rung1.to42_eval",
             "--mode", "execute", "--arm", arm, "--v", "0.277",
             "--train-seed", "0",
             "--ckpt", f"output/to42/smoke-{arm}/policy_final.pt",
             "--steps", "300", "--eval-seeds", "0", "--smoke",
             "--env-tag", "cloud", "--worker-tag", f"w{i}"],
            3600,
            (lambda arm=arm: _g0_smoke_assert(arm))))
    _pool(train_jobs, SMOKE_WORKERS)
    _pool(eval_jobs, SMOKE_WORKERS)


def stage_train() -> None:
    jobs = []
    for seed in SEEDS:      # s0 对先发（同 seed (lsel, fbkt) 成对并发）
        for arm in ARMS:
            run = f"to42r1-{arm}-s{seed}"
            jobs.append((
                f"train-{arm}-s{seed}",
                [sys.executable, "-m", "apt_g1.isaac.train_apt_isaac",
                 *BASE_ARGS, "--iters", "2000", "--seed", str(seed),
                 "--to42-sel", arm, "--out", f"output/to42/{run}"],
                6 * 3600,
                (lambda run=run: _require_train_outputs(RUNS_ROOT / run))))
    _pool(jobs, TRAIN_WORKERS)


def stage_eval() -> None:
    manifest = json.loads((RUNS_ROOT / "ckpt_selection.json").read_text("utf-8"))
    order = ([(a, s, v) for v in MID_BAND for a in ARMS for s in SEEDS] +
             [(a, s, v) for v in GRID7 if v not in MID_BAND
              for a in ARMS for s in SEEDS])
    jobs = []
    for i, (arm, seed, v) in enumerate(order):
        ckpt = manifest["runs"][arm][str(seed)]["ckpt_file"]

        def on_done(arm=arm, seed=seed, v=v):
            if not _receipt_path(arm, v, seed).exists():
                raise SystemExit(
                    f"receipt 缺失 {arm} v={v} s={seed}"
                    "（eval 子进程实际失败，rc 被 Isaac atexit 吞掉）")
            _save_bundle(None)  # 每 cell 增量打包（数据保全）

        jobs.append((
            f"eval-{arm}-v{v:.3f}-s{seed}",
            [sys.executable, "-m", "apt_g1.rung1.to42_eval",
             "--mode", "execute", "--arm", arm, "--v", f"{v:.3f}",
             "--train-seed", str(seed), "--ckpt", ckpt,
             "--steps", "3000", "--eval-seeds", "0,1,2",
             "--out-dir", str(EVAL_DIR), "--env-tag", "cloud",
             "--worker-tag", f"w{i % EVAL_WORKERS}"],
            3600, on_done))
    _pool(jobs, EVAL_WORKERS)


def stage_check() -> None:
    _pool([("checker", [sys.executable, "-m", "apt_g1.rung1.to42_checker",
                        "--eval-dir", str(EVAL_DIR),
                        "--selection-manifest",
                        str(RUNS_ROOT / "ckpt_selection.json"),
                        "--out",
                        str(_REPO / "apt_g1/outputs/to42/to42_eval_audit.json")],
            1200, None)], 1)


def stage_report() -> dict:
    table = {}
    for arm in ARMS:
        for seed in SEEDS:
            for v in GRID7:
                eps = _load_json(_receipt_path(arm, v, seed))["episodes"]
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
            contrasts[f"{seed}|{v:.3f}"] = round(
                table[f"lsel|{seed}|{v:.3f}"]["err60s"]
                - table[f"fbkt|{seed}|{v:.3f}"]["err60s"], 4)
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
    out = _save_bundle(summary)
    import hashlib

    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"[bundle] FINAL {out} sha256={h} size={out.stat().st_size}", flush=True)
    audit_p = _REPO / "apt_g1/outputs/to42/to42_eval_audit.json"
    verdict = (json.loads(audit_p.read_text("utf-8")).get("verdict")
               if audit_p.exists() else "UNKNOWN")
    sel_p = RUNS_ROOT / "ckpt_selection.json"
    windows = {}
    if sel_p.exists():
        _m = json.loads(sel_p.read_text("utf-8"))
        windows = {f"{a}-s{s}": _m["runs"][a][str(s)]["window_iters"]
                   for a in ARMS for s in SEEDS}
    result = {
        "verdict_checker": verdict,
        "ckpt_window": windows,
        "mid_band_mean_contrast_by_seed": summary.get("mid_band_mean_contrast_by_seed"),
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
    print(f"[wave-env] exe={sys.executable} cwd={os.getcwd()} "
          f"PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}", flush=True)
    print(f"=== TO42 wave v2 (worker-pool: train {TRAIN_WORKERS}x same-seed "
          f"pair / eval {EVAL_WORKERS}x mid-band-first) start "
          f"{time.strftime('%F %T')} commit={commit} stages={stages} ===",
          flush=True)
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


def stage_select() -> None:
    _pool([("select", [sys.executable, "-m", "apt_g1.rung1.to42_select",
                       "--runs-root", str(RUNS_ROOT),
                       "--out", str(RUNS_ROOT / "ckpt_selection.json")],
            600, lambda: _save_bundle(None))], 1)


if __name__ == "__main__":
    raise SystemExit(main())
