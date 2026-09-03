"""TO42 云端 wave 驱动 v3（flux task 入口，gm-run 执行；动态流水线极限版）。

角色（SCRIPT_MAP 登记）：**state-changing execution orchestration**（TO42_PLAN
§8 执行序 2–5；执行身份修订 v3 = owner 2026-09-04 授权「还不够极限」——在
v2 worker-pool 之上再加三件事：① **流水线**：任一 (arm, seed) 训练完成 →
增量机械选择（select_run，确定性）→ 立即放行该臂 7 个评测格（评测与训练
重叠，mid-band 优先）；② **显存自适应放行**：nvidia-smi 实时 free + 「最近
90s 内已放行任务的预留量」扣减，训练预留 9GB/评测 4.5GB——对未实测的单臂
显存足迹自整定；③ **冒烟零训练**：G0 wiring 检查与策略无关（fbkt 时间线 =
自然 bin 是 env 机器性质、lsel 切换 ⊆ 边界同），随机初始化 ckpt 直接验证
接线，砍掉两段冒烟训练。同时如实声明两个不为：**训练并发硬上限 = 2**
（24G 上 3×128env 显存未证实，一次 OOM 中断的代价 > 并发收益；本轮
VRAM 遥测留档供下一轮决策）；**report 不与 checker 重叠**（先审计后分析
纪律）。预计全程 4.4h（串行）→ ~2.2–2.4h。）

阶段（--stages 可选子集，默认全链，fail-fast）：
  selftest : G0 纯 torch 自检
  smoke    : 随机初始化 ckpt 双臂冒烟 eval（300 步，行内断言；零训练）
  train    : 4 runs（E47 配方 + ctrl 旗标 + τ 恒 OFF；2 并发，完成即流水）
  select   : 增量选择随训练流水进行；本阶段 = 终版 formal manifest 重算
  eval     : 28 receipts（随训练流水放行 + 尾段全宽；mid-band 优先）
  check    : to42_checker 全 PASS 才进 report（先审计后分析）
  report   : err60s 效应表 + selection 描述统计（descriptive only）
  bundle   : 终版打包 output/to42/to42_artifacts.pt + TO42_RESULT_JSON

数据保全（owner 09-03 指令）与并发安全：bundle 增量原子写（tmp→fsync→
rename+目录 fsync）且只在编排主线程落盘；receipt/train_log 全文 gz+b64 进
日志；任一 job 失败/超时 → 终止其余 → ABORT。**不要覆写 PYTHONPATH**
（gm-run 原环境 = isaacsim 解析路径；仓根靠 `python -m` cwd 语义）。
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
from apt_g1.rung1.select_checkpoint import select_run

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
RUNS_ROOT = _REPO / "output/to42"
EVAL_DIR = _REPO / "apt_g1/outputs/to42/eval"
VAE = "apt_g1/outputs/token_vae_e39/vae.pt"
REF = "apt_g1/outputs/sync/to38_ref.npz"
DECODER = "gear_sonic_deploy/policy/release/model_decoder.onnx"
ROUTER = "apt_g1/outputs/distill_final"

# 并发上限与显存预留（自适应放行的参数）
# 操作点 = TO42_PLAN §9 修订 v4（owner 09-04：2048 envs × 500it × mb4096，
# 论文式大并行；L20 48G 单臂估 ~36G → 训练并发 1（串行配对），流水线评测仍
# 与训练重叠：48−36=12G 余量供 2 个评测 worker）
TRAIN_CAP, EVAL_CAP, TOTAL_CAP = 1, 6, 6
TRAIN_VRAM_GB, EVAL_VRAM_GB = 36.0, 4.5
VRAM_RESERVE_WINDOW_S = 90  # 刚放行的任务显存爬坡期，按预留量扣减
NUM_ENVS, ITERS, PPO_MINIBATCH = 2048, 500, 4096

BASE_ARGS = [
    "--latent-mode",
    "--latent-vae-path", VAE,
    "--latent-speed-bins", "--latent-dir-bins",
    "--latent-kl-prior", "zero",
    "--progress-scale", "1.0", "--heading-scale", "0.4",
    "--to-ref", "--to-ref-npz", REF, "--to-ref-obs-zero", "--to-ref-w", "0",
    # 显式仓相对路径——train 入口默认值是 lab-ts 绝对路径（云 pod 不存在）
    "--router-model-dir", ROUTER,
    "--decoder-path", DECODER,
    "--num-envs", str(NUM_ENVS),
    "--ppo-minibatch", str(PPO_MINIBATCH),
]


# ----------------------------------------------------------------- VRAM 探测
def _vram_free_gb() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        return 99.0  # 探测失败 fail-open（OOM 由 fail-fast 兜底）


def _scheduler(pending: list) -> None:
    """统一动态池。pending = 共享 job 列表（dict：tag/cmd/kind/vram_gb/ready/
    on_done/timeout），**on_done 可在主线程向 pending 追加新 job**（流水线
    动态注入的唯一合法通道）。admit 条件：kind cap + total cap + (vram_free −
    90s 内已放行任务预留) ≥ vram_gb + ready()。失败/超时 → 终止其余 → ABORT。"""
    running: dict = {}   # proc -> dict(tag, kind, on_done, t0, timeout)
    reserved: list = []  # (monotonic, vram_gb) 最近放行的显存预留
    failures = []

    def counts():
        c = {"train": 0, "eval": 0, "util": 0}
        for r in running.values():
            c[r["kind"]] += 1
        return c

    while pending or running:
        c = counts()
        now = time.monotonic()
        eff_free = _vram_free_gb() - sum(v for t, v in reserved if now - t < VRAM_RESERVE_WINDOW_S)
        for job in list(pending):
            if len(running) >= TOTAL_CAP:
                break
            if c[job["kind"]] >= (TRAIN_CAP if job["kind"] == "train"
                                  else EVAL_CAP if job["kind"] == "eval" else 2):
                continue
            if job["kind"] != "util" and eff_free < job["vram_gb"]:
                continue
            if job.get("ready") and not job["ready"]():
                continue
            pending.remove(job)
            print(f"\n=== [{time.strftime('%F %T')}] LAUNCH {job['tag']} "
                  f"(train {c['train']}/eval {c['eval']}, free {eff_free:.1f}G) "
                  f"===", flush=True)
            print(" ".join(str(x) for x in job["cmd"]), flush=True)
            proc = subprocess.Popen([str(x) for x in job["cmd"]],
                                    cwd=str(_REPO), env=dict(os.environ))
            running[proc] = {"tag": job["tag"], "kind": job["kind"],
                             "on_done": job["on_done"],
                             "t0": time.monotonic(), "timeout": job["timeout"]}
            reserved.append((now, job["vram_gb"]))
            c[job["kind"]] += 1
            eff_free -= job["vram_gb"]
            time.sleep(1.0)
        time.sleep(2.0)
        for proc in list(running):
            rc = proc.poll()
            if rc is None:
                r = running[proc]
                if r["timeout"] and time.monotonic() - r["t0"] > r["timeout"]:
                    proc.terminate()
                    running.pop(proc)
                    failures.append((r["tag"], "TIMEOUT"))
                    print(f"=== [{time.strftime('%F %T')}] DONE {r['tag']} "
                          f"rc=TIMEOUT ===", flush=True)
                continue
            r = running.pop(proc)
            dt = (time.monotonic() - r["t0"]) / 60
            print(f"=== [{time.strftime('%F %T')}] DONE {r['tag']} rc={rc} "
                  f"({dt:.1f} min) ===", flush=True)
            if rc != 0:
                failures.append((r["tag"], f"rc={rc}"))
                continue
            try:
                if r["on_done"]:
                    r["on_done"](pending)
            except SystemExit as e:
                failures.append((r["tag"], f"post-check: {e}"))
            except Exception as e:  # noqa: BLE001
                failures.append((r["tag"], f"post-check: {type(e).__name__}: {e}"))
        if failures:
            for p in running:
                p.terminate()
            raise SystemExit(f"WAVE ABORT: {failures}")


# ----------------------------------------------------------------- 产物工具
def _receipt_path(arm: str, v: float, seed: int, smoke: bool = False) -> Path:
    base = EVAL_DIR / ("smoke" if smoke else "")
    return base / "receipts" / f"receipt_to42-{arm}-v{v:.3f}__s{seed}.json"


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _require_train_outputs(run_dir: Path) -> None:
    for f in ("policy_final.pt", "train_log.json"):
        p = run_dir / f
        if not p.exists():
            raise SystemExit(f"训练产物缺失 {p}（rc 被 Isaac atexit 吞掉）")
    import gzip

    _b64 = base64.b64encode(
        gzip.compress((run_dir / "train_log.json").read_bytes(), 9)).decode("ascii")
    print(f"TO42_GZ_B64:trainlog:{run_dir.name}:{_b64}", flush=True)
    _save_bundle(None)


def _select_one(arm: str, seed: int) -> dict:
    """增量机械选择（select_run 确定性；50-iter 窗口 argmax），条目即时落
    partial manifest 供 provenance；终版 formal manifest 由 to42_select 重算
    （逐字节同规则 → 同条目，checker 对账一致）。"""
    entry = select_run(RUNS_ROOT / f"to42r1-{arm}-s{seed}")
    pm = RUNS_ROOT / "ckpt_selection_partial.json"
    data = json.loads(pm.read_text("utf-8")) if pm.exists() else {
        "artifact": "to42-ckpt-selection-partial/v1", "runs": {a: {} for a in ARMS}}
    data["runs"][arm][str(seed)] = entry
    pm.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
    print(f"[select] {arm}-s{seed}: it{entry['window_iters'][1]} "
          f"rew={entry['window_mean_rew']} -> {Path(entry['ckpt_file']).name}",
          flush=True)
    return entry


def _g0_smoke_assert(arm: str) -> None:
    rp = _receipt_path(arm, 0.277, 0, smoke=True)
    ep = _load_json(rp)["episodes"][0]
    sel = list(base64.b64decode(ep["sel_timeline_b64"]))
    nat = int(natural_vb(torch.tensor([0.277])).clamp(0, 1)[0])
    assert len(sel) == ep["steps_done"] == 300
    if arm == "fbkt":
        assert set(sel) == {nat} and ep["sel_switch_steps"] == [], \
            f"G0 fbkt violated: {set(sel)} != {{{nat}}}"
    else:
        bad = [t for t in ep["sel_switch_steps"] if t % 25 != 0]
        assert not bad and set(sel) <= {0, 1}, f"G0 lsel violated: {bad[:3]}"
    print(f"[g0-smoke] {arm}: timeline OK (natural={nat}, "
          f"switches={ep['sel_switch_steps']})", flush=True)


def _save_bundle(summary: dict | None) -> Path:
    """增量原子打包（主线程独占调用；tmp→fsync→rename+目录 fsync）。"""
    import hashlib

    payload = {
        "artifact": "to42-artifacts/v1",
        "partial": summary is None,
        "meta": {
            "grid7": list(GRID7), "mid_band": list(MID_BAND),
            "base_args": BASE_ARGS,
            "vae": VAE, "ref_npz": REF, "decoder": DECODER, "router": ROUTER,
            "concurrency": {
                "profile": "v3 dynamic pipeline",
                "train_cap": TRAIN_CAP, "eval_cap": EVAL_CAP,
                "total_cap": TOTAL_CAP,
                "vram_gate_gb": {"train": TRAIN_VRAM_GB, "eval": EVAL_VRAM_GB},
                "train_pairing": [["to42r1-lsel-s0", "to42r1-fbkt-s0"],
                                  ["to42r1-lsel-s1", "to42r1-fbkt-s1"]],
                "eval_order": "per-(arm,seed) mid-band-first, pipelined",
                "smoke": "random-init ckpt（wiring 检查与策略无关，零训练）",
                "authorized": "owner 09-03 深夜 worker-pool + 09-04 「还不够极限」；"
                              "TO42_PLAN §9 执行身份修订 v3",
            },
        },
    }
    for name, p in (("ckpt_selection", RUNS_ROOT / "ckpt_selection.json"),
                    ("ckpt_selection_partial", RUNS_ROOT / "ckpt_selection_partial.json")):
        if p.exists():
            payload[name] = json.loads(p.read_text("utf-8"))
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
    os.replace(tmp, out)
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


# ----------------------------------------------------------------- 阶段
def stage_selftest() -> None:
    _scheduler([{"tag": "g0-selftest",
                 "cmd": [sys.executable, "-m", "apt_g1.isaac.to42_selftest"],
                 "kind": "util", "vram_gb": 0.5, "ready": None,
                 "on_done": None, "timeout": 600}])


def _random_init_ckpt() -> Path:
    """随机初始化 policy ckpt（G0 wiring 检查与策略无关——fbkt 时间线 = 自然
    bin 是 env 状态机性质，lsel 切换 ⊆ 边界同；省去两段冒烟训练）。"""
    from apt_g1.isaac.ppo_core import AptPPOPolicy

    out = RUNS_ROOT / "smoke-init"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "policy_final.pt"
    if not p.exists():
        pol = AptPPOPolicy(obs_dim=91 + 14 + 12 + 2, aux_dim=12, gate_k=2,
                           use_phase=False, latent_dim=16)
        torch.save(pol.state_dict(), p)
    return p


def stage_smoke() -> None:
    ckpt = _random_init_ckpt()
    jobs = []
    for i, arm in enumerate(ARMS):
        jobs.append({
            "tag": f"smoke-eval-{arm}",
            "cmd": [sys.executable, "-m", "apt_g1.rung1.to42_eval",
                    "--mode", "execute", "--arm", arm, "--v", "0.277",
                    "--train-seed", "0", "--ckpt", str(ckpt),
                    "--steps", "300", "--eval-seeds", "0", "--smoke",
                    "--env-tag", "cloud", "--worker-tag", f"w{i}"],
            "kind": "eval", "vram_gb": EVAL_VRAM_GB, "ready": None,
            "on_done": (lambda pending, arm=arm: _g0_smoke_assert(arm)),
            "timeout": 1800})
    _scheduler(jobs)


def stage_train_and_eval(pending: list) -> None:
    """流水线主体：训练 4 runs（2 并发，s0 对先发）+ 每臂完成 → 增量机械选择
    → on_done 向共享队列动态注入该臂 7 个评测格（mid-band 优先）。"""
    def make_on_done(arm: str, seed: int, pending: list):
        run = f"to42r1-{arm}-s{seed}"

        def _done(_pending: list):
            _require_train_outputs(RUNS_ROOT / run)
            entry = _select_one(arm, seed)
            ckpt = entry["ckpt_file"]
            vs = list(MID_BAND) + [v for v in GRID7 if v not in MID_BAND]
            for i, v in enumerate(vs):
                def on_done(arm=arm, seed=seed, v=v):
                    if not _receipt_path(arm, v, seed).exists():
                        raise SystemExit(
                            f"receipt 缺失 {arm} v={v} s={seed}"
                            "（rc 被 Isaac atexit 吞掉）")
                    _save_bundle(None)

                _pending.append({
                    "tag": f"eval-{arm}-v{v:.3f}-s{seed}",
                    "cmd": [sys.executable, "-m", "apt_g1.rung1.to42_eval",
                            "--mode", "execute", "--arm", arm,
                            "--v", f"{v:.3f}", "--train-seed", str(seed),
                            "--ckpt", ckpt,
                            "--steps", "3000", "--eval-seeds", "0,1,2",
                            "--out-dir", str(EVAL_DIR), "--env-tag", "cloud",
                            "--worker-tag", f"w{seed}{i % 3}"],
                    "kind": "eval", "vram_gb": EVAL_VRAM_GB, "ready": None,
                    "on_done": on_done, "timeout": 3600})
            _save_bundle(None)

        return _done

    jobs = []
    for seed in SEEDS:  # s0 对先发（第一批评测来自 s0）
        for arm in ARMS:
            run = f"to42r1-{arm}-s{seed}"
            jobs.append({
                "tag": f"train-{arm}-s{seed}",
                "cmd": [sys.executable, "-m", "apt_g1.isaac.train_apt_isaac",
                        *BASE_ARGS, "--iters", str(ITERS), "--seed", str(seed),
                        "--to42-sel", arm, "--out", f"output/to42/{run}"],
                "kind": "train", "vram_gb": TRAIN_VRAM_GB, "ready": None,
                "on_done": make_on_done(arm, seed, pending),
                "timeout": 6 * 3600})
    _scheduler(jobs)


def stage_select() -> None:
    """终版 formal manifest（to42_select 全量重算；与增量条目同规则同结果）。"""
    _scheduler([{"tag": "select",
                 "cmd": [sys.executable, "-m", "apt_g1.rung1.to42_select",
                         "--runs-root", str(RUNS_ROOT),
                         "--out", str(RUNS_ROOT / "ckpt_selection.json")],
                 "kind": "util", "vram_gb": 0.5, "ready": None,
                 "on_done": lambda: _save_bundle(None), "timeout": 600}])


def stage_check() -> None:
    _scheduler([{"tag": "checker",
                 "cmd": [sys.executable, "-m", "apt_g1.rung1.to42_checker",
                         "--eval-dir", str(EVAL_DIR),
                         "--selection-manifest",
                         str(RUNS_ROOT / "ckpt_selection.json"),
                         "--out",
                         str(_REPO / "apt_g1/outputs/to42/to42_eval_audit.json")],
                 "kind": "util", "vram_gb": 0.5, "ready": None,
                 "on_done": None, "timeout": 1200}])


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
                    "completed": all(e["completed"] for e in eps)}
    contrasts = {f"{s}|{v:.3f}": round(
        table[f"lsel|{s}|{v:.3f}"]["err60s"]
        - table[f"fbkt|{s}|{v:.3f}"]["err60s"], 4)
        for s in SEEDS for v in GRID7}
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
              else ["selftest", "smoke", "train", "select", "check",
                    "report", "bundle"])
    _head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=str(_REPO), capture_output=True, text=True)
    commit = _head.stdout.strip() if _head.returncode == 0 else "unknown"
    print(f"[wave-env] exe={sys.executable} cwd={os.getcwd()} "
          f"PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}", flush=True)
    print(f"=== TO42 wave v3 (dynamic pipeline: train {TRAIN_CAP}x, "
          f"eval {EVAL_CAP}x VRAM-gated {EVAL_VRAM_GB}G, per-arm pipelined) "
          f"start {time.strftime('%F %T')} commit={commit} stages={stages} ===",
          flush=True)
    t0 = time.monotonic()
    for st in stages:
        if st == "selftest":
            stage_selftest()
        elif st == "smoke":
            stage_smoke()
        elif st in ("train", "eval"):
            if not globals().get("_PIPELINE_DONE"):
                globals()["_PIPELINE_DONE"] = True
                stage_train_and_eval([])  # train+eval 流水线一体（动态注入）
        elif st == "select":
            stage_select()
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
