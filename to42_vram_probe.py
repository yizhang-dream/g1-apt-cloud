"""TO42 显存/速度探针（L20 48G，2048 envs 操作点；gm-run 独立任务入口）。

角色（SCRIPT_MAP 登记）：**DEV**（发全链前的 20min 保险；TO42_PLAN §9 修订
v4 的 entry gate——2048 envs 在 L20 上的显存足迹未实测，一次全链 OOM 的代价
远大于一次探针）。做法 = 真实训练配方跑 3 iters（fbkt 臂、seed 0），后台
线程每 2s 采样 nvidia-smi used，输出峰值显存 + 实测 dt（校准 500it 全链
墙钟估算）。判据（机械）：peak_used ≤ 46000 MB（48G − 2G cushion）且
rc=0 → PASS（打 TO42_VRAM_PROBE JSON 行）。探针产物无科学内容。
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
os.chdir(str(_REPO))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("PRIVACY_CONSENT", "Y")

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--num-envs", type=int,
                 default=int(os.environ.get("TO42_PROBE_ENVS", "2048")))
_ap.add_argument("--cushion-mb", type=int, default=2000)
_ap.add_argument("--limit-mb", type=int, default=48000)
_args = _ap.parse_args()
NUM_ENVS = _args.num_envs
LIMIT_MB = _args.limit_mb - _args.cushion_mb

BASE_ARGS = [
    "--latent-mode",
    "--latent-vae-path", "apt_g1/outputs/token_vae_e39/vae.pt",
    "--latent-speed-bins", "--latent-dir-bins",
    "--latent-kl-prior", "zero",
    "--progress-scale", "1.0", "--heading-scale", "0.4",
    "--to-ref", "--to-ref-npz", "apt_g1/outputs/sync/to38_ref.npz",
    "--to-ref-obs-zero", "--to-ref-w", "0",
    "--router-model-dir", "apt_g1/outputs/distill_final",
    "--decoder-path", "gear_sonic_deploy/policy/release/model_decoder.onnx",
    "--num-envs", str(NUM_ENVS),
    "--ppo-minibatch", "4096",
]


def main() -> int:
    samples: list = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(2.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.monotonic()
    rc = subprocess.call(
        [sys.executable, "-m", "apt_g1.isaac.train_apt_isaac", *BASE_ARGS,
         "--iters", "3", "--seed", "0", "--to42-sel", "fbkt",
         "--out", "output/to42/vram-probe"],
        cwd=str(_REPO), env=dict(os.environ))
    stop.set()
    th.join(timeout=5)
    wall = time.monotonic() - t0
    ckpt_ok = (Path(_REPO / "output/to42/vram-probe/policy_final.pt").exists())
    peak = max(samples) if samples else None
    verdict = "PASS" if (rc == 0 and ckpt_ok and peak is not None
                         and peak <= LIMIT_MB) else "FAIL"
    result = {
        "probe": "to42-vram/l20-2048envs/v1",
        "rc": rc, "ckpt_ok": ckpt_ok,
        "peak_used_mb": peak, "limit_mb": LIMIT_MB,
        "wall_min": round(wall / 60, 1),
        "n_samples": len(samples),
        "verdict": verdict,
        "note": "dt 校准见训练日志 [i/3] 行；PASS = 2048envs 可容 L20 48G",
    }
    print("TO42_VRAM_PROBE:" + json.dumps(result), flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
