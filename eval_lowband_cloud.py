"""TO39 低速带评测驱动（云任务版，2026-08-31）。

一次任务内顺序调用 5 个 cmd 点的 eval_apt_isaac.py（子进程复用同一套
实战评测代码，零逻辑分叉），JSON 产物收集到 output/lowband/ 并镜像到
/personal（若存在）。

用法（startScript）：
  gm-run eval_lowband_cloud.py --arm to38a --ckpt apt_g1/outputs/to38_ckpt/to38a_best_it150.pt \
      --extra "--to-ref-obs-zero"（b/c 臂差异走 --extra）

cmd 点固定 {0.08, 0.12, 0.15, 0.2, 0.277} × seed {0,1,2}（A 60s）。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CMDS = ["0.08", "0.12", "0.15", "0.2", "0.277"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--extra", default="", help="额外 eval 参数（引号内空格分隔）")
    cli, _ = ap.parse_known_args()

    out_dir = ROOT / "output" / "lowband"
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = cli.extra.split() if cli.extra else []

    summary = {}
    for cmd in CMDS:
        tag = f"a{cmd.replace('.', '')}"
        out_json = out_dir / f"{cli.arm}_eval_lowband_{tag}.json"
        call = [
            sys.executable,
            str(ROOT / "apt_g1" / "isaac" / "eval_apt_isaac.py"),
            "--checkpoint", cli.ckpt,
            "--tests", "A",
            "--a-cmd-vx", cmd,
            "--latent-mode",
            "--latent-vae-path", "apt_g1/outputs/token_vae_e39/vae.pt",
            "--latent-speed-bins", "--latent-dir-bins",
            "--heading-scale", "0.4",
            "--router-model-dir", "apt_g1/outputs/distill_final",
            "--decoder-path", "gear_sonic_deploy/policy/release/model_decoder.onnx",
            "--to-ref", "--to-ref-npz", "apt_g1/outputs/to38_ref.npz",
            "--out", str(out_json),
        ] + extra
        print(f"[lowband] {cli.arm} cmd={cmd} -> {out_json.name}", flush=True)
        subprocess.run(call, cwd=str(ROOT), check=False)
        if out_json.exists():
            summary[f"cmd_{cmd}"] = json.loads(out_json.read_text())["A_walk60"]
        else:
            summary[f"cmd_{cmd}"] = "MISSING"

    (out_dir / f"{cli.arm}_lowband_summary.json").write_text(json.dumps(summary, indent=1))
    if Path("/personal").is_dir():
        dst = Path("/personal") / "flux_runs" / f"{cli.arm}_lowband"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_dir, dst, dirs_exist_ok=True)
        print("mirrored to", dst, flush=True)
    print("LOWBAND DONE", cli.arm, flush=True)


if __name__ == "__main__":
    main()
