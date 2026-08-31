import json, argparse
from pathlib import Path
import numpy as np

DELTA = 0.03
T = 60.0

def a_sections(d):
    a = d.get("A_walk60", {})
    if "aux" in a:
        a = a["aux"]
    return {k: v for k, v in a.items() if k.startswith("seed")}

def load(path, arm, ck, ctag):
    for p in sorted(path.glob(f"{arm}_eval_{ck}*_{ctag}.json")):
        return json.loads(p.read_text())
    return None

def row_metrics(r, cmd):
    # 路径效率(直线度) = 60s 净位移 / (实测平均速度模长·60s)；直行≈1.0、绕圈<0.5
    eff = r["disp"] / (r["v_speed"] * T) if r["v_speed"] > 1e-6 else 0.0
    return dict(vx=r["vx"], h=r["h_min"], disp=r["disp"], done=r["completed"],
                eff=eff, err=abs(r["vx"] - cmd))

def run_arm(path, arm, ck, cmds):
    res = {}
    for cmd in cmds:
        ctag = f"a{round(cmd*1000):03d}"
        d = load(path, arm, ck, ctag)
        if d is None:
            res[cmd] = None
            continue
        rows = [row_metrics(v, cmd) for v in a_sections(d).values()]
        res[cmd] = rows
    return res

def summarize(arm_metrics, cmd, floor_ok):
    rows = arm_metrics.get(cmd)
    if not rows:
        return f"cmd {cmd}: 缺文件"
    errs = [r["err"] for r in rows if r["done"] and r["h"] >= 0.6 and r["disp"] > 0.5]
    effs = [r["eff"] for r in rows]
    det = "; ".join(f"vx={r['vx']:.2f} h={r['h']:.2f} d={r['disp']:.1f} e={r['eff']:.2f} done={r['done']}"
                    for r in rows if r["done"] and r["h"] >= 0.6 and r["disp"] > 0.5)
    return (f"cmd {cmd}: err={np.mean(errs) if errs else float('nan')}±{np.std(errs) if errs else 0:.3f}"
            f"  eff={np.mean(effs):.2f}±{np.std(effs):.2f}  | {det}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/cvgluser/ros2_data/apt_g1/outputs")
    ap.add_argument("--arms", default="to40c-ctrl,to40c-t10,to40c-t05")
    ap.add_argument("--cmds", default="0.2,0.25,0.277,0.30,0.5")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--full-arm-ckpt", default="")
    ap.add_argument("--delta", type=float, default=DELTA)
    args = ap.parse_args()
    path = Path(args.out_dir)
    cmds = [float(c) for c in args.cmds.split(",")]
    arms = args.arms.split(",")
    # floor + primary metrics
    floor_ok = {a: True for a in arms}
    metric = {a: {} for a in arms}
    for a in arms:
        m = run_arm(path, a, args.ckpt, cmds)
        metric[a] = m
        print(f"\n== {a} @ {args.ckpt} ==")
        for cmd in cmds:
            print("  " + summarize(m, cmd, floor_ok[a]))
            for r in (m.get(cmd) or []):
                if not (r["done"] and r["h"] >= 0.6 and r["disp"] > 0.5):
                    floor_ok[a] = False

    print("\n== floor ==", {a: floor_ok[a] for a in arms})

    # paired diff ctrl vs t10 / t05
    ctrl = args.arms.split(",")[0]
    for other in arms[1:]:
        diffs = []
        for cmd in cmds:
            ca = metric[ctrl].get(cmd); oa = metric[other].get(cmd)
            if not ca or not oa:
                continue
            n = min(len(ca), len(oa))
            for i in range(n):
                if ca[i]["done"] and oa[i]["done"]:
                    diffs.append(ca[i]["err"] - oa[i]["err"])
        if diffs:
            md = float(np.mean(diffs))
            # 分层互斥判定（评审修正，见 TO40C_PLAN §9）：未决窗 (0.02,0.04] 与
            # 三分支互斥；主指标须逐 cmd 达标，合并均值只作报告。
            print(f"\n配对差分 ctrl-{other} (n={len(diffs)}): {md:+.4f}")
            # per-cmd pass: each cmd diff must be <= +delta_hard (0.04) to count usable
            delta_hard = 0.04
            window_lo = 0.02
            per_cmd = []
            for cmd in cmds:
                ca = metric[ctrl].get(cmd); oa = metric[other].get(cmd)
                if not ca or not oa: continue
                n = min(len(ca), len(oa)); cd = [ca[i]["err"] - oa[i]["err"] for i in range(n) if ca[i]["done"] and oa[i]["done"]]
                if cd: per_cmd.append((cmd, float(np.mean(cd))))
            print("  per-cmd diff:", "  ".join(f"{c}:{d:+.3f}" for c,d in per_cmd))
            if window_lo < abs(md) <= delta_hard:
                layer = "未决窗：补第二训练 seed（不进三分支）"
            elif abs(md) <= window_lo:
                layer = "分支①：通道可用（等效或正向）"
            elif md < -delta_hard:
                layer = "通道正向（明显优）"
            elif md > delta_hard:
                layer = "分支③：通道毁基座（明显劣）"
            else:
                layer = "未知"
            print(f"  -> 分层判定 窗({window_lo},{delta_hard}): {layer}")
        else:
            print(f"\n配对差分 ctrl-{other}: 无有效对")

    # 2x2 cross-injection (eval-only tau on/off at cmd 0.277)
    print("\n== 2x2 cross-injection @0.277 ==")
    for a in arms:
        for x in ("off", "on"):
            p = sorted(path.glob(f"{a}_eval_{args.ckpt}_x{x}_a0277.json"))
            if not p:
                print(f"  {a} x{x}: 缺文件"); continue
            d = json.loads(p[0].read_text())
            rows = [row_metrics(v, 0.277) for v in a_sections(d).values()]
            print(f"  {a} x{x}: " + "; ".join(f"vx={r['vx']:.2f} e={r['eff']:.2f} done={r['done']}" for r in rows))

if __name__ == "__main__":
    main()