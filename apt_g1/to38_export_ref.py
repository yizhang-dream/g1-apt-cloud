"""TO38: TO36 world 解 -> RL 注入用紧凑参考表（to38_ref.npz）。

把 to36_leg_to_drake.py `world` 产出的世界系 81 样本/相（to36_world_knots.npz）
压成单张 LUT（两相拼接，时间归一化到 ψ∈[0,2π) 一个 stride），供
apt_flat_env.py 的 to_ref 注入直接查表：

  q_ref6    (M,6)  矢状 6 关节参考，SONIC/MuJoCo 序 [Lhip,Lknee,Lankle,
                  Rhip,Rknee,Rankle]，B 门符号表已乘（未减 default——
                  env 侧统一减，口径与 obs 的 jpos_rel 一致）
  tau_ref6  (M,6)  同序 τ 参考（v/v̇/τ 同映射；v1 只存不注入，备归因分析）
  pitch      (M,)   骨盆俯仰
  z          (M,)   骨盆高
  heel_rel   (M,2)  摆动脚 heel (x−骨盆x, z)——落点信号（capture 线索）
  T, v_avg          stride 周期 / 平均速度（env 时钟率与 gate 用）

符号表来自 B 门 verify FK 搜索（[-1,-1,-1,1,1,-1]，q6 相序
[Lankle,Lknee,Lhip,Rhip,Rknee,Rankle] -> MJCF），URDF 系 Isaac 同用。

用法（lab-ts，任意有 numpy 的 python）：
  python3 apt_g1/to38_export_ref.py \
      --world apt_g1/outputs/to36_world_knots.npz \
      --out apt_g1/outputs/to38_ref.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np

# B 门符号（verify 实测，FK 残差 4.3e-3）。注意 world/gait npz 的 q6 列序是
# 按角色排的（_phase_order）：[支撑 ankle, 支撑 knee, 支撑 hip, 摆动 hip,
# 摆动 knee, 摆动 ankle]，且 sign 也是按角色（支撑链 [-1,-1,-1]、摆动链
# hip/knee [+1,+1]、摆动 ankle [-1]）——同一物理关节在两相中角色互换，
# 符号随角色变。下表把两相分别映射到 SONIC 矢状序
# [Lhip,Lknee,Lankle,Rhip,Rknee,Rankle]。
# 相 l（左支撑）：Lhip=-q[2],Lknee=-q[1],Lankle=-q[0],Rhip=+q[3],Rknee=+q[4],Rankle=-q[5]
# 相 r（右支撑）：Lhip=+q[3],Lknee=+q[4],Lankle=-q[5],Rhip=-q[2],Rknee=-q[1],Rankle=-q[0]
PHASE_PERM = {
    "l": ([2, 1, 0, 3, 4, 5], [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0]),
    "r": ([3, 4, 5, 2, 1, 0], [1.0, 1.0, -1.0, -1.0, -1.0, -1.0]),
}


def resample_phase(w: np.lib.npyio.NpzFile, ph: str, m: int) -> dict:
    """把一相的 81 样本按相位时间均匀重采样到 m 点。"""
    ts = w[f"t_{ph}"]
    grid = np.linspace(0.0, float(ts[-1]), m)
    out = {}
    for k in ("q6", "u6", "pitch"):
        arr = np.atleast_2d(w[f"{k}_{ph}"]).T if w[f"{k}_{ph}"].ndim == 1 else w[f"{k}_{ph}"]
        out[k] = np.stack([np.interp(grid, ts, arr[:, j]) for j in range(arr.shape[1])], 1)
    pelvis = w[f"pxz_{ph}"]  # (81,2) 骨盆 (x,z)
    swing = "r" if ph == "l" else "l"  # l 相 = 左支撑，摆动脚 = 右
    heel = w[f"heel_{swing}"]  # (81,3) 世界系
    out["z"] = np.interp(grid, ts, pelvis[:, 1])
    out["heel_x"] = np.interp(grid, ts, heel[:, 0]) - np.interp(grid, ts, pelvis[:, 0])
    out["heel_z"] = np.interp(grid, ts, heel[:, 2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="apt_g1/outputs/to36_world_knots.npz")
    ap.add_argument("--out", default="apt_g1/outputs/to38_ref.npz")
    ap.add_argument("--m-per-phase", type=int, default=60)
    args = ap.parse_args()

    w = np.load(args.world, allow_pickle=True)
    Tl, Tr = float(w["t_l"][-1]), float(w["t_r"][-1])
    rows = []
    for ph, m in (("l", args.m_per_phase), ("r", args.m_per_phase)):
        r = resample_phase(w, ph, m)
        n = r["q6"].shape[0]
        perm, sgn = PHASE_PERM[ph]
        q_sonic = r["q6"][:, perm] * np.array(sgn)
        tau_sonic = r["u6"][:, perm] * np.array(sgn)
        rows.append(
            dict(
                q=q_sonic,
                tau=tau_sonic,
                pitch=r["pitch"][:, 0],
                z=r["z"],
                heel=np.stack([r["heel_x"], r["heel_z"]], 1),
                n=n,
            )
        )
    cat = lambda k: np.concatenate([r[k] for r in rows], 0)
    m = sum(r["n"] for r in rows)
    q = cat("q")
    # sanity（rubric 要求）：周期解首尾应近似闭合（左支撑起点 ≈ 右支撑终点）
    wrap_gap = float(np.abs(q[0] - q[-1]).max())
    scal_wrap = float(
        max(
            abs(cat("pitch")[0] - cat("pitch")[-1]),
            abs(cat("z")[0] - cat("z")[-1]),
            np.abs(cat("heel")[0] - cat("heel")[-1]).max(),
        )
    )
    assert q.shape == (m, 6) and np.isfinite(q).all(), "LUT 形状/数值异常"
    np.savez(
        args.out,
        q_ref6=q.astype(np.float32),
        tau_ref6=cat("tau").astype(np.float32),
        pitch=cat("pitch").astype(np.float32),
        z=cat("z").astype(np.float32),
        heel_rel=cat("heel").astype(np.float32),
        T=np.float64(Tl + Tr),
        T_left=np.float64(Tl),
        T_right=np.float64(Tr),
        v_avg=np.float64(float(w["v_avg"])),
        meta=np.array(
            json.dumps(
                dict(
                    world_src=args.world,
                    m_per_phase=args.m_per_phase,
                    wrap_gap_q=wrap_gap,
                    wrap_gap_scal=scal_wrap,
                    q_min=q.min(0).tolist(),
                    q_max=q.max(0).tolist(),
                )
            )
        ),
    )
    print(
        f"[to38-export] {args.out}: M={m} T={Tl + Tr:.3f}s v_avg={float(w['v_avg']):.3f}\n"
        f"  wrap_gap q={wrap_gap:.4f} rad scal={scal_wrap:.4f}（周期闭合检查）\n"
        f"  q_ref6 SONIC 序行程 min={q.min(0).round(3)} max={q.max(0).round(3)}"
    )
    if wrap_gap > 0.35 or scal_wrap > 0.25:
        raise SystemExit("[to38-export] 首尾闭合缺口过大，检查 world npz 是否单周期")


if __name__ == "__main__":
    main()
