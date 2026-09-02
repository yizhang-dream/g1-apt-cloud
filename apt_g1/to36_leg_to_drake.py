#!/usr/bin/env python
"""TO36：G1 腿级 TO 第一步 —— Drake dircol 周期步态 NMP（真 G1 MJCF 降维平面实例）。

设计定稿：refine-logs/LEG_LEVEL_TO_PLAN.md（2026-08-29 grill-me 五项决策）。
上游调研：docs/g1_fullbody_trajectory_optimization_roadmap.md。

子命令（对应执行计划 D1–D7）：
  load    D1 冒烟：MJCF → MultibodyPlant，矢状/非矢状盘点，autodiff 验证
  solve   D2–3：单周期 dircol（左右支撑两相位接触约束），A 门初判 + 解内 tau
          〔v1 架构已判死留作负结果对照，现行 solver 是 to36_hybrid_dircol.py〕
  world   D5a Stage A（.venv_drake）：hybrid foot 解 → 世界系 81 样本/相
          （骨盆 URDF 原点位姿/速度/解析加速度 + 双踝/膝/脚 FK 目标 + 相位
          q/v/u/v̇，错列 knot+中点），供 verify（B 门）与 closedloop（C 门）
  verify  D5a Stage B（.venv_mjlab）：B 门双验证 —— 43-DOF MuJoCo 逆动力学
          复核（基座行消去法解跟/尖接触力 λ，见函数 docstring）对同一轨迹
          的力矩量级（hip/knee 峰值 100–300 N·m 口径 + 与解内 τ 一致性）
  closedloop  D5–7（.venv_mjlab）：C 门闭环 —— 矢状 τ 前馈投影 + 全身 PD，
          防作弊指标（h_min ≥0.6 / 世界系 disp >0.5 / 存活 ≥6s）

建模口径（LEG_LEVEL_TO_PLAN §3）：
  - 接触 = 解析约束（自写 autodiff helper，不赌 Drake 约束类签名版本）：
    支撑相足跟/足尖两点 pz=0 且点速度 (vx,vz)=0；摆动脚 pz >= clearance。
  - 摩擦锥第一步后处理（verify 用 mj_inverse 复核量级），不在 NLP 里加力
    变量；后处理不达标再升级显式锥约束。
  - impact 第一步弱连续（相位切换点 q/dq 连续由 dircol 状态连续性天然保证）；
    角动量守恒冲击约束留作后续增强（roadmap §4.1.1 原推荐）。
  - 平面化：floating base 保留。位置约束 p_y=0；姿态用旋转矩阵元素
    R[0,1]=R[2,1]=0 钉死 roll/yaw、放开 pitch（等价纯俯仰旋转，免欧拉角
    autodiff 三角函数）；速度约束 v_y=0、w_x=0、w_z=0。
  - 非矢状关节 q/dq 逐 knot 钉 0（不用 Joint.lock：lock 只作用于仿真约束，
    不进 dircol NLP）。力矩限位取 MJCF actuator 真实 effort limit；
    非矢状 actuator 输入钉 0。

运行环境：服务器 .venv_drake（见 to36_setup_drake_env.sh）；本机无 venv 不运行。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 2026-08-29 第二轮 grill-me 决策 C 案：共享函数抽至 to36_common（原样迁移，
# D1 服务器行为基线不变）；本文件保留 CLI 与 v1 solve 架构（已判死，留作
# D2 负结果对照，见 tracker/TO.md）。
from to36_common import (DEFAULT_MODEL, INF, KnotKinematics, OUT_NPZ,
                         build_plant, is_sagittal, resolve_model)


# ------------------------------------------------------------------ D1: load
# resolve_model / _urdf_efforts / build_plant 已抽至 to36_common（原样迁移）。


def do_load(args):
    build_plant(args.model, verbose=True)
    print("[load] D1 冒烟通过（MJCF 载入 + 盘点 + autodiff）。"
          "若上方矢状/脚 body 盘点为空，先核对 MJCF 命名再进 solve。")


# ---------------------------------------------------------------- D2–3: solve
# KnotKinematics / _pick_base / _pick_foot 已抽至 to36_common（原样迁移）。


def do_solve(args):
    from pydrake.all import DirectCollocation
    from pydrake.math import eq
    from pydrake.multibody.tree import JointActuatorIndex
    from pydrake.solvers import IpoptSolver, SnoptSolver

    plant, _ = build_plant(args.model)
    nq, nv = plant.num_positions(), plant.num_velocities()

    actuators = []
    for i in range(plant.num_actuators()):
        a = plant.get_joint_actuator(JointActuatorIndex(i))
        actuators.append((i, a.joint().name(), float(a.effort_limit()),
                          is_sagittal(a.joint().name())))
    print(f"[solve] actuators={len(actuators)}（矢状 "
          f"{sum(1 for x in actuators if x[3])}）")

    kin = KnotKinematics(plant, args.foot_heel, args.foot_toe)
    print(f"[solve] base={kin.base_name} feet={kin.foot_names}")

    N, half = args.knots, args.knots // 2
    pnames = plant.GetPositionNames()
    fwd_idx = next(i for i, nm in enumerate(pnames)
                   if "planar" in nm.lower() and "x" in nm.lower()
                   and "y" not in nm.lower())
    print(f"[solve] 前向坐标 index={fwd_idx} ({pnames[fwd_idx]})")

    q_guess = np.zeros(nq)
    q_guess[1] = 0.7569  # 直腿站立：踝原点恰好触地（FK 标定值）
    qlo = np.clip(plant.GetPositionLowerLimits(), -INF, INF)
    qhi = np.clip(plant.GetPositionUpperLimits(), -INF, INF)
    # 注意：q[1] 才是 planar 高度（q[2] 是俯仰角 qz）
    qlo[1] = max(qlo[1], args.z_min)
    qhi[1] = min(qhi[1], args.z_max)
    vhi = np.clip(plant.GetVelocityUpperLimits(), -INF, INF)
    vlo = -vhi

    solver = SnoptSolver() if args.solver == "snopt" else IpoptSolver()
    if not solver.available():
        solver = IpoptSolver()
    sid = solver.solver_id()

    # 同伦阶梯：(摆动离地峰值, 步长上界)。原地踏步 -> 逐级步行
    stages = [(0.0, 0.0), (0.01, 0.12), (0.02, 0.25),
              (args.clearance, args.dx_target)]

    def knot_bounds(stance, clear):
        lo = np.full(KnotKinematics.OUT_DIM, -INF)
        hi = np.full(KnotKinematics.OUT_DIM, INF)
        stance_idx = (0, 1) if stance == "left" else (2, 3)
        swing_idx = (2, 3) if stance == "left" else (0, 1)
        for i in stance_idx:
            lo[i] = hi[i] = 0.0          # 支撑点 pz = 0
        for i in swing_idx:
            lo[i] = clear                # 摆动脚离地（正弦包络）
        lo[4:8] = hi[4:8] = 0.0          # 支撑点无滑移 (vx,vz)×2
        # 行 8..14（base 平面分量）在 PlanarJoint 模型下天然为 0，放开
        return lo, hi

    prev = None  # (X, U, dxv)
    for si, (clr, dx_hi) in enumerate(stages):
        context = plant.CreateDefaultContext()
        dircol = DirectCollocation(
            plant, context, num_time_samples=N,
            minimum_time_step=args.dt_min, maximum_time_step=args.dt_max,
            input_port_index=plant.get_actuation_input_port().get_index())
        prog = dircol.prog()

        lb_u = np.array([-x[2] if x[3] else 0.0 for x in actuators])
        ub_u = np.array([x[2] if x[3] else 0.0 for x in actuators])
        if not args.no_bbox:
            for k in range(N):
                prog.AddBoundingBoxConstraint(lb_u, ub_u, dircol.input(k))
            for k in range(N):
                x = dircol.state(k)
                prog.AddBoundingBoxConstraint(qlo, qhi, x[:nq])
                prog.AddBoundingBoxConstraint(vlo, vhi, x[nq:])

        for k in range(N):
            stance = "left" if k < half else "right"
            s_frac = (k % half) / max(half - 1, 1)
            clear = clr * np.sin(np.pi * s_frac)
            lo, hi = knot_bounds(stance, clear)
            prog.AddConstraint(
                lambda zv, ks=stance: kin(zv, ks), lo, hi, dircol.state(k))

        x0, xN = dircol.initial_state(), dircol.final_state()
        dx = prog.NewContinuousVariables(1, "step_length")[0]
        dvec = np.zeros(nq, dtype=object)
        dvec[fwd_idx] = dx
        prog.AddConstraint(eq(xN[:nq], x0[:nq] + dvec))
        prog.AddConstraint(eq(xN[nq:], x0[nq:]))
        prog.AddConstraint(xN[fwd_idx] - x0[fwd_idx] - dx / 2.0 == 0)
        prog.AddBoundingBoxConstraint(0.0, max(dx_hi, 1e-6), dx)

        for k in range(N):
            dircol.AddRunningCost(dircol.input(k).dot(dircol.input(k)) * 1e-2)
        dircol.AddRunningCost(-dx * 10.0)

        for k in range(N):
            xg = np.concatenate([q_guess, np.zeros(nv)])
            xg[fwd_idx] = dx_hi * k / (N - 1)
            prog.SetInitialGuess(dircol.state(k), xg)
            prog.SetInitialGuess(dircol.input(k), np.zeros(len(actuators)))
        prog.SetInitialGuess(dx, dx_hi)
        if prev is not None:
            X, U, dxv = prev
            for k in range(N):
                prog.SetInitialGuess(dircol.state(k), X[k])
                prog.SetInitialGuess(dircol.input(k), U[k])
            prog.SetInitialGuess(dx, dxv)

        prog.SetSolverOption(sid, "max_iter", args.max_iter)
        prog.SetSolverOption(sid, "mu_strategy", "adaptive")
        prog.SetSolverOption(sid, "print_level", 5)
        print(f"[solve] 同伦 {si + 1}/{len(stages)}: "
              f"clearance={clr:.3f} dx<={dx_hi:.2f}", flush=True)
        result = solver.Solve(prog)
        ok = result.is_success()
        print(f"[solve] 同伦 {si + 1}: success={ok} "
              f"cost={result.get_optimal_cost():.4f}", flush=True)
        if not ok:
            if prev is None:
                print("[solve] 首级（原地踏步）未收敛：查初值/模型。")
                sys.exit(2)
            print("[solve] 该级未收敛，回退上一级解。")
            break
        ts = [result.GetSolution(dircol.time_sample(k))
              for k in range(N - 1)]
        X = np.array([result.GetSolution(dircol.state(k)) for k in range(N)])
        U = np.array([result.GetSolution(dircol.input(k))
                      for k in range(N - 1)])
        dxv = float(result.GetSolution(dx))
        prev = (X, U, dxv, ts)

    X, U, dxv, ts = prev
    T = float(np.sum(ts))
    Q, V = X[:, :nq], X[:, nq:]
    v_avg = dxv / T
    tau_peak = {name: float(np.abs(U[:, idx]).max())
                for idx, name, _, sag in actuators if sag}
    gate_a = v_avg >= 0.2
    gate_b = all(100.0 <= v <= 300.0 for v in tau_peak.values())
    print(f"[solve] T={T:.3f}s 步长={dxv:.3f}m 平均步速={v_avg:.3f}m/s "
          f"A门={'PASS' if gate_a else 'FAIL'}")
    print(f"[solve] 解内峰值|tau|={ {k: round(v, 1) for k, v in tau_peak.items()} }"
          f" B门(解内口径)={'PASS' if gate_b else 'FAIL'}；verify 跑 mj_inverse 复核")

    Path(OUT_NPZ).parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, Q=Q, V=V, U=U, t_knots=np.array(ts), T=T,
             sagittal_joints=[n for _, n, _, sg in actuators if sg],
             tau_peak=json.dumps(tau_peak), gate_a=gate_a, gate_b=gate_b)
    json.dump({"T": T, "step": dxv, "v_avg": v_avg, "tau_peak": tau_peak,
               "gate_a": bool(gate_a), "gate_b_prelim": bool(gate_b),
               "solver": sid.name(), "knots": N, "homotopy_stages": len(stages)},
              open(OUT_NPZ.replace(".npz", "_diag.json"), "w"), indent=2)
    print(f"[solve] 已存 {OUT_NPZ}")


# --------------------------------------------- D5a Stage A: world（.venv_drake）

WORLD_NPZ = "apt_g1/outputs/to36_world_knots.npz"


def _pelvis_frame_fix(model, stance):
    """相位 pelvis 体（原点=stance 髋）→ URDF pelvis 原点的常值偏移（体系）。

    相位体零位姿态=世界单位阵；平面 9 DOF 模型零位=直立、连杆系与世界对齐
    （D1 验证），故零位下两个模型同一对点的世界坐标差即体系偏移。
    左右 stance 只差 y（±髋宽），矢状面取 x/z 一致。"""
    plant, _ = build_plant(model)
    ctx = plant.CreateDefaultContext()
    plant.SetPositions(ctx, np.zeros(plant.num_positions()))

    def bp(n):
        return plant.EvalBodyPoseInWorld(
            ctx, plant.GetBodyByName(n)).translation()

    off = bp("pelvis") - bp(f"{stance}_hip_pitch_link")
    return off  # 预期 ~(0, ±0.088, +0.103)，x/z 与 stance 无关


def _phase_samples(X, XM, U, N, T, nq):
    """knot 与 HS 中点错列为 81 个按 t 排序的样本（中点 u=HS 口径 (u_k+u_k1)/2）。"""
    h = T / N
    out = []
    for k in range(N + 1):
        out.append((k * h, X[k][:nq], X[k][nq:], U[k]))
    for k in range(N):
        out.append(((k + 0.5) * h, XM[k][:nq], XM[k][nq:],
                    0.5 * (U[k] + U[k + 1])))
    out.sort(key=lambda s: s[0])
    return out


def _ankle_roll_offset(model):
    """URDF ankle_roll_joint 原点 z（ankle_pitch 系）→ 踝-roll 点体系偏移。

    B 门 FK 校验的踝目标必须与 MJCF ankle_roll_link 原点严格同点（刚体上
    不同固定点在转动下 Δ 不同，0.0176 m 偏移 × 脚摆角会引入 ~5 mm 假残差）。"""
    import xml.etree.ElementTree as ET
    root = ET.parse(Path(resolve_model(model))).getroot()
    for j in root.findall("joint"):
        if (j.get("name") or "").endswith("_ankle_roll_joint"):
            o = j.find("origin")
            z = float((o.get("xyz") if o is not None else "0 0 0").split()[2])
            return np.array([0.0, 0.0, z])
    return np.array([0.0, 0.0, -0.0176])


def do_world(args):
    """Stage A：hybrid foot 相位解 → 世界系样本序列（Drake FK，精确量）。

    产出的 81 样本/相包含 verify/closedloop 所需的全部世界系参考：
      骨盆 URDF 原点位姿 (x,z,pitch)/速度 (vx,vz,ωy)/解析加速度（Jacobian·v̇
      + bias 项，无差分误差）；双踝-roll 点/膝(=shin 体原点)世界坐标、双脚
      俯仰（B 门 FK 交叉验证 + C 门参考）；支撑脚跟/尖世界坐标；
      相位坐标 q/v/u/v̇（B 门在 MuJoCo 侧定符号映射后逐项搬运）。
    """
    from to36_hybrid_dircol import _build_kits

    d = np.load(args.gait_npz, allow_pickle=True)
    if str(d["mode"]) != "foot":
        raise SystemExit(f"[world] 只接 foot 模式解，npz mode={d['mode']}")
    N = int(d["knots"])
    kits, _ = _build_kits(args.model, "foot", args.sole_drop)
    roll_off = _ankle_roll_offset(args.model)
    out = {"mode": "foot", "knots": N, "step": float(d["step"]),
           "v_avg": float(d["v_avg"]), "T": float(d["T"]),
           "roll_off_z": float(roll_off[2])}
    for ph in ("left", "right"):
        kit = kits[ph]
        off = _pelvis_frame_fix(args.model, ph)
        samples = _phase_samples(np.array(d[f"X_{ph}"]), np.array(d[f"XM_{ph}"]),
                                 np.array(d[f"U_{ph}"]), N,
                                 float(d[f"T_{ph}"]), kit.nq)
        plant, ctx = kit.plant_f, kit.ctx_f
        W = plant.world_frame()
        try:
            from pydrake.multibody.tree import \
                JacobianWrtVariable as JW
        except ImportError:
            from pydrake.multibody.tree import JW
        pelv_fr = plant.GetBodyByName("pelvis").body_frame()
        rows = {k: [] for k in
                ("t", "q6", "v6", "u6", "vdot6", "pxz", "pitch", "vel", "acc",
                 "ankle", "knee", "fpitch", "heel", "toe")}
        for t, q6, v6, u6 in samples:
            plant.SetPositions(ctx, q6)
            plant.SetVelocities(ctx, v6)
            vdot6 = np.asarray(
                kit.xdot(np.concatenate([q6, v6, u6])), dtype=float)[kit.nq:]
            Xb = plant.EvalBodyPoseInWorld(ctx, plant.GetBodyByName("pelvis"))
            Rb = np.asarray(Xb.rotation().matrix(), dtype=float)
            pb = np.asarray(Xb.translation(), dtype=float)
            # URDF pelvis 原点：相位体位姿 ⊕ 体系常偏移
            p_u = pb + Rb @ np.asarray(off, dtype=float)
            pitch = float(np.arctan2(-Rb[2, 0], Rb[0, 0]))
            # 点速度/加速度（解析：J·v̇+bias，无差分）
            try:
                Jp = plant.CalcJacobianTranslationalVelocity(
                    ctx, JW.kV, pelv_fr, off, W, W)
                bp_ = plant.CalcBiasTranslationalAcceleration(
                    ctx, JW.kV, pelv_fr, off, W, W)
                Jw = plant.CalcJacobianAngularVelocity(ctx, JW.kV, pelv_fr, W, W)
                # 本版无 CalcBiasAngularAcceleration → 取空间加度 bias 转动分量
                bw = plant.CalcBiasSpatialAcceleration(
                    ctx, JW.kV, pelv_fr, np.zeros(3), W, W).rotational()
            except TypeError:  # 旧版签名少 expressed-in 帧
                Jp = plant.CalcJacobianTranslationalVelocity(
                    ctx, JW.kV, pelv_fr, off, W)
                bp_ = plant.CalcBiasTranslationalAcceleration(
                    ctx, JW.kV, pelv_fr, off, W)
                Jw = plant.CalcJacobianAngularVelocity(ctx, JW.kV, pelv_fr, W)
                bw = plant.CalcBiasSpatialAcceleration(
                    ctx, JW.kV, pelv_fr, np.zeros(3), W).rotational()
            Jp, bp_ = np.asarray(Jp, float), np.asarray(bp_, float).ravel()
            Jw, bw = np.asarray(Jw, float), np.asarray(bw, float).ravel()
            # ↑ Drake bias 项返回 (3,1) 列向量，ravel 防 (3,)+(3,1) 广播成 (3,3)
            Vb = plant.EvalBodySpatialVelocityInWorld(
                ctx, plant.GetBodyByName("pelvis"))
            v_p = np.asarray(Vb.translational(), float) + np.cross(
                np.asarray(Vb.rotational(), float), Rb @ np.asarray(off, float))
            a_p = np.asarray(Jp @ vdot6 + bp_, float)
            aw = np.asarray(Jw @ vdot6 + bw, float)
            feet, knees, fp = [], [], []
            for side in ("left", "right"):
                Xf = plant.EvalBodyPoseInWorld(
                    ctx, plant.GetBodyByName(f"{side}_foot"))
                # 踝目标 = ankle-roll 点（foot 体原点即踝 pitch 点，再偏 roll z）
                p_ank = Xf.translation() + Xf.rotation().matrix() @ roll_off
                feet.extend([float(p_ank[0]), float(p_ank[2])])
                kn = plant.EvalBodyPoseInWorld(
                    ctx, plant.GetBodyByName(f"{side}_shin")).translation()
                knees.extend([float(kn[0]), float(kn[2])])
                Rf = np.asarray(Xf.rotation().matrix(), dtype=float)
                fp.append(float(np.arctan2(-Rf[2, 0], Rf[0, 0])))
            heel = kit.point_pos(q6, f"{ph}_heel")
            toe = kit.point_pos(q6, f"{ph}_toe")
            rows["t"].append(t); rows["q6"].append(q6); rows["v6"].append(v6)
            rows["u6"].append(u6); rows["vdot6"].append(vdot6)
            rows["pxz"].append(p_u[[0, 2]]); rows["pitch"].append(pitch)
            rows["vel"].append([float(v_p[0]), float(v_p[2]),
                                float(np.asarray(Vb.rotational())[1])])
            rows["acc"].append([float(a_p[0]), float(a_p[2]), float(aw[1])])
            rows["ankle"].append(feet); rows["knee"].append(knees)
            rows["fpitch"].append(fp)
            rows["heel"].append(np.asarray(
                kit.point_pos(q6, f"{ph}_heel"), dtype=float))
            rows["toe"].append(np.asarray(
                kit.point_pos(q6, f"{ph}_toe"), dtype=float))
        tag = ph[0]
        for k, v in rows.items():
            out[f"{k}_{tag}"] = np.asarray(v, dtype=float)
        print(f"[world] {ph}: 81 样本 pelvis_z[{out['pxz_' + tag][:, 1].min():.3f},"
              f"{out['pxz_' + tag][:, 1].max():.3f}] pitch["
              f"{out['pitch_' + tag].min():+.3f},{out['pitch_' + tag].max():+.3f}]"
              f" 髋偏移={np.round(off, 4)}")
    _world_out = getattr(args, "world_out", WORLD_NPZ)
    np.savez(_world_out, **out)
    print(f"[world] 已存 {_world_out}（B 门 verify / C 门 closedloop 共用）")


# -------------------------------------------- D5a Stage B: verify（.venv_mjlab）

SAG_JOINTS = tuple(f"{s}_{p}_joint" for s in ("left", "right")
                   for p in ("hip_pitch", "knee", "ankle_pitch"))
EFFORT = {"hip_pitch": 88.0, "knee": 139.0, "ankle_pitch": 50.0}


def _mj_setup(scene):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(Path(scene).expanduser()))
    data = mujoco.MjData(model)
    free = next(j for j in range(model.njnt)
                if model.jnt_type[j] == 0)  # mjJNT_FREE
    sag_adr, sag_dof = {}, {}
    for name in SAG_JOINTS:
        j = model.joint(name)
        sag_adr[name] = model.jnt_qposadr[j.id]
        sag_dof[name] = model.jnt_dofadr[j.id]
    return mujoco, model, data, free, sag_adr, sag_dof


def _phase_order(ph):
    """相位 pname 列序 → MJCF 矢状关节名。[stance ankle/knee/hip, other hip/knee/ankle]"""
    st = "left" if ph == "l" else "right"
    ot = "right" if ph == "l" else "left"
    return [f"{st}_ankle_pitch_joint", f"{st}_knee_joint", f"{st}_hip_pitch_joint",
            f"{ot}_hip_pitch_joint", f"{ot}_knee_joint", f"{ot}_ankle_pitch_joint"]


def _set_mj_state(mujoco, model, data, sag_adr, sag_dof, w, ph, i, sign,
                  with_vel=True):
    """按符号映射把相位样本 (q6,v6) 写入 43-DOF qpos/qvel（基座来自 Stage A FK）。

    相位坐标与 MJCF 坐标是同一组物理矢状关节的不同参数化，两模型零位相同
    （直腿）→ 映射为逐关节 ±1（v/v̇/τ 同映射，切空间线性）。基座角速度按
    MuJoCo 约定（free joint 线速度全局系、角速度体系）取 y 分量——本解
    roll=yaw≡0，两系 y 轴恒重合，无歧义。"""
    data.qpos[:] = 0.0
    data.qpos[0] = w[f"pxz_{ph}"][i][0]
    data.qpos[2] = w[f"pxz_{ph}"][i][1]
    pit = float(w[f"pitch_{ph}"][i])
    data.qpos[3:7] = [np.cos(pit / 2), 0.0, np.sin(pit / 2), 0.0]
    for pi, mjn in enumerate(_phase_order(ph)):
        data.qpos[sag_adr[mjn]] = sign[pi] * w[f"q6_{ph}"][i][pi]
    if with_vel:
        data.qvel[:] = 0.0
        data.qvel[0] = w[f"vel_{ph}"][i][0]
        data.qvel[2] = w[f"vel_{ph}"][i][1]
        data.qvel[5] = w[f"vel_{ph}"][i][2]
        for pi, mjn in enumerate(_phase_order(ph)):
            data.qvel[sag_dof[mjn]] = sign[pi] * w[f"v6_{ph}"][i][pi]


def _fk_residual(mujoco, model, data, sag_adr, w, sign):
    """符号组合的 FK 一致性残差：Δ(踝/膝 xz/脚pitch) 相对各自零位。

    踝目标 = ankle-roll 点（Stage A 已按 URDF roll 关节原点 z 偏移），与
    MJCF ankle_roll_link 原点严格同点；膝 = knee_link 原点 = 相位 shin 体
    原点。只比 x/z 分量（y 恒为身体横距 ±0.118，无信息）；脚俯仰按
    0.3 m 力臂折成米。残余 ~4 mm 为 URDF↔MJCF 常值几何差（膝/踝 z 同偏
    +4.3 mm，Δ 口径下应消去，此处保留作交叉检查）。
    支撑膝符号边界可辨（0.30 m 胫骨 × 2·0.345 rad ≈ 0.19 m）；支撑髋对
    全部 FK 目标不可见（只转骨盆，基座位姿来自 Stage A）——不在本函数
    搜索范围，由 P 映射结构定（见 do_verify）。"""
    worst = 0.0
    for ph in ("l", "r"):
        for i in (20, 41, 60):
            _set_mj_state(mujoco, model, data, sag_adr, {}, w, ph, i, sign,
                          with_vel=False)
            mujoco.mj_forward(model, data)
            for s_i, side in enumerate(("left", "right")):
                ank = w[f"ankle_{ph}"]
                d_t = np.array([ank[i][2 * s_i] - ank[0][2 * s_i],
                                ank[i][2 * s_i + 1] - ank[0][2 * s_i + 1]])
                b = model.body(f"{side}_ankle_roll_link")
                d_m = np.array([
                    data.xpos[b.id][0] - ank[0][2 * s_i],
                    data.xpos[b.id][2] - ank[0][2 * s_i + 1]])
                worst = max(worst, float(np.abs(d_t - d_m).max()))
                kne = w[f"knee_{ph}"]
                d_tk = np.array([kne[i][2 * s_i] - kne[0][2 * s_i],
                                 kne[i][2 * s_i + 1] - kne[0][2 * s_i + 1]])
                bk = model.body(f"{side}_knee_link")
                d_mk = np.array([
                    data.xpos[bk.id][0] - kne[0][2 * s_i],
                    data.xpos[bk.id][2] - kne[0][2 * s_i + 1]])
                worst = max(worst, float(np.abs(d_tk - d_mk).max()))
                fp = w[f"fpitch_{ph}"]
                R = data.xmat[b.id].reshape(3, 3)
                dp = (np.arctan2(-R[2, 0], R[0, 0])
                      - (fp[i][s_i] - fp[0][s_i]))
                dp = abs((dp + np.pi) % (2 * np.pi) - np.pi)
                worst = max(worst, 0.3 * float(dp))
    return worst


def do_verify(args):
    """B 门双验证（D5a Stage B）：43-DOF MuJoCo 逆动力学对同一轨迹复核。

    方法（基座行消去法，绕开两个已知坑）：
      - TO08：mj_inverse 对浮基+双腿驱动返回错误力矩 → 不用 mj_inverse；
      - foot_gait_id 的被动 λ 近似（qfrc_constraint 取 mj_forward 的坍塌解）
        → λ 改由**无驱动基座 6 行**精确解出：
            (M·qacc + qfrc_bias − qfrc_passive)[base] = (Jᵀλ)[base]
        跟/尖两点接触（与 NLP 的 weld 同构），6 方程 6 未知；再由矢状关节行
        取 τ = 同式 − (Jᵀλ)[sag]。这是相位 plant「结构接触」在真实模型上的
        等价复刻——无差分、无被动近似。
    判定三层：① 数字口径 hip/knee 峰值 100–300 N·m（计划 DoD 原文）；
    ② 双验证一致性：MJ-ID vs 解内 τ 逐关节峰值比 ∈ [0.5, 2] 且 corr > 0.7
    （计划「两套量级一致才算过」的操作化）；③ effort 限位可行。
    统计口径：内点样本（剔除 k=0/k=N 双支撑+冲击瞬间——两脚同时触地时
    跟/尖 6 元 λ 不定，且速度有冲击跳变）。
    """
    w = np.load(WORLD_NPZ, allow_pickle=True)
    mujoco, model, data, free, sag_adr, sag_dof = _mj_setup(args.scene)

    # ---- 符号映射：FK 可辨的 5 位暴力搜索（0/1/3/4/5 = 支撑踝/膝 + 摆动
    # 髋/膝/踝），支撑髋（位置 2）FK 不可见、由 P 映射结构定 = −1
    # （probe_coord_map 的 flipped 行：同关节 stance 约定 = −swing 约定，
    # 而 swing 约定与 MJCF 同号——MJCF hip 亦为 parent=pelvis,child=thigh）。
    # 阈值 1e-2：正确组合的残差 ~4 mm（URDF↔MJCF 常值几何差），错误组合
    # ≥0.02 m（支撑膝边界 0.19 m 量级）。
    from itertools import product
    best_sign, best_res = None, np.inf
    for s in product((1.0, -1.0), repeat=5):
        cand = np.array([s[0], s[1], -1.0, s[2], s[3], s[4]])
        r = _fk_residual(mujoco, model, data, sag_adr, w, cand)
        if r < best_res:
            best_res, best_sign = r, cand
    print(f"[verify] 符号映射: {best_sign.astype(int).tolist()} "
          f"(位2支撑髋=-1 由 P 映射定) FK 残差 {best_res:.2e}（阈值 1e-2，"
          "正确组合应 ~4 mm = URDF↔MJCF 常值几何差）")
    if best_res > 1e-2:
        raise SystemExit("[verify] FK 符号映射失败：相位解与 43-DOF 模型"
                         "运动学对不上，先查 Stage A 目标/场景模型。")
    sign = best_sign

    results = {n: {"drake": [], "mj": [], "mj_static": []} for n in SAG_JOINTS}
    t_rows = []
    lam_rows, fk_worst = [], 0.0
    for ph in ("l", "r"):
        stance = "left" if ph == "l" else "right"
        n_s = w[f"t_{ph}"].shape[0]
        for i in range(n_s):
            excl = i in (0, n_s - 1)  # 双支撑+冲击瞬间
            _set_mj_state(mujoco, model, data, sag_adr, sag_dof, w, ph, i,
                          sign)
            mujoco.mj_forward(model, data)
            # FK 全样本校验（含边界——支撑膝符号的 0.19 m 信号恰在相位端点；
            # 边界只对 λ/ID 奇异，对 FK 无奇异性）
            for s_i, side in enumerate(("left", "right")):
                ank = w[f"ankle_{ph}"]
                d_t = np.array([ank[i][2 * s_i] - ank[0][2 * s_i],
                                ank[i][2 * s_i + 1] - ank[0][2 * s_i + 1]])
                b = model.body(f"{side}_ankle_roll_link")
                d_m = np.array([
                    data.xpos[b.id][0] - ank[0][2 * s_i],
                    data.xpos[b.id][2] - ank[0][2 * s_i + 1]])
                fk_worst = max(fk_worst, float(np.abs(d_t - d_m).max()))
            if excl:
                continue
            # qacc（基座解析加度 + 关节符号搬运 v̇）→ rhs
            qacc = np.zeros(model.nv)
            qacc[0] = w[f"acc_{ph}"][i][0]
            qacc[2] = w[f"acc_{ph}"][i][1]
            qacc[5] = w[f"acc_{ph}"][i][2]
            for pi, mjn in enumerate(_phase_order(ph)):
                qacc[sag_dof[mjn]] = sign[pi] * w[f"vdot6_{ph}"][i][pi]
            M = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M, data.qM)
            rhs = M @ qacc + data.qfrc_bias - data.qfrc_passive
            # 支撑脚跟/尖接触点（世界坐标自 MJCF FK，与 NLP weld 同构）
            fb = model.body(f"{stance}_ankle_roll_link").id
            Rf = data.xmat[fb].reshape(3, 3)
            pts = [data.xpos[fb] + Rf @ np.array([-args.heel, 0.0, -args.sole]),
                   data.xpos[fb] + Rf @ np.array([args.toe, 0.0, -args.sole])]
            Jc = np.zeros((6, model.nv))
            for ci, p in enumerate(pts):
                jp = np.zeros((3, model.nv))
                try:  # mujoco 3.5 签名：需 (3,1) 点 + body id
                    mujoco.mj_jac(model, data, jp, None,
                                  np.ascontiguousarray(p).reshape(3, 1), fb)
                except TypeError:
                    mujoco.mj_jac(model, data, jp, p)
                Jc[3 * ci:3 * ci + 3] = jp
            A = Jc[:, :6].T  # 基座 6 行 × λ 6 元
            # 跟/尖两点接触的 A 恒奇异（rank 5）：沿跟-尖连线的「挤压模态」
            # （heel +fx / toe −fx，两点同 z）对基座行与踝行的净贡献均为零
            # ——纯脚掌内压，不影响任何 ID 行。取最小范数解，τ_sag 唯一。
            lam = np.linalg.lstsq(A, rhs[:6], rcond=None)[0]
            tau_row = rhs - Jc.T @ lam
            lam0 = np.linalg.lstsq(A, data.qfrc_bias[:6], rcond=None)[0]
            tau_row0 = data.qfrc_bias - Jc.T @ lam0
            lam_rows.append([float(w[f"t_{ph}"][i]), float(ph == "r"), *lam])
            t_rows.append(float(w[f"t_{ph}"][i] + (1.2 if ph == "r" else 0.0)))
            for pi, mjn in enumerate(_phase_order(ph)):
                d_ = sag_dof[mjn]
                results[mjn]["mj"].append(float(tau_row[d_]))
                results[mjn]["drake"].append(
                    float(sign[pi] * w[f"u6_{ph}"][i][pi]))
                results[mjn]["mj_static"].append(float(tau_row0[d_]))
    print(f"[verify] FK 全样本校验 max 残差 {fk_worst:.2e} m"
          f"（阈值 2e-2，含 URDF↔MJCF ~4 mm 常值几何差）")
    if fk_worst > 2e-2:
        raise SystemExit("[verify] FK 校验失败：映射只在部分样本成立")
    # 模型差异盘点：MJCF 阻尼/armature/摩擦（Drake 相位 plant 均为 0，
    # 若 MJCF 非零则是双验证量级差的系统性来源）
    sag_ids = [model.joint(n).id for n in SAG_JOINTS]
    sag_dofs = [model.jnt_dofadr[i] for i in sag_ids]
    dmp = np.round(model.dof_damping[sag_dofs], 3).tolist()
    arm = np.round(model.dof_armature[sag_dofs], 3).tolist()
    frl = np.round(model.dof_frictionloss[sag_dofs], 3).tolist()
    print(f"[verify] MJCF 矢状关节 damping={dmp} armature={arm} "
          f"frictionloss={frl}")

    # ---- 汇总与三层判定
    print("\n[verify] B 门双验证（内点样本；λ 由基座 6 行消去法解出）：")
    print(f"{'joint':>26} | {'解内峰':>7} | {'MJ-ID峰':>7} | {'比值':>5} | "
          f"{'corr':>5} | {'静态峰':>6} | effort")
    gate_range, gate_consist, gate_effort = True, True, True
    summary = {}
    for name in SAG_JOINTS:
        r = results[name]
        pk_d = max(abs(v) for v in r["drake"])
        pk_m = max(abs(v) for v in r["mj"])
        pk_s = max(abs(v) for v in r["mj_static"])
        corr = float(np.corrcoef(r["drake"], r["mj"])[0, 1])
        ratio = pk_m / max(pk_d, 1e-9)
        eff = next(v for k, v in EFFORT.items() if k in name)
        gate_on = "hip_pitch" in name or "_knee" in name
        ok_r = (100.0 <= pk_m <= 300.0) if gate_on else True
        ok_c = 0.5 <= ratio <= 2.0 and corr > 0.7
        ok_e = pk_m <= eff
        gate_range &= ok_r
        gate_consist &= ok_c
        gate_effort &= ok_e
        summary[name] = {"drake_peak": pk_d, "mj_peak": pk_m,
                         "mj_static_peak": pk_s, "ratio": ratio, "corr": corr}
        print(f"{name:>26} | {pk_d:7.1f} | {pk_m:7.1f} | {ratio:5.2f} | "
              f"{corr:+5.2f} | {pk_s:6.1f} | {eff:.0f}"
              f"{' ok' if ok_e else ' 超限!'}")
    lam = np.asarray(lam_rows)
    fz = lam[:, 4] + lam[:, 7]  # heel_z + toe_z
    mu = float(np.max(np.abs(lam[:, [2, 5]]) / np.maximum(
        np.abs(fz[:, None]), 1.0)))
    print(f"[verify] λ 检查: Fz[{fz.min():.0f},{fz.max():.0f}]N "
          f"(机器人 mg≈{model.body_subtreemass[0] * 9.81:.0f}N) "
          f"水平/垂直 max {mu:.2f}（参考锥 0.7）")
    # 逐相分解（摆动链 vs 支撑链）：支撑链常值系统差的归因输出
    per_phase = {}
    n_half = len(t_rows) // 2
    for name in SAG_JOINTS:
        dr, mj = results[name]["drake"], results[name]["mj"]
        for lo, tag in ((0, "L"), (n_half, "R")):
            a, b = np.asarray(dr[lo:lo + n_half]), np.asarray(
                mj[lo:lo + n_half])
            c = (float(np.corrcoef(a, b)[0, 1])
                 if np.std(a) > 1e-9 and np.std(b) > 1e-9 else None)
            per_phase[f"{name}:{tag}"] = {
                "corr": c, "rms_diff": float(np.sqrt(np.mean((a - b) ** 2))),
                "drake_rng": [float(a.min()), float(a.max())],
                "mj_rng": [float(b.min()), float(b.max())]}
    swing_ok = all(v["corr"] is not None and v["corr"] > 0.7
                   for k, v in per_phase.items()
                   if (k.endswith(":L") and "_right_" in k)
                   or (k.endswith(":R") and "_left_" in k))
    print(f"[verify] 逐相分解: 摆动链 corr>0.7 = {'PASS' if swing_ok else 'FAIL'}"
          f"（支撑链常值差归因 = URDF↔MJCF 整机 CoM x 差 ~1.4 cm × 356 N ≈ "
          f"−5 N·m，腿链 FK 同位形仅 4 mm 差、总质量一致——上体质量分布"
          f"模型差异，非解/方法错误）")
    print(f"[verify] B门 ①数字口径(hip/knee 100–300 N·m)="
          f"{'PASS' if gate_range else 'FAIL'} "
          f"②双验证一致(比值0.5–2且corr>0.7)="
          f"{'PASS' if gate_consist else 'FAIL'} "
          f"③effort可行={'PASS' if gate_effort else 'FAIL'}")
    Path("apt_g1/outputs").mkdir(parents=True, exist_ok=True)
    json.dump({"per_joint": summary,
               "per_phase": per_phase, "swing_chain_ok": bool(swing_ok),
               "gate_b_range": bool(gate_range),
               "gate_b_consistency": bool(gate_consist),
               "gate_b_effort": bool(gate_effort),
               "lam_fz": [float(fz.min()), float(fz.max())], "mu_max": mu,
               "sign": sign.astype(int).tolist(),
               "sole": args.sole, "heel": args.heel, "toe": args.toe},
              open("apt_g1/outputs/to36_verify_b.json", "w"), indent=2)
    print("[verify] 已存 apt_g1/outputs/to36_verify_b.json")
    np.savez("apt_g1/outputs/to36_verify_b_samples.npz", t=np.asarray(t_rows),
             **{n.replace("_joint", ""):
                np.stack([results[n]["drake"], results[n]["mj"],
                          results[n]["mj_static"]])
                for n in SAG_JOINTS})
    print("[verify] 逐样本时序已存 apt_g1/outputs/to36_verify_b_samples.npz")


# ------------------------------------------------- D5–7: C 门 closedloop

# SONIC PD 增益（MuJoCo 29 非手 DOF 序，与 eval_torque_gait.py 同源复制）
KP = np.array([
    99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
    99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
    40.17924, 28.50125, 28.50125,
    14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
    14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
], dtype=np.float64)
KD = np.array([
    6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
    6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
    2.55789, 1.81445, 1.81445,
    0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
    0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
], dtype=np.float64)
EFF = np.array([
    88, 88, 88, 139, 50, 50,
    88, 88, 88, 139, 50, 50,
    50, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
], dtype=np.float64)


def do_closedloop(args):
    """C 门闭环（D5–7）：F9 foot 周期解在 43-DOF MuJoCo 真接触下的回放。

    控制律（eval_torque_gait harness 模式，参考改为 TO 轨迹）：
      矢状 6 关节： τ = ff_scale·τ_TO(t) + kp·(q_TO(t)−q) − kd·q̇ (+ velff)
      其余非手 23： PD 稳在 0（B 门同口径：上体/手臂锁定在 URDF 零位）
      手 14： ctrl 0
    参考时钟：相位拼接 L→R→L…，世界 x 偏移 = (2c+p)·step；初始状态 =
    左相位样本 0（冲击后极限环状态，非静止站立）。
    防作弊指标（计划 DoD）：存活 ≥6s + h_min ≥0.6 + 世界系 disp >0.5 m。
    归因诊断（计划降级阶梯「不得混归因」）：jerr(跟踪) / 相位滞后 /
    触地误差 / 每半周期实际步长——jerr 小而倒 = 轨迹/冲击问题；
    jerr 早期发散 = 稳定层缺失（TO18–22 已知模式）。"""
    w = np.load(WORLD_NPZ, allow_pickle=True)
    mujoco, model, data, free, sag_adr, sag_dof = _mj_setup(args.scene)

    # 符号映射（与 verify 同法：FK 搜索 + 位 2 由 P 映射定）
    from itertools import product
    best_sign, best_res = None, np.inf
    for s in product((1.0, -1.0), repeat=5):
        cand = np.array([s[0], s[1], -1.0, s[2], s[3], s[4]])
        r = _fk_residual(mujoco, model, data, sag_adr, w, cand)
        if r < best_res:
            best_res, best_sign = r, cand
    if best_res > 1e-2:
        raise SystemExit(f"[cl] 符号映射失败（残差 {best_res:.2e}），先跑 verify")
    sign = best_sign
    print(f"[cl] 符号映射 {sign.astype(int).tolist()}（残差 {best_res:.1e}）")

    step, Tl, Tr = float(w["step"]), float(w["t_l"][-1]), float(w["t_r"][-1])
    T = Tl + Tr
    n_s = w["t_l"].shape[0]

    # 参考插值器：t_local → (q_ref6, tau_ref6, pxz_ref, pitch_ref)（线性）
    def ref_local(ph, tau):
        t_arr, lo, hi = 0.0, float(w[f"t_{ph}"][-1]), None
        ts = w[f"t_{ph}"]
        j = min(int(np.searchsorted(ts, tau)), n_s - 1)
        j0 = max(j - 1, 0)
        span = ts[j] - ts[j0]
        a = 0.0 if span <= 0 else (tau - ts[j0]) / span
        a = min(max(a, 0.0), 1.0)
        q = (1 - a) * w[f"q6_{ph}"][j0] + a * w[f"q6_{ph}"][j]
        u = (1 - a) * w[f"u6_{ph}"][j0] + a * w[f"u6_{ph}"][j]
        px = (1 - a) * w[f"pxz_{ph}"][j0] + a * w[f"pxz_{ph}"][j]
        pi_ = (1 - a) * w[f"pitch_{ph}"][j0] + a * w[f"pitch_{ph}"][j]
        return q, u, px, float(pi_)

    # 初始状态：左相位样本 0（冲击后）
    _set_mj_state(mujoco, model, data, sag_adr, sag_dof, w, "l", 0, sign)
    mujoco.mj_forward(model, data)
    x0, h0 = float(data.qpos[0]), float(data.qpos[2])

    # 非手关节遍历（KP/KD/EFF 的序）与矢状关节在其中的位置
    qpos_adr, dof_adr, act_ids = [], [], []
    for act_id in range(model.nu):
        jid = model.actuator_trnid[act_id, 0]
        name = model.joint(jid).name
        if "hand" in name:
            continue
        act_ids.append(act_id)
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])
    qpos_adr, dof_adr, act_ids = (np.asarray(x, int) for x in
                                  (qpos_adr, dof_adr, act_ids))
    jname = [model.joint(model.actuator_trnid[a, 0]).name for a in act_ids]
    sag_rows = {n: jname.index(n) for n in SAG_JOINTS}  # KP/KD/EFF 行号

    dt = float(model.opt.timestep)
    decim = max(int(round(args.dt_ctrl / dt)), 1)
    n_steps = int(args.seconds / (dt * decim))
    h_min, fall = h0, None
    jerr2 = np.zeros(6)
    pitch_rng = [1e9, -1e9]
    td_err, steps_done = [], []
    swing_prev = None
    # 相位-空间锁定（--phase-lock）：开环相位钟与实际前进解耦时（实际
    # vx 由接触/PD 重塑，典型 2× 超前），身体冲过支撑脚而参考摆动腿还在
    # 半空 → 前扑。相位钟按 vx 实际/参考比变速（低通限幅），参考 x 偏移
    # 同步换成相位积分口径，消除空间超前。
    t_phase = 0.0
    vx_f = float(w["v_avg"])

    def pitch_now():
        w_, x_, y_, z_ = data.qpos[3:7]
        return float(np.arcsin(np.clip(2 * (w_ * y_ - x_ * z_), -1, 1)))
    for k in range(n_steps):
        if args.phase_lock or args.event_phase:
            # 统一累积相位钟：phase_lock 变速（vx 低通限幅 0.5–2×），
            # event_phase 门控（摆动脚悬空时相位停在 0.95T 等落地）。
            if args.phase_lock:
                vx_f = 0.9 * vx_f + 0.1 * float(data.qvel[0])
                rate = float(np.clip(vx_f / max(w["v_avg"], 1e-6), 0.5, 2.0))
            else:
                rate = 1.0
            ph_now = "l" if (t_phase % T) < Tl else "r"
            tl_now = (t_phase % T) if ph_now == "l" else (t_phase % T) - Tl
            if args.event_phase:
                swing2 = "right" if ph_now == "l" else "left"
                sw_b2 = model.body(f"{swing2}_ankle_roll_link")
                if tl_now > 0.95 * (Tl if ph_now == "l" else Tr) and \
                        float(data.xpos[sw_b2.id][2]) > 0.02:
                    rate = 0.0
            t_phase += decim * dt * rate
            t = k * decim * dt
            c, ph_t = int(t_phase // T), t_phase % T
        else:
            t = k * decim * dt
            c, ph_t = int(t // T), t % T
        ph = "l" if ph_t < Tl else "r"
        tau_local = ph_t if ph == "l" else ph_t - Tl
        q6, u6, pxz_r, pitch_r = ref_local(ph, tau_local)
        x_off = (2 * c + (0 if ph == "l" else 1)) * step
        # 参考关节角/力矩（符号映射）；世界系参考仅诊断
        q_ref = {n: sign[i] * q6[i] for i, n in enumerate(_phase_order(ph))}
        u_ref = {n: sign[i] * u6[i] for i, n in enumerate(_phase_order(ph))}
        stance_side = "left" if ph == "l" else "right"
        swing_side = "right" if ph == "l" else "left"
        if args.capture > 0.0 and 0.15 * T < tau_local < 0.95 * T:
            # capture 式落足自适应（LIPM 反馈稳层最小版）：摆动脚落点
            # 按 CoM 状态前移 x_sw = com + vx·T_rem/2·k，在线解析 IK
            # （foot_gait_id 同款，MJCF 符号：hip=−θ_h、knee=θ_k）替换
            # 摆动腿 hip/knee 参考，踝按平脚公式 −hip−knee（TO22）。
            # 只动摆动腿；落点限幅 ±0.15 m 防飞。
            T_ph = Tl if ph == "l" else Tr
            com_x = float(data.subtree_com[0][0])
            vx_c = float(data.qvel[0])
            x_land = com_x + vx_c * 0.5 * (T_ph - tau_local) * args.capture
            com0_x = float(w["pxz_l"][0][0])  # 相位系原点（骨盆零位 x）
            dx = float(np.clip(x_land - com0_x, -0.15, 0.15))
            hz = 0.657  # 髋高（G1 直立标定）
            fz = 0.056  # 踝高（平脚）
            dx = float(np.clip(dx, -0.20, 0.20))
            d = np.hypot(dx, hz - fz)
            d = min(max(d, 1e-6), 0.6406 - 1e-6)
            cosk = (d * d - 0.3406 ** 2 - 0.30 ** 2) / (2 * 0.3406 * 0.30)
            thk = np.arccos(np.clip(cosk, -1.0, 1.0))
            # foot_gait_id 口径：dz = 踝z−髋z = −(hz−fz)，phi = atan2(dx, −dz)
            phi = np.arctan2(dx, hz - fz)
            delta = np.arcsin(np.clip(0.30 * np.sin(thk) / d, -1, 1))
            thh = phi + delta
            hip_mj, knee_mj = -thh, thk
            ank_mj = -hip_mj - knee_mj
            for pat, qv in (("hip_pitch", hip_mj), ("knee", knee_mj),
                            ("ankle_pitch", ank_mj)):
                nm = f"{swing_side}_{pat}_joint"
                if nm in q_ref:
                    q_ref[nm] = qv
        for _ in range(decim):
            q = data.qpos[qpos_adr]
            qd = data.qvel[dof_adr]
            torque = args.kp_scale * KP * (0.0 - q) - args.kd_scale * KD * qd
            for n, row in sag_rows.items():
                e = q_ref[n] - q[row]
                kp_eff = args.kp_scale * KP[row]
                kd_eff = args.kd_scale * KD[row]
                # 踝单独 boost：支撑踝防「绕脚尖/脚跟翻转」——TO 相位模型
                # 支撑脚 weld 钉死全掌刚性，MuJoCo 真脚可绕掌缘翻转（CoP
                # 走出支撑面即分岔），踝 PD 是唯一的脚俯仰执行器
                if "ankle" in n:
                    kp_eff *= args.ankle_kp
                    kd_eff *= args.ankle_kd
                torque[row] = (kp_eff * e - kd_eff * qd[row]
                               + args.ff_scale * u_ref[n])
                # B 门归因补偿：URDF↔MJCF 整机 CoM x 差 1.4 cm → 支撑链
                # τ_mj − τ_drake ≈ −5 N·m（ID 实测）——按 MJCF 符号补 +bias
                if stance_side in n:
                    torque[row] += args.bias_chain
                    if "ankle" in n:
                        torque[row] += args.bias_ankle
                        torque[row] += args.stab_pitch * (
                            pitch_r - pitch_now()) - args.stab_kd * float(
                            data.qvel[5])
                jerr2[list(SAG_JOINTS).index(n)] += e * e
            torque = np.clip(torque, -EFF, EFF)
            ctrl = np.zeros(model.nu)
            ctrl[act_ids] = torque
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        h_min = min(h_min, float(data.qpos[2]))
        pitch_rng[0] = min(pitch_rng[0], pitch_now())
        pitch_rng[1] = max(pitch_rng[1], pitch_now())
        if float(data.qpos[2]) < 0.2 or not np.all(np.isfinite(data.qpos)):
            fall = t
            break
        # 触地检测（相位末 5% 窗口内摆动脚接近地面 → 记录落点误差）
        ph_frac = tau_local / (Tl if ph == "l" else Tr)
        fb = model.body(f"{swing_side}_ankle_roll_link")
        sw_z = float(data.xpos[fb.id][2])
        if ph_frac > 0.95 and sw_z < 0.045:
            sw_x = float(data.xpos[fb.id][0]) - x_off
            plan = step if ph == "l" else step  # 摆动跟点落地 x（相位系）
            td_err.append(sw_x - plan)
        if k % 25 == 0:
            q_now = data.qpos[qpos_adr]
            je = max(abs(q_ref[n] - q_now[sag_rows[n]]) for n in SAG_JOINTS)
            print(f"[cl] t={t:5.2f} {ph} h={data.qpos[2]:.3f} "
                  f"x={data.qpos[0]:+.2f} "
                  f"pitch={np.degrees(pitch_now()):+5.1f}° "
                  f"sag_jerr_max={je:.3f} sw_z={sw_z:.3f}", flush=True)

    n_eff = max(k, 1) * decim
    disp = float(data.qpos[0] - x0)
    jerr_rms = np.sqrt(jerr2 / max(k * decim, 1))
    survive = fall if fall is not None else args.seconds
    gate = (survive >= 6.0 and h_min >= 0.6 and disp > 0.5)
    print(f"\n[cl] C门判定: 存活 {survive:.2f}s（门槛 6）h_min {h_min:.3f}"
          f"（门槛 0.6）世界系 disp {disp:+.2f} m（门槛 >0.5）"
          f"vx={disp / max(survive, 1e-9):+.3f} m/s → "
          f"{'PASS' if gate else 'FAIL'}")
    print(f"[cl] 归因: sag jerr_rms={np.round(jerr_rms, 3).tolist()} rad "
          f"(序 {list(SAG_JOINTS)})")
    print(f"[cl] 归因: pelvis pitch [{np.degrees(pitch_rng[0]):+.1f},"
          f"{np.degrees(pitch_rng[1]):+.1f}]°  触地误差(相位系) "
          f"{np.round(td_err, 3).tolist() if td_err else '无样本'}")
    json.dump({"survive": survive, "h_min": h_min, "disp": disp,
               "fall": fall, "gate_c": bool(gate),
               "jerr_rms": jerr_rms.tolist(),
               "pitch_rng_deg": np.degrees(pitch_rng).tolist(),
               "td_err": td_err,
               "params": {"ff": args.ff_scale, "kp": args.kp_scale,
                          "kd": args.kd_scale, "seconds": args.seconds}},
              open("apt_g1/outputs/to36_closedloop.json", "w"), indent=2)
    print("[cl] 已存 apt_g1/outputs/to36_closedloop.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="load/solve 用模型（URDF 自动注入浮基；MJCF 仅当网格"
                         "为 obj 时可用）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("load", help="D1 载入冒烟")
    p.set_defaults(fn=do_load)

    p = sub.add_parser("solve", help="D2–3 dircol 周期解（A 门 + 解内 B 门初判）")
    p.add_argument("--knots", type=int, default=24)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--dt-min", type=float, default=0.01)
    p.add_argument("--dt-max", type=float, default=0.05)
    p.add_argument("--foot-heel", type=float, default=-0.04,
                   help="足跟接触点（脚系 x 偏移，m）")
    p.add_argument("--foot-toe", type=float, default=0.10,
                   help="足尖接触点（脚系 x 偏移，m）")
    p.add_argument("--clearance", type=float, default=0.02)
    p.add_argument("--dx-target", type=float, default=0.4,
                   help="末级同伦的步长目标（m）")
    p.add_argument("--z-min", type=float, default=0.60)
    p.add_argument("--z-max", type=float, default=0.90)
    p.add_argument("--solver", choices=("ipopt", "snopt"), default="ipopt")
    p.add_argument("--no-bbox", action="store_true",
                   help="调试开关：不加 q/v/u 盒约束（检验 IPOPT DOF 核算口径）")
    p.set_defaults(fn=do_solve)

    p = sub.add_parser(
        "world", help="D5a Stage A：hybrid foot 解 → 世界系样本（.venv_drake）")
    p.add_argument("--gait-npz", default="apt_g1/outputs/to36_hybrid_gait.npz",
                   help="hybrid 解 npz（to36_hybrid_dircol.py solve 的 OUT_NPZ）")
    p.add_argument("--world-out", default=WORLD_NPZ,
                   help="world npz 输出路径；默认 = canonical to36_world_knots.npz。"
                        "TO41 批量导 7 材料 LUT 时必须显式指向独立路径（禁覆盖 canonical）")
    p.add_argument("--sole-drop", type=float, default=0.04,
                   help="pointe 兼容参数（foot 模式自动解析 URDF，忽略）")
    p.set_defaults(fn=do_world)

    p = sub.add_parser(
        "verify", help="D5a Stage B：B 门 43-DOF ID 复核（.venv_mjlab）")
    p.add_argument("--scene",
                   default="~/ros2_data/GR00T-WholeBodyControl/gear_sonic/"
                           "data/robot_model/model_data/g1/scene_43dof.xml",
                   help="MuJoCo 原生 43-DOF scene")
    p.add_argument("--gait-npz", default=None, help="（保留位，未用）")
    p.add_argument("--sole", type=float, default=0.0526,
                   help="踝-roll 原点→掌底垂距（与 NLP _parse_foot_geometry 同源）")
    p.add_argument("--heel", type=float, default=0.05, help="跟点 x 偏距（m）")
    p.add_argument("--toe", type=float, default=0.12, help="尖点 x 偏距（m）")
    p.set_defaults(fn=do_verify)

    p = sub.add_parser(
        "closedloop", help="D5–7 C 门：TO 轨迹闭环回放（.venv_mjlab）")
    p.add_argument("--scene",
                   default="~/ros2_data/GR00T-WholeBodyControl/gear_sonic/"
                           "data/robot_model/model_data/g1/scene_43dof.xml")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--dt-ctrl", type=float, default=0.02, help="控制周期（s）")
    p.add_argument("--kp-scale", type=float, default=0.5)
    p.add_argument("--kd-scale", type=float, default=1.0)
    p.add_argument("--ff-scale", type=float, default=1.0,
                   help="TO 力矩前馈缩放")
    p.add_argument("--bias-chain", type=float, default=0.0,
                   help="支撑链常值力矩补偿（N·m；B 门 ID 实测 τ_mj−τ_dr≈−5，"
                        "补偿 +5）")
    p.add_argument("--bias-ankle", type=float, default=0.0,
                   help="支撑踝额外常值补偿（N·m）")
    p.add_argument("--stab-pitch", type=float, default=0.0,
                   help="骨盆 pitch 反馈增益（N·m/rad，加在支撑踝；降级阶梯"
                        "「简单反馈稳层」）")
    p.add_argument("--stab-kd", type=float, default=0.0,
                   help="骨盆 pitch 速率阻尼（N·m·s/rad）")
    p.add_argument("--ankle-kp", type=float, default=1.0,
                   help="踝 PD kp 倍率（防脚掌绕掌缘翻转，双侧）")
    p.add_argument("--ankle-kd", type=float, default=1.0,
                   help="踝 PD kd 倍率")
    p.add_argument("--phase-lock", action="store_true",
                   help="相位-空间锁定：相位钟按 vx 实际/参考比变速（低通"
                        "限幅 0.5–2×），消除开环时钟与实际前进的解耦")
    p.add_argument("--event-phase", action="store_true",
                   help="事件驱动相位门控：摆动脚悬空时相位钟停在 0.95T"
                        "等触地（治脚未落地相位先切）")
    p.add_argument("--capture", type=float, default=0.0,
                   help="capture 落足自适应增益（0=关；摆动落点 = com + "
                        "vx·T_rem/2·k，在线 IK 重算摆动腿参考）")
    p.set_defaults(fn=do_closedloop)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
