#!/usr/bin/env python
"""TO36 共享模块：v1（to36_leg_to_drake.py）与 hybrid（to36_hybrid_dircol.py）
共用的模型构造 / 工具函数。2026-08-29 第二轮 grill-me 决策 4（代码组织 C 案）
从 v1 原样抽出，函数体未改语义——D1 服务器已验证的行为基线。

包含：
  常量        DEFAULT_MODEL / OUT_NPZ / PREPARED_URDF / SAGITTAL_PATTERNS / INF
  URDF 预处理  resolve_model（浮基+ weld 降维，含 ElementTree 防注释坑）
  plant 构造   build_plant（PlanarJoint 平面基座 + 矢状 actuator 注入）
  kin 工具     KnotKinematics / body_names / is_sagittal / _pick_base / _pick_foot
运行环境：服务器 .venv_drake；本机无 venv 不运行（AGENTS.md 环境约定）。
"""

from pathlib import Path

import numpy as np

DEFAULT_MODEL = ("~/ros2_data/GR00T-WholeBodyControl/gear_sonic/data/"
                 "robot_model/model_data/g1/g1_29dof_with_hand.urdf")
OUT_NPZ = "apt_g1/outputs/to36_periodic_gait.npz"
PREPARED_URDF = "apt_g1/outputs/to36_g1_floating.urdf"
SAGITTAL_PATTERNS = ("hip_pitch", "knee", "ankle_pitch")
INF = 1e6  # 「无界」量级（不用 np.inf：IPOPT 对大数界更稳）


def is_sagittal(name: str) -> bool:
    low = name.lower()
    return ("left" in low or "right" in low) and \
        any(p in low for p in SAGITTAL_PATTERNS)


def _cross(a, b):
    """float/AutoDiffXd 通用三维叉积。"""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def body_names(plant):
    """pydrake 无 GetBodyNames()，按 BodyIndex 枚举。"""
    from pydrake.multibody.tree import BodyIndex
    return [plant.get_body(BodyIndex(i)).name()
            for i in range(plant.num_bodies())]


def resolve_model(model: str) -> str:
    """模型路径解析（纯 ElementTree，避免字符串匹配被注释块骗过——2026-08-29
    实测 `<!-- <link name="world">` 含目标子串导致 prep 被跳过）。

    MJCF 直接返回（Drake MJCF parser 只吃 .obj 网格，G1 全 STL 会报错）。
    URDF：① 根 link 显式 weld 到 world（防 Drake 自动加浮基——平面基座改由
    build_plant 的 PlanarJoint 给出）；② 非矢状 revolute/continuous/prismatic
    → fixed（weld）。为什么 weld 而不用 bbox 钉 0：IPOPT 消元固定变量后其
    动力学等式行仍全数保留，37 locked DOF × N knots 会把自由度核算打成负
    （实测 "too few degrees of freedom, n_x=768, n_c=2665"）。
    为什么用 PlanarJoint 而不是浮基+平面化约束：浮基 7 DOF + 每 knot 6 条
    平面等式使 NLP 结构性欠自由度（每 knot 变量 32 vs 等式 37，IPOPT
    n_x < n_c 必报 too few DOF）；PlanarJoint 让模型字面成为平面实例
    （3 平面 DOF + 6 矢状关节）。
    """
    p = Path(model).expanduser()
    if p.suffix != ".urdf":
        return str(p)
    import xml.etree.ElementTree as ET
    root = ET.parse(p).getroot()

    fixed_cnt = 0
    for j in root.iter("joint"):
        jn = j.get("name", "")
        if j.get("type") in ("revolute", "continuous", "prismatic") \
                and not is_sagittal(jn):
            j.set("type", "fixed")
            fixed_cnt += 1

    dst = Path(PREPARED_URDF)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(ET.tostring(root, encoding="unicode") + "\n")
    print(f"[prep] 浮基+weld 降维 URDF 已生成: {dst}（源: {p.name}，"
          f"weld 非矢状关节 {fixed_cnt} 个）")
    return str(dst)


def _urdf_efforts(path: Path) -> dict:
    """URDF <limit effort> → {joint_name: N·m}。Unitree URDF 不带 transmission，
    Drake 解析后 actuators=0，须在 Finalize 前 AddJointActuator（2026-08-29
    D1 实测 actuators=0 命中后补的修复）。"""
    import xml.etree.ElementTree as ET
    eff = {}
    for j in ET.parse(path).getroot().findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("effort") is not None:
            eff[j.get("name")] = float(lim.get("effort"))
    return eff


def build_plant(model: str, verbose: bool = False):
    """URDF/MJCF → 连续时间 MultibodyPlant（矢状关节注入 actuator）；返回盘点。"""
    from pydrake.all import MultibodyPlant, Parser
    from pydrake.multibody.tree import JointIndex

    src = Path(resolve_model(model))
    plant = MultibodyPlant(time_step=0.0)
    Parser(plant).AddModels(str(src))
    if src.suffix == ".urdf":
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.multibody.tree import FixedOffsetFrame, PlanarJoint
        # 平面基座：F/C 均为 RotX(+90°)：Fx=Wx（前进）、Fy=Wz（高度）、
        # Fz=+Wy（俯仰）；C 与 F 同旋转保证零位=直立
        F = plant.AddFrame(FixedOffsetFrame(
            "planar_base_F", plant.world_frame(),
            RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2.0),
                           [0, 0, 0])))
        # 子帧 C 与 F 同旋转：零位时 pelvis 姿态 = F·C⁻¹ = I（直立）
        C = plant.AddFrame(FixedOffsetFrame(
            "planar_base_C", plant.GetFrameByName("pelvis"),
            RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2.0),
                           [0, 0, 0])))
        plant.AddJoint(PlanarJoint("planar_base", F, C))
    added_act = []
    if src.suffix == ".urdf":
        efforts = _urdf_efforts(src)
        for side in ("left", "right"):
            for pat in ("hip_pitch", "knee", "ankle_pitch"):
                jn = f"{side}_{pat}_joint"
                try:
                    joint = plant.GetJointByName(jn)
                except RuntimeError:
                    continue
                plant.AddJointActuator(jn + "_motor", joint,
                                       efforts.get(jn, 300.0))
                added_act.append(jn)
    plant.Finalize()
    sagittal, others = [], []
    for i in range(plant.num_joints()):
        j = plant.get_joint(JointIndex(i))
        if j.num_positions() == 0:
            continue
        (sagittal if is_sagittal(j.name()) else others).append(
            (j.name(), j.num_positions()))
    info = {"nq": plant.num_positions(), "nv": plant.num_velocities(),
            "num_bodies": plant.num_bodies(),
            "num_actuators": plant.num_actuators(),
            "actuated_joints": added_act,
            "sagittal": sagittal, "others": others,
            "body_names": body_names(plant)}
    if verbose:
        print(f"[load] nq={info['nq']} nv={info['nv']} "
              f"bodies={info['num_bodies']} actuators={info['num_actuators']} "
              f"(注入: {added_act})")
        print(f"[load] 矢状关节 {len(sagittal)}: {[n for n, _ in sagittal]}")
        print(f"[load] 非矢状关节 {len(others)}: {[n for n, _ in others]}")
        plant_ad = plant.ToAutoDiffXd()
        print(f"[load] autodiff plant OK (nq={plant_ad.num_positions()})")
    return plant, info


class KnotKinematics:
    """逐 knot 输出（14 维，float 与 AutoDiffXd 双模通用）：

      [0:4]   双足跟/尖世界 z（左跟,左尖,右跟,右尖）
      [4:8]   支撑脚跟/尖世界速度 (vx,vz)x2
      [8:11]  base 世界速度 (vy, wx, wz)   —— PlanarJoint 模型下天然 ~0，放开
      [11:14] base 平面位姿（p_y, R[0,1], R[2,1]）—— 同上，放开
    """

    OUT_DIM = 14

    def __init__(self, plant, foot_heel, foot_toe):
        self.nq, self.nv = plant.num_positions(), plant.num_velocities()
        self.plant_f = plant
        self.ctx_f = plant.CreateDefaultContext()
        self.ad = plant.ToAutoDiffXd()
        self.ctx_ad = self.ad.CreateDefaultContext()
        self.base_name = _pick_base(plant)
        self.foot_names = {s: _pick_foot(plant, s)
                           for s in ("left", "right")}
        self.heel = np.array([foot_heel, 0.0, 0.0])
        self.toe = np.array([foot_toe, 0.0, 0.0])

    def _env(self, ad_mode):
        plant = self.ad if ad_mode else self.plant_f
        ctx = self.ctx_ad if ad_mode else self.ctx_f
        return plant, ctx

    def __call__(self, vars, stance_side):
        ad_mode = vars.dtype == object
        plant, ctx = self._env(ad_mode)
        base = plant.GetBodyByName(self.base_name)
        feet = {s: plant.GetBodyByName(n)
                for s, n in self.foot_names.items()}
        q, v = vars[:self.nq], vars[self.nq:self.nq + self.nv]
        plant.SetPositions(ctx, q)
        plant.SetVelocities(ctx, v)

        def point(body, pt):
            X_WB = plant.EvalBodyPoseInWorld(ctx, body)
            R, p0 = X_WB.rotation().matrix(), X_WB.translation()
            V = plant.EvalBodySpatialVelocityInWorld(ctx, body)
            w, vv = V.rotational(), V.translational()
            p_BQ = R @ pt
            return p0 + p_BQ, vv + _cross(w, p_BQ)

        out = []
        for s in ("left", "right"):
            b = feet[s]
            for pt in (self.heel, self.toe):
                out.append(point(b, pt)[0][2])
        sb = feet[stance_side]
        for pt in (self.heel, self.toe):
            pQ, vQ = point(sb, pt)
            out.extend([vQ[0], vQ[2]])
        Xb = plant.EvalBodyPoseInWorld(ctx, base)
        Vb = plant.EvalBodySpatialVelocityInWorld(ctx, base)
        Rb = Xb.rotation().matrix()
        out.extend([Vb.translational()[1], Vb.rotational()[0],
                    Vb.rotational()[2]])
        out.extend([Xb.translation()[1], Rb[0, 1], Rb[2, 1]])
        return np.array(out)


def _pick_base(plant):
    names = body_names(plant)
    for cand in ("pelvis", "torso", "base"):
        for n in names:
            if cand in n.lower():
                return n
    raise RuntimeError(f"找不到 base body，bodies={names}")


def _pick_foot(plant, side):
    names = body_names(plant)
    for pat in ("foot", "ankle_roll", "ankle"):
        cand = [n for n in names if side in n.lower() and pat in n.lower()]
        if cand:
            return cand[-1]
    raise RuntimeError(f"找不到 {side} 脚 body，bodies={names}")
