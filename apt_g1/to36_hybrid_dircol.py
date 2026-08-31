#!/usr/bin/env python
"""TO36 D3：hybrid 双相位 dircol 周期步态（点足锁踝先行 → 真脚升级）。

设计定稿：refine-logs/LEG_LEVEL_TO_PLAN.md §2 第二轮访谈增补（2026-08-29，
grill-me 四项决策：点足先行 / 出口=真脚 A 门 / impact=模板冲击 / 双卡点 /
代码组织=新文件+common）。上游：tracker/TO.md TO36 D2 记录（v1 单 plant
钉足约束架构已判死——方程无接触反力，inf_pr 发散）。

模型（Path B：程序化 5 体平面 plant，Drake API 直接构建）：
  - URDF 手术两试两败（都记录在 tracker/TO.md TO36-D3）：① pin 小腿 link
    时 URDF revolute 转轴恒过 child 原点 → pivot 错在膝点、掌点画弧；
    ② 锚点 dummy 插踝关节会切断 pelvis 的世界通路（Drake 补浮基），而重定
    根需倒装膝/髋活动关节——活动关节父/子互换在 URDF 不可表达（旋转中心
    漂移）。故弃 URDF 手术，改程序化建体。
  - 5 体：{stance}_shin（原点=膝点，惯量=小腿+踝+脚链膝点聚合）、
    {stance}_thigh（髋点）、pelvis（stance 髋点；惯量=盆+躯干+双臂+手在该
    髋点聚合）、对侧 thigh/shin。惯量经 D1 已验证的平面 9 DOF plant 用
    CalcSpatialInertia 聚合，几何（大腿/小腿长）零位 FK 抽取。
  - 关节：world→(rev, 轴过掌点=sole_f)→shin→(膝)→thigh→(髋)→pelvis→(对侧
    髋)→thigh'→(膝')→shin'，nq=nv=5、nu=4（pin 不驱动）。支撑接触由结构
    保证（转轴过掌点，compass-gait 模板），动力学方程天然含约束反力。
  - 髋点在 pelvis 系共点（平面抽象）：真实双髋 ±y 偏置只进入 Iyy（聚合
    惯量平行轴已含），不影响矢状动力学。

NLP（对 v1 的三点结构修正）：
  1. 单 MathematicalProgram 手写 Hermite–Simpson 隐式中点两段转录（pydrake
     无公开多相位 dircol 向导）：每区间约束
     x_{k+1} − x_k − (T/N)/6·(f_k + 4·f_mid + f_{k+1}) = 0，动力学经
     autodiff plant 黑箱求值（_hs_eval）。
  2. 相位接口 = q/v 连续（数值探针求两相位坐标间的常值线性映射 P，
     probe_coord_map；两相位世界嵌入差一个刚体场，探针与验证一律相对
     TO 侧锚点＝对侧掌点）+ 刚体冲击：辅助变量 (v_aux, λ) 满足
     M(q)(v_pre − v_aux) = Jᵀλ、J·v_aux = 0（新支撑点速度归零）——
     免手写矩阵求逆，autodiff 友好。整周期 L→R→L 两次冲击闭合。
  3. 摆动掌点离地 sin 半波包络 + 速度门 2·step ≥ v_min·(T_L+T_R)；
     同伦阶梯（clr, v_min）默认 [(0.01,0.05),(0.02,0.12),(0.03,0.20)]，
     每 stage 重建 prog（Drake 不支持删约束），上一级解作初值。

周期口径：整周期两相位（不做镜像对称假设）；相对坐标（每相掌点即原点），
step = 摆动掌点末端 x，v_avg = 2·step/T。不强行 v=0 原地踏步。

子命令：
  pin     相位 plant 冒烟：几何/惯量聚合盘点 + P 探针 + autodiff finite
  solve   hybrid 双相位周期解（--mode pointe），存 to36_hybrid_gait.npz
  check   解后自检（A 门初判 + 物理合理性，读 npz 不起 Drake）

foot（真脚+踝主动 6 DOF）待点足里程碑过后在同 builder 上扩展（Q2 决策：
点足先行，真脚为 pin/weld 结构差分）。B 门 verify / C 门 closedloop 沿用
to36_leg_to_drake.py（npz 布局不同，verify 适配是 D4 工作）。
运行环境：服务器 .venv_drake；本机不运行。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from to36_common import (DEFAULT_MODEL, INF, _urdf_efforts, body_names,
                         build_plant, resolve_model)

OUT_NPZ = "apt_g1/outputs/to36_hybrid_gait.npz"
EXPECT_NQ = 5  # pin + 双腿 hip/knee


# ------------------------------------------------------- 模型构建（Path B）

LEG_LINKS = ("hip_pitch_link", "knee_link", "ankle_pitch_link",
             "ankle_roll_link")


class PhaseGeometry:
    """从 D1 平面 9 DOF plant 抽取的几何与聚合惯量（左右对称，按 stance 抽）。

    体/坐标系约定（零位 = 直腿站立、stance 掌点在世界原点）：
      {stance}_shin   原点=膝点；惯量 = 膝链（knee+ankle_pitch+ankle_roll）
                      关于膝点聚合
      {stance}_thigh  原点=髋点；惯量 = hip_pitch_link
      pelvis          原点=stance 髋点；惯量 = 其余全体（盆+躯干+双臂+手）
                      关于该髋点聚合（对侧腿 ±y 偏置只进 Iyy，矢状动力学
                      诚实的平面抽象）
    """

    def __init__(self, model, sole_drop, stance, mode="pointe"):
        plant, _ = build_plant(model)
        ctx = plant.CreateDefaultContext()
        plant.SetPositions(ctx, np.zeros(plant.num_positions()))

        def bp(n):
            return plant.EvalBodyPoseInWorld(
                ctx, plant.GetBodyByName(n)).translation()

        p_hip = bp(f"{stance}_hip_pitch_link")
        p_knee = bp(f"{stance}_knee_link")
        p_ankle = bp(f"{stance}_ankle_pitch_link")
        self.mode = mode
        self.stance = stance
        self.L_thigh = float(np.linalg.norm(p_knee - p_hip))
        idx = lambda n: plant.GetBodyByName(n).index()
        fr = lambda n: plant.GetFrameByName(n)
        leg_names = {f"{s}_{l}" for s in ("left", "right") for l in LEG_LINKS}
        pelvis_set = [idx(n) for n in body_names(plant)
                      if n != "world" and n not in leg_names]
        efforts = _urdf_efforts(Path(resolve_model(model)))
        self.effort_hip = efforts.get(f"{stance}_hip_pitch_joint", 88.0)
        self.effort_knee = efforts.get(f"{stance}_knee_joint", 139.0)
        self.masses = {}

        if mode == "pointe":
            # 踝+脚整体并入小腿（点足：掌点=踝下 −sole_drop）
            self.L_sole = float(np.linalg.norm(p_ankle - p_knee)) + sole_drop
            self.I_shin = plant.CalcSpatialInertia(
                ctx, fr(f"{stance}_knee_link"),
                [idx(f"{stance}_{l}") for l in
                 ("knee_link", "ankle_pitch_link", "ankle_roll_link")])
        elif mode == "foot":
            # 真脚：小腿 = knee_link（膝→踝）；脚 = ankle_pitch+roll（踝点聚合）。
            # 跟/尖接触点与掌底高从 URDF collision 球解析（G1 实测：跟 −0.05 /
            # 尖 +0.12 / 掌底踝下 0.0526 m），失败回退默认并打印。
            self.L_shin = float(np.linalg.norm(p_ankle - p_knee))
            self.toe_x, self.heel_x, sd = _parse_foot_geometry(model)
            self.sole_drop = sd
            self.I_shin = plant.CalcSpatialInertia(
                ctx, fr(f"{stance}_knee_link"), [idx(f"{stance}_knee_link")])
            self.I_foot = plant.CalcSpatialInertia(
                ctx, fr(f"{stance}_ankle_pitch_link"),
                [idx(f"{stance}_ankle_pitch_link"),
                 idx(f"{stance}_ankle_roll_link")])
            self.effort_ankle = efforts.get(f"{stance}_ankle_pitch_joint",
                                            50.0)
            self.masses["foot"] = self.I_foot.get_mass()
        else:
            raise ValueError(f"未知 mode: {mode}")

        self.I_thigh = plant.CalcSpatialInertia(
            ctx, fr(f"{stance}_hip_pitch_link"),
            [idx(f"{stance}_hip_pitch_link")])
        self.I_pelvis = plant.CalcSpatialInertia(
            ctx, fr(f"{stance}_hip_pitch_link"), pelvis_set)
        self.masses.update({
            "shin": self.I_shin.get_mass(),
            "thigh": self.I_thigh.get_mass(),
            "pelvis": self.I_pelvis.get_mass(),
        })
        extra = (f" toe/heel/drop={self.toe_x:.3f}/{self.heel_x:.3f}/"
                 f"{self.sole_drop:.4f}") if mode == "foot" else \
                (f" L_sole={self.L_sole:.4f}")
        print(f"[geo] {stance}[{mode}]: L_thigh={self.L_thigh:.4f}{extra} "
              f"m(shin/thigh/pelvis"
              f"{'/foot' if mode == 'foot' else ''})="
              f"{self.masses['shin']:.2f}/{self.masses['thigh']:.2f}/"
              f"{self.masses['pelvis']:.2f}"
              f"{'/' + format(self.masses.get('foot', 0), '.2f') if mode == 'foot' else ''} kg "
              f"effort(hip/knee{'/ankle' if mode == 'foot' else ''})="
              f"{self.effort_hip}/{self.effort_knee}"
              f"{'/' + format(self.effort_ankle, '.0f') if mode == 'foot' else ''}")


def _parse_foot_geometry(model):
    """URDF collision 球 → (toe_x, heel_x, sole_drop)。G1 ankle_roll_link
    四球（跟 ±0.025 双球 x=−0.05、尖 ±0.03 双球 x=+0.12，z=−0.03，
    r=0.005）+ ankle_roll_joint origin z=−0.0176 → 掌底 = 踝 pitch 轴下
    0.0526。解析失败回退实测默认值。"""
    toe_x, heel_x, drop = 0.12, 0.05, 0.0526
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(resolve_model(model)).getroot()
        for l in root.findall("link"):
            if (l.get("name") or "").endswith("_ankle_roll_link"):
                xs, zmin, r = [], None, 0.0
                for c in l.findall("collision"):
                    s = c.find("geometry/sphere")
                    if s is None:
                        continue
                    o = c.find("origin")
                    xyz = [float(v) for v in
                           (o.get("xyz") if o is not None else
                            "0 0 0").split()]
                    xs.append(xyz[0])
                    zmin = xyz[2] if zmin is None else min(zmin, xyz[2])
                    r = max(r, float(s.get("radius")))
                roll_z = 0.0
                for j in root.findall("joint"):
                    if (j.get("name") or "").endswith("_ankle_roll_joint"):
                        o = j.find("origin")
                        roll_z = float((o.get("xyz") if o is not None
                                        else "0 0 0").split()[2])
                if xs and zmin is not None:
                    toe_x = max(toe_x, max(xs))
                    heel_x = max(heel_x, -min(xs))
                    drop = -(roll_z + zmin - r)
                break
    except Exception as e:
        print(f"[geo] 脚掌几何解析失败（用默认 {toe_x}/{heel_x}/{drop}）：{e}")
    return float(toe_x), float(heel_x), float(drop)


def build_phase_plant(geo, stance, slope_deg=0.0):
    """程序化平面 plant（零位 = 双脚平贴地面、stance 掌底在世界原点）。

    pointe（5 体 5 DOF）：world→pin(掌点)→shin→knee→thigh→hip→pelvis
    →swing_hip→swing_thigh→swing_knee→swing_shin。
    foot（7 体 6 DOF，计划 §2.1 决策 6「pin 换 weld + 解锁踝」）：支撑脚
    WeldJoint 钉世界（全掌刚体接触，跟/尖两点结构零速）+ 踩踝解锁：
    world→weld→{stance}_foot→ankle→shin→knee→thigh→hip→pelvis→swing_hip
    →swing_thigh→swing_knee→swing_shin→swing_ankle→swing_foot。
    体坐标约定：shin 原点=膝点、thigh 原点=髋点、pelvis 原点=stance 髋点、
    foot 原点=踝 pitch 点（与 PhaseGeometry 惯量聚合点一致）；关节 child
    帧用偏移帧补零位重合。
    slope_deg：重力倾斜同伦——世界 x 正向为下坡，
    g = 9.81·[sinγ, 0, −cosγ]。斜坡上周期解必须「耗散=重力做功」，零触地
    滚动解结构性不存在（能量单调增），真实冲击步态成为唯一周期族；退火
    slope→0 时力矩接管能量注入（vmax3–6 实验定论：平地冷/热启动都进不了
    v_td 盆，种子只能从斜坡被动步态长出来）。"""
    from pydrake.all import (FixedOffsetFrame, MultibodyPlant,
                             RevoluteJoint, RigidTransform, WeldJoint)
    other = "right" if stance == "left" else "left"
    plant = MultibodyPlant(time_step=0.0)
    W = plant.world_frame()
    Y = np.array([0.0, 1.0, 0.0])
    if geo.mode == "foot":
        foot_s = plant.AddRigidBody(f"{stance}_foot", geo.I_foot)
        shin_s = plant.AddRigidBody(f"{stance}_shin", geo.I_shin)
        thigh_s = plant.AddRigidBody(f"{stance}_thigh", geo.I_thigh)
        pelvis = plant.AddRigidBody("pelvis", geo.I_pelvis)
        thigh_w = plant.AddRigidBody(f"{other}_thigh", geo.I_thigh)
        shin_w = plant.AddRigidBody(f"{other}_shin", geo.I_shin)
        foot_w = plant.AddRigidBody(f"{other}_foot", geo.I_foot)
        # weld：零位掌底贴地 → weld X_PC 抬高 sole_drop（踝点在世界
        # (0,0,+sole_drop)，跟/尖两球底 z=0）
        plant.AddJoint(WeldJoint(
            f"{stance}_foot_weld", W, foot_s.body_frame(),
            RigidTransform([0.0, 0.0, geo.sole_drop])))
        shin_ankle_f = plant.AddFrame(FixedOffsetFrame(
            f"{stance}_shin_ankle_f", shin_s,
            RigidTransform([0.0, 0.0, -geo.L_shin])))
        thigh_knee_f = plant.AddFrame(FixedOffsetFrame(
            f"{stance}_thigh_knee_f", thigh_s, RigidTransform(
                [0.0, 0.0, -geo.L_thigh])))
        knee_wf = plant.AddFrame(FixedOffsetFrame(
            "swing_knee_f", thigh_w, RigidTransform([0.0, 0.0, -geo.L_thigh])))
        shinw_ankle_f = plant.AddFrame(FixedOffsetFrame(
            f"{other}_shin_ankle_f", shin_w,
            RigidTransform([0.0, 0.0, -geo.L_shin])))
        # 根 = 支撑踝（脚已钉地，踝动小腿）；父=foot 帧（原点=踝点），
        # 子=shin 踝偏移帧
        plant.AddJoint(RevoluteJoint(f"{stance}_ankle",
                                     foot_s.body_frame(), shin_ankle_f, Y))
        plant.AddJoint(RevoluteJoint(f"{stance}_knee", shin_s.body_frame(),
                                     thigh_knee_f, Y))
        plant.AddJoint(RevoluteJoint(f"{stance}_hip", thigh_s.body_frame(),
                                     pelvis.body_frame(), Y))
        plant.AddJoint(RevoluteJoint(f"{other}_hip", pelvis.body_frame(),
                                     thigh_w.body_frame(), Y))
        plant.AddJoint(RevoluteJoint(f"{other}_knee", knee_wf,
                                     shin_w.body_frame(), Y))
        plant.AddJoint(RevoluteJoint(f"{other}_ankle",
                                     foot_w.body_frame(), shinw_ankle_f, Y))
        for jn, lim in ((f"{stance}_ankle", geo.effort_ankle),
                        (f"{stance}_knee", geo.effort_knee),
                        (f"{stance}_hip", geo.effort_hip),
                        (f"{other}_hip", geo.effort_hip),
                        (f"{other}_knee", geo.effort_knee),
                        (f"{other}_ankle", geo.effort_ankle)):
            plant.AddJointActuator(jn + "_motor", plant.GetJointByName(jn),
                                   lim)
        plant.Finalize()
        if slope_deg:
            ga = np.deg2rad(slope_deg)
            g = 9.81
            plant.gravity_field().set_gravity_vector(
                np.array([g * np.sin(ga), 0.0, -g * np.cos(ga)]))
        return plant

    shin_s = plant.AddRigidBody(f"{stance}_shin", geo.I_shin)
    thigh_s = plant.AddRigidBody(f"{stance}_thigh", geo.I_thigh)
    pelvis = plant.AddRigidBody("pelvis", geo.I_pelvis)
    thigh_w = plant.AddRigidBody(f"{other}_thigh", geo.I_thigh)
    shin_w = plant.AddRigidBody(f"{other}_shin", geo.I_shin)

    Y = np.array([0.0, 1.0, 0.0])
    # 体坐标系约定：shin 原点=膝点、thigh 原点=髋点、pelvis 原点=stance 髋点
    # （与 PhaseGeometry 的惯量聚合点一致）。joint child frame 用偏移帧补齐
    # 零位重合条件。
    sole_f = plant.AddFrame(FixedOffsetFrame(
        "stance_sole_f", shin_s, RigidTransform([0.0, 0.0, -geo.L_sole])))
    thigh_knee_f = plant.AddFrame(FixedOffsetFrame(
        "stance_thigh_knee_f", thigh_s, RigidTransform(
            [0.0, 0.0, -geo.L_thigh])))
    knee_wf = plant.AddFrame(FixedOffsetFrame(
        "swing_knee_f", thigh_w, RigidTransform([0.0, 0.0, -geo.L_thigh])))
    plant.AddJoint(RevoluteJoint("stance_pin", W, sole_f, Y))
    plant.AddJoint(RevoluteJoint(f"{stance}_knee", shin_s.body_frame(),
                                 thigh_knee_f, Y))
    plant.AddJoint(RevoluteJoint(f"{stance}_hip", thigh_s.body_frame(),
                                 pelvis.body_frame(), Y))
    plant.AddJoint(RevoluteJoint(f"{other}_hip", pelvis.body_frame(),
                                 thigh_w.body_frame(), Y))
    plant.AddJoint(RevoluteJoint(f"{other}_knee", knee_wf,
                                 shin_w.body_frame(), Y))
    for jn, lim in ((f"{stance}_knee", geo.effort_knee),
                    (f"{stance}_hip", geo.effort_hip),
                    (f"{other}_hip", geo.effort_hip),
                    (f"{other}_knee", geo.effort_knee)):
        plant.AddJointActuator(jn + "_motor", plant.GetJointByName(jn), lim)
    plant.Finalize()
    if slope_deg:
        ga = np.deg2rad(slope_deg)
        g = 9.81
        plant.gravity_field().set_gravity_vector(
            np.array([g * np.sin(ga), 0.0, -g * np.cos(ga)]))
    return plant


class PhaseKit:
    """单相位 plant 的双模（double/AutoDiffXd）求值器。

    q 顺序 = GetPositionNames()（[stance_pin, stance_knee, stance_hip,
    other_hip, other_knee]）；u 顺序 = actuator 注入顺序。接触点集
    contacts：键 f"{side}_sole"，值 (body, offset)。
    """

    def __init__(self, plant, contacts, stance):
        self.plant_f = plant
        self.stance = stance
        self.contacts = contacts
        self.nq, self.nv = plant.num_positions(), plant.num_velocities()
        self.nu = plant.num_actuators()
        self.ctx_f = plant.CreateDefaultContext()
        self.ad = plant.ToAutoDiffXd()
        self.ctx_ad = self.ad.CreateDefaultContext()
        self.pnames = plant.GetPositionNames()
        self._fr_cache = {}

    def _env(self, ad):
        return (self.ad, self.ctx_ad) if ad else (self.plant_f, self.ctx_f)

    def frame(self, plant, name):
        key = (id(plant), name)
        if key not in self._fr_cache:
            self._fr_cache[key] = plant.GetFrameByName(name)
        return self._fr_cache[key]

    def point_pos(self, q, cname):
        """接触点世界坐标（float/AutoDiffXd 双模）。q 可为全状态 (nq+nv)
        或纯位形 (nq)——统一取前 nq（避免 SetPositions 尺寸断言失败）。"""
        plant, ctx = self._env(q.dtype == object)
        body, off = self.contacts[cname]
        plant.SetPositions(ctx, q[:self.nq])
        X = plant.EvalBodyPoseInWorld(ctx, plant.GetBodyByName(body))
        return X.translation() + X.rotation().matrix() @ off

    def point_jac(self, q, cname):
        """3×nv 世界系平移雅可比。CalcJacobianTranslationalVelocity 的
        帧参数个数跨版本有差，双签名兜底（D1 风格防御）。"""
        plant, ctx = self._env(q.dtype == object)
        body, off = self.contacts[cname]
        plant.SetPositions(ctx, q[:self.nq])
        try:
            from pydrake.multibody.tree import \
                JacobianWrtVariable as JacobianWrt
        except ImportError:
            from pydrake.multibody.tree import JacobianWrt
        fr = self.frame(plant, body)
        world = plant.world_frame()
        try:
            return plant.CalcJacobianTranslationalVelocity(
                ctx, JacobianWrt.kV, fr, off, world, world)
        except TypeError:
            return plant.CalcJacobianTranslationalVelocity(
                ctx, JacobianWrt.kV, fr, off, world)

    def xdot(self, xu):
        """[x(nq+nv), u(nu)] → 连续动力学 xdot(nv)。"""
        n = self.nq + self.nv
        x, u = xu[:n], xu[n:]
        plant, ctx = self._env(xu.dtype == object)
        plant.SetPositions(ctx, x[:self.nq])
        plant.SetVelocities(ctx, x[self.nq:n])
        plant.get_actuation_input_port().FixValue(ctx, u)
        return plant.EvalTimeDerivatives(ctx).CopyToVector()

    def mass(self, q):
        plant, ctx = self._env(q.dtype == object)
        plant.SetPositions(ctx, q)
        return plant.CalcMassMatrix(ctx)


def _iter_link_bodies(plant, side, pats):
    names = body_names(plant)
    for pat in pats:
        for n in names:
            if side in n.lower() and pat in n.lower():
                yield n


def _stance_ref_body(plant, stance):
    names = body_names(plant)
    for name in (f"{stance}_foot", f"{stance}_shin"):  # foot 优先（foot 模式）
        if name in names:
            return name
    raise RuntimeError(f"找不到锚点体，bodies={names}")


def _point_state(plant, ctx, body_name, off):
    """(点世界坐标, 点世界速度)——空间速度 + ω×r。"""
    body = plant.GetBodyByName(body_name)
    X = plant.EvalBodyPoseInWorld(ctx, body)
    V = plant.EvalBodySpatialVelocityInWorld(ctx, body)
    r = X.rotation().matrix() @ off
    return X.translation() + r, V.translational() + np.cross(
        V.rotational(), r)


def probe_coord_map(kit_from, kit_to):
    """两相位坐标间的常值线性映射 P（nq_to × nq_from）：q_to = P·q_from。

    两相位是同一 5 体链的不同支点坐标化，但**世界嵌入不同**（各自掌点钉在
    世界原点，整机构型差一个刚体场）。共同参照 = TO 侧锚点（其被钉死的
    掌点）：TO 里速度恒 0（J 行直接置零），FROM 里是摆动掌点（用点速度
    ω×r 修正）。相对量下刚性场消去 → lstsq 精确取列；随机 q/v 三重双验。
    """
    nf, nt = kit_from.nq, kit_to.nq
    pf, cf = kit_from._env(False)
    pt, ct = kit_to._env(False)
    try:
        from pydrake.multibody.tree import \
            JacobianWrtVariable as JacobianWrt
    except ImportError:
        from pydrake.multibody.tree import JacobianWrt

    names_f = set(body_names(kit_from.plant_f))
    cand = {f"{s}_{b}" for s in ("left", "right")
            for b in ("thigh", "shin", "foot")} | {"pelvis"}
    body_list = sorted(b for b in cand if b in names_f)
    ref_name = _stance_ref_body(kit_to.plant_f, kit_to.stance)
    ref_off = kit_to.ref_off

    # 解析映射（链式角累计；直腿零位是速度探针的奇异位形——所有连杆共线、
    # Jacobian 秩 1，lstsq 必出垃圾，故弃数值探针取列）：
    #   pointe：pin_t = Σ q_f（TO 支撑小腿绝对角 = FROM 全链累计，链末无踝）。
    #   foot：踝关节倒装（parent=foot、child=shin）→ 链末脚绝对角 =
    #     Σ5 − a2（Σ5 = 除摆动踝外全链累计）。图映射：
    #     根行 ts_ankle = Σ5（=小腿绝对角，精确恒等）；膝/髋行 = −对侧同名
    #     （re-root 反转）；摆动踝行 = +支撑踝（re-root 反转 × 倒装反转，
    #     双负得正）。手推 + probe 三重位形验证兜底。
    def pidx(kit, sub):
        for i, nm in enumerate(kit.pnames):
            if sub in nm:
                return i
        raise RuntimeError(f"pname 中找不到 {sub}：{kit.pnames}")

    fs, ts = kit_from.stance, kit_to.stance
    fo = "right" if fs == "left" else "left"
    to_ = "right" if ts == "left" else "left"
    P = np.zeros((nt, nf))
    is_foot = any("ankle" in nm for nm in kit_to.pnames)
    root_pat = "stance_pin" if not is_foot else f"{ts}_ankle"
    for j in range(nf):
        P[pidx(kit_to, root_pat), j] = 1.0
    if is_foot:
        # 根行扣除摆动踝列：ts_ankle = Σ5（脚绝对角 Σ5−a2 由触地约束归零）
        P[pidx(kit_to, root_pat), pidx(kit_from, f"{fo}_ankle")] = 0.0
        # 摆动踝行 = +fs_ankle（双重翻转）
        P[pidx(kit_to, f"{to_}_ankle"), pidx(kit_from, f"{fs}_ankle")] = 1.0
    flipped = ([(f"{ts}_knee", f"{fo}_knee"), (f"{ts}_hip", f"{fo}_hip"),
                (f"{to_}_hip", f"{fs}_hip"), (f"{to_}_knee", f"{fs}_knee")])
    for t_sub, f_sub in flipped:
        P[pidx(kit_to, t_sub), pidx(kit_from, f_sub)] = -1.0

    rng = np.random.default_rng(0)
    i_fa = pidx(kit_from, f"{fo}_ankle") if is_foot else None
    for _ in range(3):
        qr, vr = rng.normal(size=nf) * 0.3, rng.normal(size=nf) * 0.5
        if is_foot:
            # foot 模式两卡形状流形不同（各自支撑脚平贴），P 只在「双脚
            # 平贴」交集（平贴约束 Σ5 − a2 = 0，线性）上精确——接口解被
            # 触地等式约束钉在该交集上。探针采样投影到交集：把摆动踝列
            # 调成使脚绝对角（角速度）为零。
            qr[i_fa] = qr.sum() - qr[i_fa]
            vr[i_fa] = vr.sum() - vr[i_fa]
        pf.SetPositions(cf, qr)
        pf.SetVelocities(cf, vr)
        rel_p, rel_v = [], []
        for b in body_list:
            body = pf.GetBodyByName(b)
            XW = pf.EvalBodyPoseInWorld(cf, body)
            V = pf.EvalBodySpatialVelocityInWorld(cf, body)
            rel_p.append(XW.translation())
            rel_v.append(V.translational())
        pr_f, vr_f = _point_state(pf, cf, ref_name, ref_off)
        rel_p = [p - pr_f for p in rel_p]
        rel_v = [v - vr_f for v in rel_v]
        pt.SetPositions(ct, np.dot(P, qr))
        pt.SetVelocities(ct, np.dot(P, vr))
        pr_t, vr_t = _point_state(pt, ct, ref_name, ref_off)
        err = 0.0
        for i, b in enumerate(body_list):
            body = pt.GetBodyByName(b)
            XW = pt.EvalBodyPoseInWorld(ct, body)
            V = pt.EvalBodySpatialVelocityInWorld(ct, body)
            err = max(err,
                      float(np.abs(XW.translation() - pr_t - rel_p[i]).max()),
                      float(np.abs(V.translational() - vr_t - rel_v[i]).max()))
        if err > 1e-6:
            raise RuntimeError(
                f"相位坐标映射探针失败（残差 {err:.2e}）：两相位形状空间"
                "不一致或锚点参照失效")
    print("[probe] P 解析构造 + 三重形状验证通过")
    return P


# ------------------------------------------------------------- NLP 约束件

def _hs_eval(kit, z, N, nu):
    """分离式 Hermite–Simpson 缺陷 + 中点插值一致性。
    z = [x_k, u_k, x_mid, x_{k+1}, u_{k+1}, T]，返回 2(nq+nv) 维残差：
      [defect]  x_{k+1} − x_k − (T/N)/6·(f_k + 4·f_m + f_{k+1})
      [interp]  x_m − ½(x_k + x_{k+1}) − (T/N)/8·(f_k − f_{k+1})
    〔2026-08-30 修复：原实现只有 defect（压缩式却无中点插值约束），
    x_mid 实为自由变量——可任意充当「动力学平均调节器」，knot 间跳变
    不受物理约束（混叠伪解的机制，F5 平地零力矩损耗 45 J/周期、能量
    审计 drift 22 J 暴露）。分离式两行补全后配点才真实成立。〕"""
    n = kit.nq + kit.nv
    i = 0
    x_k = z[i:i + n]; i += n
    u_k = z[i:i + nu]; i += nu
    x_m = z[i:i + n]; i += n
    x_k1 = z[i:i + n]; i += n
    u_k1 = z[i:i + nu]; i += nu
    T = z[i:]
    h = T[0] / N
    f_k = kit.xdot(np.concatenate([x_k, u_k]))
    f_m = kit.xdot(np.concatenate([x_m, (u_k + u_k1) / 2.0]))
    f_k1 = kit.xdot(np.concatenate([x_k1, u_k1]))
    defect = x_k1 - x_k - (h / 6.0) * (f_k + 4.0 * f_m + f_k1)
    interp = x_m - 0.5 * (x_k + x_k1) - (h / 8.0) * (f_k - f_k1)
    return np.concatenate([defect, interp])


def make_impact(kit_pre, cnames_pre, eps=0.0):
    """冲击约束（pre 相位坐标内求值）。z = [q, v_pre, v_aux, λ]，
    行 = nv + 2·npts：
      M(q)(v_pre − v_aux) − Jᵀλ = 0（线/角动量冲量方程）
      J·(v_aux + eps·v_pre) = 0 —— 恢复系数同伦：eps=1 完全弹性弹跳
      （可行域大，起步用）；eps=0 真实刚性落地（恢复系数 0）。
    v_aux = post 速度（pre 坐标表达），经 P 映射进 post 相位。"""
    nv = kit_pre.nv
    nc = 2 * len(cnames_pre)

    def fn(z):
        q = z[:nv]
        v_pre, v_aux, lam = z[nv:2 * nv], z[2 * nv:3 * nv], z[3 * nv:]
        M = kit_pre.mass(q)
        J = np.vstack([kit_pre.point_jac(q, c)[[0, 2], :]
                       for c in cnames_pre])
        # object(AutoDiff) 数组用 np.dot：@/matmul 对 object dtype 有坑
        return np.concatenate(
            [np.dot(M, v_pre - v_aux) - np.dot(J.T, lam),
             np.dot(J, v_aux + eps * v_pre)])

    return fn, nc


def make_interface(P):
    """相位接口 q/v 连续。实际变量布局（AddConstraint 侧）：
    z = [x_pre(nq+nv), v_aux(nv), x_post(nq+nv)]，行 = 2·nq_post：
    [q_post − P·q_pre; v_post − P·v_aux]。
    〔bug 修复 2026-08-30：本函数原按 [q_pre, v_aux, q_post, v_post] 布局
    切片，与实际传入的整块 x_pre/x_post 错位——一直在强制 v_aux=P·q_pre、
    q_post=P·v_pre 两行胡乱等式，真实相位连续从未生效。D3 至今全部
    「已收敛」hybrid 解的相位缝因此作废（vmax1–12 + 19:57 轮），暴露线索
    = dump 的 lr 接口 P 映射差恒 0.4–0.5 + 接口诊断残差 O(7)。〕"""
    nt = P.shape[0]
    npre = P.shape[1]  # nq_pre（本文件 nq≡nv=5）

    def fn(z):
        nx = 2 * npre            # x_pre 整块长度（q_pre + v_pre）
        q_pre = z[:npre]
        v_aux = z[nx:nx + npre]
        q_post = z[nx + npre:nx + 2 * npre]
        v_post = z[nx + 2 * npre:nx + 3 * npre]
        return np.concatenate([q_post - np.dot(P, q_pre),
                               v_post - np.dot(P, v_aux)])

    return fn, 2 * nt


def _make_gate(kit, cname, v_min):
    """速度门（行 1）：2·step − v_min·(T_L+T_R) ≥ 0。
    z = [x_L_end(nq+nv), T_L, T_R]，step = 摆动点末端 x。"""
    n = kit.nq + kit.nv

    def fn(z):
        step = kit.point_pos(z[:n], cname)[0]
        return np.array([2.0 * step - v_min * (z[n] + z[n + 1])])

    return fn


# ------------------------------------------------------------- 构建

def _build_kits(model, mode, sole_drop, verbose=True, slope_deg=0.0):
    if mode not in ("pointe", "foot"):
        raise ValueError(mode)
    kits, Ps = {}, {}
    for stance in ("left", "right"):
        geo = PhaseGeometry(model, sole_drop, stance, mode=mode)
        plant = build_phase_plant(geo, stance, slope_deg=slope_deg)
        if mode == "pointe":
            contacts = {
                f"{s}_sole": (f"{s}_shin", np.array([0.0, 0.0, -geo.L_sole]))
                for s in ("left", "right")} | {
                "pelvis": ("pelvis", np.zeros(3))}
            ref_off = np.array([0.0, 0.0, -geo.L_sole])
            expect_nq, step_contact = 5, "right_sole"
        else:
            # 跟/尖两接触点（foot 帧原点=踝点，z 轴下 sole_drop 到掌底平面）
            contacts = {
                f"{s}_{pt}": (f"{s}_foot",
                              np.array([dx, 0.0, -geo.sole_drop]))
                for s in ("left", "right")
                for pt, dx in (("heel", -geo.heel_x), ("toe", geo.toe_x))
            } | {"pelvis": ("pelvis", np.zeros(3))}
            ref_off = np.array([0.0, 0.0, -geo.sole_drop])
            expect_nq, step_contact = 6, "right_heel"
        kit = PhaseKit(plant, stance=stance, contacts=contacts)
        kit.ref_off = ref_off
        kit.step_contact = step_contact
        kits[stance] = kit
        if kit.nq != expect_nq:
            raise RuntimeError(f"{stance} 相位 nq={kit.nq} ≠ {expect_nq}，"
                               f"position_names={kit.pnames}")
        if verbose:
            print(f"[kit] {stance}: nq={kit.nq} nv={kit.nv} nu={kit.nu} "
                  f"q={kit.pnames}")
    Ps[("left", "right")] = probe_coord_map(kits["left"], kits["right"])
    Ps[("right", "left")] = probe_coord_map(kits["right"], kits["left"])
    return kits, Ps


def _u_bounds(kit):
    from pydrake.multibody.tree import JointActuatorIndex
    ulo, uhi = [], []
    for i in range(kit.plant_f.num_actuators()):
        a = kit.plant_f.get_joint_actuator(JointActuatorIndex(i))
        lim = float(a.effort_limit())
        ulo.append(-lim if lim > 0 else -1e4)
        uhi.append(lim if lim > 0 else 1e4)
    return np.array(ulo), np.array(uhi)


def _build_stage_prog(kits, Ps, N, npts, clr, v_min, eps, pin_lim, v_td,
                      args):
    """单 stage 的 prog（Drake 不支持删约束，同伦每级重建）。返回
    (prog, sv)；sv 为扁平变量字典，跨 stage 传递初值。

    eps：冲击同伦。λ 界 = (1−eps)·300——eps=1 时 λ=0，接口退化为速度弱
    连续（v_aux=v_pre，物理错但可行域大，计划 §3 fallback）；eps=0 恢复
    真实刚体冲击。pin_lim：支撑 pin 转角限幅，防大回环甩腿伪解。"""
    from pydrake.all import MathematicalProgram
    kits_l, kits_r = kits["left"], kits["right"]
    nq_l, nv_l = kits_l.nq, kits_l.nv
    nq_r, nv_r = kits_r.nq, kits_r.nv
    prog = MathematicalProgram()
    sv = {}
    for ph, kit in (("left", kits_l), ("right", kits_r)):
        n = kit.nq + kit.nv
        tag = ph[0].upper()
        sv[f"X_{ph}"] = [prog.NewContinuousVariables(n, f"x{tag}{k}")
                         for k in range(N + 1)]
        sv[f"U_{ph}"] = [prog.NewContinuousVariables(kit.nu, f"u{tag}{k}")
                         for k in range(N + 1)]
        sv[f"XM_{ph}"] = [prog.NewContinuousVariables(n, f"xm{tag}{k}")
                          for k in range(N)]
    sv["T_left"] = prog.NewContinuousVariables(1, "TL")[0]
    sv["T_right"] = prog.NewContinuousVariables(1, "TR")[0]
    sv["v_aux_lr"] = prog.NewContinuousVariables(nv_l, "vauxLR")
    sv["v_aux_rl"] = prog.NewContinuousVariables(nv_r, "vauxRL")
    sv["lam_lr"] = prog.NewContinuousVariables(2 * npts, "lamLR")
    sv["lam_rl"] = prog.NewContinuousVariables(2 * npts, "lamRL")
    for key in ("lam_lr", "lam_rl"):
        prog.AddBoundingBoxConstraint(-300.0, 300.0, sv[key])

    # 盒约束：q/v 用 plant 限位（INF 截断）+ 速度物理限幅，u ±effort，
    # T 相位时长。pin 另加转角限幅（同伦参数）：Drake 程序化 revolute
    # 默认无界，不锁则解会走「大回环甩腿」（pin 一次转多圈、零力矩）。
    # V_CAP 同时是抗混叠关键：N 粗（h 大）+ 速度无界 → 真实运动在 knot
    # 间失控，配点条件逐点自洽但 knot 序列能量不守恒（混叠伪解，F5 实测
    # 平地零力矩"损耗"45 J/周期）。V_CAP 须满足 h·(v/L) ≪ 1。
    V_CAP = args.v_cap
    for ph, kit in (("left", kits_l), ("right", kits_r)):
        plant = kit.plant_f
        xlo = np.concatenate([
            np.clip(plant.GetPositionLowerLimits(), -INF, INF),
            np.maximum(np.clip(plant.GetVelocityLowerLimits(), -INF, INF),
                       -V_CAP)])
        xhi = np.concatenate([
            np.clip(plant.GetPositionUpperLimits(), -INF, INF),
            np.minimum(np.clip(plant.GetVelocityUpperLimits(), -INF, INF),
                       V_CAP)])
        pin_i = next((i for i, nm in enumerate(kit.pnames)
                      if "stance_pin" in nm), None)  # foot 模式无 pin
        if pin_i is not None:
            xlo[pin_i] = max(xlo[pin_i], -pin_lim)
            xhi[pin_i] = min(xhi[pin_i], pin_lim)
        # 膝/髋/踝行程限幅：封「摆动膝绕圈（|q|>π）+ 髋大角度贴地」类伪解
        # 〔F11 修正（2026-08-30，B/C 门联动发现）〕：原对称膝盒 ±2.0 放行
        # 了支撑膝过伸（F9 相位 +0.345 → MJCF −0.345）与摆动膝反屈（相位
        # −0.90 → MJCF −0.90），而真实 G1 膝限位 [−0.087, 2.88]（屈曲为正）
        # ——F9 步态对真实膝运动学不可行，C 门闭环被 MJCF 限位约束全力对抗。
        # 修正 = 相位感知盒：支撑膝与 MJCF 反号（FK 验证的符号映射）→
        # [−2.88, +0.087]；摆动膝同号 → [−0.087, +2.88]。踝同理收紧到
        # MJCF [−0.873, 0.524] 的相位镜像（两角色踝均反号 → [−0.524, 0.873]）。
        # pointe 模式保持旧对称盒（该线已按 §2.1 决策 7 结案）。
        is_foot_mode = any("ankle" in nm for nm in kit.pnames)
        for side in ("left", "right"):
            for pat, lim in (("knee", 2.0), ("hip", 1.6), ("ankle", 0.7)):
                ji = next((i for i, nm in enumerate(kit.pnames)
                           if f"{side}_{pat}" in nm), None)
                if ji is None:
                    continue
                xlo[ji] = max(xlo[ji], -lim)
                xhi[ji] = min(xhi[ji], lim)
                if is_foot_mode and pat in ("knee", "ankle"):
                    if pat == "knee":
                        if side == kit.stance:
                            xlo[ji], xhi[ji] = -2.88, 0.087
                        else:
                            xlo[ji], xhi[ji] = -0.087, 2.88
                    else:  # ankle：支撑/摆动均与 MJCF 反号
                        xlo[ji], xhi[ji] = -0.524, 0.873
        ulo, uhi = _u_bounds(kit)
        for k in range(N + 1):
            prog.AddBoundingBoxConstraint(xlo, xhi, sv[f"X_{ph}"][k])
            prog.AddBoundingBoxConstraint(ulo, uhi, sv[f"U_{ph}"][k])
        for k in range(N):
            prog.AddBoundingBoxConstraint(xlo, xhi, sv[f"XM_{ph}"][k])
        prog.AddBoundingBoxConstraint(args.t_min, args.t_max, sv[f"T_{ph}"])

    # HS 缺陷 + 中点插值一致性（分离式，每相各自的 kit；lambda 默认参防闭包
    # 晚绑定）。残差 2n 行 = [defect(n); interp(n)]。
    for ph, kit in (("left", kits_l), ("right", kits_r)):
        n = kit.nq + kit.nv
        for k in range(N):
            z = np.concatenate([sv[f"X_{ph}"][k], sv[f"U_{ph}"][k],
                                sv[f"XM_{ph}"][k], sv[f"X_{ph}"][k + 1],
                                sv[f"U_{ph}"][k + 1], [sv[f"T_{ph}"]]])
            prog.AddConstraint(
                lambda zz, kk=kit: _hs_eval(kk, zz, N, kk.nu),
                np.zeros(2 * n), np.zeros(2 * n), z)

    # 摆动掌点：首末 knot 触地（z=0 等式：离地/落地时刻）+ 中段 sin 半波
    # 离地包络。无首末触地约束时摆动腿可穿地蹭步长（D3 首解漏洞：步长
    # 1.362 m = 劈叉极限、力矩全零、swing_min_z=-0.011 穿地）。
    swing = {"left": "right", "right": "left"}
    for ph, kit in (("left", kits_l), ("right", kits_r)):
        sw = swing[ph]
        cnames = [c for c in kit.contacts if c.startswith(sw)]
        for k in (0, N):
            for cn in cnames:
                prog.AddConstraint(
                    lambda x, kk=kit, c=cn: np.array([kk.point_pos(x, c)[2]]),
                    [-args.td_tol], [args.td_tol], sv[f"X_{ph}"][k])
        for k in range(1, N):
            clr_k = clr * np.sin(np.pi * k / N)
            for cn in cnames:
                prog.AddConstraint(
                    lambda x, kk=kit, c=cn, c0=clr_k:
                        np.array([kk.point_pos(x, c)[2] - c0]),
                    [0.0], [INF], sv[f"X_{ph}"][k])
        # 中点变量同样约束（否则摆动腿可在 knot 之间穿地——D3 第二解漏洞：
        # knot 级 swing_min_z=0 但力矩全零 + KE 1270 J 的甩腿解）
        for k in range(N):
            clr_m = clr * np.sin(np.pi * (k + 0.5) / N)
            for cn in cnames:
                prog.AddConstraint(
                    lambda x, kk=kit, c=cn, c0=clr_m:
                        np.array([kk.point_pos(x, c)[2] - c0]),
                    [0.0], [INF], sv[f"XM_{ph}"][k])

    # 髋高下限（knot+中点）：防蹲蹭伪解——C 门 h_min 防作弊指标前移进
    # NLP（D3 第三解：pelvis 贴地 0.005、准静态零力矩自洽的蹲蹭解）
    h_min = getattr(args, "h_min", 0.62)
    for ph, kit in (("left", kits_l), ("right", kits_r)):
        for k in range(N + 1):
            prog.AddConstraint(
                lambda x, kk=kit: np.array([kk.point_pos(x, "pelvis")[2]]),
                [h_min], [INF], sv[f"X_{ph}"][k])
        for k in range(N):
            prog.AddConstraint(
                lambda x, kk=kit: np.array([kk.point_pos(x, "pelvis")[2]]),
                [h_min], [INF], sv[f"XM_{ph}"][k])

    # 强制真实落地：触地时刻摆动掌点法向（z）速度 ≤ −v_td。软着陆
    # （触地速度≈0）让冲击损耗趋零、最小力矩解退化为准静态蹭步
    # （D3 实测：eps 阶梯在 0.05→0 墙级停滞，cost 0.0047 打满迭代）。
    # v_td 本身作同伦参数（0→0.3）随阶梯爬坡。
    swing_td = {"left": "right", "right": "left"}
    if v_td > 0.0:
        for ph, kit in (("left", kits_l), ("right", kits_r)):
            sw = swing_td[ph]
            cnames = [c for c in kit.contacts if c.startswith(sw)]
            for cn in cnames:
                prog.AddConstraint(
                    lambda x, kk=kit, c=cn: np.array([
                        np.dot(kk.point_jac(x, c)[2, :], x[kk.nq:]) + v_td]),
                    [-INF], [0.0], sv[f"X_{ph}"][N])

    # 相位接口：冲击（冲量方程 + 恢复系数同伦 eps）+ q/v 连续
    imp_lr, nc_lr = make_impact(
        kits_l, [c for c in kits_l.contacts if c.startswith("right")], eps)
    imp_rl, nc_rl = make_impact(
        kits_r, [c for c in kits_r.contacts if c.startswith("left")], eps)
    if_lr, _ = make_interface(Ps[("left", "right")])
    if_rl, _ = make_interface(Ps[("right", "left")])
    prog.AddConstraint(imp_lr, np.zeros(nv_l + nc_lr),
                       np.zeros(nv_l + nc_lr),
                       np.concatenate([sv["X_left"][N], sv["v_aux_lr"],
                                       sv["lam_lr"]]))
    prog.AddConstraint(if_lr, np.zeros(nq_r + nv_r), np.zeros(nq_r + nv_r),
                       np.concatenate([sv["X_left"][N], sv["v_aux_lr"],
                                       sv["X_right"][0]]))
    prog.AddConstraint(imp_rl, np.zeros(nv_r + nc_rl),
                       np.zeros(nv_r + nc_rl),
                       np.concatenate([sv["X_right"][N], sv["v_aux_rl"],
                                       sv["lam_rl"]]))
    prog.AddConstraint(if_rl, np.zeros(nq_l + nv_l), np.zeros(nq_l + nv_l),
                       np.concatenate([sv["X_right"][N], sv["v_aux_rl"],
                                       sv["X_left"][0]]))

    # 速度门：2·step ≥ v_min·(T_L+T_R)
    gate = _make_gate(kits_l, kits_l.step_contact, v_min)
    prog.AddConstraint(gate, [0.0], [INF],
                       np.concatenate([sv["X_left"][N],
                                       [sv["T_left"], sv["T_right"]]]))

    # 成本：速度最大化（步长奖励 + 相位时长惩罚）+ 力矩小正则。D3 续实测：
    # 最小力矩权重 1e-2 下真实量级解（|u|~100 N·m → 力矩成本 ~1e4）在成本上
    # 被零力矩准静态解（成本 ≈0）压死——19:57 轮 stage3 解 tau≈1e-7、
    # v_avg 0.152、A/B 双 FAIL，即「软着陆盆」的根源是成本尺度而非纯盆地
    # 问题。速度项主导（w_step·step 上不封顶、w_time 压 T 到下界）后动态
    # 解才有成本激励；w_tau 默认 1e-2 保旧口径，速度最大化跑法传 1e-4。
    kits_l_ = kits_l
    step_c = kits_l.step_contact

    def progress_cost(z):
        n = kits_l_.nq + kits_l_.nv
        step = kits_l_.point_pos(z[:n], step_c)[0]
        return -args.w_step * step + args.w_time * (z[n] + z[n + 1])

    prog.AddCost(progress_cost,
                 np.concatenate([sv["X_left"][N],
                                 [sv["T_left"], sv["T_right"]]]))
    for ph in ("left", "right"):
        for k in range(N + 1):
            prog.AddCost(np.dot(sv[f"U_{ph}"][k], sv[f"U_{ph}"][k]) * args.w_tau)

    from pydrake.solvers import IpoptSolver
    sid = IpoptSolver().solver_id()
    prog.SetSolverOption(sid, "max_iter", args.max_iter)
    prog.SetSolverOption(sid, "mu_strategy", "adaptive")
    prog.SetSolverOption(sid, "print_level", args.print_level)
    prog.SetSolverOption(sid, "constr_viol_tol", args.constr_viol_tol)
    return prog, sv


def _set_guess(prog, sv, best, kits, N):
    if best is None:
        # 行走形态构造初值（v_td 版）：摆动腿「后蹬→前摆→屈膝前伸落地」，
        # 末端膝仍屈 ~0.25 且在伸展 → 触地法向速度为负（v_td 约束的种子）。
        # 旧版摆动髋/膝末端回 0（直腿+零速触地，直腿处 FK 对膝角不敏感
        # → 触地垂速恒 ≈0），恰是 vmax3/4 卡死的零触地速度流形：从该流形
        # 上热启动（vmax3）或冷启动（vmax4）都无法长出真实触地速度。
        for ph, kit in kits.items():
            nq = kit.nq
            idx = lambda sub: next(i for i, nm in enumerate(kit.pnames)
                                   if sub in nm)
            is_foot = any("ankle" in nm for nm in kit.pnames)
            i_pin = None if is_foot else idx("stance_pin")
            i_knee_s = idx(f"{kit.stance}_knee")
            i_hip_s = idx(f"{kit.stance}_hip")
            ot = "right" if kit.stance == "left" else "left"
            i_hip_w = idx(f"{ot}_hip")
            i_knee_w = idx(f"{ot}_knee")
            i_ankle_w = idx(f"{ot}_ankle") if is_foot else None
            T0 = 0.7
            s = np.linspace(0.0, 1.0, N + 1)
            Q = np.zeros((N + 1, nq))
            if i_pin is not None:
                Q[:, i_pin] = -0.25 + 0.5 * s
            # 摆动髋：后位 −0.25 → 中段前摆峰 → 末端前伸 +0.35 落地
            Q[:, i_hip_w] = -0.25 + 0.60 * s + 0.10 * np.sin(np.pi * s)
            # 摆动膝：蹬地 0.30 → 中段抬腿 0.65 → 末端屈 0.25 且仍伸展
            Q[:, i_knee_w] = 0.30 + 0.35 * np.sin(np.pi * s) - 0.05 * s
            Q[:, i_hip_s] = -0.15 * np.sin(np.pi * s)
            # 支撑膝微屈（0.06→0.16→0.06）：缓冲 + 防直腿奇异。
            # 〔F11〕foot 模式支撑膝相位盒 ≤+0.087（MJCF 屈曲为正、相位反号）
            # → 初值取负弯曲镜像（pointe 旧口径保留正值）
            bend = 0.06 + 0.10 * np.sin(np.pi * s)
            Q[:, i_knee_s] = -bend if is_foot else bend
            if is_foot:
                # 踝：脚保持世界系平贴（leg 绝对角摆动 → 踝角反向补偿）；
                # 摆动腿绝对角 ≈ 髋+膝累计的负值近似，取 −0.3·sin 包络
                Q[:, i_ankle_w] = -0.3 * np.sin(np.pi * s)
            V = np.zeros_like(Q)
            V[1:-1] = (Q[2:] - Q[:-2]) / (2 * T0 / N)
            V[0] = (Q[1] - Q[0]) / (T0 / N)
            V[-1] = (Q[-1] - Q[-2]) / (T0 / N)
            for k in range(N + 1):
                prog.SetInitialGuess(
                    sv[f"X_{ph}"][k], np.concatenate([Q[k], V[k]]))
                prog.SetInitialGuess(sv[f"U_{ph}"][k], np.zeros(kit.nu))
            for k in range(N):
                prog.SetInitialGuess(
                    sv[f"XM_{ph}"][k],
                    np.concatenate([(Q[k] + Q[k + 1]) / 2,
                                    (V[k] + V[k + 1]) / 2]))
        prog.SetInitialGuess(sv["T_left"], 0.7)
        prog.SetInitialGuess(sv["T_right"], 0.7)
        return
    for name, val in best.items():
        var = sv[name]
        if isinstance(var, list):
            for k, v in enumerate(var):
                prog.SetInitialGuess(v, val[k])
        else:
            prog.SetInitialGuess(var, val)


def _parse_stages(s):
    """'clr:v_min[:eps[:pin_lim[:v_td[:slope_deg]]]]' 逗号分隔。eps=恢复系数同伦
    （1=弹性弹跳、0=真实刚性落地）；pin_lim=支撑 pin 转角限幅；v_td=触地
    法向速度下限（0=不限，防软着陆蹭步）；slope_deg=重力倾斜同伦（斜坡
    被动步态种子，0=平地）。省略字段取默认 eps=0、pin_lim=0.7、v_td=0、
    slope=0。"""
    out = []
    for tok in s.split(","):
        parts = tok.split(":")
        clr, v_min = float(parts[0]), float(parts[1])
        eps = float(parts[2]) if len(parts) > 2 else 0.0
        plim = float(parts[3]) if len(parts) > 3 else 0.7
        vtd = float(parts[4]) if len(parts) > 4 else 0.0
        slope = float(parts[5]) if len(parts) > 5 else 0.0
        out.append((clr, v_min, eps, plim, vtd, slope))
    return out


# ------------------------------------------------------------- 子命令

def do_pin(args):
    from pydrake.autodiffutils import InitializeAutoDiff
    kits, Ps = _build_kits(args.model, args.mode, args.sole_drop)
    for stance, kit in kits.items():
        for cname in kit.contacts:
            p = kit.point_pos(np.zeros(kit.nq), cname)
            print(f"[pin] {stance} {cname}: 零位世界坐标 = "
                  f"({float(p[0]):+.4f}, {float(p[1]):+.4f}, "
                  f"{float(p[2]):+.4f})")
        x0 = InitializeAutoDiff(np.zeros(kit.nq + kit.nv + kit.nu))
        xd = kit.xdot(x0)
        vals = np.array([float(v.value()) for v in xd])
        assert np.all(np.isfinite(vals)), f"{stance} xdot 非有限: {vals}"
        M = kit.mass(InitializeAutoDiff(np.zeros(kit.nq)))
        Mv = np.array([[float(c.value()) for c in row] for row in M])
        assert np.all(np.isfinite(Mv)), f"{stance} 质量阵非有限"
        print(f"[pin] {stance}: autodiff xdot/mass OK "
              f"(xdot max|{np.abs(vals).max():.2e}|, M 对称差 "
              f"{np.abs(Mv - Mv.T).max():.1e})")
    for (a, b), P in Ps.items():
        print(f"[pin] P[{a}→{b}]（nq_{b}×nq_{a}）=\n{np.round(P, 4)}")
    print("[pin] 冒烟通过。pointe 零位：双掌点应 ≈ 世界原点，"
          "stance 侧由 pin 结构保证；swing 侧 (0,0,0) 验证 pelvis 高度。")


def _diag_infeasible(kits, Ps, sol, N, tag, eps=0.0):
    """失败级诊断 + 物理审计。返回审计 dict（全 None 异常时）。
    eps 须传本级同伦值：冲击方程审计必须用同口径（对弹性级按刚性判会
    误判——RV3 教训）。"""
    try:
        return _diag_infeasible_impl(kits, Ps, sol, N, tag, eps)
    except Exception as e:
        print(f"[diag:{tag}] 诊断器自身失败（忽略）：{type(e).__name__}: {e}",
              flush=True)
        return None


AUDIT_THRESH = {"hs": 1e-6, "impact": 1e-6, "interface": 1e-6,
                "drift": 2.0, "ke_drop_min": -1e-6}


def _audit_pass(a):
    """物理审计判据：HS/冲击/接口残差严格小 + 相内能量漂移小 + 冲击不产能。
    IPOPT 证书既非必要（近可行点证不出来）也非充分（混叠伪解有证书）——
    以本审计为唯一验收口径（F6/F5 双向教训）。"""
    if a is None:
        return False
    return (a["hs"] < AUDIT_THRESH["hs"] and a["impact"] < AUDIT_THRESH["impact"]
            and a["interface"] < AUDIT_THRESH["interface"]
            and a["drift"] < AUDIT_THRESH["drift"]
            and a["ke_drop"] >= AUDIT_THRESH["ke_drop_min"])


def _diag_infeasible_impl(kits, Ps, sol, N, tag, eps=0.0):
    n = kits["left"].nq + kits["left"].nv
    Tl, Tr = float(sol["T_left"]), float(sol["T_right"])
    audit = {"hs": 0.0, "impact": 0.0, "interface": 0.0, "drift": 0.0,
             "ke_drop": 0.0}
    for ph in ("left", "right"):
        kit = kits[ph]
        X = np.array(sol[f"X_{ph}"], dtype=float)
        U = np.array(sol[f"U_{ph}"], dtype=float)
        XM = np.array(sol[f"XM_{ph}"], dtype=float)
        T = Tl if ph == "left" else Tr
        h = T / N
        # HS 缺陷 + 插值一致性（复刻 _hs_eval 分离式残差）
        def hs_res(k):
            z = np.concatenate([X[k], U[k], XM[k], X[k + 1], U[k + 1], [T]])
            return _hs_eval(kit, z, N, kit.nu)
        hs = max(float(np.abs(hs_res(k)).max()) for k in range(N))
        audit["hs"] = max(audit["hs"], hs)
        # 摆动脚 z 与触地垂速（取摆动侧任一接触点）
        swing = "right" if ph == "left" else "left"
        cn = sorted(c for c in kit.contacts if c.startswith(swing))[0]
        zs = [float(kit.point_pos(X[k][:kit.nq], cn)[2]) for k in range(N + 1)]
        v_td = float(np.dot(kit.point_jac(X[N][:kit.nq], cn)[2, :],
                            X[N][kit.nq:]))
        pz = [float(kit.point_pos(X[k][:kit.nq], "pelvis")[2])
              for k in range(N + 1)]
        print(f"[diag:{tag}] {ph}: T={T:.3f} HS残差max={hs:.2e} "
              f"swing_z[min={min(zs):+.4f},k0={zs[0]:+.4f},kN={zs[-1]:+.4f}] "
              f"v_td={v_td:+.3f} pelvis_z[{min(pz):.3f},{max(pz):.3f}] "
              f"|tau|max={np.abs(U).max():.1f}", flush=True)
    # 冲击 + 接口残差
    for tag2, ph, aux, lam in (("lr", "left", "v_aux_lr", "lam_lr"),
                               ("rl", "right", "v_aux_rl", "lam_rl")):
        kit = kits[ph]
        kit_post = kits["right" if ph == "left" else "left"]
        Pn = Ps[("left", "right") if ph == "left" else ("right", "left")]
        xn = np.array(sol[f"X_{ph}"][-1], dtype=float)
        z = np.concatenate([xn, np.array(sol[aux], dtype=float),
                            np.array(sol[lam], dtype=float)])
        imp, _ = make_impact(
            kit, [c for c in kit.contacts if c.startswith(
                "right" if ph == "left" else "left")], eps)
        r_imp = np.abs(imp(z)).max()
        # 接口残差（变量布局与修复后的 make_interface 一致：x_pre 整块）
        intf, _ = make_interface(Pn)
        x0_post = np.array(sol[f"X_{'right' if ph == 'left' else 'left'}"][0],
                           dtype=float)
        z_if = np.concatenate([xn, np.array(sol[aux], dtype=float), x0_post])
        r_if = np.abs(intf(z_if)).max()
        q_pre = xn[:kit.nq]
        v_pre = xn[kit.nq:]
        M = kit.mass(q_pre)
        ke_pre = 0.5 * float(v_pre @ M @ v_pre)
        ke_aux = 0.5 * float(np.array(sol[aux]) @ M @ np.array(sol[aux]))
        audit["impact"] = max(audit["impact"], float(r_imp))
        audit["interface"] = max(audit["interface"], float(r_if))
        audit["ke_drop"] = min(audit["ke_drop"], ke_pre - ke_aux)
        print(f"[diag:{tag}] 冲击{tag2}: 残差max={r_imp:.2e} "
              f"KE {ke_pre:.3f}→{ke_aux:.3f} 接口残差max={r_if:.2e} "
              f"隐含post_pin={float(Pn[0] @ q_pre):+.3f} "
              f"λ={np.round(np.array(sol[lam], dtype=float), 2)} "
              f"v_aux={np.round(np.array(sol[aux], dtype=float), 2)}",
              flush=True)
    # 能量审计（混叠检测）
    for ph in ("left", "right"):
        kit = kits[ph]
        plant, ctx = kit.plant_f, kit.ctx_f
        X = np.array(sol[f"X_{ph}"], dtype=float)
        es = []
        for k in range(N + 1):
            q, v = X[k][:kit.nq], X[k][kit.nq:]
            plant.SetPositions(ctx, q)
            plant.SetVelocities(ctx, v)
            es.append(0.5 * float(v @ plant.CalcMassMatrix(ctx) @ v)
                      + float(plant.CalcPotentialEnergy(ctx)))
        drift = max(abs(es[k + 1] - es[k]) for k in range(N))
        audit["drift"] = max(audit["drift"], float(drift))
        print(f"[diag:{tag}] 能量审计 {ph}: 相内 knot 间 |dE| max = "
              f"{drift:.3f} J", flush=True)
    print(f"[diag:{tag}] 审计汇总: hs={audit['hs']:.2e} "
          f"impact={audit['impact']:.2e} interface={audit['interface']:.2e} "
          f"drift={audit['drift']:.3f} J ke_drop_min={audit['ke_drop']:.3f} "
          f"→ {'PASS' if _audit_pass(audit) else 'FAIL'}", flush=True)
    return audit


def do_solve(args):
    global OUT_NPZ
    if getattr(args, "out_npz", None):
        OUT_NPZ = args.out_npz
        print(f"[dump] 输出 = {OUT_NPZ}")
    from pydrake.solvers import IpoptSolver

    mode = args.mode
    npts = 1 if mode == "pointe" else 2
    N = args.knots
    stages = _parse_stages(args.stages)
    solver = IpoptSolver()

    best, last_ok = None, False
    kits = Ps = None
    cur_slope = None
    if getattr(args, "guess_npz", None):
        # 跨 run 链式热启动：成本权重（w_step/w_time）是全局 CLI 参数，
        # 长程退火须分多次 run 换权重时，用上一 run 的解做本级种子。
        # 跨 N 重采样：seed 的 knots 与本级不同时按归一化相位插值
        # （knots/中点网格各自均匀；v_aux/lam 维度不变直接沿用）。
        g = np.load(args.guess_npz, allow_pickle=True)
        N_old = int(g["knots"])
        best = {k: g[k] for k in ("v_aux_lr", "v_aux_rl", "lam_lr", "lam_rl")}
        for ph in ("left", "right"):
            Xo = np.array(g[f"X_{ph}"], dtype=float)
            XMo = np.array(g[f"XM_{ph}"], dtype=float)
            Uo = np.array(g[f"U_{ph}"], dtype=float)
            if N_old == N:
                best[f"X_{ph}"] = Xo
                best[f"XM_{ph}"] = XMo
                best[f"U_{ph}"] = Uo
                continue
            so_k = np.linspace(0.0, 1.0, N_old + 1)
            sn_k = np.linspace(0.0, 1.0, N + 1)
            so_m = (np.arange(N_old) + 0.5) / N_old
            sn_m = (np.arange(N) + 0.5) / N
            best[f"X_{ph}"] = np.stack(
                [np.interp(sn_k, so_k, Xo[:, j]) for j in range(Xo.shape[1])],
                axis=1)
            best[f"XM_{ph}"] = np.stack(
                [np.interp(sn_m, so_m, XMo[:, j]) for j in range(XMo.shape[1])],
                axis=1)
            best[f"U_{ph}"] = np.stack(
                [np.interp(sn_k, so_k, Uo[:, j]) for j in range(Uo.shape[1])],
                axis=1)
        best["T_left"] = float(g["T_left"])
        best["T_right"] = float(g["T_right"])
        print(f"[solve] 从 {args.guess_npz} 载入热启动种子"
              f"（knots {N_old}→{N} 重采样）", flush=True)
    for si, (clr, v_min, eps, plim, vtd, slope) in enumerate(stages):
        # 斜度变化才重建相位 plant（gravity_field 属 plant 实例）；
        # best 按变量名热启动，跨斜度传递不受影响
        if kits is None or slope != cur_slope:
            kits, Ps = _build_kits(args.model, mode, args.sole_drop,
                                   slope_deg=slope)
            cur_slope = slope
        prog, sv = _build_stage_prog(kits, Ps, N, npts, clr, v_min,
                                     eps, plim, vtd, args)
        _set_guess(prog, sv, best, kits, N)
        print(f"[solve] stage {si + 1}/{len(stages)}: clr={clr} "
              f"v_min={v_min} eps={eps} pin_lim={plim} v_td={vtd} "
              f"slope={slope}°", flush=True)
        result = solver.Solve(prog)
        ok_ipopt = result.is_success()
        print(f"[solve] stage {si + 1}: success={ok_ipopt} "
              f"cost={result.get_optimal_cost():.4f}", flush=True)
        if not ok_ipopt and args.retries > 0 and best is not None:
            # 混合重启：上级可行解 × 本级失败点各半——失败点常带本级新结构
            # （如刚性冲击下力矩已出现、仅冲击残差 ~4e-3），混合种子介于两
            # 盆之间，常可收口（vmax9b stage8 诊断定向）
            sol_fail = {
                name: (np.array([result.GetSolution(v) for v in var])
                       if isinstance(var, list)
                       else np.array(result.GetSolution(var)))
                for name, var in sv.items()}
            blend = {name: 0.5 * (np.asarray(best[name]) + sol_fail[name])
                     for name in sv}
            prog, sv = _build_stage_prog(kits, Ps, N, npts, clr, v_min,
                                         eps, plim, vtd, args)
            _set_guess(prog, sv, blend, kits, N)
            print(f"[solve] stage {si + 1}: 混合重启（0.5·上级解+0.5·失败点）",
                  flush=True)
            result = solver.Solve(prog)
            ok_ipopt = result.is_success()
            print(f"[solve] stage {si + 1} retry: success={ok_ipopt} "
                  f"cost={result.get_optimal_cost():.4f}", flush=True)
        # 物理审计验收（唯一口径）：IPOPT 证书既非必要（F6 近可行点证不出）
        # 也非充分（F5 混叠伪解有证书）。审计过 = 接受，无论求解器怎么说。
        sol = {name: (np.array([result.GetSolution(v) for v in var])
                      if isinstance(var, list)
                      else result.GetSolution(var))
               for name, var in sv.items()}
        audit = _diag_infeasible(kits, Ps, sol, N, f"s{si + 1}", eps=eps)
        ok = _audit_pass(audit)
        if not ok_ipopt and ok:
            print(f"[solve] stage {si + 1}: IPOPT 未签证书但物理审计 PASS，"
                  "审计验收采纳本解。", flush=True)
        if ok_ipopt and not ok:
            print(f"[solve] stage {si + 1}: IPOPT 有证书但物理审计 FAIL，"
                  "判伪解弃用。", flush=True)
        if not ok:
            if best is None:
                print("[solve] 首级未收敛：先跑 pin 冒烟；再查初值/包络，"
                      "或 --stages 放大首级 clearance。")
                sys.exit(2)
            print("[solve] 该级未收敛，沿用上一级解收尾（降级记录进 diag）。")
            break
        best = sol
        last_ok = ok
        # 级间速览（远程 grep 监控用）：step 取左相位末端右掌点 x，
        # 与 _dump 同口径；|tau| 取双相全 knot 最大
        kl = kits["left"]
        xn = best["X_left"][-1]
        st = float(kl.point_pos(xn[:kl.nq], kl.step_contact)[0])
        tt = float(best["T_left"]) + float(best["T_right"])
        tau_max = max(np.abs(np.array(best["U_left"], dtype=float)).max(),
                      np.abs(np.array(best["U_right"], dtype=float)).max())
        print(f"[solve] stage {si + 1}: v_avg={2.0 * st / tt:.3f} m/s "
              f"step={st:.3f} m T={tt:.3f} s |tau|max={tau_max:.1f} N·m",
              flush=True)

    _dump(kits, Ps, best, N, mode, stages, last_ok)


def _dump(kits, Ps, best, N, mode, stages, last_ok):
    kits_l = kits["left"]
    Xl, Xr = np.array(best["X_left"]), np.array(best["X_right"])
    Ul, Ur = np.array(best["U_left"]), np.array(best["U_right"])
    Tl, Tr = float(best["T_left"]), float(best["T_right"])
    T = Tl + Tr
    cname = kits_l.step_contact
    step = float(kits_l.point_pos(Xl[-1, :kits_l.nq], cname)[0])
    v_avg = 2.0 * step / T
    gate_a = v_avg >= 0.2

    from pydrake.multibody.tree import JointActuatorIndex
    act_names = []
    for i in range(kits_l.plant_f.num_actuators()):
        a = kits_l.plant_f.get_joint_actuator(JointActuatorIndex(i))
        act_names.append(a.joint().name())
    tau_peak = {jn: float(max(np.abs(Ul[:, i]).max(), np.abs(Ur[:, i]).max()))
                for i, jn in enumerate(act_names)}
    gate_b = all(100.0 <= v <= 300.0 for v in tau_peak.values())

    swing_min_z = min(float(kits_l.point_pos(Xl[k, :kits_l.nq], cname)[2])
                      for k in range(N + 1))

    # 冲击 KE 自检：KE_pre − KE_aux（pre 坐标，同 q 同 M）应 ≥ 0；
    # P 一致性：post KE 经 P 映到 post 相位坐标应与 aux KE 一致
    ke = {}
    for tag, ph, aux in (("lr", "left", "v_aux_lr"),
                         ("rl", "right", "v_aux_rl")):
        kit_pre = kits[ph]
        xn = best[f"X_{ph}"][-1]
        q, v_pre = xn[:kit_pre.nq], xn[kit_pre.nq:]
        v_aux = best[aux]
        M = kit_pre.mass(q)
        ke_pre = 0.5 * float(v_pre @ M @ v_pre)
        ke_aux = 0.5 * float(v_aux @ M @ v_aux)
        P = Ps[(("left", "right") if ph == "left" else ("right", "left"))]
        kit_post = kits[("right" if ph == "left" else "left")]
        Mp = kit_post.mass(P @ q)
        ke_post = 0.5 * float((P @ v_aux) @ Mp @ (P @ v_aux))
        ke[tag] = {"pre": ke_pre, "aux": ke_aux, "post_mapped": ke_post,
                   "drop": ke_pre - ke_aux,
                   "p_err": abs(ke_aux - ke_post)}
        print(f"[dump] 冲击 {tag}: KE {ke_pre:.2f}→{ke_aux:.2f} "
              f"(Δ={ke_pre - ke_aux:.2f}, P 映射差 "
              f"{abs(ke_aux - ke_post):.1e})")

    # 能量审计（抗混叠）：相内逐 knot 机械能漂移——HS 配点在采样点自洽
    # 不代表物理正确（F5 混叠伪解：平地零力矩却冲击损耗 45 J/周期）。
    # 守恒动力学下漂移应 ≈ 离散误差（<1 J 量级）；大漂移 = 解作废。
    energy_drift = {}
    for ph in ("left", "right"):
        kit = kits[ph]
        plant, ctx = kit.plant_f, kit.ctx_f
        X = np.array(best[f"X_{ph}"], dtype=float)
        es = []
        for k in range(N + 1):
            q, v = X[k][:kit.nq], X[k][kit.nq:]
            plant.SetPositions(ctx, q)
            plant.SetVelocities(ctx, v)
            es.append(0.5 * float(v @ plant.CalcMassMatrix(ctx) @ v)
                      + float(plant.CalcPotentialEnergy(ctx)))
        drift = max(abs(es[k + 1] - es[k]) for k in range(N))
        energy_drift[ph] = drift
        print(f"[dump] 能量审计 {ph}: 相内 knot 间 |dE| max = {drift:.3f} J"
              f"（>2 J 判混叠伪解，加 knot/收 v-cap 重解）", flush=True)

    print(f"[dump] 输出 = {OUT_NPZ}")
    Path(OUT_NPZ).parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, X_left=Xl, X_right=Xr, U_left=Ul, U_right=Ur,
             XM_left=np.array(best["XM_left"]), XM_right=np.array(best["XM_right"]),
             v_aux_lr=np.array(best["v_aux_lr"]), v_aux_rl=np.array(best["v_aux_rl"]),
             lam_lr=np.array(best["lam_lr"]), lam_rl=np.array(best["lam_rl"]),
             T_left=Tl, T_right=Tr, T=T, step=step, v_avg=v_avg,
             P_lr=Ps[("left", "right")], P_rl=Ps[("right", "left")],
             tau_peak=json.dumps(tau_peak), gate_a=gate_a, gate_b=gate_b,
             swing_min_z=swing_min_z, mode=mode, knots=N,
             ke_drop_lr=ke["lr"]["drop"], ke_drop_rl=ke["rl"]["drop"],
             energy_drift_left=energy_drift["left"],
             energy_drift_right=energy_drift["right"],
             position_names_left=np.array(kits_l.pnames),
             position_names_right=np.array(kits["right"].pnames))
    json.dump({"mode": mode, "knots": N, "T": T, "T_left": Tl,
               "T_right": Tr, "step": step, "v_avg": v_avg,
               "tau_peak": tau_peak, "gate_a": bool(gate_a),
               "gate_b_prelim": bool(gate_b), "swing_min_z": swing_min_z,
               "stages": stages, "last_stage_ok": bool(last_ok),
               "ke": ke}, open(OUT_NPZ.replace(".npz", "_diag.json"), "w"),
              indent=2)
    # 解形态速览（放在落盘之后，只增不拦）：全部坐标列行程 + pelvis 高度
    # 行走 vs 甩腿/劈叉的判读依据
    def q_range(X, i, nm):
        qcol = np.asarray(X)[:, i]
        short = nm.replace("DefaultModelInstance_", "").replace("_q", "")
        return f"{short}[{qcol.min():+.2f},{qcol.max():+.2f}]"
    rng_s = " ".join(q_range(Xl, i, nm)
                     for i, nm in enumerate(kits_l.pnames))
    pl, cl = kits_l._env(False)
    pz = []
    for k in range(N + 1):
        pl.SetPositions(cl, Xl[k, :kits_l.nq])
        pz.append(float(pl.EvalBodyPoseInWorld(
            cl, pl.GetBodyByName("pelvis")).translation()[2]))
    print(f"[dump] 行程: {rng_s} pelvis_z[{min(pz):.3f},{max(pz):.3f}]")
    print(f"[dump] T={T:.3f}s 步长={step:.3f}m 步速={v_avg:.3f}m/s "
          f"A门={'PASS' if gate_a else 'FAIL'} "
          f"峰值|tau|={ {k: round(v, 1) for k, v in tau_peak.items()} } "
          f"swing_min_z={swing_min_z:.3f}")
    print(f"[dump] 已存 {OUT_NPZ} + _diag.json；B 门 verify 复核走 "
          f"to36_leg_to_drake.py（D4 适配）")


def do_check(args):
    """解后自检（读 npz，不起 Drake）：A 门 + 离地 + 冲击 KE。"""
    if getattr(args, "out_npz", None):
        global OUT_NPZ
        OUT_NPZ = args.out_npz
    d = np.load(OUT_NPZ, allow_pickle=True)
    print(f"[check] mode={d['mode']} knots={int(d['knots'])} "
          f"T={float(d['T']):.3f}s（L {float(d['T_left']):.3f} / "
          f"R {float(d['T_right']):.3f}）")
    print(f"[check] 步长={float(d['step']):.3f}m "
          f"步速={float(d['v_avg']):.3f}m/s  "
          f"A门={'PASS' if bool(d['gate_a']) else 'FAIL'}  "
          f"B门(解内)={'PASS' if bool(d['gate_b']) else 'FAIL'}")
    print(f"[check] 峰值|tau|={json.loads(str(d['tau_peak']))}")
    print(f"[check] swing min z = {float(d['swing_min_z']):.4f}（>0 防蹭地）")
    print(f"[check] 冲击 KE 损失 lr={float(d['ke_drop_lr']):.2f} "
          f"rl={float(d['ke_drop_rl']):.2f}（应 ≥0；<0 = 冲击方程坏）")
    if "energy_drift_left" in d:
        el, er = float(d["energy_drift_left"]), float(d["energy_drift_right"])
        ok = "OK" if max(el, er) < 2.0 else "混叠伪解！加 knot/收 v-cap 重解"
        print(f"[check] 能量审计 相内|dE|max L={el:.3f} R={er:.3f} J → {ok}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="原始 G1 URDF（几何/惯量经 D1 平面模型抽取）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pin", help="相位 plant 冒烟（先于 solve）")
    p.add_argument("--mode", choices=("pointe", "foot"), default="pointe")
    p.add_argument("--sole-drop", type=float, default=0.04,
                   help="踝原点→掌点垂距（m）")
    p.set_defaults(fn=do_pin)

    p = sub.add_parser("solve", help="hybrid 双相位周期解（A 门）")
    p.add_argument("--mode", choices=("pointe", "foot"), default="pointe")
    p.add_argument("--knots", type=int, default=14)
    p.add_argument("--max-iter", type=int, default=4000)
    p.add_argument("--print-level", type=int, default=5)
    p.add_argument("--t-min", type=float, default=0.25)
    p.add_argument("--t-max", type=float, default=1.2)
    p.add_argument("--sole-drop", type=float, default=0.04)
    p.add_argument("--pin-limit", type=float, default=0.7,
                   help="支撑 pin 转角限幅（rad），防大回环甩腿解")
    p.add_argument("--h-min", type=float, default=0.62,
                   help="髋高下限（m），防蹲蹭伪解（C 门 h_min 前移）")
    p.add_argument("--td-tol", type=float, default=0.0,
                   help="触地/起飞摆动掌点 z=0 的容差带（m，默认精确等式；"
                        "刚性冲击过渡级放宽 ~2e-3 助 IPOPT 收口——vmax9b 诊断"
                        "失败点距可行仅 4.1e-3 冲击残差）")
    p.add_argument("--constr-viol-tol", type=float, default=1e-4,
                   help="IPOPT constr_viol_tol（默认同 IPOPT 1e-4；刚性过渡"
                        "级可放宽到 1e-2 接受近可行点，物理一致性由 check/"
                        "verify 复核）")
    p.add_argument("--v-cap", type=float, default=20.0,
                   help="关节速度物理限幅（rad/s）。默认 20 防 IPOPT 飞野；"
                        "抗混叠解须按 h·(v/L)≪1 收紧（foot 建议 6）")
    p.add_argument("--retries", type=int, default=0,
                   help="失败级混合重启次数（0=关；1=失败点×上级解各半作"
                        "种子重解一次）")
    p.add_argument("--guess-npz", default=None,
                   help="跨 run 热启动：上一 run 落盘的解 npz（需含 "
                        "v_aux/lam/XM 的新版 dump）")
    p.add_argument("--w-step", type=float, default=3.0,
                   help="步长进度奖励权重（负成本，拉向动态步态）")
    p.add_argument("--w-time", type=float, default=0.0,
                   help="相位时长惩罚权重（正成本，压 T 提步速；速度最大化用）")
    p.add_argument("--w-tau", type=float, default=1e-2,
                   help="力矩正则权重（默认 1e-2=旧最小力矩口径；"
                        "速度最大化跑法建议 1e-4）")
    p.add_argument("--v-td", type=float, default=0.3,
                   help="触地法向速度约束（m/s，强制真实落地防软着陆蹭步）")
    p.add_argument("--stages",
                   default="0.005:0.02:1:1.2,0.01:0.05:0.6:1.0,"
                           "0.02:0.14:0.2:0.8,0.02:0.16:0.1:0.75:0.1,"
                           "0.02:0.18:0.05:0.72:0.2,0.03:0.20:0:0.7:0.3",
                   help="同伦阶梯 clr:v_min:eps:pin_lim:v_td[:slope_deg]"
                        "（eps=恢复系数同伦；v_td=触地法向速度下限；"
                        "slope_deg=重力倾斜同伦：斜坡被动步态种子，退火到 0"
                        "后力矩接管——平地冷/热启动均进不了 v_td 盆"
                        "（vmax3–6 定论））")
    p.add_argument("--out-npz", default=None,
                   help="落盘 npz 路径（默认 apt_g1/outputs/to36_hybrid_gait.npz；"
                        "并行求解必须各自指定——TO37 mid45/mid55 共名覆盖丢失教训）")
    p.set_defaults(fn=do_solve)

    p = sub.add_parser("check", help="解后自检（读 npz）")
    p.add_argument("--out-npz", default=None, help="与 solve --out-npz 对应")
    p.set_defaults(fn=do_check)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
