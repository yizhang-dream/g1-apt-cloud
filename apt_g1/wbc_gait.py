"""QP whole-body controller on the LIPM gait (TO23/TO24).

TO23 v1 findings baked in:
  - dynamics equality MUST include qfrc_passive (joint damping) or the QP
    model diverges from the simulator as velocities build (1.1 s -> 8.3 s fix)
  - contact constraints are SOFT (slack with huge weight): the QP never goes
    primal infeasible around stance switches
  - lam_z lower bound must sit at the lam_z index (v1 bounded lam_x by mistake)

TO24 v2 upgrades:
  - unified two-contact QP: BOTH feet are always modeled. A foot's contact
    state is just per-step parameters: in contact -> fz_lb=5, slack weight
    1e4, lambda nominal mg/2; free -> fz_lb=0, slack weight 1e-2 (constraint
    vacuous), lambda nominal 0. One solver instance, continuous warm start,
    double support falls out for free (both feet in contact).
  - DOUBLE SUPPORT windows around each stance switch (default +-40 ms): the
    soft config's per-cycle CoM sag signature pointed at instantaneous
    support switching with residual touchdown velocity.
  - LATERAL SWAY CoM reference: y_ref = sway * cos(2 pi t / T) leans the CoM
    toward the stance foot; commanding y=0 in single support asks the CoP to
    hold a constant m*g*hip_width lateral moment, which saturates the foot.

Run on the SERVER under .venv_mjlab.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/cvgluser/ros2_data")
sys.path.insert(0, "/home/cvgluser/ros2_data/apt_g1")
sys.path.insert(0, "/home/cvgluser/ros2_data/GR00T-WholeBodyControl")

import mujoco
import casadi as ca

from foot_gait_id import ANKLE_H, HIP_DZ, HIP_H, LEFT_KNEE, RIGHT_KNEE
from eval_torque_gait import EFFORT, SAG, setup as setup5
from lipm_gait_id import build_gait

G = 9.81
MASS = 35.07


def body_vel(model, data, bid):
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
    return v


class LipmMPC:
    """Simplified-model MPC reference layer (TO28).

    Discrete LIPM  cdd = w^2 (c - p)  over a horizon of CoP decisions p_k,
    constrained to the support polygon timeline (SS: stance-foot box,
    DS: hull of both feet). The WBC then tracks the MPC's planned CoM
    (position / velocity / feedforward acceleration) instead of the
    open-loop trapezoid -- the preview lets the CoP go to the foot edge at
    the right time to catch the lateral divergence, which reactive CoP
    tracking at +-0.03 m authority could not.
    """

    def __init__(self, n, dt, w_pos=60.0, w_vel=10.0, w_p=1.0, w_term=200.0):
        self.n, self.dt = n, dt
        self.w_pos, self.w_vel, self.w_p, self.w_term = w_pos, w_vel, w_p, w_term
        p = ca.MX.sym("p", 2 * n)          # [px_0..n-1, py_0..n-1]
        c0 = ca.MX.sym("c0", 2)            # measured com (x, y)
        v0 = ca.MX.sym("v0", 2)
        cref = ca.MX.sym("cref", 2 * n)
        vref = ca.MX.sym("vref", 2 * n)
        pc = ca.MX.sym("pc", 2 * n)        # CoP centers (regularizer)
        pmin = ca.MX.sym("pmin", 2 * n)
        pmax = ca.MX.sym("pmax", 2 * n)
        w2 = ca.MX.sym("w2")
        cost = 0
        c, v = c0, v0
        for k in range(n):
            pk = ca.vertcat(p[k], p[n + k])
            ck = ca.vertcat(cref[k], cref[n + k])
            vk = ca.vertcat(vref[k], vref[n + k])
            cost += w_pos * ca.sumsqr(c - ck) + w_vel * ca.sumsqr(v - vk)                 + w_p * ca.sumsqr(pk - ca.vertcat(pc[k], pc[n + k]))
            cn = c + self.dt * v + 0.5 * self.dt**2 * w2 * (c - pk)
            vn = v + self.dt * w2 * (c - pk)
            c, v = cn, vn
        cost += w_term * (ca.sumsqr(c - ca.vertcat(cref[n - 1], cref[2 * n - 1]))
                          + ca.sumsqr(v - ca.vertcat(vref[n - 1], vref[2 * n - 1])))
        g = ca.vertcat(p - pmin, pmax - p)   # pmin <= p <= pmax
        prob = {"x": p, "f": cost, "g": g,
                "p": ca.vertcat(c0, v0, cref, vref, pc, pmin, pmax, w2)}
        self.solver = ca.qpsol("mpc", "osqp", prob, {
            "error_on_fail": False, "osqp": {"verbose": False, "polish": True}})
        self.prev = None

    def solve(self, c0, v0, cref, vref, pc, pmin, pmax, w2):
        args = {
            "p": np.concatenate([c0, v0, cref, vref, pc, pmin, pmax, [w2]]),
            "lbx": np.full(2 * self.n, -1e9), "ubx": np.full(2 * self.n, 1e9),
            "lbg": np.full(4 * self.n, -1e10), "ubg": np.full(4 * self.n, 1e10),
        }
        if self.prev is not None:
            args["x0"] = self.prev
        r = self.solver(**args)
        p = np.array(r["x"]).flatten()
        self.prev = p
        return p  # [px(0..n-1), py(0..n-1)]


class CentroidalMPC:
    """Centroidal MPC with angular-momentum states (TO29).

    States: [cx, vx, cy, vy, Hx(roll AM), Hz(yaw AM)]
    Inputs: [px, py (CoP), fx, fy (horizontal GRF)]  (fz = m g quasi-static)
    Dynamics:  cdd = f/m
               Hdx = (py-cy) m g + h fy        (roll moment about CoM)
               Hdz = (px-cx) fy - (py-cy) fx   (yaw moment about CoM)
    In the LIPM the force is tied to the CoM-CoP line so these moments
    vanish identically -- which is exactly why the roll channel was
    invisible to the LipmMPC. Here the MPC can plan lateral CoM motion
    WITHOUT roll angular momentum by shifting the CoP off the force line,
    subject to the support-polygon box.
    """

    def __init__(self, n, dt, mass, h_com, mu=0.4,
                 w_pos=60.0, w_vel=10.0, w_H=400.0, w_p=1.0, w_f=0.05, w_term=200.0):
        self.n, self.dt, self.mass, self.h = n, dt, mass, h_com
        nu = 4 * n
        u = ca.MX.sym("u", nu)               # [px_n, py_n, fx_n, fy_n]
        c0 = ca.MX.sym("c0", 2); v0 = ca.MX.sym("v0", 2)
        H0 = ca.MX.sym("H0", 2)              # [Hx, Hz] measured
        cref = ca.MX.sym("cref", 2 * n); vref = ca.MX.sym("vref", 2 * n)
        pc = ca.MX.sym("pc", 2 * n)
        pmin = ca.MX.sym("pmin", 2 * n); pmax = ca.MX.sym("pmax", 2 * n)
        mg = ca.MX.sym("mg")
        cost = 0
        c, v = c0, v0
        Hx, Hz = H0[0], H0[1]
        for k in range(n):
            px, py = u[k], u[n + k]
            fx, fy = u[2 * n + k], u[3 * n + k]
            cost += w_pos * ca.sumsqr(c - ca.vertcat(cref[k], cref[n + k]))                 + w_vel * ca.sumsqr(v - ca.vertcat(vref[k], vref[n + k]))                 + w_H * (Hx * Hx + Hz * Hz)                 + w_p * ca.sumsqr(ca.vertcat(px, py) - ca.vertcat(pc[k], pc[n + k]))                 + w_f * (fx * fx + fy * fy)
            # integrate
            cx1 = c[0] + dt * v[0] + 0.5 * dt**2 * fx / mass
            cy1 = c[1] + dt * v[1] + 0.5 * dt**2 * fy / mass
            vx1 = v[0] + dt * fx / mass
            vy1 = v[1] + dt * fy / mass
            Hx = Hx + dt * ((py - cy1) * mg + self.h * fy)
            Hz = Hz + dt * ((px - cx1) * fy - (py - cy1) * fx)
            c, v = ca.vertcat(cx1, cy1), ca.vertcat(vx1, vy1)
        cost += w_term * (ca.sumsqr(c - ca.vertcat(cref[n - 1], cref[2 * n - 1]))
                          + ca.sumsqr(v - ca.vertcat(vref[n - 1], vref[2 * n - 1]))
                          + w_H / w_term * (Hx * Hx + Hz * Hz))
        g = ca.vertcat(
            u[:2 * n] - ca.vertcat(pmin[:2 * n] if False else 0 * u[:2 * n]),  # placeholder
        )
        # constraints: pmin <= p <= pmax  and  |f| <= mu*mg
        g = ca.vertcat(u[:n] - pmin[:n], pmax[:n] - u[:n],
                       u[n:2 * n] - pmin[n:], pmax[n:] - u[n:2 * n],
                       u[2 * n:] - mu * mg, mu * mg - u[2 * n:])
        prob = {"x": u, "f": cost, "g": g,
                "p": ca.vertcat(c0, v0, H0, cref, vref, pc, pmin, pmax, mg)}
        self.solver = ca.qpsol("cmpc", "osqp", prob, {
            "error_on_fail": False, "osqp": {"verbose": False, "polish": True}})
        self.prev = None

    def solve(self, c0, v0, H0, cref, vref, pc, pmin, pmax, mg):
        nu = 4 * self.n
        args = {
            "p": np.concatenate([c0, v0, H0, cref, vref, pc, pmin, pmax, [mg]]),
            "lbx": np.full(nu, -1e9), "ubx": np.full(nu, 1e9),
            "lbg": np.full(8 * self.n, -1e10), "ubg": np.full(8 * self.n, 1e10),
        }
        if self.prev is not None:
            args["x0"] = self.prev
        r = self.solver(**args)
        st = self.solver.stats().get("return_status", "")
        if "solved" not in st and "uccess" not in st:
            self.n_fail = getattr(self, "n_fail", 0) + 1
            if self.prev is not None:
                return self.prev
        u = np.array(r["x"]).flatten()
        if not np.all(np.isfinite(u)) or np.abs(u).max() > 1e5:
            self.n_fail = getattr(self, "n_fail", 0) + 1
            return np.zeros(4 * self.n)
        self.prev = u
        return u   # [px_n, py_n, fx_n, fy_n]


def lateral_orbit(t, T, d, omega):
    """Periodic LATERAL LIPM orbit (support alternates +-d each T/2).

    y(0)=0, ydot(0)=ydot0=omega*d*tanh(omega*T/4); returns y, ydot, yddot.
    """
    Ts = T / 2.0
    k = int(np.floor(t / Ts))
    tau = t - k * Ts
    pk = d if k % 2 == 0 else -d
    yd0 = omega * d * np.tanh(omega * T / 4.0)
    xi0 = -pk
    xid0 = yd0 if k % 2 == 0 else -yd0
    ch, sh = np.cosh(omega * tau), np.sinh(omega * tau)
    xi = xi0 * ch + (xid0 / omega) * sh
    xid = xi0 * omega * sh + xid0 * ch
    xidd = omega * omega * xi
    return pk + xi, xid, xidd


class WBC2:
    """Unified two-contact task-space ID QP.

    CoP box from the REAL sole geometry (measured in scene_43dof.xml):
    contact spheres at heel x=-0.05 / toe x=+0.12, width y=+-0.03. The
    earlier +-0.05 lateral box let the QP plan CoP outside the real foot ->
    the sole rolls onto its edge in sim (pelvis roll divergence helper).
    """

    def __init__(self, nv, ntau, mu=0.7, cop_x=0.08, cop_y=0.03):
        a = ca.MX.sym("a", nv)
        tau = ca.MX.sym("tau", ntau)
        lam1 = ca.MX.sym("lam1", 6)
        lam2 = ca.MX.sym("lam2", 6)
        s1 = ca.MX.sym("s1", 6)
        s2 = ca.MX.sym("s2", 6)
        x = ca.vertcat(a, tau, lam1, lam2, s1, s2)

        M = ca.MX.sym("M", nv, nv)
        h = ca.MX.sym("h", nv)
        B = ca.MX.sym("B", nv, ntau)
        Jcom = ca.MX.sym("Jcom", 3, nv)
        Jtor = ca.MX.sym("Jtor", 3, nv)
        Jsw = ca.MX.sym("Jsw", 3, nv)
        Jc1 = ca.MX.sym("Jc1", 6, nv)
        Jc2 = ca.MX.sym("Jc2", 6, nv)
        rhs_com = ca.MX.sym("rhs_com", 3)
        rhs_tor = ca.MX.sym("rhs_tor", 3)
        rhs_sw = ca.MX.sym("rhs_sw", 3)
        tau_ff = ca.MX.sym("tau_ff", ntau)
        lam_nom = ca.MX.sym("lam_nom", 12)
        rhs_post = ca.MX.sym("rhs_post", ntau)
        JH = ca.MX.sym("JH", 2, nv)      # centroidal AM jacobian rows (roll, yaw)
        rhs_H = ca.MX.sym("rhs_H", 2)
        w = ca.MX.sym("w", 10)

        lam_cat = ca.vertcat(lam1, lam2)
        f = (
            w[0] * ca.sumsqr(Jcom @ a - rhs_com)
            + w[1] * ca.sumsqr(Jtor @ a - rhs_tor)
            + w[2] * ca.sumsqr(Jsw @ a - rhs_sw)
            + w[3] * ca.sumsqr(tau - tau_ff)
            + w[4] * ca.sumsqr(lam_cat - lam_nom)
            + w[5] * ca.sumsqr(a)
            + w[6] * ca.sumsqr(s1)
            + w[7] * ca.sumsqr(s2)
            # posture task: softly pull ALL body joints toward the planned
            # pose (occupies the null space; without it the waist folds to
            # -86 deg and the free arms flail while the CoM task is happy)
            + w[8] * ca.sumsqr(B.T @ a - rhs_post)
            # TO30 angular-momentum task (column-space form): A_G rows map
            # accelerations to centroidal angular momentum rate; tracking H
            # here is nullspace-consistent, unlike the pelvis-feedforward
            # attempt (TO29) that fought the orientation task.
            + w[9] * ca.sumsqr(JH @ a - rhs_H)
        )
        g_dyn = M @ a + h - B @ tau - Jc1.T @ lam1 - Jc2.T @ lam2
        g_con = ca.vertcat(Jc1 @ a - s1, Jc2 @ a - s2)

        def cone(lam):
            return ca.vertcat(
                lam[0] - mu * lam[2], -lam[0] - mu * lam[2],
                lam[1] - mu * lam[2], -lam[1] - mu * lam[2],
                lam[4] - cop_x * lam[2], -lam[4] - cop_x * lam[2],
                lam[3] - cop_y * lam[2], -lam[3] - cop_y * lam[2],
            )

        g = ca.vertcat(g_dyn, g_con, cone(lam1), cone(lam2))
        nlp = {
            "x": x, "f": f, "g": g,
            "p": ca.vertcat(
                ca.reshape(M, -1, 1), h, ca.reshape(B, -1, 1),
                ca.reshape(Jcom, -1, 1), ca.reshape(Jtor, -1, 1),
                ca.reshape(Jsw, -1, 1), ca.reshape(Jc1, -1, 1), ca.reshape(Jc2, -1, 1),
                rhs_com, rhs_tor, rhs_sw, tau_ff, lam_nom, rhs_post,
                ca.reshape(JH, -1, 1), rhs_H, w,
            ),
        }
        self.solver = ca.qpsol("wbc_qp", "qpoases", nlp, {
            "error_on_fail": False, "printLevel": "none",
        })
        self.nv, self.ntau = nv, ntau
        self.np = nlp["p"].shape[0]
        self.prev = None
        self.last_good = None
        self.n_fail = 0

    def solve(self, p, eff, fz_lb, rhs_con, a_lb=None):
        """fz_lb: (2,) lower bounds for the two feet's lam_z.
        rhs_con: (12,) contact-equality RHS (-vdrift per foot).
        a_lb: optional (nv,) lower bounds on accelerations (joint-limit guard)."""
        nv, nt = self.nv, self.ntau
        lb = np.full(nv + nt + 24, -1e9)
        if a_lb is not None:
            lb[:nv] = a_lb
        lb[nv:nt + nv] = -eff
        lb[nv + nt + 2] = fz_lb[0]        # lam1 z
        lb[nv + nt + 8] = fz_lb[1]        # lam2 z
        ub = np.full(nv + nt + 24, 1e9)
        ub[nv:nt + nv] = eff
        lbg = np.concatenate([np.zeros(nv), rhs_con, np.full(16, -1e10)])
        # TO32 bug fix: the 16 cone rows (friction cone + CoP box per foot)
        # used to get ub=+1e10 -- VACUOUS. Every QP since the WBC2 refactor
        # solved with completely unconstrained contact wrenches (lambda was
        # only regularized toward nominal), i.e. the QP could hallucinate
        # forces MuJoCo's real feet never provide. Enforce cone(...) <= 0.
        ubg = np.concatenate([np.zeros(nv), rhs_con, np.zeros(16)])
        args = {"p": p, "lbx": lb, "ubx": ub, "lbg": lbg, "ubg": ubg}
        if self.prev is not None:
            args["x0"] = self.prev
        r = self.solver(**args)
        status = self.solver.stats().get("return_status", "")
        if "uccess" not in status and "solved" not in status:
            self.n_fail += 1
            if self.last_good is not None:
                return self.last_good
        x = np.array(r["x"]).flatten()
        self.prev = x
        tau = x[nv:nv + nt]
        out = (tau, x[nv + nt:nv + nt + 6], x[nv + nt + 6:nv + nt + 12])
        self.last_good = out
        return out


def main(v=0.3, T=0.5, z_c=0.70, seconds=10.0, seed=0,
         w_com=100.0, w_torso=60.0, w_sw=100.0, w_tau=1e-3, w_lam=1e-3,
         w_a=1e-3, w_con_on=1e4, w_con_off=1e-2,
         kp_com=20.0, kd_com=8.0, kp_torso=60.0, kd_torso=15.0,
         kp_sw=150.0, kd_sw=30.0, ff_scale=1.0, h_clear=0.06,
         ds_win=0.04, sway=0.03, hip_h=HIP_H,
         w_post=30.0, kp_post=40.0, kd_post=10.0, knee_guard=0.0,
         k_place_x=0.8, k_place_y=0.8, stance_widen=0.0, mpc=0,
         mpc_horizon=0.8, mpc_dt=0.04, kd_mom=0.0,
         w_h_task=0.0, kd_h=1.0, foot_halfy=0.03):
    np.random.seed(seed)
    model, data, qpos_adr, dof_adr, act_ids = setup5()
    nv = model.nv
    ntau = len(dof_adr)

    X, Xd, Q, Qd, tau_tab = build_gait(v, T, z_c, h_clear, hip_h=hip_h)
    n = tau_tab.shape[0]
    Xdd = np.gradient(Xd, T / n)  # LIPM fore-aft accel feedforward
    phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
    S = v * T
    L = S / 4.0

    B = np.zeros((nv, ntau))
    for i, d in enumerate(dof_adr):
        B[d, i] = 1.0
    lam_nom_half = np.array([0.0, 0.0, MASS * G / 2, 0.0, 0.0, 0.0])
    lam_zero = np.zeros(6)

    LFOOT = model.body("left_ankle_roll_link").id
    RFOOT = model.body("right_ankle_roll_link").id
    # TO24: orientation task anchors the PELVIS (the leg root), not the chest:
    # v2 tracked torso_link and the pelvis was free to tip (29 deg within 0.2 s
    # of single support while the chest stayed upright via waist counter-bend).
    TORSO = model.body("pelvis").id

    data.qpos[0:3] = [0.0, 0.0, hip_h + HIP_DZ]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = Q[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    com0 = data.subtree_com[0].copy()
    foot_y = {LFOOT: float(data.xpos[LFOOT][1]) * (1.0 + stance_widen),
              RFOOT: float(data.xpos[RFOOT][1]) * (1.0 + stance_widen)}
    com_z_ref = float(com0[2])
    sway_sign = 1.0 if foot_y[LFOOT] > 0 else -1.0

    wbc = WBC2(nv, ntau, cop_y=foot_halfy)
    mpc_n = max(4, int(round(mpc_horizon / mpc_dt)))
    mpc_solver = None
    if mpc == 1:
        mpc_solver = LipmMPC(mpc_n, mpc_dt)
    elif mpc == 2:
        mpc_solver = CentroidalMPC(mpc_n, mpc_dt, MASS, com_z_ref)
    om2 = G / z_c

    def gait_phase(tt):
        kh = int(tt // (T / 2.0))
        th_ = tt - kh * (T / 2.0)
        st, sw = (LFOOT, RFOOT) if kh % 2 == 0 else (RFOOT, LFOOT)
        ids = th_ < ds_win or th_ > T / 2.0 - ds_win
        return kh, th_, st, sw, ids

    def plan_boxes(tt):
        # support-polygon box (xmin,xmax,ymin,ymax) + CoP center at time tt
        kh, th_, st, sw, ids = gait_phase(tt)
        ys, yw = foot_y[st], foot_y[sw]
        xs = L + (S / 2.0) * kh
        if ids:
            ylo, yhi = min(ys, yw) - foot_halfy, max(ys, yw) + foot_halfy
            xlo, xhi = min(xs, 0.0) - 0.08, max(xs, 0.0) + 0.12
            return xlo, xhi, ylo, yhi, 0.5 * (xs + 0.0), 0.5 * (ys + yw)
        return xs - 0.08, xs + 0.12, ys - foot_halfy, ys + foot_halfy, xs, ys

    dt_ctrl = 0.02
    n_steps = int(seconds / dt_ctrl)
    x0 = float(data.qpos[0])
    h_min = float(data.qpos[2])
    h_sum, h_n = 0.0, 0
    fall = None

    for step in range(n_steps):
        t = step * dt_ctrl
        phi = 2.0 * np.pi * (t % T / T)
        th = t % (T / 2.0)
        com_x_ref = v * t + float(np.interp(phi, phases, X))
        com_vx_ref = float(np.interp(phi, phases, Xd))

        k_half = int(t // (T / 2.0))  # half-cycle index (fix: th<T/2 was always true)
        if k_half % 2 == 0:
            stance_foot, swing_foot = LFOOT, RFOOT
            toe_x, heel_x = S / 2.0 + L - S, S / 2.0 + L
        else:
            stance_foot, swing_foot = RFOOT, LFOOT
            toe_x, heel_x = L, S + L
        # TO27 trapezoid sway (quasi-static weight transfer): transfer the CoM
        # to the stance foot's lateral position DURING the double-support
        # window (both feet down -> wide support polygon, CoP authority is
        # easy), then HOLD it over the stance foot through single support.
        # The earlier sin profile kept the CoM moving through SS = permanently
        # off-equilibrium; hold + transfer is the classical recipe.
        ramp = min(1.0, t / T)  # fade in over the first cycle
        sstp = lambda x: 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
        y_st = foot_y[stance_foot] * sway * ramp
        y_ot = foot_y[swing_foot] * sway * ramp
        if th < ds_win:                      # DS early: other -> stance
            a = sstp(th / max(ds_win, 1e-6))
            com_y_ref, com_vy_ref = y_ot + (y_st - y_ot) * a, 0.0
        elif th > T / 2.0 - ds_win:          # DS late: stance -> other
            a = sstp((th - (T / 2.0 - ds_win)) / max(ds_win, 1e-6))
            com_y_ref, com_vy_ref = y_st + (y_ot - y_st) * a, 0.0
        else:                                # SS: hold over the stance foot
            com_y_ref, com_vy_ref = y_st, 0.0
        com_ay_ff = 0.0

        if mpc_solver is not None:
            # TO28: replace the open-loop CoM references with the LIPM-MPC plan
            mujoco.mj_forward(model, data)
            mujoco.mj_subtreeVel(model, data)
            c0m = data.subtree_com[0][:2].copy()
            v0m = data.subtree_linvel[0][:2].copy()
            cref = np.zeros(2 * mpc_n)
            vref = np.zeros(2 * mpc_n)
            pc = np.zeros(2 * mpc_n)
            pmin = np.zeros(2 * mpc_n)
            pmax = np.zeros(2 * mpc_n)
            for k in range(mpc_n):
                tk = t + k * mpc_dt
                cref[k] = v * tk
                vref[k] = v
                kh2, th2, st2, sw2, ids2 = gait_phase(tk)
                ramp2 = min(1.0, tk / T)
                ys2 = foot_y[st2] * sway * ramp2
                yo2 = foot_y[sw2] * sway * ramp2
                if th2 < ds_win:
                    yr2 = yo2 + (ys2 - yo2) * sstp(th2 / max(ds_win, 1e-6))
                elif th2 > T / 2.0 - ds_win:
                    yr2 = ys2 + (yo2 - ys2) * sstp((th2 - (T / 2.0 - ds_win)) / max(ds_win, 1e-6))
                else:
                    yr2 = ys2
                cref[mpc_n + k] = yr2
                xlo, xhi, ylo, yhi, pcx, pcy = plan_boxes(tk)
                pmin[k], pmax[k] = xlo, xhi
                pmin[mpc_n + k], pmax[mpc_n + k] = ylo, yhi
                pc[k], pc[mpc_n + k] = pcx, pcy
            if mpc == 2:
                H0m = np.array([data.subtree_angmom[0][0], data.subtree_angmom[0][2]])
                uk = mpc_solver.solve(c0m, v0m, H0m, cref, vref, pc, pmin, pmax, MASS * G)
                f0 = np.array([uk[2 * mpc_n], uk[3 * mpc_n]])
                c1 = c0m + mpc_dt * v0m + 0.5 * mpc_dt**2 * f0 / MASS
                v1 = v0m + mpc_dt * f0 / MASS
                com_x_ref, com_vx_ref = float(c1[0]), float(v1[0])
                com_y_ref, com_vy_ref = float(c1[1]), float(v1[1])
                mpc_ax, mpc_ay = float(f0[0] / MASS), float(f0[1] / MASS)
            else:
                pk = mpc_solver.solve(c0m, v0m, cref, vref, pc, pmin, pmax, om2)
                p0 = np.array([pk[0], pk[mpc_n]])
                c1 = c0m + mpc_dt * v0m + 0.5 * mpc_dt**2 * om2 * (c0m - p0)
                v1 = v0m + mpc_dt * om2 * (c0m - p0)
                com_x_ref, com_vx_ref = float(c1[0]), float(v1[0])
                com_y_ref, com_vy_ref = float(c1[1]), float(v1[1])
                a_ff_mpc = om2 * (c0m - p0)
                mpc_ax, mpc_ay = float(a_ff_mpc[0]), float(a_ff_mpc[1])
        else:
            mpc_ax, mpc_ay = None, None

        in_ds = th < ds_win or th > T / 2.0 - ds_win
        feet = [LFOOT, RFOOT]
        contact = {f: (in_ds or f == stance_foot) for f in feet}

        u = (th % (T / 2.0)) / (T / 2.0)
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        # TO26 reactive foot placement (capture stepping): the landing target
        # absorbs the current CoM error so the NEXT stance catches the fall
        # (continuous CoP authority alone cannot stabilize the lateral
        # pendulum during single support on these narrow feet)
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        com_now = data.subtree_com[0].copy()
        comv_now = data.subtree_linvel[0].copy()
        ex = com_now[0] - com_x_ref
        ey = com_now[1] - com_y_ref
        evy = comv_now[1] - com_vy_ref
        heel_eff = heel_x + k_place_x * ex
        sw_y_ref = foot_y  # placeholder replaced below
        sw_x = toe_x + (heel_eff - toe_x) * s
        sw_z = ANKLE_H + h_clear * (1.0 - np.cos(2.0 * np.pi * u)) / 2.0
        ds_du = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
        sw_vx = (heel_x - toe_x) * ds_du * 2.0 / T
        sw_vz = h_clear * np.sin(2.0 * np.pi * u) * np.pi / (T / 2.0)

        mujoco.mj_forward(model, data)
        qvel = data.qvel.copy()
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M, data.qM)
        h = (data.qfrc_bias - data.qfrc_passive).copy()
        Jcom = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(model, data, Jcom, 0)
        Jcs, rhs_con, fz_lb = [], np.zeros(12), np.zeros(2)
        for fi, f in enumerate(feet):
            Jl = np.zeros((3, nv)); Jr = np.zeros((3, nv))
            mujoco.mj_jacBody(model, data, Jl, Jr, f)
            Jc = np.vstack([Jl, Jr])
            Jcs.append(Jc)
            vf = body_vel(model, data, f)
            rhs_con[6 * fi:6 * fi + 6] = -(vf - Jc @ qvel)
            fz_lb[fi] = 5.0 if contact[f] else 0.0
        sw_y_tgt = foot_y[swing_foot] + k_place_y * (ey + 0.25 * evy) if not in_ds else 0.0
        Jsw_lin = np.zeros((3, nv))
        if not in_ds:
            Jl = np.zeros((3, nv)); _r = np.zeros((3, nv))
            mujoco.mj_jacBody(model, data, Jl, _r, swing_foot)
            Jsw_lin = Jl
        Jtor_lin = np.zeros((3, nv)); Jtor_rot = np.zeros((3, nv))
        mujoco.mj_jacBody(model, data, Jtor_lin, Jtor_rot, TORSO)

        mujoco.mj_subtreeVel(model, data)
        com = data.subtree_com[0].copy()
        com_v = data.subtree_linvel[0].copy()
        v_drift_com = com_v - Jcom @ qvel
        v_sw = body_vel(model, data, swing_foot)
        v_drift_sw = v_sw[:3] - Jsw_lin @ qvel
        wq, xq, yq, zq = data.xquat[TORSO]
        pitch = np.arcsin(np.clip(2.0 * (wq * yq - zq * xq), -1, 1))
        roll = np.arctan2(2.0 * (wq * xq + yq * zq), 1.0 - 2.0 * (xq**2 + yq**2))
        yaw = np.arctan2(2.0 * (wq * zq + xq * yq), 1.0 - 2.0 * (yq**2 + zq**2))
        v_torso = body_vel(model, data, TORSO)[3:]
        v_drift_tor = v_torso - Jtor_rot @ qvel

        ax_ff = mpc_ax if mpc_ax is not None else float(np.interp(phi, phases, Xdd))
        ay_ff = mpc_ay if mpc_ay is not None else com_ay_ff
        a_com_cmd = np.array([
            ax_ff + kp_com * (com_x_ref - com[0]) + kd_com * (com_vx_ref - com_v[0]),
            ay_ff + kp_com * (com_y_ref - com[1]) + kd_com * (com_vy_ref - com_v[1]),
            kp_com * (com_z_ref - com[2]) + kd_com * (0.0 - com_v[2]),
        ]) - v_drift_com
        # TO29 angular-momentum damping (flywheel): the MPC plans H~0 but the
        # WBC needs an execution channel -- measured whole-body angular
        # momentum about the CoM feeds the pelvis task as extra damping,
        # resisting the swing-leg reaction that drove ROLL/YAW divergence.
        Hx_m = float(data.subtree_angmom[0][0])
        Hz_m = float(data.subtree_angmom[0][2])
        a_tor_cmd = np.array([
            kp_torso * (0.0 - roll) + kd_torso * (0.0 - v_torso[0]) - kd_mom * Hx_m,
            kp_torso * (0.0 - pitch) + kd_torso * (0.0 - v_torso[1]),
            kp_torso * (0.0 - yaw) + kd_torso * (0.0 - v_torso[2]) - kd_mom * Hz_m,
        ]) - v_drift_tor
        sw_pos = data.xpos[swing_foot].copy()
        if in_ds:
            a_sw_cmd = np.zeros(3)
        else:
            a_sw_cmd = np.array([
                kp_sw * (sw_x - sw_pos[0]) + kd_sw * (sw_vx - v_sw[0]),
                kp_sw * (sw_y_tgt - sw_pos[1]) + kd_sw * (0.0 - v_sw[1]),
                kp_sw * (sw_z - sw_pos[2]) + kd_sw * (sw_vz - v_sw[2]),
            ]) - v_drift_sw

        tau_ff = np.zeros(ntau)
        for k, j in enumerate(SAG):
            tau_ff[j] = ff_scale * float(np.interp(phi, phases, tau_tab[:, k]))

        q_cur = data.qpos[qpos_adr].copy()
        qd_cur = data.qvel[dof_adr].copy()
        q_post_ref = np.array([np.interp(phi, phases, Q[:, j])
                               for j in range(Q.shape[1])])
        rhs_post = kp_post * (q_post_ref - q_cur) - kd_post * qd_cur
        w_vec = np.array([w_com, w_torso, w_sw, w_tau, w_lam, w_a,
                          w_con_on, w_con_on, w_post, w_h_task])
        if w_h_task > 0.0:
            # A_G roll/yaw rows: angular momentum is linear in qvel, so each
            # column = subtree_angmom with qvel = e_j (49 cheap mj_subtreeVel)
            qvel_save = data.qvel.copy()
            JH = np.zeros((2, nv))
            for j in range(nv):
                data.qvel[:] = 0.0
                data.qvel[j] = 1.0
                mujoco.mj_subtreeVel(model, data)
                JH[0, j] = data.subtree_angmom[0][0]   # roll AM
                JH[1, j] = data.subtree_angmom[0][2]   # yaw AM
            data.qvel[:] = qvel_save
            mujoco.mj_subtreeVel(model, data)
            H_meas2 = np.array([data.subtree_angmom[0][0], data.subtree_angmom[0][2]])
            rhs_H = -kd_h * H_meas2   # regulate centroidal AM to the MPC plan (~0)
        else:
            JH = np.zeros((2, nv))
            rhs_H = np.zeros(2)
        for fi, f in enumerate(feet):
            if not contact[f]:
                w_vec[6 + fi] = w_con_off
        lam_nom = np.concatenate([
            lam_nom_half if contact[LFOOT] else lam_zero,
            lam_nom_half if contact[RFOOT] else lam_zero,
        ])

        p = np.concatenate([
            M.reshape(-1, order="F"), h, B.reshape(-1, order="F"),
            Jcom.reshape(-1, order="F"), Jtor_rot.reshape(-1, order="F"),
            Jsw_lin.reshape(-1, order="F"),
            Jcs[0].reshape(-1, order="F"), Jcs[1].reshape(-1, order="F"),
            a_com_cmd, a_tor_cmd, a_sw_cmd, tau_ff, lam_nom, rhs_post,
            JH.reshape(-1, order="F"), rhs_H, w_vec,
        ])
        assert p.shape[0] == wbc.np, (p.shape[0], wbc.np)
        # TO26 joint-limit guard: when a knee nears full extension, forbid
        # further extension acceleration in the QP (the swing-down PD used to
        # push the knee through 0 into hyperextension; the limit impact force
        # is unmodeled -- same class of bug as qfrc_passive)
        a_lb = np.full(nv, -1e9)
        if knee_guard > 0.0:
            for k in (LEFT_KNEE, RIGHT_KNEE):
                if float(data.qpos[qpos_adr[k]]) < knee_guard:
                    a_lb[dof_adr[k]] = 0.0
        tau, lam1, lam2 = wbc.solve(p, EFFORT, fz_lb, rhs_con, a_lb)
        if step % 10 == 0 or step < 60:
            sl = wbc.prev
            o1 = nv + ntau
            s1n = float(np.linalg.norm(sl[o1 + 12:o1 + 18])) if sl is not None else -1
            s2n = float(np.linalg.norm(sl[o1 + 18:o1 + 24])) if sl is not None else -1
            waist = float(data.qpos[qpos_adr[12]])
            pelv_quat = data.xquat[model.body("pelvis").id]
            wp, xp, yp, zp = pelv_quat
            pelv_pitch = np.degrees(np.arcsin(np.clip(2 * (wp * yp - zp * xp), -1, 1)))
            pelv_roll = np.degrees(np.arctan2(2 * (wp * xp + yp * zp), 1 - 2 * (xp**2 + yp**2)))
            pelv_yaw = np.degrees(np.arctan2(2 * (wp * zp + xp * yp), 1 - 2 * (yp**2 + zp**2)))
            sat_names = [model.joint(model.actuator_trnid[act_ids[j], 0]).name
                         for j in np.argsort(-np.abs(tau))[:3] if abs(tau[j]) >= EFFORT[j] - 0.5]
            print(f"    [d] t={t:.2f} slack=({s1n:.3f},{s2n:.3f}) waist={waist:+.2f} "
                  f"pelv=({pelv_roll:+5.1f}R,{pelv_pitch:+5.1f}P,{pelv_yaw:+5.1f}Y) z_err={com[2]-com_z_ref:+.3f} "
                  f"sw@({sw_pos[0]:+.2f},{sw_pos[1]:+.2f},{sw_pos[2]:.3f})"
                  f"err=({sw_pos[0]-sw_x:+.2f},{sw_pos[1]-foot_y[swing_foot]:+.2f},{sw_pos[2]-sw_z:+.2f})"
                  f" tgt=({sw_x:.2f},{sw_z:.3f}) "
                  f"swq=(h{q_cur[0 if swing_foot==LFOOT else 6]:+.2f},k{q_cur[3 if swing_foot==LFOOT else 9]:+.2f}) "
                  f"satN={'+'.join(sat_names) if sat_names else '-'}", flush=True)
        ctrl = np.zeros(model.nu)
        ctrl[act_ids] = tau
        data.ctrl[:] = ctrl
        for _ in range(4):
            mujoco.mj_step(model, data)
        h_min = min(h_min, float(data.qpos[2]))
        h_sum += float(data.qpos[2]); h_n += 1
        if not np.all(np.isfinite(data.qpos)):
            fall = t
            break
        if step % 25 == 0 or step == n_steps - 1:
            sat = int(np.sum(np.abs(tau) >= EFFORT - 0.5))
            print(f"  t={t:5.2f} com=({com[0]-com_x_ref:+.2f},{com[1]-com_y_ref:+.2f}) "
                  f"vx={com_v[0]:+.2f}(ref {com_vx_ref:+.2f}) pitch={np.degrees(pitch):+5.1f} "
                  f"sw_err={np.linalg.norm(sw_pos-np.array([sw_x,foot_y[swing_foot],sw_z])):.2f} "
                  f"lam=({lam1[2]:5.0f},{lam2[2]:5.0f}) sat={sat} h={data.qpos[2]:.3f}"
                  f"{' DS' if in_ds else ''}", flush=True)
        if float(data.qpos[2]) < 0.2:
            fall = t
            break

    disp = float(data.qpos[0] - x0)
    nn = n_steps if fall is None else max(1, step + 1)
    print(f"=== TO24 QP-WBC v2 (v={v} T={T} ds={ds_win} sway={sway} "
          f"kp=({kp_com},{kp_torso},{kp_sw}) {seconds}s) ===")
    print(f"  fall={fall}  h_min={h_min:.3f}  h_mean={h_sum/max(1,h_n):.3f}  disp={disp:+.2f}m  "
          f"vx={disp/(nn*dt_ctrl):+.2f} m/s  qp_fails={wbc.n_fail}/{nn}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--z-c", type=float, default=0.70)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-com", type=float, default=100.0)
    ap.add_argument("--w-torso", type=float, default=60.0)
    ap.add_argument("--w-sw", type=float, default=100.0)
    ap.add_argument("--w-tau", type=float, default=1e-3)
    ap.add_argument("--w-lam", type=float, default=1e-3)
    ap.add_argument("--kp-com", type=float, default=20.0)
    ap.add_argument("--kp-torso", type=float, default=60.0)
    ap.add_argument("--kp-sw", type=float, default=150.0)
    ap.add_argument("--ff-scale", type=float, default=1.0)
    ap.add_argument("--ds-win", type=float, default=0.04)
    ap.add_argument("--sway", type=float, default=1.0,
                    help="lateral sway scale x (foot_offset - 5cm), 0 = centered")
    ap.add_argument("--hip-h", type=float, default=HIP_H,
                    help="gait hip height (TO25: raise toward 0.68 = straighter knees)")
    ap.add_argument("--w-post", type=float, default=30.0)
    ap.add_argument("--kp-post", type=float, default=40.0)
    ap.add_argument("--kd-post", type=float, default=5.0)
    ap.add_argument("--knee-guard", type=float, default=0.0,
                    help="forbid knee extension accel below this angle (0=off; "
                         "hard bound proved harmful -- QP distorts other tasks)")
    ap.add_argument("--k-place-x", type=float, default=0.8)
    ap.add_argument("--k-place-y", type=float, default=0.8)
    ap.add_argument("--stance-widen", type=float, default=0.0,
                    help="scale foot lateral targets (0.3 = 30%% wider stance)")
    ap.add_argument("--mpc", type=int, default=0,
                    help="MPC reference layer: 1 = LIPM (TO28), 2 = centroidal+angular "
                         "momentum (TO29, adds the roll/yaw channel)")
    ap.add_argument("--mpc-horizon", type=float, default=0.8)
    ap.add_argument("--mpc-dt", type=float, default=0.04)
    ap.add_argument("--kd-mom", type=float, default=0.0,
                    help="angular-momentum damping on the pelvis task (TO29)")
    ap.add_argument("--w-h-task", type=float, default=0.0,
                    help="TO30: centroidal angular-momentum task weight (0 = off)")
    ap.add_argument("--kd-h", type=float, default=1.0)
    ap.add_argument("--foot-halfy", type=float, default=0.03,
                    help="TO31 widened-feet sim test: lateral CoP half-width; "
                         "must match the APT_SCENE variant's sole (real = 0.03)")
    a = ap.parse_args()
    main(a.v, a.T, a.z_c, a.seconds, a.seed, w_com=a.w_com, w_torso=a.w_torso,
         w_sw=a.w_sw, w_tau=a.w_tau, w_lam=a.w_lam,
         kp_com=a.kp_com, kp_torso=a.kp_torso, kp_sw=a.kp_sw,
         ff_scale=a.ff_scale, ds_win=a.ds_win, sway=a.sway, hip_h=a.hip_h,
         w_post=a.w_post, kp_post=a.kp_post, kd_post=a.kd_post,
         knee_guard=a.knee_guard, k_place_x=a.k_place_x, k_place_y=a.k_place_y,
         stance_widen=a.stance_widen, mpc=a.mpc,
         mpc_horizon=a.mpc_horizon, mpc_dt=a.mpc_dt, kd_mom=a.kd_mom,
         w_h_task=a.w_h_task, kd_h=a.kd_h, foot_halfy=a.foot_halfy)
