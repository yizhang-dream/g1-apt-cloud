"""Isaac Lab (DirectRLEnv) flat-ground APT env for Unitree G1.

Pipeline mirrors the MuJoCo ``apt_g1.envs.mujoco_g1_flat_env`` loop:

    command + proprio history -> frozen phase router -> 64-d token
    token + 10-frame history  -> frozen SONIC ONNX decoder -> 29-d joint targets
    q_des[lower 12] += aux_scale * aux            (aux learned by RL)

Implemented on top of ``isaaclab.envs.DirectRLEnv`` so we can train with
thousands of parallel GPU environments (the missing piece in MuJoCo) and add
the paper's RL-stage mechanisms (2 Hz gait-gate hold + feedback obs, latent KL,
exploration-bonus decay) from the trainer side.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaacsim.core.utils.torch.rotations import quat_rotate_inverse

from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
from gear_sonic.envs.manager_env.robots import g1
from gear_sonic.envs.manager_env.robots.g1 import G1_MUJOCO_TO_ISAACLAB_DOF

from apt_g1.encoder import Command
from apt_g1.isaac.batched_router import BatchedPhaseRouter
from apt_g1.isaac.elevation_map import (
    build_env_grids,
    sample_elevation,
    wrap_height_function,
)
from apt_g1.isaac.sonic_decoder_torch import SonicTorchDecoder
from apt_g1.isaac.to42_gate import To42Gate


# ---------------------------------------------------------------------------
# SONIC constants (copied from apt_g1.envs.mujoco_g1_flat_env to avoid the
# mujoco dependency in the Isaac venv). MuJoCo joint order 0-11 = legs,
# 12-14 = waist, 15-28 = arms.
# ---------------------------------------------------------------------------
SONIC_ACTION_SCALE_MUJOCO = np.array(
    [
        0.3506614664, 0.3506614664, 0.5475464652, 0.3506614664,
        0.4385773139, 0.4385773139, 0.3506614664, 0.3506614664,
        0.5475464652, 0.3506614664, 0.4385773139, 0.4385773139,
        0.5475464652, 0.4385773139, 0.4385773139, 0.4385773139,
        0.4385773139, 0.4385773139, 0.4385773139, 0.4385773139,
        0.0745008703, 0.0745008703, 0.4385773139, 0.4385773139,
        0.4385773139, 0.4385773139, 0.4385773139, 0.0745008703,
        0.0745008703,
    ],
    dtype=np.float32,
)

SONIC_DEFAULT_ANGLES_MUJOCO = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)

# Lower-body aux joints (12): both legs' hip yaw/roll/pitch, knee,
# ankle pitch/roll -- indices in G1_ISAACLab_ORDER (SONIC order).
LOWER_AUX_NAMES = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def _sonic_default_isaac() -> np.ndarray:
    """SONIC default angles in G1_ISAACLab_ORDER (29,)."""
    return SONIC_DEFAULT_ANGLES_MUJOCO[G1_MUJOCO_TO_ISAACLAB_DOF].astype(np.float32)


def _sonic_scale_isaac() -> np.ndarray:
    """SONIC action scales in G1_ISAACLab_ORDER (29,)."""
    return SONIC_ACTION_SCALE_MUJOCO[G1_MUJOCO_TO_ISAACLAB_DOF].astype(np.float32)


@configclass
class AptFlatG1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s: float = 20.0
    decimation: int = 4
    action_space: int = 14  # phase(2) + aux(12)
    observation_space: int = 91
    state_space: int = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=4)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=True
    )

    # robot (29-DOF G1 body, gear_sonic physics config)
    robot: ArticulationCfg = g1.G1_CYLINDER_MODEL_12_DEX_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # APT / SONIC
    use_sonic_prior: bool = True
    sonic_decoder_path: str = (
        "gear_sonic_deploy/policy/release/model_decoder.onnx"
    )
    router_model_dir: str = "apt_g1/outputs/distill_final"
    aux_scale: float = 0.2
    aux_clip: float = 1.0
    aux_l2_scale: float = 0.0
    aux_rate_scale: float = 0.0
    yaw_sigma2: float = 0.25
    vel_sigma2: float = 0.25
    progress_scale: float = 0.0  # forward-progress bonus (forces traversal)
    anti_stop_scale: float = 0.0  # penalty for vx < 0.3 (anti-idle/anti-backward)
    anti_stop_thresh: float = 0.3
    # E44v3: direct yaw-rate penalty (kills the spin gait that decoder fine-tune
    # otherwise discovers; track_xy/progress use body-frame vx which a turning
    # robot still scores high on).
    yaw_rate_penalty: float = 0.0
    phase_mode: bool = False  # True: policy selects phase (joint RL)
    phase_anchor: bool = False  # True: phase = router clock + bounded offset
    phase_anchor_ema: bool = True  # True: EMA-smooth the router clock (like encode_batch)
    phase_offset_scale: float = 0.15  # max offset magnitude when anchored
    latent_mode: bool = False  # True: policy outputs 16-d latent -> VAE -> token -> SONIC
    # E44: decoder fine-tuning. Policy action = 29-d normalized joint targets
    # (decoder output space); the policy owns the trainable SONIC decoder, the
    # env only applies q_des = default + action*scale. obs adds 930-d proprio
    # history + 2-d walk-clock phase (see decft_policy.py layout).
    decft_mode: bool = False
    latent_vae_path: str = ""  # frozen TokenWindowVAE checkpoint
    latent_phase_rate: float = 0.0  # 0 -> read from pca.npz (walk cadence)
    # E28: command-conditioned gait cadence. When True, the latent phase clock
    # advances per-env at base_rate * clamp(cmd_vx / ref, 0, max) instead of a
    # fixed scalar, letting the policy speed up/slow down the cycle to track the
    # commanded vx (the frozen decoder D(z, phi) already generalizes over phi).
    latent_cmd_phase_rate: bool = False
    latent_phase_rate_ref: float = 0.6  # cmd vx at which cadence == base walk rate
    latent_phase_rate_max: float = 2.0  # clamp on (cmd_vx / ref) multiplier
    stillness_vx_scale: float = 0.05  # forward-speed (vx^2) penalty weight in stillness
    # E31: speed-conditioned VAE decoder. When True the frozen decoder takes a
    # speed bin derived from the commanded vx (D(z, phase, v_bin) -> token), so
    # the manifold itself encodes gait speed (vs E27/E29's single-speed manifold).
    latent_speed_bins: bool = False
    latent_vae_n_bins: int = 3
    # E35: direction+speed-conditioned VAE decoder. When True the frozen decoder
    # takes (z, phase, v_bin, psi_bin) where psi_bin is the 8-bin heading of the
    # commanded direction (bin 4 = +x forward, build_exp3_dataset formula). The
    # heading is fully determined by the command condition, so the policy's z
    # only picks speed — fixing E31's systematic yaw drift.
    latent_dir_bins: bool = False
    latent_vae_n_dbins: int = 8
    # E48: full-joint residual escape channel (RuN/ReSkill-style,
    # LITERATURE_SURVEY_FROZEN_DECODER.md solution 2). In latent mode the
    # action becomes [z(16), res(29)] and q_des = q_decoder(z) +
    # res_scale * clamp(res) over ALL 29 joints -- vs the old weak aux
    # (12-d lower body, scale 0.2) that E15-E22b showed is never enough.
    latent_residual: bool = False
    # E49: direct-token RL. Policy action = raw 64-d token coordinates
    # (UNBOUNDED -- no tanh, no FSQ quantization); the env maps
    # token = token_mean + token_alpha * token_std * a and feeds the frozen
    # SONIC decoder directly (no VAE). The walk clock still free-runs at the
    # latent arm's cadence; token_phase_obs (E49-B) appends [sin phi, cos phi]
    # of that clock to the policy obs (attribution arm: phi as obs only,
    # never a reward or an output prior).
    token_mode: bool = False
    token_phase_obs: bool = False
    token_alpha: float = 1.0
    token_stats: str = ""  # npz with mean/std/rate from official g1-mode tokens
    token_std_floor: float = 1e-3
    # E49-A-tanh: restricted-range ablation (anti-drift stabilizer). The
    # unbounded arm found walking in <=50 iters but PPO then drifted off the
    # decodable token region and the gait collapsed (s0: it_50 3/3 29m ->
    # final 0/3 backward shuffle; s1 diverged even faster). "tanh" re-bounds
    # the mapping to mean +/- alpha*std per dim -- the pre-registered
    # restricted-range arm from the protocol, repurposed as the immediate fix.
    token_bound: str = "none"  # "none" (unbounded) | "tanh"
    res_scale: float = 0.4  # rad; max joint-target offset from the prior
    res_clip: float = 1.0
    res_l2_scale: float = 0.0  # penalty on the raw residual (keep prior dominant)
    # E48c: zero the residual for the first N control steps (z-head learns a
    # working controller first; the residual is only freed afterwards --
    # ReSkill-style residual on top of a working base policy).
    res_freeze_steps: int = 0
    # E32: heading/velocity-direction reward. yaw_scale multiplies the base
    # track_yaw term; heading_scale adds exp(-(vy/vx vs cmd heading)^2)-style
    # alignment reward to fight the high-speed yaw drift E31 showed.
    yaw_scale: float = 0.5  # default = E31 base behavior
    heading_scale: float = 0.0

    # TO38: TO36 F11b reference injection (obs + cmd-gated tracking reward).
    # The reference clock psi free-runs at the solution's natural stride
    # period T -- it deliberately does NOT share the decoder walk clock
    # (pca cadence ~0.49 s/cycle vs TO 2.4 s/stride; forcing them together
    # would drag the decoder gait off the VAE manifold). obs block (12):
    # [sin psi, cos psi, q_ref6_rel, pitch, z, heel_x_rel, heel_z].
    to_ref: bool = False  # master flag: appends a 12-d obs block (zeros if no npz)
    to_ref_npz: str = ""  # LUT from apt_g1/to38_export_ref.py; "" -> zero block (paired control arm)
    # control arm: load the LUT anyway so the clock and tracking diagnostics
    # run identically, but zero the obs block (paired isolation)
    to_ref_obs_zero: bool = False
    to_ref_w: float = 0.0  # tracking reward weight (0 = reward off)
    to_ref_sigma2: float = 0.1  # tracking kernel width (rad^2)
    to_ref_gate2: float = 0.0036  # cmd-proximity gate sigma^2 (default 0.06^2)
    # TO40-C: gated torque feedforward (paper hybrid-control analog, leg level).
    # tau_ff(psi) from the same LUT, applied ONLY when cmd is near the solution
    # speed (gate reuse) -- injection as position offset dq = w*tau/kp on the 6
    # sagittal joints (implicit PD semantics guaranteed; effort superposition
    # left for later). Obs dim unchanged -> to38b stays a valid control.
    to_tau: bool = False
    to_tau_w: float = 1.0
    # NOTE: kp must match the **actuator stiffness actually used** (implicit PD:
    # dq 精确等价 τ_ff 的前提是 kp 用对)。腿每侧排序 [hip_pitch, hip_roll,
    # hip_yaw, knee, ...]，踝的 stiffness = 2×STIFFNESS_5020 = 28.50125，不是
    # hip_yaw 的 40.17924（曾错位，已修正）。运行优先读 sim 实际值（见 init）。
    to_tau_kp: tuple = (99.09843, 99.09843, 28.50125,
                        99.09843, 99.09843, 28.50125)  # Lh,Lk,La,Rh,Rk,Ra

    # privileged local elevation map (teacher-style terrain observation)
    use_elevation: bool = False
    elev_grid: int = 9          # grid_n x grid_n patch
    elev_res: float = 0.15      # meters per cell
    elev_lookahead: float = 0.6  # patch center ahead of the robot (m)

    # learned gait/group selection (paper's gait logit analog)
    use_gate_sel: bool = False
    gate_groups: tuple = ((0, -1.0, 4), (1, 0.2, 4), (2, -1.0, 4))  # idle/slow/walk fwd

    # TO42: learned regime selection on the frozen decoder substrate
    # (TO42_PLAN §3)。"off" = 行为与 TO41 canonical 逐位一致；"lsel" = 策略
    # Bernoulli 位在 2 Hz 决策边界采纳（边界间锁存 0.5 s，gate 布尔只在真切换
    # 步为 True，语义同 _gate_tick）；"fbkt" = selection 槽位每步写
    # clamp(bucketize(cmd), 0, 1)、gate 恒 False、策略位被忽略（配对基线臂；
    # eval 网格 v≤0.325<0.533 上与冻结自然分配逐位一致）。obs 追加
    # [sel_state, gate_bool] 两维（两臂一致），action 追加 sel 位一维（16→17）。
    to42_sel: str = "off"
    to42_hold_steps: int = 25  # 25 control steps @ 50 Hz = 0.5 s
    to42_n_sel: int = 2        # Rung 1 selector 值域 {vb0, vb1}（vb2 不进）

    # command sampling (paper ranges come later; start MuJoCo-parity)
    vx_min: float = 0.0
    vx_max: float = 0.8
    vy_min: float = 0.0
    vy_max: float = 0.0
    yaw_min: float = 0.0
    yaw_max: float = 0.0

    # 2 Hz gait-gate hold (paper: gait selection at 2 Hz, decoder held 0.5 s)
    use_2hz_gate: bool = True
    gate_hold_steps: int = 25  # 25 control steps @ 50 Hz = 0.5 s

    # disturbance (MuJoCo C2 semantics: scheduled once per episode at reset)
    disturbance_prob: float = 0.0
    disturbance_force_range: tuple[float, float] = (200.0, 500.0)

    # terminations
    fall_height: float = 0.2
    termination_penalty: float = -10.0
    reset_grace_steps: int = 25  # ignore falls right after reset (terrain init collisions)


class AptFlatG1Env(DirectRLEnv):
    cfg: AptFlatG1EnvCfg

    def __init__(self, cfg: AptFlatG1EnvCfg, render_mode: str | None = None, **kwargs):
        self._sonic_default = _sonic_default_isaac()
        self._sonic_scale = _sonic_scale_isaac()
        self._body_names = G1_ISAACLab_ORDER
        self._lower_aux_idx = torch.tensor(
            [self._body_names.index(n) for n in LOWER_AUX_NAMES], dtype=torch.long
        )
        # record the exact height grid before the terrain is generated
        self._elev_key = None
        if cfg.use_elevation and cfg.terrain.terrain_type == "generator":
            sub = cfg.terrain.terrain_generator.sub_terrains
            if len(sub) == 1:
                name = next(iter(sub))
                if hasattr(sub[name], "noise_range"):
                    noise = float(sub[name].noise_range[1])
                    self._elev_key = (cfg.terrain.terrain_generator.seed, noise)
                    wrap_height_function(sub[name], self._elev_key)
                # non-random terrains (stairs/stones/discrete): no grid to wrap,
                # the observation falls back to an all-zero elevation patch.
                # NOTE: observation_space is bumped by the train/eval scripts
                # (before policy creation); do not bump here to avoid double-add.
        self._sonic_default_t = None  # set after device known
        self._sonic_scale_t = None
        super().__init__(cfg, render_mode, **kwargs)
        if self._elev_key is not None:
            self._elev_grids, self._elev_origins, self._elev_hscale = build_env_grids(
                self.terrain, self._elev_key, self.num_envs
            )
            self._elev_grids = self._elev_grids.to(self.device)
            self._elev_origins = self._elev_origins.to(self.device)
        else:
            self._elev_grids = None

        self._sonic_default_t = torch.from_numpy(self._sonic_default).to(self.device)
        self._sonic_scale_t = torch.from_numpy(self._sonic_scale).to(self.device)

        self._body_idx = torch.tensor(
            [self.robot.joint_names.index(n) for n in self._body_names],
            dtype=torch.long,
            device=self.device,
        )
        self._all_dof = torch.arange(self.robot.num_joints, device=self.device)
        # root body index for disturbance
        self._root_body_idx, _ = self.robot.find_bodies("pelvis")

        # TO38: sagittal joints in SONIC/MuJoCo order (L hip/knee/ankle pitch,
        # R hip/knee/ankle pitch) -- the to_ref LUT columns are already in this
        # order with the B-gate sign map applied (to38_export_ref.py).
        self._sag_idx = torch.tensor([0, 3, 4, 6, 9, 10], dtype=torch.long, device=self.device)
        self._to_q = self._to_scal = self._to_def_sag = self._to_tau = None
        if self.cfg.to_ref or self.cfg.to_tau:
            self._to_phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            if self.cfg.to_ref_npz:
                lut = np.load(self.cfg.to_ref_npz)
                T = float(lut["T"])
                self._to_m = lut["q_ref6"].shape[0]
                self._to_rate = math.tau / (T * self.cfg.sim.dt * self.cfg.decimation)
                self._to_vavg = float(lut["v_avg"])
                self._to_q = torch.from_numpy(lut["q_ref6"]).float().to(self.device)
                self._to_scal = torch.from_numpy(
                    np.concatenate([lut["pitch"][:, None], lut["z"][:, None], lut["heel_rel"]], 1)
                ).float().to(self.device)
                self._to_tau = torch.from_numpy(lut["tau_ref6"]).float().to(self.device)
                # TO40-C: kp 用 sim 实际执行器 stiffness（隐式 PD 的真实增益），
                # 而非硬编码；只读默认值，不 set 回（不改动基座动力学）。读取失败
                # 回退 cfg.to_tau_kp 并打印警告。
                # default_joint_stiffness 按 sim 的 URDF joint 顺序排，
                # **不是** SONIC 顺序；因此要按关节名逐一取回，重排到 sagittal6
                # [Lhip,Lknee,Lankle,Rhip,Rknee,Rankle]（与 LUT 列序一致）。
                _sag_names = ["left_hip_pitch_joint", "left_knee_joint",
                              "left_ankle_pitch_joint", "right_hip_pitch_joint",
                              "right_knee_joint", "right_ankle_pitch_joint"]
                _sim_kp = None
                if hasattr(self.robot.data, "default_joint_stiffness") and \
                        self.robot.data.default_joint_stiffness is not None:
                    _stiff = self.robot.data.default_joint_stiffness[0]
                    try:
                        _uids = [self.robot.joint_names.index(n) for n in _sag_names]
                        _sim_kp = _stiff[_uids]
                    except ValueError:
                        _sim_kp = None
                if _sim_kp is not None and torch.isfinite(_sim_kp).all() and \
                        (_sim_kp > 0).all():
                    self._to_kp = _sim_kp.to(torch.float32).to(self.device)
                    print(f"[to40c] kp from sim (sagittal6) = "
                          f"{[round(float(v), 4) for v in self._to_kp]}")
                else:
                    self._to_kp = torch.tensor(
                        list(self.cfg.to_tau_kp), dtype=torch.float32, device=self.device)
                    print("[to40c] WARN: sim default_joint_stiffness 不可用，回退 cfg "
                          f"{list(self.cfg.to_tau_kp)}")
                self._to_def_sag = self._sonic_default_t[self._sag_idx]

        if self.cfg.use_sonic_prior:
            self._decoder = SonicTorchDecoder(self.cfg.sonic_decoder_path, device=self.device)
            self._router = BatchedPhaseRouter(self.cfg.router_model_dir, device=self.device)
            self._vae = None
            if self.cfg.latent_mode or self.cfg.decft_mode or self.cfg.token_mode:
                import numpy as _np

                # walk-clock cadence (latent and decft modes drive the VAE
                # phase from this clock; the VAE itself is only loaded in
                # latent mode -- decft loads it inside DecFtPolicy). E49 token
                # mode runs the SAME clock (fixed cadence from the stats npz)
                # so clock state stays comparable to E45 even though no VAE
                # consumes it.
                if self.cfg.token_mode:
                    stats = _np.load(self.cfg.token_stats)
                    self._token_mean = torch.as_tensor(
                        _np.asarray(stats["mean"], dtype=_np.float32),
                        device=self.device,
                    )
                    std = _np.maximum(
                        _np.asarray(stats["std"], dtype=_np.float32),
                        self.cfg.token_std_floor,
                    )
                    self._token_std = torch.as_tensor(std, device=self.device)
                    self._latent_phase_rate = float(
                        self.cfg.latent_phase_rate or float(stats["rate"])
                    )
                else:
                    pca = _np.load(
                        str(Path(self.cfg.latent_vae_path).parent / "pca.npz")
                    )
                    self._latent_phase_rate = float(
                        self.cfg.latent_phase_rate or pca["rate"]
                    )
                self._latent_phase = torch.zeros(
                    self.num_envs, dtype=torch.float32, device=self.device
                )
            if self.cfg.latent_mode:
                from apt_g1.isaac.token_window_vae import (
                    DirSpeedPhaseTokenVAE,
                    PhaseTokenVAE,
                    SpeedPhaseTokenVAE,
                )

                if self.cfg.latent_dir_bins:
                    vae = DirSpeedPhaseTokenVAE(
                        n_vbins=self.cfg.latent_vae_n_bins,
                        n_dbins=self.cfg.latent_vae_n_dbins,
                    ).to(self.device)
                elif self.cfg.latent_speed_bins:
                    vae = SpeedPhaseTokenVAE(n_bins=self.cfg.latent_vae_n_bins).to(self.device)
                else:
                    vae = PhaseTokenVAE().to(self.device)
                vae.load_state_dict(
                    torch.load(self.cfg.latent_vae_path, map_location=self.device),
                    strict=False,  # checkpoint also carries the encoder; only the decoder is needed
                )
                vae.eval()
                self._vae = vae
            self._router_state = (
                self._router.reset_state(self.num_envs)
                if not self.cfg.phase_mode
                else None
            )
            self._modes_list = torch.tensor(
                sorted({md["group"][0] for md in self._router.meta.values()}), dtype=torch.long
            )
        else:
            self._decoder = None
            self._router = None
            self._router_state = None
            self._modes_list = torch.tensor([0], dtype=torch.long)
        self._gate_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._held_groups_np: np.ndarray | None = None
        self._gate_tick = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gate_sel = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.cfg.use_gate_sel:
            self._gate_gi = [
                self._router.gmap.get(g, -1) for g in self.cfg.gate_groups
            ]
            assert all(gi >= 0 for gi in self._gate_gi), (
                f"gate groups not in router: {self.cfg.gate_groups}"
            )

        # TO42: per-env regime selector state machine（与 selftest 同一份代码）
        self._to42 = None
        if self.cfg.to42_sel != "off":
            assert self.cfg.latent_mode and not self.cfg.decft_mode, (
                "to42 selection rides on the latent decode path only")
            self._to42 = To42Gate(
                self.num_envs, self.device,
                hold_steps=self.cfg.to42_hold_steps,
                mode=self.cfg.to42_sel,
                vx_max=self.cfg.vx_max,
                n_bins=self.cfg.latent_vae_n_bins,
                n_sel=self.cfg.to42_n_sel,
            )

        # history buffers (oldest -> newest along dim 1)
        self._hist_ang_vel = torch.zeros(
            self.num_envs, 10, 3, dtype=torch.float32, device=self.device
        )
        self._hist_joint_pos = torch.zeros(
            self.num_envs, 10, 29, dtype=torch.float32, device=self.device
        )
        self._hist_joint_vel = torch.zeros(
            self.num_envs, 10, 29, dtype=torch.float32, device=self.device
        )
        self._hist_last_actions = torch.zeros(
            self.num_envs, 10, 29, dtype=torch.float32, device=self.device
        )
        self._hist_gravity = torch.zeros(
            self.num_envs, 10, 3, dtype=torch.float32, device=self.device
        )

        self._commands = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.router_commands: list[Command | None] = [None] * self.num_envs
        self._last_phase = torch.zeros(
            self.num_envs,
            64 if self.cfg.token_mode else (16 if self.cfg.latent_mode else 2),
            dtype=torch.float32,
            device=self.device,
        )
        self._phase_ema = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self._phase_groups = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._anchor_prev = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self._anchor_groups = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._last_aux = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        self._prev_aux = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        self._aux_rate = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        self._last_res = torch.zeros(self.num_envs, 29, dtype=torch.float32, device=self.device)
        self._q_des = torch.zeros(self.num_envs, 29, dtype=torch.float32, device=self.device)
        # E49: 复位前终末状态观测（_reset_idx 在 super() 前截留），惰性分配
        self._final_obs = None
        # E49 诊断步骤③：最近一个控制步的奖励分项快照（_get_rewards 内写入；
        # 训练侧 --diag-log 开启时逐步读取 GPU 累积）
        self._last_rew_terms = None
        self._disturb = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._disturb_dir = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._disturb_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------- env internals
    def _commands_from_cfg(self, env_ids: torch.Tensor) -> torch.Tensor:
        n = len(env_ids)
        vx = torch.empty(n, device=self.device).uniform_(self.cfg.vx_min, self.cfg.vx_max)
        vy = torch.empty(n, device=self.device).uniform_(self.cfg.vy_min, self.cfg.vy_max)
        yaw = torch.empty(n, device=self.device).uniform_(self.cfg.yaw_min, self.cfg.yaw_max)
        return torch.stack([vx, vy, yaw], dim=1)

    def _build_commands_list(self) -> list[Command]:
        out = []
        for i in range(self.num_envs):
            if self.router_commands[i] is not None:
                out.append(self.router_commands[i])
            else:
                out.append(
                    Command.from_vxvy(
                        float(self._commands[i, 0]),
                        float(self._commands[i, 1]),
                        float(self._commands[i, 2]),
                    )
                )
        return out

    def _router_groups(self, cmds) -> np.ndarray:
        if self._router is None:
            return np.zeros(self.num_envs, dtype=np.int64)
        if self.cfg.use_2hz_gate and self._held_groups_np is not None:
            return self._held_groups_np
        return self._router.select_groups(cmds)

    def _proprio_np(self) -> np.ndarray:
        """930-d proprio vector (N, 930) in the router/decoder layout."""
        parts = [
            self._hist_ang_vel,
            self._hist_joint_pos,
            self._hist_joint_vel,
            self._hist_last_actions,
            self._hist_gravity,
        ]
        return (
            torch.cat([p.reshape(self.num_envs, -1) for p in parts], dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _proprio_t(self) -> torch.Tensor:
        """930-d proprio history as a GPU tensor (decft obs component)."""
        parts = [
            self._hist_ang_vel,
            self._hist_joint_pos,
            self._hist_joint_vel,
            self._hist_last_actions,
            self._hist_gravity,
        ]
        return torch.cat([p.reshape(self.num_envs, -1) for p in parts], dim=1)

    def _base_lin_vel(self) -> torch.Tensor:
        return self.robot.data.root_lin_vel_b

    def _base_ang_vel(self) -> torch.Tensor:
        return self.robot.data.root_ang_vel_b

    def _world_to_body(self, vec_w: torch.Tensor) -> torch.Tensor:
        return quat_rotate_inverse(
            self.robot.data.root_quat_w, vec_w.reshape(-1, 3)
        )

    def _decoder_obs_parts(self, tokens) -> tuple[torch.Tensor, ...]:
        """History tensors in the layout the torch decoder expects.

        tokens 接受 numpy 数组（router/phase 路径，维持原 from_numpy 搬运）
        或已在 GPU 上的 tensor（latent 路径，零拷贝直通，消除每控制步
        GPU->CPU->GPU 往返；dtype/数值逐位不变）。
        """
        if isinstance(tokens, torch.Tensor):
            tok = tokens.detach().to(self.device)
        else:
            tok = torch.from_numpy(tokens).to(self.device)
        return (
            tok,
            self._hist_ang_vel,
            self._hist_joint_pos,
            self._hist_joint_vel,
            self._hist_last_actions,
            self._hist_gravity,
        )

    def _push_history(self, ang_vel, joint_pos_rel, joint_vel, last_actions, gravity):
        def _push(buf: torch.Tensor, value: torch.Tensor):
            buf[:, :-1] = buf[:, 1:].clone()
            buf[:, -1] = value.detach()

        _push(self._hist_ang_vel, ang_vel)
        _push(self._hist_joint_pos, joint_pos_rel)
        _push(self._hist_joint_vel, joint_vel)
        _push(self._hist_last_actions, last_actions)
        _push(self._hist_gravity, gravity)

    def _compute_q_des(
        self, phase: torch.Tensor, aux: torch.Tensor, res: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Decode token prior + aux (+ optional full-joint residual) into (N, 29) targets."""
        cmds = self._build_commands_list()
        # 热循环去同步：930 维 proprio 历史要经过一次 GPU->CPU 搬运，只在真正
        # 消费它的分支里计算——latent 分支只吃策略 z + walk clock（VAE 不看
        # proprio），phase 分支仅 anchor 子分支经 phase_raw_batch（numpy API）
        # 消费，router 分支经 encode_batch（numpy API）消费。
        if self.cfg.latent_mode:
            # E27/E28: the policy's latent z (passed as "phase") is decoded by
            # the frozen phase-conditioned VAE at the current walk-clock phase.
            with torch.no_grad():
                phi = self._latent_phase
                sc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)
                if self.cfg.latent_dir_bins:
                    # E35: speed bin from commanded vx + direction bin from the
                    # commanded heading (atan2 of cmd vx/vy, bin 4 = +x fwd).
                    n = self.cfg.latent_vae_n_bins
                    cmd_v = self._commands[:, 0]
                    edges = torch.linspace(0.0, self.cfg.vx_max, n + 1)[1:-1].to(cmd_v.device)
                    if self._to42 is not None:
                        vb = self._to42.state  # TO42: 锁存的 selector 状态
                    else:
                        vb = torch.bucketize(cmd_v, edges).clamp(0, n - 1)
                    ang = torch.atan2(self._commands[:, 1], self._commands[:, 0])
                    db = torch.floor((ang + math.pi) / (2.0 * math.pi) * 8).long() % 8
                    # decode 结果保持 GPU tensor 直送 _decoder_obs_parts，
                    # 去掉 .cpu().numpy() + from_numpy 的每步往返（数值逐位不变）
                    tokens = self._vae.decode(phase, sc, vb, db).detach()
                elif self.cfg.latent_speed_bins:
                    # E31: pick the speed bin from the commanded vx (bins
                    # trained on walk phase-rate thirds: slow/mid/fast).
                    n = self.cfg.latent_vae_n_bins
                    cmd_v = self._commands[:, 0]
                    edges = torch.linspace(0.0, self.cfg.vx_max, n + 1)[1:-1].to(cmd_v.device)
                    if self._to42 is not None:
                        vb = self._to42.state  # TO42: 锁存的 selector 状态
                    else:
                        vb = torch.bucketize(cmd_v, edges).clamp(0, n - 1)
                    tokens = self._vae.decode(phase, sc, vb).detach()
                else:
                    tokens = self._vae.decode(phase, sc).detach()
                if self.cfg.latent_cmd_phase_rate:
                    # E28: advance the clock per-env scaled by commanded vx so
                    # the gait cadence can track the speed command (the policy
                    # sees cmd vx as obs[0] and picks z accordingly).
                    mult = torch.clamp(
                        self._commands[:, 0] / self.cfg.latent_phase_rate_ref,
                        min=0.0,
                        max=self.cfg.latent_phase_rate_max,
                    )
                    self._latent_phase = (phi + self._latent_phase_rate * mult) % math.tau
                else:
                    # E27: fixed scalar walk cadence.
                    self._latent_phase = (phi + self._latent_phase_rate) % math.tau
        elif self.cfg.token_mode:
            # E49: the policy's raw 64-d action maps LINEARLY onto the
            # official token stats (a unbounded -- alpha sets the initial
            # exploration scale only, the reachable range stays open). The
            # frozen decoder consumes the token directly. The walk clock
            # still advances so the E49-B phase obs carries the same phi the
            # latent arm feeds its VAE at the matching control step.
            with torch.no_grad():
                phi = self._latent_phase
                if self.cfg.token_bound == "tanh":
                    tokens = (
                        self._token_mean
                        + self.cfg.token_alpha * self._token_std * torch.tanh(phase)
                    )
                else:
                    tokens = (
                        self._token_mean
                        + self.cfg.token_alpha * self._token_std * phase
                    )
                self._latent_phase = (phi + self._latent_phase_rate) % math.tau
        elif self.cfg.phase_mode:
            groups = self._router_groups(cmds)
            tokens = np.zeros((self.num_envs, 64), dtype=np.float32)
            # temporal EMA smoothing of the policy-selected phase (like the router)
            p = phase.detach()
            g_t = torch.from_numpy(groups).to(self.device)
            changed = g_t != self._phase_groups
            if changed.any():
                self._phase_ema[changed] = p[changed]
            else:
                p = 0.3 * self._phase_ema + 0.7 * p
            self._phase_ema = p
            self._phase_groups = g_t
            if self.cfg.phase_anchor:
                # anchored readout: keep the frozen PhaseNet's gait clock and let
                # the policy only nudge a bounded offset (phase = normalize(clock
                # + scale * offset)). This preserves full walk-cadence rotation
                # that free-phase RL (E23/E24) failed to reproduce.
                proprio = self._proprio_np()  # phase_raw_batch 为 numpy API
                sc_r, _ = self._router.phase_raw_batch(
                    proprio, cmds, force_groups=groups
                )
                sc_r = torch.from_numpy(sc_r).to(self.device)
                if self.cfg.phase_anchor_ema:
                    # mirror the frozen router's encode_batch EMA (0.3 old + 0.7
                    # new); reset the EMA where the router group changed.
                    g_ch = torch.from_numpy(groups).to(self.device)
                    changed = g_ch != self._anchor_groups
                    if changed.any():
                        self._anchor_prev[changed] = sc_r[changed]
                    else:
                        sc_r = 0.3 * self._anchor_prev + 0.7 * sc_r
                    self._anchor_prev = sc_r.clone()
                    self._anchor_groups = g_ch
                pn = p / p.norm(dim=1, keepdim=True).clamp_min(1e-6)
                sc = sc_r + self.cfg.phase_offset_scale * pn
                sc = sc / sc.norm(dim=1, keepdim=True).clamp_min(1e-6)
                p_np = sc.cpu().numpy()
            else:
                p_np = p.cpu().numpy()
            for gi in self._router.nets:
                mask = groups == gi
                if not mask.any():
                    continue
                phi = np.arctan2(p_np[mask, 0], p_np[mask, 1])
                n_bins = self._router.n_bins[gi]
                # continuous readout: interpolate the two nearest prototypes by
                # the fractional phase (token stays on the prototype hull and
                # phase->token becomes smooth, giving RL gradients; validated in
                # MuJoCo: interp_router_flat.json walk 3/3).
                x = (phi + np.pi) / (2.0 * np.pi) * n_bins
                b0 = np.floor(x).astype(np.int64) % n_bins
                frac = (x - np.floor(x)).astype(np.float32)[:, None]
                b1 = (b0 + 1) % n_bins
                protos_t = self._router.protos[gi]
                tokens[mask] = (1.0 - frac) * protos_t[b0] + frac * protos_t[b1]
        else:
            proprio = self._proprio_np()  # encode_batch 为 numpy API
            tokens, self._router_state = self._router.encode_batch(
                proprio, cmds, state=self._router_state, ema=0.3,
                force_groups=self._router_groups(cmds),
            )
        action_t = self._decoder.decode(*self._decoder_obs_parts(tokens))
        default = self._sonic_default_t
        scale = self._sonic_scale_t
        q_des = default + action_t * scale
        aux_c = torch.clamp(aux, -self.cfg.aux_clip, self.cfg.aux_clip)
        q_des[:, self._lower_aux_idx] += self.cfg.aux_scale * aux_c
        if res is not None:
            # E48: full-joint residual on top of the frozen-decoder prior
            res_c = torch.clamp(res, -self.cfg.res_clip, self.cfg.res_clip)
            q_des = q_des + self.cfg.res_scale * res_c
        return q_des

    # --------------------------------------------------------------- RL API
    def _pre_physics_step(self, actions: torch.Tensor):
        self._sample_disturbance()
        self._update_gate()
        if self.cfg.to_ref or self.cfg.to_tau:
            # TO38/40: advance the free-running reference clock once per control
            # step (period = the TO solution's natural stride time).
            self._to_phase = (self._to_phase + self._to_rate) % math.tau
        if self.cfg.decft_mode:
            # E44: action = 29-d normalized joint targets (decoder output space).
            # The policy owns the trainable SONIC decoder; advance the walk
            # clock here and expose (sin, cos) of the POST-advance phase so the
            # policy's token decode (next act()) matches the env's phase clock.
            self._q_des = (
                self._sonic_default_t + actions * self._sonic_scale_t
            ).detach()
            phi = self._latent_phase
            if self.cfg.latent_cmd_phase_rate:
                mult = torch.clamp(
                    self._commands[:, 0] / self.cfg.latent_phase_rate_ref,
                    min=0.0,
                    max=self.cfg.latent_phase_rate_max,
                )
                self._latent_phase = (phi + self._latent_phase_rate * mult) % math.tau
            else:
                self._latent_phase = (phi + self._latent_phase_rate) % math.tau
            self._last_phase = torch.stack(
                [torch.sin(self._latent_phase), torch.cos(self._latent_phase)], dim=1
            )
            self._last_aux = torch.zeros(
                self.num_envs, 12, dtype=torch.float32, device=self.device
            )
        else:
            if self.cfg.use_gate_sel:
                aux = actions[:, :12]
                self._gate_sel = actions[:, 12].long()
                phase = torch.zeros_like(aux[:, :2])
            elif self.cfg.latent_mode:
                phase = actions[:, :16]
                if self._to42 is not None:
                    # TO42: action = [z(16), sel bit(1)]；边界处采纳（lsel）
                    # 或忽略（fbkt），见 to42_gate.To42Gate
                    sel_bit = (actions[:, 16] > 0.5).long()
                    self._to42.step(self._commands[:, 0], sel_bit)
                if self.cfg.latent_residual:
                    # E48: action = [z(16), res(29)] -- res is the full-joint
                    # escape channel added on top of the frozen-decoder prior.
                    res = actions[:, 16:45]
                    if self.cfg.res_freeze_steps > 0 and (
                        self.common_step_counter < self.cfg.res_freeze_steps
                    ):
                        # E48c: residual still frozen -> pure z/prior control
                        res = torch.zeros_like(res)
                else:
                    res = None
                aux = torch.zeros(
                    self.num_envs, 12, dtype=torch.float32, device=self.device
                )
            elif self.cfg.token_mode:
                # E49: action = raw 64-d token coordinates; the env maps it
                # in _compute_q_des (mean + alpha*std*a, unbounded). Obs
                # feedback = the RAW policy output (same convention as the
                # latent arm's z feedback), not the mapped token; A/B arms
                # differ only in the +2 phase obs.
                phase = actions
                aux = torch.zeros(
                    self.num_envs, 12, dtype=torch.float32, device=self.device
                )
                res = None
            else:
                phase = actions[:, :2]
                aux = actions[:, 2:14]
                res = None
            self._q_des = self._compute_q_des(phase, aux, res).detach()
            self._last_phase = phase.detach()
            self._last_aux = aux.detach()
            if res is not None:
                self._last_res = res.detach()
        self._aux_rate = self._last_aux - self._prev_aux
        self._prev_aux = self._last_aux
        self._actions = actions.clone()

    def _apply_action(self):
        # apply disturbance force to pelvis during the whole control step
        if self._disturb.any():
            forces = torch.zeros(
                self.num_envs, 1, 3, dtype=torch.float32, device=self.device
            )
            forces[self._disturb, 0, :] = self._disturb_dir[self._disturb]
            self.robot.set_external_force_and_torque(
                forces, torch.zeros_like(forces), body_ids=[self._root_body_idx[0]]
            )
        full = torch.zeros(
            self.num_envs, self.robot.num_joints, dtype=torch.float32, device=self.device
        )
        full[:, self._body_idx] = self._q_des
        if self.cfg.to_tau and self._to_tau is not None:
            # TO40-C: gated torque feedforward as position offset dq = w*tau/kp
            # on the sagittal joints (cmd-gated; obs/pairing untouched).
            _q, _s, tau = self._to_ref_lookup()
            w = torch.exp(-((self._commands[:, 0] - self._to_vavg) ** 2)
                          / self.cfg.to_ref_gate2)
            dq = tau * (self.cfg.to_tau_w * w).unsqueeze(1) / self._to_kp
            full[:, self._body_idx[self._sag_idx]] = (
                full[:, self._body_idx[self._sag_idx]] + dq)
        self.robot.set_joint_position_target(full, joint_ids=None)

    def _to_ref_lookup(self) -> tuple[torch.Tensor, torch.Tensor]:
        """TO38: per-env reference at the current psi. Linear interp on the
        stride LUT (two dircol phases concatenated, time-normalized)."""
        x = self._to_phase / math.tau * self._to_m
        i0 = x.floor().long().clamp(0, self._to_m - 1)
        i1 = (i0 + 1) % self._to_m
        a = (x - x.floor()).unsqueeze(1)
        q = (1 - a) * self._to_q[i0] + a * self._to_q[i1]
        s = (1 - a) * self._to_scal[i0] + a * self._to_scal[i1]
        tau = ((1 - a) * self._to_tau[i0] + a * self._to_tau[i1]
               if self._to_tau is not None else None)
        return q, s, tau

    def _get_observations(self) -> dict:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        gravity = self.robot.data.projected_gravity_b
        jpos_rel = self.robot.data.joint_pos[:, self._body_idx] - torch.from_numpy(
            self._sonic_default
        ).to(self.device)
        jvel = self.robot.data.joint_vel[:, self._body_idx]
        mode_oh = torch.zeros(
            self.num_envs, len(self._modes_list), dtype=torch.float32, device=self.device
        )
        mode_oh[torch.arange(self.num_envs, device=self.device), self._gate_mode] = 1.0
        parts = [
            base_lin_vel,
            base_ang_vel,
            gravity,
            jpos_rel,
            jvel,
            self._commands,
            self._last_phase,
            self._last_aux,
            mode_oh,
            self._gate_tick.float().unsqueeze(-1),
            self.robot.data.root_pos_w[:, 2:3],
        ]
        if self.cfg.use_elevation:
            if self._elev_grids is not None:
                root_xy = self.robot.data.root_pos_w[:, :2]
                root_z = self.robot.data.root_pos_w[:, 2]
                q = self.robot.data.root_quat_w
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                yaw = torch.atan2(
                    2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
                )
                elev = sample_elevation(
                    self._elev_grids,
                    self._elev_origins,
                    self._elev_hscale,
                    root_xy,
                    root_z,
                    yaw,
                    self.cfg.elev_grid,
                    self.cfg.elev_res,
                    self.cfg.elev_lookahead,
                )
            else:
                # flat ground: privileged patch is all zeros
                elev = torch.zeros(
                    self.num_envs,
                    self.cfg.elev_grid * self.cfg.elev_grid,
                    device=self.device,
                )
            parts.append(elev)
        if self.cfg.latent_residual:
            # E48: residual action feedback (like _last_aux for the aux modes)
            parts.append(self._last_res)
        if self.cfg.token_phase_obs:
            # E49-B: the walk clock made visible. _latent_phase was advanced
            # inside _compute_q_des earlier this control step, so obs carries
            # the phi that will pair with the NEXT action's token -- the same
            # action-phi pairing the latent arm's VAE decode uses.
            parts.append(
                torch.stack(
                    [torch.sin(self._latent_phase), torch.cos(self._latent_phase)],
                    dim=1,
                )
            )
        if self.cfg.decft_mode:
            # E44: full 930-d proprio history (decoder input) + walk-clock phase
            parts.append(self._proprio_t())
            parts.append(self._last_phase)
        if self.cfg.to_ref:
            if self._to_q is not None and not self.cfg.to_ref_obs_zero:
                q_ref, scal, _ = self._to_ref_lookup()
                clocks = torch.stack(
                    [torch.sin(self._to_phase), torch.cos(self._to_phase)], dim=1
                )
                parts.append(torch.cat([clocks, q_ref - self._to_def_sag, scal], dim=1))
            else:
                # paired control arm (or no LUT): same obs dim, zero block
                parts.append(
                    torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
                )
        if self._to42 is not None:
            # TO42: [sel_state, gate_bool] feedback（两臂同槽位）
            parts.append(torch.stack(
                [self._to42.state.float(), self._to42.gate.float()], dim=1))
        obs = torch.cat(
            [
                *parts,
            ],
            dim=-1,
        )
        assert obs.shape[1] == self.cfg.observation_space, obs.shape
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        gravity = self.robot.data.projected_gravity_b
        track_xy = torch.exp(
            -((base_lin_vel[:, 0] - self._commands[:, 0]) ** 2) / self.cfg.vel_sigma2
        )
        track_yaw = torch.exp(
            -((base_ang_vel[:, 2] - self._commands[:, 2]) ** 2) / self.cfg.yaw_sigma2
        )
        upright = torch.exp(-(gravity[:, :2].norm(dim=1) ** 2) / 0.1)
        height = torch.exp(-((self.robot.data.root_pos_w[:, 2] - 0.76) ** 2) / 0.02)
        stillness = (
            -self.cfg.stillness_vx_scale * base_lin_vel[:, 0] ** 2
            - 0.05 * base_lin_vel[:, 1] ** 2
            - 0.05 * (base_ang_vel[:, 0] ** 2 + base_ang_vel[:, 1] ** 2)
        )
        reward = (
            1.0 * track_xy
            + self.cfg.yaw_scale * track_yaw
            + 0.1 * upright
            + 0.5 * height
            + stillness
        )
        if self.cfg.yaw_rate_penalty > 0.0:
            reward = reward - self.cfg.yaw_rate_penalty * (base_ang_vel[:, 2] ** 2)
        if self.cfg.heading_scale > 0.0:
            # E32: reward velocity direction aligned with commanded heading.
            # world-frame vx/vy vs command frame (yaw from root quat).
            yaw = torch.atan2(
                2.0 * (self.robot.data.root_quat_w[:, 0] * self.robot.data.root_quat_w[:, 3]
                       + self.robot.data.root_quat_w[:, 1] * self.robot.data.root_quat_w[:, 2]),
                1.0 - 2.0 * (self.robot.data.root_quat_w[:, 2] ** 2
                             + self.robot.data.root_quat_w[:, 3] ** 2),
            )
            # rotate world vel into command frame (cmd heading = +x of cmd frame)
            v_cx = base_lin_vel[:, 0] * torch.cos(yaw) + base_lin_vel[:, 1] * torch.sin(yaw)
            v_cy = -base_lin_vel[:, 0] * torch.sin(yaw) + base_lin_vel[:, 1] * torch.cos(yaw)
            sp = torch.clamp(torch.sqrt(v_cx ** 2 + v_cy ** 2), min=1e-3)
            heading = torch.clamp(v_cx / sp, -1.0, 1.0)  # 1 = moving along +x cmd
            reward = reward + self.cfg.heading_scale * (0.5 + 0.5 * heading)
        if self.cfg.progress_scale > 0.0:
            reward = reward + self.cfg.progress_scale * torch.clamp(
                base_lin_vel[:, 0], 0.0, 1.0
            )
        if self.cfg.to_ref and self.cfg.to_ref_w > 0.0 and self._to_q is not None:
            # TO38: sagittal-joint tracking of the TO reference, gated by
            # commanded-speed proximity to the solution speed -- envs commanded
            # away from v_TO are untouched by the reference (isolation).
            q_ref, _, _ = self._to_ref_lookup()
            q_sag = self.robot.data.joint_pos[:, self._body_idx][:, self._sag_idx]
            err = ((q_sag - q_ref) ** 2).sum(-1)
            gate = torch.exp(
                -((self._commands[:, 0] - self._to_vavg) ** 2) / self.cfg.to_ref_gate2
            )
            reward = reward + self.cfg.to_ref_w * torch.exp(
                -err / self.cfg.to_ref_sigma2
            ) * gate
        if self.cfg.anti_stop_scale > 0.0:
            reward = reward - self.cfg.anti_stop_scale * torch.clamp(
                self.cfg.anti_stop_thresh - base_lin_vel[:, 0], min=0.0, max=None
            )
        if self.cfg.aux_l2_scale > 0.0:
            reward = reward - self.cfg.aux_l2_scale * (self._last_aux ** 2).sum(-1)
        if self.cfg.aux_rate_scale > 0.0:
            reward = reward - self.cfg.aux_rate_scale * (self._aux_rate ** 2).sum(-1)
        if self.cfg.res_l2_scale > 0.0 and self.cfg.latent_residual:
            reward = reward - self.cfg.res_l2_scale * (self._last_res ** 2).sum(-1)
        reward = reward - self.cfg.termination_penalty * self.reset_terminated.float()
        # E49 诊断步骤③：分项快照，return 前一次性存（GPU tensor 原样引用，
        # 纯记录不改计算图/数值；heading/progress/yaw_rate 等默认 0 的分支不记）。
        # 未开启 diag 时每步被覆盖，无累积开销。vx_err = 当时口径的 cmd 跟踪误差。
        self._last_rew_terms = {
            "track_xy": track_xy,
            "track_yaw": track_yaw,
            "upright": upright,
            "height": height,
            "stillness": stillness,
            "vx_err": base_lin_vel[:, 0] - self._commands[:, 0],
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        fallen = self.robot.data.root_pos_w[:, 2] < self.cfg.fall_height
        not_finite = ~torch.isfinite(self.robot.data.root_pos_w[:, 2])
        terminated = fallen | not_finite
        if self.cfg.reset_grace_steps > 0:
            terminated &= self.episode_length_buf >= self.cfg.reset_grace_steps
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        # E49: 在 super() 复位之前截留复位前终末状态的观测，供训练侧对超时步
        # （trunc&~done）做价值自举——Isaac 的 step 在返回前已把 obs 换成复位后
        # 新局的观测，训练侧拿不到终末状态价值。两个约束：
        # ① 此时 reset 尚未发生，robot.data 仍是上一局的终末物理状态；
        # ② decft 模式下该 obs 缺本步终末的 proprio history 帧（_post_step_history
        #    在 step 返回后才 push），E49 token/latent 模式不含 history 块，不受影响。
        env_ids_pre = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_pre.numel() > 0:
            final_obs = self._get_observations()["policy"]
            if self._final_obs is None:
                self._final_obs = torch.zeros(
                    self.num_envs, final_obs.shape[1],
                    dtype=torch.float32, device=self.device,
                )
            self._final_obs[env_ids_pre] = final_obs[env_ids_pre].detach()
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = len(env_ids)

        # root state: standing at SONIC default above the local terrain height
        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, 2] = (
            self.scene.env_origins[env_ids, 2]
            + 0.76
            + torch.empty(n, device=self.device).uniform_(-0.02, 0.02)
        )
        default_root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.robot.write_root_state_to_sim(default_root, env_ids)

        # joint state: SONIC default angles, zero velocity
        default_jp = torch.zeros(
            n, self.robot.num_joints, dtype=torch.float32, device=self.device
        )
        default_jp[:, self._body_idx] = self._sonic_default_t
        default_jv = torch.zeros_like(default_jp)
        self.robot.write_joint_state_to_sim(default_jp, default_jv, env_ids=env_ids)

        # commands
        self._commands[env_ids] = self._commands_from_cfg(env_ids)
        if self._to42 is not None:
            # TO42: 自然 bin 中性起步；count 归零（首边界在 reset 后第
            # to42_hold_steps 步）
            self._to42.reset(env_ids, self._commands[env_ids, 0])

        # history: fill with the standing frame
        gravity = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float32, device=self.device
        ).repeat(n, 1)
        zero29 = torch.zeros(n, 29, dtype=torch.float32, device=self.device)
        self._hist_ang_vel[env_ids] = 0.0
        self._hist_joint_pos[env_ids] = 0.0
        self._hist_joint_vel[env_ids] = 0.0
        self._hist_last_actions[env_ids] = 0.0
        self._hist_gravity[env_ids] = gravity[:, None, :].expand(n, 10, 3).clone()

        # router EMA + gate
        if self._router_state is not None:
            self._router_state = self._router.reset_state(self.num_envs)
        self._gate_mode[env_ids] = 0
        self._gate_tick[env_ids] = False
        self._gate_count[env_ids] = 0
        self._gate_sel[env_ids] = 0
        if self.cfg.use_gate_sel:
            for i in env_ids.tolist():
                self.router_commands[i] = None
        if self.cfg.use_2hz_gate and self._router is not None:
            self._held_groups_np = self._router.select_groups(self._build_commands_list())
        self._last_phase[env_ids] = 0.0
        if self.cfg.latent_mode or self.cfg.decft_mode or self.cfg.token_mode:
            self._latent_phase[env_ids] = torch.rand(
                len(env_ids), dtype=torch.float32, device=self.device
            ) * math.tau
        if self.cfg.to_ref or self.cfg.to_tau:
            # random stride phase at spawn (same treatment as the walk clock;
            # the policy must handle any phase offset since it cannot observe
            # the latent clock either)
            self._to_phase[env_ids] = torch.rand(
                len(env_ids), dtype=torch.float32, device=self.device
            ) * math.tau
        self._phase_ema[env_ids] = 0.0
        self._phase_groups[env_ids] = -1
        self._anchor_prev[env_ids] = 0.0
        self._anchor_groups[env_ids] = -1
        self._last_aux[env_ids] = 0.0
        self._prev_aux[env_ids] = 0.0
        self._aux_rate[env_ids] = 0.0
        self._last_res[env_ids] = 0.0
        self._disturb[env_ids] = False
        # schedule a single push per episode with probability disturbance_prob
        self._disturb_step[env_ids] = -1
        self._disturb_dir[env_ids] = 0.0
        if self.cfg.disturbance_prob > 0.0:
            n = len(env_ids)
            sched = torch.rand(n, device=self.device) < self.cfg.disturbance_prob
            if sched.any():
                step_min = 50
                step_max = max(60, self.max_episode_length - 150)
                steps = torch.randint(step_min, step_max, (n,), device=self.device)
                mag = torch.empty(n, device=self.device).uniform_(
                    self.cfg.disturbance_force_range[0],
                    self.cfg.disturbance_force_range[1],
                )
                theta = torch.empty(n, device=self.device).uniform_(0.0, 2.0 * math.pi)
                self._disturb_step[env_ids] = torch.where(
                    sched, steps, torch.tensor(-1, device=self.device)
                )
                self._disturb_dir[env_ids, 0] = torch.cos(theta) * mag
                self._disturb_dir[env_ids, 1] = torch.sin(theta) * mag

        # default q_des (SONIC default)
        self._q_des[env_ids] = self._sonic_default_t

    # --------------------------------------------------------------- stepping extras
    def _sample_disturbance(self):
        # one scheduled push per episode (matches MuJoCo C2)
        self._disturb = self.episode_length_buf == self._disturb_step

    def _update_gate(self):
        if not self.cfg.use_2hz_gate or self._router is None:
            return
        # paper-style gate: re-evaluate the gait decision at 2 Hz, latch the
        # selected decoder group for 0.5 s, and set the Boolean signal ONLY on
        # an actual decision (group change) -- not on a free-running clock.
        self._gate_count += 1
        boundary = self._gate_count % self.cfg.gate_hold_steps == 0
        self._gate_tick = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if boundary.any():
            cmds = self._build_commands_list()
            if self.cfg.use_gate_sel:
                groups = np.array(
                    [self._gate_gi[g] for g in self._gate_sel.cpu().tolist()],
                    dtype=np.int64,
                )
                # canonical commands for the selected groups keep the router
                # phase nets in-distribution (reward still tracks _commands).
                for i, gidx in enumerate(self._gate_sel.cpu().tolist()):
                    g = self.cfg.gate_groups[gidx]
                    mdir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    self.router_commands[i] = Command(
                        mode=g[0], speed=g[1], mdir=mdir, fdir=mdir
                    )
            else:
                groups = self._router.select_groups(cmds)
            if self._held_groups_np is None:
                self._held_groups_np = groups
            else:
                changed = groups != self._held_groups_np
                self._held_groups_np[changed] = groups[changed]
                self._gate_tick = torch.from_numpy(changed).to(self.device)
            modes = np.array(
                [self._router.group_rows[gi][0] for gi in self._held_groups_np],
                dtype=np.int64,
            )
            mode_t = torch.from_numpy(modes).to(self.device)
            for i, m in enumerate(self._modes_list.tolist()):
                self._gate_mode[mode_t == m] = i

    def _post_step_history(self):
        """Push the current frame into history after physics stepping."""
        base_ang_vel = self.robot.data.root_ang_vel_b
        gravity = self.robot.data.projected_gravity_b
        jpos = self.robot.data.joint_pos[:, self._body_idx]
        jvel = self.robot.data.joint_vel[:, self._body_idx]
        default = self._sonic_default_t
        scale = self._sonic_scale_t
        last_actions = (self._q_des - default) / scale
        self._push_history(base_ang_vel, jpos - default, jvel, last_actions, gravity)

    def step(self, action: torch.Tensor):
        obs, rew, term, trunc, extras = super().step(action)
        self._post_step_history()
        if self._disturb.any():
            zeros = torch.zeros(
                self.num_envs, 1, 3, dtype=torch.float32, device=self.device
            )
            self.robot.set_external_force_and_torque(
                zeros, zeros, body_ids=[self._root_body_idx[0]]
            )
        return obs, rew, term, trunc, extras
