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
    phase_mode: bool = False  # True: policy selects phase (joint RL)
    phase_anchor: bool = False  # True: phase = router clock + bounded offset
    phase_anchor_ema: bool = True  # True: EMA-smooth the router clock (like encode_batch)
    phase_offset_scale: float = 0.15  # max offset magnitude when anchored
    latent_mode: bool = False  # True: policy outputs 16-d latent -> VAE -> token -> SONIC
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

    # privileged local elevation map (teacher-style terrain observation)
    use_elevation: bool = False
    elev_grid: int = 9          # grid_n x grid_n patch
    elev_res: float = 0.15      # meters per cell
    elev_lookahead: float = 0.6  # patch center ahead of the robot (m)

    # learned gait/group selection (paper's gait logit analog)
    use_gate_sel: bool = False
    gate_groups: tuple = ((0, -1.0, 4), (1, 0.2, 4), (2, -1.0, 4))  # idle/slow/walk fwd

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

        if self.cfg.use_sonic_prior:
            self._decoder = SonicTorchDecoder(self.cfg.sonic_decoder_path, device=self.device)
            self._router = BatchedPhaseRouter(self.cfg.router_model_dir, device=self.device)
            self._vae = None
            if self.cfg.latent_mode:
                import numpy as _np

                from apt_g1.isaac.token_window_vae import PhaseTokenVAE

                vae = PhaseTokenVAE().to(self.device)
                vae.load_state_dict(
                    torch.load(self.cfg.latent_vae_path, map_location=self.device),
                    strict=False,  # checkpoint also carries the encoder; only the decoder is needed
                )
                vae.eval()
                self._vae = vae
                pca = _np.load(
                    str(Path(self.cfg.latent_vae_path).parent / "pca.npz")
                )
                self._latent_phase_rate = float(
                    self.cfg.latent_phase_rate or pca["rate"]
                )
                self._latent_phase = torch.zeros(
                    self.num_envs, dtype=torch.float32, device=self.device
                )
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
            16 if self.cfg.latent_mode else 2,
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
        self._q_des = torch.zeros(self.num_envs, 29, dtype=torch.float32, device=self.device)
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

    def _base_lin_vel(self) -> torch.Tensor:
        return self.robot.data.root_lin_vel_b

    def _base_ang_vel(self) -> torch.Tensor:
        return self.robot.data.root_ang_vel_b

    def _world_to_body(self, vec_w: torch.Tensor) -> torch.Tensor:
        return quat_rotate_inverse(
            self.robot.data.root_quat_w, vec_w.reshape(-1, 3)
        )

    def _decoder_obs_parts(self, tokens_np: np.ndarray) -> tuple[torch.Tensor, ...]:
        """History tensors in the layout the torch decoder expects."""
        return (
            torch.from_numpy(tokens_np).to(self.device),
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

    def _compute_q_des(self, phase: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Decode token prior + aux into (N, 29) SONIC-order joint targets."""
        cmds = self._build_commands_list()
        proprio = self._proprio_np()
        if self.cfg.latent_mode:
            # E27/E28: the policy's latent z (passed as "phase") is decoded by
            # the frozen phase-conditioned VAE at the current walk-clock phase.
            with torch.no_grad():
                phi = self._latent_phase
                sc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)
                tokens = self._vae.decode(phase, sc).detach().cpu().numpy()
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
        return q_des

    # --------------------------------------------------------------- RL API
    def _pre_physics_step(self, actions: torch.Tensor):
        self._sample_disturbance()
        self._update_gate()
        if self.cfg.use_gate_sel:
            aux = actions[:, :12]
            self._gate_sel = actions[:, 12].long()
            phase = torch.zeros_like(aux[:, :2])
        elif self.cfg.latent_mode:
            phase = actions
            aux = torch.zeros(
                self.num_envs, 12, dtype=torch.float32, device=self.device
            )
        else:
            phase = actions[:, :2]
            aux = actions[:, 2:14]
        self._q_des = self._compute_q_des(phase, aux).detach()
        self._last_phase = phase.detach()
        self._last_aux = aux.detach()
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
        self.robot.set_joint_position_target(full, joint_ids=None)

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
            + 0.5 * track_yaw
            + 0.1 * upright
            + 0.5 * height
            + stillness
        )
        if self.cfg.progress_scale > 0.0:
            reward = reward + self.cfg.progress_scale * torch.clamp(
                base_lin_vel[:, 0], 0.0, 1.0
            )
        if self.cfg.anti_stop_scale > 0.0:
            reward = reward - self.cfg.anti_stop_scale * torch.clamp(
                self.cfg.anti_stop_thresh - base_lin_vel[:, 0], min=0.0, max=None
            )
        if self.cfg.aux_l2_scale > 0.0:
            reward = reward - self.cfg.aux_l2_scale * (self._last_aux ** 2).sum(-1)
        if self.cfg.aux_rate_scale > 0.0:
            reward = reward - self.cfg.aux_rate_scale * (self._aux_rate ** 2).sum(-1)
        reward = reward - self.cfg.termination_penalty * self.reset_terminated.float()
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
        if self.cfg.latent_mode:
            self._latent_phase[env_ids] = torch.rand(
                len(env_ids), dtype=torch.float32, device=self.device
            ) * math.tau
        self._phase_ema[env_ids] = 0.0
        self._phase_groups[env_ids] = -1
        self._anchor_prev[env_ids] = 0.0
        self._anchor_groups[env_ids] = -1
        self._last_aux[env_ids] = 0.0
        self._prev_aux[env_ids] = 0.0
        self._aux_rate[env_ids] = 0.0
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
