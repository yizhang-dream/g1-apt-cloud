"""Direct MuJoCo G1 flat-ground environment for the APT-SONIC experiment.

This environment does not use Isaac Lab. It loads the G1 MJCF model, builds the
994-dimensional SONIC decoder observation from a 10-frame history, decodes a
64-d latent token into 29 body joint targets, applies the APT auxiliary action
on the 12 lower-body joints, and steps MuJoCo with PD control.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

try:
    from scipy.spatial.transform import Rotation
except ImportError:  # pragma: no cover - optional dependency
    Rotation = None

try:
    from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand
except ImportError:  # pragma: no cover - optional dependency
    ElasticBand = None


# G1 ordering constants from
# gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp.
# IsaacLab order is what the SONIC decoder expects for joint history and actions.
G1_ISAACLAB_TO_MUJOCO_DOF = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16,
    20, 22, 24, 26, 28,
]
G1_MUJOCO_TO_ISAACLAB_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
    20, 27, 21, 28,
]

# action_scale = 0.25 * effort_limit / stiffness, in MuJoCo joint order.
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

SONIC_KP_MUJOCO = np.array(
    [
        99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
        99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
        40.17924, 28.50125, 28.50125,
        14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
        14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 16.77833, 16.77833,
    ],
    dtype=np.float32,
)

SONIC_KD_MUJOCO = np.array(
    [
        6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
        6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
        2.55789, 1.81445, 1.81445,
        0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
        0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 1.06814, 1.06814,
    ],
    dtype=np.float32,
)


def quat_rotate(q, v):
    q = q / np.linalg.norm(q)
    qv = q[1:]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q[0] * v)


def quat_rotate_inverse(q, v):
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    return quat_rotate(q_inv, v)


class MujocoG1FlatEnv:
    def __init__(
        self,
        sonic_decoder,
        repo_root: str | Path,
        robot_scene: str = "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml",
        wbc_config_path: str | Path = (
            "gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"
        ),
        sim_dt: float = 0.005,
        control_decimation: int = 4,
        episode_length_s: float = 20.0,
        aux_scale: float = 0.2,
        stand_only: bool = False,
        use_elastic_band: bool = False,
        band_scale: float = 1.0,
        command_vx_min: float = 0.0,
        command_vx_max: float = 1.0,
        command_vy_min: float = -0.5,
        command_vy_max: float = 0.5,
        command_yaw_min: float = -0.5,
        command_yaw_max: float = 0.5,
        reference_token_sequence: np.ndarray | None = None,
        residual_scale: float = 0.1,
        base_token: np.ndarray | None = None,
        token_vae=None,
        skill_tokens: np.ndarray | None = None,
        token_seq_vae=None,
        add_phase_obs: bool = True,
        joint_seq_vae=None,
        joint_reset_path: str | Path | None = None,
        phase_router=None,
        disturbance_prob: float = 0.0,
        disturbance_force_range: tuple[float, float] = (0.0, 0.0),
    ):
        self.repo_root = Path(repo_root)
        self.sonic_decoder = sonic_decoder
        self.sim_dt = sim_dt
        self.control_decimation = control_decimation
        self.episode_length = int(episode_length_s / (sim_dt * control_decimation))
        self.aux_scale = aux_scale
        self.stand_only = stand_only
        self.use_elastic_band = use_elastic_band
        self.band_scale = band_scale
        self.command_vx_min = command_vx_min
        self.command_vx_max = command_vx_max
        self.command_vy_min = command_vy_min
        self.command_vy_max = command_vy_max
        self.command_yaw_min = command_yaw_min
        self.command_yaw_max = command_yaw_max
        self.reference_token_sequence = (
            None
            if reference_token_sequence is None
            else np.asarray(reference_token_sequence, dtype=np.float32)
        )
        self.residual_scale = residual_scale
        self.base_token = (
            None if base_token is None else np.asarray(base_token, dtype=np.float32)
        )
        self.token_vae = token_vae
        self.skill_tokens = (
            None
            if skill_tokens is None
            else np.asarray(skill_tokens, dtype=np.float32)
        )
        self.token_seq_vae = token_seq_vae
        self.add_phase_obs = add_phase_obs
        self.joint_seq_vae = joint_seq_vae
        self.phase_router = phase_router
        self.router_command = None
        self.disturbance_prob = disturbance_prob
        self.disturbance_force_range = disturbance_force_range
        self._disturb = None
        self.joint_reset = None
        if joint_reset_path is not None:
            self.joint_reset = np.load(joint_reset_path).astype(np.float32)

        with open(self.repo_root / wbc_config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.model = mujoco.MjModel.from_xml_path(str(self.repo_root / robot_scene))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_dt

        self._setup_joints()
        self._history_len = 10
        self._reset_history()
        self.command = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        if self.use_elastic_band:
            if ElasticBand is None:
                raise ImportError("use_elastic_band=True requires gear_sonic ElasticBand")
            self.elastic_band = ElasticBand()

    def _setup_joints(self):
        self.body_act_ids = []
        self.body_qpos_adr = []
        self.body_dof_adr = []
        self.body_names = []

        for act_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[act_id, 0]
            name = self.model.joint(joint_id).name
            if "hand" in name:
                continue
            self.body_act_ids.append(act_id)
            self.body_qpos_adr.append(self.model.jnt_qposadr[joint_id])
            self.body_dof_adr.append(self.model.jnt_dofadr[joint_id])
            self.body_names.append(name)

        self.body_act_ids = np.asarray(self.body_act_ids, dtype=int)
        self.body_qpos_adr = np.asarray(self.body_qpos_adr, dtype=int)
        self.body_dof_adr = np.asarray(self.body_dof_adr, dtype=int)
        self.num_body = len(self.body_names)

        self.kp = SONIC_KP_MUJOCO
        self.kd = SONIC_KD_MUJOCO
        self.effort_limit = np.asarray(
            [self.cfg["motor_effort_limit_list"][act_id] for act_id in self.body_act_ids],
            dtype=np.float32,
        )
        self.default_motor_angles = SONIC_DEFAULT_ANGLES_MUJOCO.copy()
        self.motor_pos_lower = np.asarray(
            self.cfg["motor_pos_lower_limit_list"][: self.num_body], dtype=np.float32
        )
        self.motor_pos_upper = np.asarray(
            self.cfg["motor_pos_upper_limit_list"][: self.num_body], dtype=np.float32
        )
        self.lower_body_indices = np.arange(12, dtype=int)
        self.isaaclab_to_mujoco = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF, dtype=int)
        self.mujoco_to_isaaclab = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF, dtype=int)
        self.sonic_scale_mujoco = SONIC_ACTION_SCALE_MUJOCO
        self.sonic_default_isaac = self.default_motor_angles[self.mujoco_to_isaaclab]
        self.sonic_scale_isaac = self.sonic_scale_mujoco[self.mujoco_to_isaaclab]
        self.lower_isaaclab_indices = self.mujoco_to_isaaclab[:12]

    def _reset_history(self):
        self.history = {
            "base_angular_velocity": np.zeros((self._history_len, 3), dtype=np.float32),
            "body_joint_positions": np.zeros((self._history_len, self.num_body), dtype=np.float32),
            "body_joint_velocities": np.zeros((self._history_len, self.num_body), dtype=np.float32),
            "last_actions": np.zeros((self._history_len, self.num_body), dtype=np.float32),
            "gravity_dir": np.zeros((self._history_len, 3), dtype=np.float32),
        }

    def _push_history(self, key: str, value: np.ndarray):
        arr = self.history[key]
        arr[:-1] = arr[1:]
        arr[-1] = value

    def _fill_history_from_state(self):
        qpos, qvel = self._get_body_state()
        self.history["base_angular_velocity"][:] = self._get_base_angular_velocity()
        self.history["body_joint_positions"][:] = qpos
        self.history["body_joint_velocities"][:] = qvel
        self.history["last_actions"][:] = 0.0
        self.history["gravity_dir"][:] = self._get_gravity_dir()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0.0, 0.0, 0.76])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.body_qpos_adr] = (
            self.joint_reset
            if self.joint_reset is not None
            else self.default_motor_angles
        )
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._reset_history()
        self._fill_history_from_state()
        if self.phase_router is not None:
            # Reset EMA phase state when the episode restarts.
            self.phase_router.reset()
        self._disturb = None
        if self.disturbance_prob > 0 and np.random.rand() < self.disturbance_prob:
            step = int(np.random.randint(50, max(60, self.episode_length - 150)))
            mag = float(
                np.random.uniform(self.disturbance_force_range[0], self.disturbance_force_range[1])
            )
            d = np.random.choice(["x", "-x", "y", "-y"])
            vec = {
                "x": [mag, 0.0, 0.0],
                "-x": [-mag, 0.0, 0.0],
                "y": [0.0, mag, 0.0],
                "-y": [0.0, -mag, 0.0],
            }[d]
            self._disturb = (step, np.asarray(vec, dtype=np.float64))
        if self.stand_only:
            self.command = np.zeros(3, dtype=np.float32)
        else:
            self.command = np.array(
                [
                    np.random.uniform(self.command_vx_min, self.command_vx_max),
                    np.random.uniform(self.command_vy_min, self.command_vy_max),
                    np.random.uniform(self.command_yaw_min, self.command_yaw_max),
                ],
                dtype=np.float32,
            )
        self.step_count = 0
        return self.get_obs()

    def _get_body_state(self):
        qpos = self.data.qpos[self.body_qpos_adr].copy()
        qvel = self.data.qvel[self.body_dof_adr].copy()
        return qpos, qvel

    def _get_gravity_dir(self):
        q = self.data.qpos[3:7].copy()
        return quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0])).astype(np.float32)

    def _get_base_angular_velocity(self):
        q = self.data.qpos[3:7].copy()
        return quat_rotate_inverse(q, self.data.qvel[3:6].copy()).astype(np.float32)

    def _get_base_linear_velocity(self):
        q = self.data.qpos[3:7].copy()
        return quat_rotate_inverse(q, self.data.qvel[0:3].copy()).astype(np.float32)

    def get_obs(self) -> np.ndarray:
        qpos, qvel = self._get_body_state()
        obs = [
            self._get_base_linear_velocity(),
            self._get_base_angular_velocity(),
            self._get_gravity_dir(),
            qpos,
            qvel,
            self.history["last_actions"][-1],
            self.command,
        ]
        if self.reference_token_sequence is not None and self.add_phase_obs:
            obs.append(np.asarray([self._get_motion_phase()], dtype=np.float32))
        return np.concatenate(obs).astype(np.float32)

    def _get_motion_phase(self) -> float:
        if self.reference_token_sequence is None:
            return 0.0
        length = len(self.reference_token_sequence)
        return float((self.step_count % length) / length)

    def _get_sonic_history(self):
        qpos = self.history["body_joint_positions"]
        qvel = self.history["body_joint_velocities"]
        last_actions = self.history["last_actions"]
        return {
            "base_angular_velocity": self.history["base_angular_velocity"],
            "body_joint_positions": (
                (qpos - self.default_motor_angles)[..., self.mujoco_to_isaaclab]
            ),
            "body_joint_velocities": qvel[..., self.mujoco_to_isaaclab],
            "last_actions": last_actions[..., self.mujoco_to_isaaclab],
            "gravity_dir": self.history["gravity_dir"],
        }

    def _decode_body_action(
        self,
        token: np.ndarray,
        aux: np.ndarray,
        skill: np.ndarray | None = None,
        phase: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (q_des in MuJoCo order, effective normalized action in MuJoCo order)."""
        token = np.asarray(token, dtype=np.float32)
        if self.joint_seq_vae is not None:
            import torch

            with torch.no_grad():
                decoded = self.joint_seq_vae.decode(
                    torch.from_numpy(token).float().unsqueeze(0)
                )
                q_des_mujoco = decoded[0, 0].numpy().astype(np.float32)
            aux = np.clip(np.asarray(aux, dtype=np.float32), -1.0, 1.0)
            q_des_mujoco[:12] += self.aux_scale * aux
            q_des_mujoco = np.clip(
                q_des_mujoco, self.motor_pos_lower, self.motor_pos_upper
            )
            effective_action_mujoco = (
                (q_des_mujoco - self.default_motor_angles) / self.sonic_scale_mujoco
            )
            return q_des_mujoco, effective_action_mujoco
        if self.token_seq_vae is not None:
            import torch

            with torch.no_grad():
                decoded = self.token_seq_vae.decode(
                    torch.from_numpy(token).float().unsqueeze(0)
                )
                token = decoded[0, 0].numpy()
        if self.skill_tokens is not None and skill is not None:
            token = self.skill_tokens[int(skill)] + self.residual_scale * np.clip(
                token, -1.0, 1.0
            )
        if self.token_vae is not None:
            import torch

            with torch.no_grad():
                token = (
                    self.token_vae.decode(
                        torch.from_numpy(token).float().unsqueeze(0)
                    )
                    .squeeze(0)
                    .numpy()
                )
        if self.reference_token_sequence is not None:
            ref = self.reference_token_sequence[self.step_count % len(self.reference_token_sequence)]
            token = ref + self.residual_scale * np.clip(token, -1.0, 1.0)
        elif self.base_token is not None:
            token = self.base_token + self.residual_scale * np.clip(token, -1.0, 1.0)
        if self.phase_router is not None:
            # Frozen distilled token prior: ignore the policy token and use the
            # phase-router output for the current command + proprio history.
            from apt_g1.encoder import Command

            cmd = (
                self.router_command
                if self.router_command is not None
                else Command.from_vxvy(
                    float(self.command[0]), float(self.command[1]), 0.0
                )
            )
            if phase is not None:
                # Policy-selected phase (RL latent action): normalize and look up
                # the corresponding prototype token.
                p = np.asarray(phase, dtype=np.float32)
                n = float(np.linalg.norm(p))
                if n < 1e-6:
                    n = 1.0
                phi = float(np.arctan2(p[1] / n, p[0] / n))
                gi = self.phase_router.select_group(cmd)
                n_bins = len(self.phase_router.protos[gi])
                b = int(np.floor((phi + np.pi) / (2 * np.pi) * n_bins) % n_bins)
                token = self.phase_router.protos[gi][b]
            else:
                token = self.phase_router.encode(cmd, self._get_sonic_history())

        action_isaac = self.sonic_decoder.decode(
            token,
            self._get_sonic_history(),
        )[0].astype(np.float32)
        aux = np.clip(np.asarray(aux, dtype=np.float32), -1.0, 1.0)
        q_des_isaac = self.sonic_default_isaac + action_isaac * self.sonic_scale_isaac
        q_des_isaac[self.lower_isaaclab_indices] += self.aux_scale * aux
        q_des_mujoco = q_des_isaac[self.isaaclab_to_mujoco]
        q_des_mujoco = np.clip(
            q_des_mujoco, self.motor_pos_lower, self.motor_pos_upper
        )
        effective_action_mujoco = (
            (q_des_mujoco - self.default_motor_angles) / self.sonic_scale_mujoco
        )
        return q_des_mujoco, effective_action_mujoco

    def _step_physics(self, q_des: np.ndarray):
        for _ in range(self.control_decimation):
            if self.use_elastic_band:
                self._apply_elastic_band()
            qpos, qvel = self._get_body_state()
            torque = (
                self.kp[: self.num_body] * (q_des - qpos)
                - self.kd[: self.num_body] * qvel
            )
            torque = np.clip(
                torque,
                -self.effort_limit[: self.num_body],
                self.effort_limit[: self.num_body],
            )
            ctrl = np.zeros(self.model.nu, dtype=np.float32)
            ctrl[self.body_act_ids] = torque
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    def _apply_elastic_band(self):
        """Exact copy of the SONIC MuJoCo sim loop's virtual spring on the pelvis."""
        pelvis_id = self.model.body("pelvis").id
        pose = np.concatenate(
            [
                self.data.xpos[pelvis_id],
                self.data.xquat[pelvis_id],
                np.zeros(6),
            ]
        )
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            pelvis_id,
            pose[7:13],
            0,
        )
        pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
        force_torque = self.elastic_band.Advance(pose)
        self.data.xfrc_applied[pelvis_id] = force_torque * self.band_scale

    def step(self, action: dict[str, np.ndarray]):
        if self._disturb is not None and self._disturb[0] == self.step_count:
            pid = self.model.body("pelvis").id
            self.data.xfrc_applied[pid, :3] = self._disturb[1]
        q_des, effective_action = self._decode_body_action(
            action["token"], action["aux"], action.get("skill"), action.get("phase")
        )
        self._step_physics(q_des)
        if self._disturb is not None and self._disturb[0] == self.step_count:
            pid = self.model.body("pelvis").id
            self.data.xfrc_applied[pid, :3] = 0.0

        qpos, qvel = self._get_body_state()
        self._push_history("base_angular_velocity", self._get_base_angular_velocity())
        self._push_history("body_joint_positions", qpos)
        self._push_history("body_joint_velocities", qvel)
        self._push_history("last_actions", effective_action)
        self._push_history("gravity_dir", self._get_gravity_dir())
        self.step_count += 1

        reward = self._compute_reward()
        terminated = self._check_termination()
        if terminated:
            if self.step_count < self.episode_length:
                reward -= 10.0
            return self.reset(), reward, terminated, {}
        return self.get_obs(), reward, terminated, {}

    def _compute_reward(self) -> float:
        if not np.all(np.isfinite(self.data.qpos)):
            return 0.0
        lin_vel = self._get_base_linear_velocity()
        ang_vel = self._get_base_angular_velocity()
        track_xy = np.exp(-((lin_vel[0] - self.command[0]) ** 2) / 0.25)
        track_yaw = np.exp(-((ang_vel[2] - self.command[2]) ** 2) / 0.25)
        upright = np.exp(-(np.linalg.norm(self._get_gravity_dir()[:2]) ** 2) / 0.1)
        height = np.exp(-((self.data.qpos[2] - 0.76) ** 2) / 0.02)
        stillness = -0.05 * (lin_vel[0] ** 2 + lin_vel[1] ** 2) - 0.05 * (
            ang_vel[0] ** 2 + ang_vel[1] ** 2
        )
        reward = (
            1.0 * track_xy
            + 0.5 * track_yaw
            + 0.1 * upright
            + 0.5 * height
            + stillness
        )
        return float(reward)

    def _check_termination(self) -> bool:
        if not np.all(np.isfinite(self.data.qpos)):
            return True
        if self.data.qpos[2] < 0.2:
            return True
        return self.step_count >= self.episode_length
