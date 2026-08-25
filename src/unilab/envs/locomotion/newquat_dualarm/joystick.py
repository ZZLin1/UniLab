from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.base import Sensor
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.common.terrain_spawn import BaseSpawnManager
from unilab.envs.locomotion.newquat_dualarm.base import (
    DUALARM_SCENE,
    LEG_ACTION_DIM,
    MODEL_ACTION_DIM,
    NewquatDualArmBaseCfg,
    NewquatDualArmBaseEnv,
    build_position_gains,
)


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.278]


@dataclass
class NewquatDualArmDomainRandConfig(DomainRandConfig):
    randomize_kp: bool = False
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    randomize_kd: bool = False
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float
    target_foot_height: float = 0.1
    contact_forces_threshold: float = 200.0
    undesired_contact_threshold: float = 0.1


@dataclass
class JoystickSensor(Sensor):
    local_linvel = "local_linvel"
    gyro = "gyro"
    feet_pos = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]
    feet_force = ["FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"]
    feet_vel = ["FL_global_linvel", "FR_global_linvel", "RL_global_linvel", "RR_global_linvel"]


@registry.envcfg("NewquatDualArmJoystickFlat")
@dataclass
class NewquatDualArmJoystickFlatCfg(NewquatDualArmBaseCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(model_file=str(DUALARM_SCENE)))
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)
    domain_rand: NewquatDualArmDomainRandConfig = field(default_factory=NewquatDualArmDomainRandConfig)


class NewquatDualArmJoystickDRProvider(LocomotionDRProvider):
    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        return getattr(env, "_base_kp", None), getattr(env, "_base_kd", None)

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates,
                linvel,
                gyro,
                gravity,
                dof_pos[:, :LEG_ACTION_DIM],
                dof_vel[:, :LEG_ACTION_DIM],
                env.feet_phase[env_ids],
            ),
        )


@registry.env("NewquatDualArmJoystickFlat", sim_backend="mujoco")
@registry.env("NewquatDualArmJoystickFlat", sim_backend="motrix")
class NewquatDualArmWalkTask(NewquatDualArmBaseEnv):
    _cfg: NewquatDualArmJoystickFlatCfg

    def __init__(self, cfg: NewquatDualArmJoystickFlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend_kwargs: dict[str, Any] = {
            "base_name": cfg.asset.base_name,
            "push_body_name": cfg.domain_rand.push_body_name,
            "motrix_max_iterations": cfg.motrix_max_iterations,
            "post_step_forward_sensor": cfg.post_step_forward_sensor,
        }
        if backend_type != "motrix" or cfg.control_config.override_position_gains:
            backend_kwargs["position_actuator_gains"] = build_position_gains(cfg.control_config)
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            **backend_kwargs,
        )
        super().__init__(cfg, backend, num_envs, policy_action_dim=LEG_ACTION_DIM)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._spawn = BaseSpawnManager()
        self._base_kp, self._base_kd = (None, None)
        if backend_type != "motrix":
            self._base_kp, self._base_kd = backend.get_actuator_gains()
        self._init_undesired_contact_sensors()
        self._init_reward_functions()
        self._init_domain_randomization(NewquatDualArmJoystickDRProvider())
        self.phase = np.zeros((num_envs,), dtype=np.float32)
        self.feet_phase = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self.gait_frequency = 2.0
        self.feet_force = np.zeros((num_envs, len(cfg.sensor.feet_force), 3), dtype=np.float32)
        self.feet_pos = np.zeros((num_envs, len(cfg.sensor.feet_pos), 3), dtype=np.float32)
        self.feet_vel = np.zeros((num_envs, len(cfg.sensor.feet_vel), 3), dtype=np.float32)
        self._last_dof_vel_for_acc = np.zeros((num_envs, LEG_ACTION_DIM), dtype=get_global_dtype())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 49, "critic": 52}

    def _init_undesired_contact_sensors(self) -> None:
        foot_contacts = set(self._cfg.sensor.feet_force)
        self._undesired_contact_sensor_names = tuple(
            name
            for name in self._backend.get_sensor_names()
            if name.endswith("_contact") and name not in foot_contacts
        )

    def _undesired_contact_counts(self, threshold: float) -> np.ndarray:
        sensor_names = getattr(self, "_undesired_contact_sensor_names", ())
        if not sensor_names:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        values = np.asarray(self._backend.get_sensor_data_batch(sensor_names), dtype=get_global_dtype())
        return np.sum(values > float(threshold), axis=1).astype(get_global_dtype())

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "torques": rewards.torques,
            "dof_vel": self._reward_dof_vel,
            "energy": rewards.energy,
            "dof_acc": rewards.dof_acc,
            "similar_to_default": rewards.similar_to_default,
            "alive": rewards.alive,
            "orientation": rewards.orientation,
            "swing_feet_z": self._reward_swing_feet_z,
            "contact": self._reward_contact,
            "contact_forces": self._reward_contact_forces,
            "feet_slide": self._reward_feet_slide,
            "foot_drag": self._reward_foot_drag,
            "undesired_contacts": self._reward_undesired_contacts,
        }

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else actions
        )
        leg_ctrl = exec_actions * self._cfg.control_config.action_scale + self.default_angles_full[:LEG_ACTION_DIM]
        arm_ctrl = np.broadcast_to(
            self.default_angles_full[LEG_ACTION_DIM:MODEL_ACTION_DIM],
            (self._num_envs, MODEL_ACTION_DIM - LEG_ACTION_DIM),
        )
        return np.concatenate([leg_ctrl, arm_ctrl], axis=1, dtype=get_global_dtype())

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = super().reset(env_indices)
        dof_vel = self.get_dof_vel()[:, :LEG_ACTION_DIM]
        self._last_dof_vel_for_acc[np.asarray(env_indices, dtype=np.intp)] = dof_vel[
            np.asarray(env_indices, dtype=np.intp)
        ]
        return obs, info

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self.phase = np.fmod(self.phase + self._cfg.ctrl_dt * self.gait_frequency, 1.0)
        self.feet_phase[:, 0] = self.phase
        self.feet_phase[:, 3] = self.phase
        self.feet_phase[:, 1] = (self.phase + 0.5) % 1
        self.feet_phase[:, 2] = (self.phase + 0.5) % 1

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos_full = self.get_dof_pos()
        dof_vel_full = self.get_dof_vel()
        dof_pos = dof_pos_full[:, :LEG_ACTION_DIM]
        dof_vel = dof_vel_full[:, :LEG_ACTION_DIM]
        self._refresh_feet_buffers()
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        state.info["torques"] = self._estimate_pd_torques(state.info, dof_pos, dof_vel)
        self._update_commands(state.info)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel, self.feet_phase)
        terminated = gravity[:, 2] <= 0.5
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _refresh_feet_buffers(self) -> None:
        self.feet_force[:, :, :] = 0.0
        for i, name in enumerate(self._cfg.sensor.feet_force):
            values = np.asarray(self._backend.get_sensor_data(name), dtype=get_global_dtype())
            if values.ndim == 2 and values.shape[1] > 1:
                strength = np.linalg.norm(values, axis=1)
            else:
                strength = np.ravel(values)
            self.feet_force[:, i, 2] = strength
        for i, name in enumerate(self._cfg.sensor.feet_pos):
            self.feet_pos[:, i, :] = self._backend.get_sensor_data(name)
        for i, name in enumerate(self._cfg.sensor.feet_vel):
            self.feet_vel[:, i, :] = self._backend.get_sensor_data(name)

    def _update_commands(self, info: dict[str, Any]) -> None:
        commands = np.asarray(info["commands"], dtype=get_global_dtype())
        interval = max(int(round(float(self._cfg.commands.resampling_time) / self._cfg.ctrl_dt)), 1)
        steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
        mask = (steps > 0) & ((steps % interval) == 0)
        if np.any(mask):
            low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
            high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
            commands[mask] = np.random.uniform(low=low, high=high, size=(int(np.sum(mask)), 3))
        info["commands"] = commands

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel, feet_phase):
        noise_cfg = self._cfg.noise_config
        diff = self._obs_noise(dof_pos - self.default_angles, noise_cfg.scale_joint_angle)
        dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        obs = np.concatenate([gyro, -gravity, diff, dof_vel, last_actions, command, feet_phase], axis=1, dtype=get_global_dtype())
        critic = np.concatenate([obs, linvel], axis=1, dtype=get_global_dtype())
        return {"obs": obs, "critic": critic}

    def _compute_reward(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        cfg = self._reward_cfg
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            gravity=gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=cfg.tracking_sigma,
            base_height_target=cfg.base_height_target,
            base_height=np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype()),
        )
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _reward_swing_feet_z(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        is_swing = self.feet_phase >= 0.6
        height_error = np.square(self.feet_pos[:, :, 2] - self._reward_cfg.target_foot_height)
        return np.sum(np.exp(-height_error / 0.01) * is_swing, axis=1) / len(self._cfg.sensor.feet_pos)

    def _reward_foot_drag(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        foot_contact = self.get_foot_contact()
        is_swing = foot_contact < 0.5
        safe_height = self._reward_cfg.target_foot_height / 2.0
        error = np.square(np.clip(safe_height - self.feet_pos[..., 2], 0.0, None)) * is_swing
        return np.sum(error, axis=1)

    def _reward_contact(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        contact = self.feet_force[:, :, 2] > 0.1
        res = np.zeros(self._num_envs, dtype=np.float32)
        for i in range(len(self._cfg.sensor.feet_force)):
            is_contact = (self.feet_phase[:, i] < 0.6) | (self.gait_frequency < 1.0e-8)
            res += (contact[:, i] == is_contact).astype(np.float32)
        return res / len(self._cfg.sensor.feet_force)

    def _reward_dof_vel(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        return np.sum(np.abs(ctx.dof_vel), axis=1)

    def _reward_contact_forces(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        force = np.max(np.abs(self.feet_force), axis=2)
        return np.sum(np.clip(force - self._reward_cfg.contact_forces_threshold, 0.0, None), axis=1)

    def _reward_undesired_contacts(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        return self._undesired_contact_counts(self._reward_cfg.undesired_contact_threshold)

    def _reward_feet_slide(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        contact = np.max(np.abs(self.feet_force), axis=2) > 0.1
        return np.sum(np.linalg.norm(self.feet_vel[:, :, :2], axis=2) * contact, axis=1)

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = (dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt
        self._last_dof_vel_for_acc[:] = dof_vel
        return np.asarray(qacc, dtype=get_global_dtype())

    def _estimate_pd_torques(self, info: dict[str, Any], dof_pos: np.ndarray, dof_vel: np.ndarray) -> np.ndarray:
        actions = np.asarray(info.get("current_actions", np.zeros((dof_pos.shape[0], LEG_ACTION_DIM))), dtype=get_global_dtype())
        targets = actions * self._cfg.control_config.action_scale + self.default_angles
        return (
            float(self._cfg.control_config.leg_kp) * (targets - dof_pos)
            - float(self._cfg.control_config.leg_kd) * dof_vel
        ).astype(get_global_dtype())
