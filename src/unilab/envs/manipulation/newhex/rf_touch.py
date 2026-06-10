from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationCapabilities, DomainRandomizationProvider, ResetPlan
from unilab.dr.dr_utils import zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_apply_inverse
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.base import Sensor
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.newhex.base import NewhexBaseCfg, NewhexBaseEnv

FOOT_ORDER: tuple[str, ...] = ("RF", "RM", "RB", "LF", "LM", "LB")
SUPPORT_FOOT_INDICES = np.asarray([1, 2, 3, 4, 5], dtype=np.int32)


@dataclass
class InitState:
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.59])


@dataclass
class TargetSamplingConfig:
    low: list[float] = field(default_factory=lambda: [0.2, -0.2, 0.0])
    high: list[float] = field(default_factory=lambda: [0.8, 0.6, 1.2])
    success_distance: float = 0.05


@dataclass
class LoadConfig:
    body_name: str = "RF_foot_link"
    force_range: list[float] = field(default_factory=lambda: [0.0, 50.0])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    target_sigma: float = 0.1
    support_still_sigma: float = 0.02
    support_vel_sigma: float = 0.05
    contact_threshold: float = 0.1


@dataclass
class RFTouchSensor(Sensor):
    local_linvel = "local_linvel"
    gyro = "gyro"
    upvector = "upvector"
    feet_pos = ["RF_pos", "RM_pos", "RB_pos", "LF_pos", "LM_pos", "LB_pos"]
    feet_force = [
        "RF_foot_contact",
        "RM_foot_contact",
        "RB_foot_contact",
        "LF_foot_contact",
        "LM_foot_contact",
        "LB_foot_contact",
    ]
    feet_vel = ["RF_vel", "RM_vel", "RB_vel", "LF_vel", "LM_vel", "LB_vel"]


@registry.envcfg("NewhexRFTouch")
@dataclass
class NewhexRFTouchCfg(NewhexBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "newhex6" / "scene.xml")
        )
    )
    max_episode_seconds: float = 10.0
    init_state: InitState = field(default_factory=InitState)
    sensor: RFTouchSensor = field(default_factory=RFTouchSensor)
    target_sampling: TargetSamplingConfig = field(default_factory=TargetSamplingConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    reward_config: RewardConfig | None = None


class NewhexRFTouchDRProvider(DomainRandomizationProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        del env, capabilities

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:3] = np.asarray(env.cfg.init_state.pos, dtype=get_global_dtype())

        env._sample_targets(env_ids, qpos[:, 0:2])
        info_updates: dict[str, Any] = {
            "commands": env._compute_commands_for_envs(env_ids),
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "target_pos_w": env._target_pos_w[env_ids].copy(),
        }
        return ResetPlan(env_ids=env_ids, qpos=qpos, qvel=qvel, info_updates=info_updates)

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        env._refresh_feet_buffers(rows=env_ids)
        env._support_foot_start_pos_w[env_ids] = env.feet_pos[env_ids][:, SUPPORT_FOOT_INDICES, :]
        env._rf_load_force[env_ids] = env._sample_rf_load(len(env_ids))
        command = env._compute_commands_for_envs(env_ids)
        info_updates["commands"] = command
        info_updates["target_pos_w"] = env._target_pos_w[env_ids].copy()

        gyro = env.get_gyro()[env_ids]
        gravity = env._backend.get_sensor_data(env.cfg.sensor.upvector)[env_ids]
        dof_pos = env.get_dof_pos()[env_ids]
        dof_vel = env.get_dof_vel()[env_ids]
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(info_updates, gyro, gravity, dof_pos, dof_vel),
        )


@registry.env("NewhexRFTouch", sim_backend="mujoco")
@registry.env("NewhexRFTouch", sim_backend="motrix")
class NewhexRFTouchEnv(NewhexBaseEnv):
    _cfg: NewhexRFTouchCfg

    def __init__(self, cfg: NewhexRFTouchCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            position_actuator_gains={"kp": cfg.control_config.Kp, "kd": cfg.control_config.Kd},
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._reward_cfg = cfg.reward_config
        self._target_pos_w = np.zeros((num_envs, 3), dtype=get_global_dtype())
        self._rf_load_force = np.zeros((num_envs,), dtype=get_global_dtype())
        self._rf_body_id = self._backend.get_body_id(cfg.load.body_name)
        self._rf_body_ids = np.asarray([self._rf_body_id], dtype=np.int32)
        self.feet_pos = np.zeros((num_envs, len(cfg.sensor.feet_pos), 3), dtype=get_global_dtype())
        self.feet_vel = np.zeros((num_envs, len(cfg.sensor.feet_vel), 3), dtype=get_global_dtype())
        self.feet_force = np.zeros(
            (num_envs, len(cfg.sensor.feet_force), 3), dtype=get_global_dtype()
        )
        self._support_foot_start_pos_w = np.zeros(
            (num_envs, len(SUPPORT_FOOT_INDICES), 3), dtype=get_global_dtype()
        )
        self._last_dof_vel_for_acc = np.zeros((num_envs, self._num_action), dtype=get_global_dtype())
        self._init_reward_functions()
        self._init_domain_randomization(NewhexRFTouchDRProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 69, "critic": 69}

    def get_playback_extra_data(self) -> dict[str, np.ndarray]:
        return {
            "marker_positions": self._target_pos_w.copy(),
            "box_centers": self._target_box_center_w(),
            "box_sizes": np.broadcast_to(
                self._target_box_size(),
                (self._num_envs, 3),
            ).copy(),
        }

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "base_lin_vel_zero": self._reward_base_lin_vel_zero,
            "base_ang_vel_zero": self._reward_base_ang_vel_zero,
            "support_feet_still": self._reward_support_feet_still,
            "support_feet_vel": self._reward_support_feet_vel,
            "support_feet_contact": self._reward_support_feet_contact,
            "rf_target": self._reward_rf_target,
            "action_rate": rewards.action_rate,
            "torques": rewards.torques,
            "dof_vel": self._reward_dof_vel,
            "dof_acc": rewards.dof_acc,
            "energy": rewards.energy,
        }

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        dof_vel = self.get_dof_vel()
        if dof_vel.shape[0] == self._num_envs:
            self._last_dof_vel_for_acc[env_ids] = dof_vel[env_ids]
        return obs, info

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        return gravity[:, 2] <= 0.5

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        ctrl = super().apply_action(actions, state)
        self._apply_rf_load()
        return ctrl

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._refresh_feet_buffers()

        state.info["commands"] = self._compute_commands_for_envs(
            np.arange(self._num_envs, dtype=np.int32)
        )
        state.info["target_pos_w"] = self._target_pos_w.copy()
        state.info["target_sampling_box_center_w"] = self._target_box_center_w()
        state.info["target_sampling_box_size"] = self._target_box_size()
        state.info["rf_load_force"] = self._rf_load_force.copy()
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        state.info["torques"] = self._estimate_pd_torques(state.info, dof_pos, dof_vel)

        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _sample_targets(self, env_ids: np.ndarray, base_xy: np.ndarray) -> None:
        cfg = self._cfg.target_sampling
        low = np.asarray(cfg.low, dtype=get_global_dtype())
        high = np.asarray(cfg.high, dtype=get_global_dtype())
        samples = np.random.uniform(low=low, high=high, size=(len(env_ids), 3)).astype(
            get_global_dtype()
        )
        self._target_pos_w[env_ids, 0:2] = base_xy + samples[:, 0:2]
        self._target_pos_w[env_ids, 2] = samples[:, 2]

    def _sample_rf_load(self, num_reset: int) -> np.ndarray:
        low, high = self._cfg.load.force_range
        return np.asarray(
            np.random.uniform(float(low), float(high), size=(num_reset,)),
            dtype=get_global_dtype(),
        )

    def _apply_rf_load(self) -> None:
        force = np.zeros((self._num_envs, 1, 3), dtype=get_global_dtype())
        force[:, 0, 2] = -self._rf_load_force
        self._backend.apply_body_force(self._rf_body_ids, force)

    def _compute_commands_for_envs(self, env_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(env_ids, dtype=np.intp)
        rf_pos = self._backend.get_sensor_data(self._cfg.sensor.feet_pos[0])[rows]
        base_quat = self._backend.get_base_quat()[rows]
        delta_w = self._target_pos_w[rows] - rf_pos
        return np.asarray(np_quat_apply_inverse(base_quat, delta_w), dtype=get_global_dtype())

    def _compute_obs(
        self, info: dict[str, Any], gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        command = np.asarray(info["commands"], dtype=get_global_dtype())
        last_actions = info.get("current_actions", np.zeros_like(diff))
        obs = np.concatenate(
            [gyro, -gravity, diff, dof_vel, last_actions, command],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": obs.copy()}

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        del gravity
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=self._reward_cfg.tracking_sigma,
        )
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=True,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _refresh_feet_buffers(self, rows: np.ndarray | None = None) -> None:
        target = slice(None) if rows is None else np.asarray(rows, dtype=np.intp)
        for i, sensor_name in enumerate(self._cfg.sensor.feet_pos):
            self.feet_pos[target, i, :] = self._backend.get_sensor_data(sensor_name)[target]
        for i, sensor_name in enumerate(self._cfg.sensor.feet_vel):
            self.feet_vel[target, i, :] = self._backend.get_sensor_data(sensor_name)[target]
        self.feet_force[target, :, :] = 0.0
        for i, sensor_name in enumerate(self._cfg.sensor.feet_force):
            values = self._backend.get_sensor_data(sensor_name)[target]
            self.feet_force[target, i, 2] = np.ravel(values)

    def _reward_base_lin_vel_zero(self, ctx: RewardContext) -> np.ndarray:
        error = np.sum(np.square(ctx.linvel), axis=1)
        return np.exp(-error / self._reward_cfg.tracking_sigma)  # type: ignore[no-any-return]

    def _reward_base_ang_vel_zero(self, ctx: RewardContext) -> np.ndarray:
        error = np.sum(np.square(ctx.gyro), axis=1)
        return np.exp(-error / self._reward_cfg.tracking_sigma)  # type: ignore[no-any-return]

    def _reward_support_feet_still(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        delta = self.feet_pos[:, SUPPORT_FOOT_INDICES, :] - self._support_foot_start_pos_w
        error = np.sum(np.square(delta[..., :2]), axis=(1, 2))
        return np.exp(-error / self._reward_cfg.support_still_sigma)  # type: ignore[no-any-return]

    def _reward_support_feet_vel(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        speed_sq = np.sum(np.square(self.feet_vel[:, SUPPORT_FOOT_INDICES, :2]), axis=(1, 2))
        return np.exp(-speed_sq / self._reward_cfg.support_vel_sigma)  # type: ignore[no-any-return]

    def _reward_support_feet_contact(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        contact = self.feet_force[:, SUPPORT_FOOT_INDICES, 2] > self._reward_cfg.contact_threshold
        return np.mean(contact.astype(get_global_dtype()), axis=1)  # type: ignore[no-any-return]

    def _reward_rf_target(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        distance = np.linalg.norm(self.feet_pos[:, 0, :] - self._target_pos_w, axis=1)
        success = distance <= self._cfg.target_sampling.success_distance
        shaped = np.exp(-np.square(distance) / self._reward_cfg.target_sigma)
        return np.where(success, 1.0, shaped).astype(get_global_dtype())

    def _reward_dof_vel(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        return np.sum(np.abs(ctx.dof_vel), axis=1)  # type: ignore[no-any-return]

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt)
        self._last_dof_vel_for_acc[:] = dof_vel
        return np.asarray(qacc, dtype=get_global_dtype())

    def _estimate_pd_torques(
        self, info: dict[str, Any], dof_pos: np.ndarray, dof_vel: np.ndarray
    ) -> np.ndarray:
        actions = np.asarray(
            info.get("current_actions", np.zeros((dof_pos.shape[0], self._num_action))),
            dtype=get_global_dtype(),
        )
        if self._cfg.control_config.simulate_action_latency:
            actions = np.asarray(info.get("last_actions", actions), dtype=get_global_dtype())
        targets = actions * self._cfg.control_config.action_scale + self.default_angles
        torques = (
            float(self._cfg.control_config.Kp) * (targets - dof_pos)
            - float(self._cfg.control_config.Kd) * dof_vel
        )
        return np.asarray(torques, dtype=get_global_dtype())

    def _target_box_center_w(self) -> np.ndarray:
        low = np.asarray(self._cfg.target_sampling.low, dtype=get_global_dtype())
        high = np.asarray(self._cfg.target_sampling.high, dtype=get_global_dtype())
        base_xy = self._backend.get_base_pos()[:, :2]
        center = np.zeros((self._num_envs, 3), dtype=get_global_dtype())
        center[:, :2] = base_xy + 0.5 * (low[:2] + high[:2])
        center[:, 2] = 0.5 * (low[2] + high[2])
        return center

    def _target_box_size(self) -> np.ndarray:
        low = np.asarray(self._cfg.target_sampling.low, dtype=get_global_dtype())
        high = np.asarray(self._cfg.target_sampling.high, dtype=get_global_dtype())
        return high - low
