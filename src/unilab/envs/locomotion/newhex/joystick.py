from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
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
from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.envs.locomotion.newhex.base import NewhexBaseCfg, NewhexBaseEnv


@dataclass
class InitState:
    pos = [0.0, 0.0, 1.5]


@dataclass
class NewhexDomainRandConfig(DomainRandConfig):
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float
    target_foot_height: float = 0.1
    contact_forces_threshold: float = 200.0


@dataclass
class JoystickSensor(Sensor):
    local_linvel = "local_linvel"
    gyro = "gyro"
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


@registry.envcfg("NewhexJoystickFlat")
@dataclass
class NewhexJoystickFlatCfg(NewhexBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g000_p036" / "scene.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)
    domain_rand: NewhexDomainRandConfig = field(default_factory=NewhexDomainRandConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)


class NewhexJoystickDomainRandomizationProvider(LocomotionDRProvider):
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
                info_updates, linvel, gyro, gravity, dof_pos, dof_vel, env.feet_phase[env_ids]
            ),
        )


def _resolve_scene_xml_path(path: str, model_file: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.is_file():
        return candidate.resolve()
    return (model_file.parent / candidate).resolve()


def _scene_home_keyframe_height(scene: SceneCfg) -> float:
    model_file = Path(scene.model_file).resolve()
    candidate_paths = [model_file]
    candidate_paths.extend(_resolve_scene_xml_path(path, model_file) for path in scene.fragment_files)

    for path in candidate_paths:
        root = ET.parse(path).getroot()
        key = root.find("./keyframe/key[@name='home']")
        if key is None:
            key = root.find("./keyframe/key")
        if key is None:
            continue
        qpos_text = key.get("qpos")
        if qpos_text is None:
            raise ValueError(f"scene keyframe in {path} is missing qpos=...")
        qpos = [float(value) for value in qpos_text.split()]
        if len(qpos) < 3:
            raise ValueError(f"scene keyframe in {path} must include floating-base xyz qpos")
        return qpos[2]

    searched = ", ".join(str(path) for path in candidate_paths)
    raise ValueError(f"Newhex scene must define a home keyframe with qpos; searched: {searched}")


def _apply_scene_base_height_target(cfg: NewhexJoystickFlatCfg) -> None:
    assert cfg.reward_config is not None
    cfg.reward_config.base_height_target = _scene_home_keyframe_height(cfg.scene) - 0.05


@registry.env("NewhexJoystickFlat", sim_backend="mujoco")
@registry.env("NewhexJoystickFlat", sim_backend="motrix")
class NewhexWalkTask(NewhexBaseEnv):
    _cfg: NewhexJoystickFlatCfg

    def __init__(self, cfg: NewhexJoystickFlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        _apply_scene_base_height_target(cfg)

        self._scene_terrain_origins: np.ndarray | None = None
        scene_cfg = cfg.scene
        terrain_generator = scene_cfg.terrain.generator if scene_cfg.terrain is not None else None

        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            position_actuator_gains={"kp": cfg.control_config.Kp, "kd": cfg.control_config.Kd},
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        self._terrain_surface_sampler = getattr(backend, "terrain_surface_sampler", None)
        self._terrain_surface_sample_height = self._resolve_terrain_surface_sample_height()
        terrain_origins = getattr(backend, "terrain_origins", None)
        if terrain_origins is not None:
            self._scene_terrain_origins = terrain_origins
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._init_reward_functions()
        self._init_domain_randomization(NewhexJoystickDomainRandomizationProvider())
        if self._scene_terrain_origins is not None and terrain_generator is not None:
            self._spawn = TerrainSpawnManager(
                num_envs,
                self._scene_terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=self._terrain_surface_sampler,
            )

        self.phase = np.zeros((num_envs,), dtype=np.float32)
        self.feet_phase = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self.gait_frequency = 2
        self.feet_force = np.zeros((num_envs, len(cfg.sensor.feet_force), 3), dtype=np.float32)
        self.feet_pos = np.zeros((num_envs, len(cfg.sensor.feet_pos), 3), dtype=np.float32)
        self.feet_vel = np.zeros((num_envs, len(cfg.sensor.feet_vel), 3), dtype=np.float32)
        self._last_dof_vel_for_acc = np.zeros(
            (num_envs, self._num_action), dtype=get_global_dtype()
        )

    def get_playback_model(self, env_index: int | None = None) -> Any:
        return super().get_playback_model(env_index)

    def _resolve_terrain_surface_sample_height(
        self,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        sampler = self._terrain_surface_sampler
        if sampler is None:
            return None

        sample_height = getattr(sampler, "sample_height", None)
        if not callable(sample_height):
            raise TypeError("terrain_surface_sampler must expose sample_height(xy)")
        return cast(Callable[[np.ndarray], np.ndarray], sample_height)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 75, "critic": 78}

    def _init_reward_functions(self):
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

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self.phase = np.fmod(self.phase + self._cfg.ctrl_dt * self.gait_frequency, 1.0)
        tripod_a = (0, 2, 4)
        tripod_b = (1, 3, 5)
        phase_a = self.phase
        phase_b = (self.phase + 0.5) % 1.0
        self.feet_phase[:, tripod_a] = phase_a[:, None]
        self.feet_phase[:, tripod_b] = phase_b[:, None]

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self.feet_force[:, :, :] = 0
        for i in range(len(self._cfg.sensor.feet_force)):
            self.feet_force[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_force[i])
        for i in range(len(self._cfg.sensor.feet_pos)):
            self.feet_pos[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_pos[i])
        for i in range(len(self._cfg.sensor.feet_vel)):
            self.feet_vel[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_vel[i])

        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        state.info["torques"] = self._estimate_pd_torques(state.info, dof_pos, dof_vel)

        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(
            state.info, linvel, gyro, gravity, dof_pos, dof_vel, self.feet_phase
        )
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        done = state.terminated | state.truncated
        if np.any(done):
            done_indices = np.where(done)[0]
            stats = self._spawn.update_on_done(
                done_indices, self._backend.get_base_pos()[done_indices]
            )
            if stats:
                if "log" not in state.info:
                    state.info["log"] = {}
                for k, v in stats.items():
                    state.info["log"][f"terrain_curriculum/{k}"] = float(v)
        return state

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel, feet_phase
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        obs = np.concatenate(
            [gyro, -gravity, diff, dof_vel, last_actions, command, feet_phase],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate([obs, linvel], axis=1, dtype=get_global_dtype())
        return {"obs": obs, "critic": critic}

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
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
            base_height=self._reward_base_height_values(),
        )
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    # ── reward functions (robot-specific) ────────────────────────────

    def _reward_base_height_values(self) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        sample_height = self._terrain_surface_sample_height
        if sample_height is None:
            return np.asarray(base_pos[:, 2], dtype=get_global_dtype())

        surface = np.asarray(sample_height(base_pos[:, :2]), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2] - surface, dtype=get_global_dtype())

    def _reward_swing_feet_z(self, ctx: RewardContext) -> np.ndarray:
        is_swing = self.feet_phase >= 0.6
        target_height = 0.1
        height_error = np.square(self.feet_pos[:, :, 2] - target_height)
        swing_rew = np.exp(-height_error / 0.01) * is_swing
        reward: np.ndarray = np.sum(swing_rew, axis=1) / len(self._cfg.sensor.feet_pos)
        return reward

    def _reward_foot_drag(self, ctx: RewardContext) -> np.ndarray:
        foot_pos = self.get_foot_pos()
        foot_heights = foot_pos[..., 2]
        foot_contact = self.get_foot_contact()
        is_swing = foot_contact < 0.5
        safe_height = self._reward_cfg.target_foot_height / 2.0
        height_error = np.clip(safe_height - foot_heights, 0.0, None)
        error = np.square(height_error) * is_swing
        drag_penalty: np.ndarray = np.sum(error, axis=1)
        return drag_penalty

    def _reward_contact(self, ctx: RewardContext) -> np.ndarray:
        contact = self.feet_force[:, :, 2] > 0.1
        res = np.zeros(self._num_envs, dtype=np.float32)
        for i in range(len(self._cfg.sensor.feet_force)):
            is_contact = (self.feet_phase[:, i] < 0.6) | (self.gait_frequency < 1.0e-8)
            res += (contact[:, i] == is_contact).astype(np.float32)
        return res / len(self._cfg.sensor.feet_force)

    def _reward_dof_vel(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        return np.sum(np.abs(ctx.dof_vel), axis=1)  # type: ignore[no-any-return]

    def _reward_contact_forces(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        force = np.max(np.abs(self.feet_force), axis=2)
        excess = np.clip(force - self._reward_cfg.contact_forces_threshold, 0.0, None)
        return np.sum(excess, axis=1)  # type: ignore[no-any-return]

    def _reward_feet_slide(self, ctx: RewardContext) -> np.ndarray:
        del ctx
        contact = np.max(np.abs(self.feet_force), axis=2) > 0.1
        lateral_speed = np.linalg.norm(self.feet_vel[:, :, :2], axis=2)
        return np.sum(lateral_speed * contact, axis=1)  # type: ignore[no-any-return]

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt)
        self._last_dof_vel_for_acc[:] = dof_vel
        return np.asarray(qacc, dtype=get_global_dtype())

    def _estimate_pd_torques(
        self, info: dict, dof_pos: np.ndarray, dof_vel: np.ndarray
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
