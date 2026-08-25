from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import SimBackend
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_apply
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)

GO2_DUALARM_SCENE = ASSETS_ROOT_PATH / "robots" / "go2_dual_arm" / "scene_flat.xml"
B2_DUALARM_SCENE = ASSETS_ROOT_PATH / "robots" / "b2_dual_arm" / "mjcf" / "scene_flat.xml"
ANYMAL_C_DUALARM_SCENE = ASSETS_ROOT_PATH / "robots" / "anymal_c_dual_arm" / "scene_flat.xml"
DUALARM_SCENE = B2_DUALARM_SCENE
LEG_ACTION_DIM = 12
MODEL_ACTION_DIM = 24


@dataclass
class NoiseConfig(BaseNoiseConfig):
    pass


@dataclass
class ControlConfig:
    action_scale: float = 0.25
    simulate_action_latency: bool = False
    Kp: float = 35.0
    Kd: float = 0.5
    override_position_gains: bool = True
    leg_kp: float = 280.0
    leg_kd: float = 2.0
    arm_kp: list[float] = field(default_factory=lambda: [95.0, 115.0, 100.0, 52.0, 54.0, 55.0])
    arm_kd: list[float] = field(default_factory=lambda: [3.5, 3.8, 2.5, 1.5, 1.5, 1.5])


@dataclass
class Asset:
    base_name: str = "base_link"
    foot_name: str = "foot"
    ground: str = "floor"
    right_endpoint_body: str = "right_link6"


@dataclass
class NewquatDualArmBaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


def build_position_gains(cfg: ControlConfig) -> dict[str, np.ndarray]:
    leg_kp = np.full((LEG_ACTION_DIM,), float(cfg.leg_kp), dtype=np.float64)
    leg_kd = np.full((LEG_ACTION_DIM,), float(cfg.leg_kd), dtype=np.float64)
    arm_kp = np.asarray(cfg.arm_kp, dtype=np.float64)
    arm_kd = np.asarray(cfg.arm_kd, dtype=np.float64)
    if arm_kp.shape != (6,) or arm_kd.shape != (6,):
        raise ValueError("control_config.arm_kp and arm_kd must each have shape (6,)")
    return {
        "kp": np.concatenate([leg_kp, arm_kp, arm_kp]),
        "kd": np.concatenate([leg_kd, arm_kd, arm_kd]),
    }


class NewquatDualArmBaseEnv(LocomotionBaseEnv):
    _cfg: NewquatDualArmBaseCfg

    def __init__(
        self,
        cfg: NewquatDualArmBaseCfg,
        backend: SimBackend,
        num_envs: int = 1,
        *,
        policy_action_dim: int,
    ):
        self._policy_action_dim = int(policy_action_dim)
        super().__init__(cfg, backend, num_envs)
        self._model_action_dim = backend.num_actuators
        if self._model_action_dim != MODEL_ACTION_DIM:
            raise ValueError(f"NewquatDualArm expects 24 actuators, got {self._model_action_dim}")
        self._leg_slice = slice(0, LEG_ACTION_DIM)
        self._left_arm_slice = slice(12, 18)
        self._right_arm_slice = slice(18, 24)

    def _init_action_space(self) -> None:
        ctrl_range = self._backend.get_actuator_ctrl_range()
        exposed = ctrl_range[: self._policy_action_dim]
        self._action_space = gym.spaces.Box(
            exposed[:, 0],
            exposed[:, 1],
            (self._policy_action_dim,),
            dtype=float,
        )  # type: ignore[assignment]

    def _init_buffers(self) -> None:
        dtype = get_global_dtype()
        self._init_qpos = np.asarray(self._backend.get_keyframe_qpos(self._keyframe_name), dtype=dtype)
        self.default_angles_full = np.asarray(self._init_qpos[-MODEL_ACTION_DIM:], dtype=dtype)
        self.default_angles = self.default_angles_full[: self._policy_action_dim].copy()
        self._init_qvel = np.asarray(self._backend.get_init_qvel(), dtype=dtype)

    def get_foot_pos(self) -> np.ndarray:
        names = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]
        return np.stack([self._backend.get_sensor_data(name) for name in names], axis=1)

    def get_foot_contact(self) -> np.ndarray:
        names = ["FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"]
        contacts = []
        for name in names:
            values = np.asarray(self._backend.get_sensor_data(name), dtype=get_global_dtype())
            if values.ndim == 2 and values.shape[1] > 1:
                contacts.append(np.linalg.norm(values, axis=1))
            else:
                contacts.append(np.ravel(values))
        return np.stack(contacts, axis=1)

    def right_endpoint_world_pos(self) -> np.ndarray:
        local = np.asarray(self._backend.get_sensor_data("right_endpoint_pos"), dtype=get_global_dtype())
        ref_pos = np.asarray(
            self._backend.get_sensor_data("right_armbasepoint_world_pos"), dtype=get_global_dtype()
        )
        ref_quat = np.asarray(
            self._backend.get_sensor_data("right_armbasepoint_world_quat"), dtype=get_global_dtype()
        )
        if self._backend.backend_type == "motrix":
            ref_quat = ref_quat[:, [3, 0, 1, 2]]
        return ref_pos + np_quat_apply(ref_quat, local)
