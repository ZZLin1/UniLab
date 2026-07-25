from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    PdControlConfig,
)

UNDESIRED_CONTACT_SENSOR_PREFIX = "undesired_"


@dataclass
class NoiseConfig(BaseNoiseConfig):
    pass


@dataclass
class ControlConfig(PdControlConfig):
    Kp: float = 80.0
    Kd: float =0.5


@dataclass
class Asset:
    base_name = "base_link"
    foot_name = "foot"
    ground = "floor"


@dataclass
class NewhexBaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


class NewhexBaseEnv(LocomotionBaseEnv):
    _cfg: NewhexBaseCfg

    def _init_undesired_contact_sensors(self) -> None:
        self._undesired_contact_sensor_names = tuple(
            name
            for name in self._backend.get_sensor_names()
            if name.startswith(UNDESIRED_CONTACT_SENSOR_PREFIX)
        )

    def _undesired_contact_counts(self, threshold: float) -> np.ndarray:
        sensor_names = getattr(self, "_undesired_contact_sensor_names", ())
        if not sensor_names:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        values = np.asarray(
            self._backend.get_sensor_data_batch(sensor_names),
            dtype=get_global_dtype(),
        )
        return np.sum(values > float(threshold), axis=1).astype(get_global_dtype())

    def get_foot_pos(self) -> np.ndarray:
        """Get foot positions. Returns shape (num_envs, 6, 3)"""
        foot_names = ["RF_pos", "RM_pos", "RB_pos", "LF_pos", "LM_pos", "LB_pos"]
        foot_pos = [self._backend.get_sensor_data(name) for name in foot_names]
        return np.stack(foot_pos, axis=1)

    def get_foot_contact(self) -> np.ndarray:
        """Get foot contact forces. Returns shape (num_envs, 6)"""
        contact_names = ["RF_foot_contact", "RM_foot_contact", "RB_foot_contact", "LF_foot_contact", "LM_foot_contact", "LB_foot_contact"]
        contacts = [self._backend.get_sensor_data(name)[:, 0] for name in contact_names]
        return np.stack(contacts, axis=1)
