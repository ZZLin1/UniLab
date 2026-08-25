from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
    PdControlConfig,
)


@dataclass
class NoiseConfig(BaseNoiseConfig):
    pass


@dataclass
class ControlConfig(PdControlConfig):
    Kp: float = 280.0 
    Kd: float = 2.0


@dataclass
class Asset:
    base_name = "base_link"
    foot_name = "foot"
    ground = "floor"


@dataclass
class NewquatBaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


class NewquatBaseEnv(LocomotionBaseEnv):
    _cfg: NewquatBaseCfg

    def get_foot_pos(self) -> np.ndarray:
        foot_names = ["RF_pos", "LF_pos", "RH_pos", "LH_pos"]
        foot_pos = [self._backend.get_sensor_data(name) for name in foot_names]
        return np.stack(foot_pos, axis=1)

    def get_foot_contact(self) -> np.ndarray:
        contact_names = ["RF_foot_contact", "LF_foot_contact", "RH_foot_contact", "LH_foot_contact"]
        contacts = []
        for name in contact_names:
            values = np.asarray(self._backend.get_sensor_data(name))
            if values.ndim == 2 and values.shape[1] > 1:
                contacts.append(np.linalg.norm(values, axis=1))
            else:
                contacts.append(np.ravel(values))
        return np.stack(contacts, axis=1)
