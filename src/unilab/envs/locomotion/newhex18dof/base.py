from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.newhex.base import (
    Asset as NewhexAsset,
    NewhexBaseCfg,
    NewhexBaseEnv,
    NoiseConfig,
)

FOOT_ORDER: tuple[str, ...] = ("RF", "RM", "RB", "LF", "LM", "LB")


@dataclass
class RobotConfig:
    name: str = "el4090"


@dataclass
class ControlConfig:
    action_scale: float = 0.25
    simulate_action_latency: bool = False
    Kp: float = 80.0
    Kd: float = 0.8


@dataclass
class Asset(NewhexAsset):
    base_name = "BASE"
    foot_name = "foot"
    ground = "floor"


@dataclass
class Newhex18DofBaseCfg(NewhexBaseCfg):
    robot: RobotConfig = field(default_factory=RobotConfig)
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    asset: Asset = field(default_factory=Asset)


ROBOT_SCENES: dict[str, Path] = {
    "el4090": ASSETS_ROOT_PATH / "robots" / "el4090" / "scene_newhex18dof.xml",
    "elspider_air": ASSETS_ROOT_PATH / "robots" / "elspider_air" / "scene_newhex18dof.xml",
}

ROBOT_BASE_NAMES: dict[str, str] = {
    "el4090": "BASE",
    "elspider_air": "trunk",
}

ROBOT_PD_GAINS: dict[str, tuple[float, float]] = {
    "el4090": (80.0, 0.8),
    "elspider_air": (50.0, 0.5),
}

ROBOT_RF_LOAD_BODIES: dict[str, str] = {
    "el4090": "RF_foot",
    "elspider_air": "RF_SHANK",
}


def apply_robot_defaults(cfg: Newhex18DofBaseCfg) -> None:
    robot_name = cfg.robot.name
    if robot_name not in ROBOT_SCENES:
        supported = ", ".join(sorted(ROBOT_SCENES))
        raise ValueError(f"Unsupported newhex18dof robot {robot_name!r}; expected one of: {supported}")

    cfg.scene.model_file = str(ROBOT_SCENES[robot_name])
    cfg.asset.base_name = ROBOT_BASE_NAMES[robot_name]
    cfg.control_config.Kp, cfg.control_config.Kd = ROBOT_PD_GAINS[robot_name]


class Newhex18DofBaseEnv(NewhexBaseEnv):
    _cfg: Newhex18DofBaseCfg

    def get_foot_contact(self) -> np.ndarray:
        contacts = []
        for foot in FOOT_ORDER:
            values = np.asarray(
                self._backend.get_sensor_data(f"{foot}_foot_contact"),
                dtype=get_global_dtype(),
            )
            if values.ndim == 2 and values.shape[1] > 1:
                contacts.append(np.linalg.norm(values, axis=1))
            else:
                contacts.append(np.ravel(values))
        return np.stack(contacts, axis=1)
