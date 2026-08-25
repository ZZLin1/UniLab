from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.manipulation.newhex.rf_touch import (
    InitState,
    NewhexRFTouchEnv,
    RFTouchSensor,
    RewardConfig,
    TargetSamplingConfig,
)
from unilab.envs.locomotion.newhex18dof.base import (
    Newhex18DofBaseCfg,
    ROBOT_RF_LOAD_BODIES,
    apply_robot_defaults,
)


@dataclass
class LoadConfig:
    body_name: str = ""
    force_range: list[float] = field(default_factory=lambda: [0.0, 50.0])


@registry.envcfg("Newhex18DofRFTouch")
@dataclass
class Newhex18DofRFTouchCfg(Newhex18DofBaseCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(model_file=""))
    max_episode_seconds: float = 10.0
    init_state: InitState = field(default_factory=InitState)
    sensor: RFTouchSensor = field(default_factory=RFTouchSensor)
    target_sampling: TargetSamplingConfig = field(default_factory=TargetSamplingConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    reward_config: RewardConfig | None = None


@registry.env("Newhex18DofRFTouch", sim_backend="mujoco")
@registry.env("Newhex18DofRFTouch", sim_backend="motrix")
class Newhex18DofRFTouchEnv(NewhexRFTouchEnv):
    _cfg: Newhex18DofRFTouchCfg

    def __init__(
        self, cfg: Newhex18DofRFTouchCfg, num_envs: int = 1, backend_type: str = "mujoco"
    ):
        apply_robot_defaults(cfg)
        if not cfg.load.body_name:
            cfg.load.body_name = ROBOT_RF_LOAD_BODIES[cfg.robot.name]
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 63, "critic": 63}

    def _refresh_feet_buffers(self, rows: np.ndarray | None = None) -> None:
        target = slice(None) if rows is None else np.asarray(rows, dtype=np.intp)
        for i, sensor_name in enumerate(self._cfg.sensor.feet_pos):
            self.feet_pos[target, i, :] = self._backend.get_sensor_data(sensor_name)[target]
        for i, sensor_name in enumerate(self._cfg.sensor.feet_vel):
            self.feet_vel[target, i, :] = self._backend.get_sensor_data(sensor_name)[target]
        self.feet_force[target, :, :] = 0.0
        for i, sensor_name in enumerate(self._cfg.sensor.feet_force):
            values = np.asarray(self._backend.get_sensor_data(sensor_name)[target], dtype=get_global_dtype())
            if values.ndim == 2 and values.shape[1] > 1:
                contact_strength = np.linalg.norm(values, axis=1)
            else:
                contact_strength = np.ravel(values)
            self.feet_force[target, i, 2] = contact_strength
