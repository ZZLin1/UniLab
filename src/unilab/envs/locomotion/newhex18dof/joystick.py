from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.newhex.joystick import (
    InitState,
    JoystickSensor,
    Commands,
    NewhexDomainRandConfig,
    NewhexWalkTask,
    RewardConfig,
    TerrainCurriculumCfg,
)
from unilab.envs.locomotion.newhex18dof.base import (
    Newhex18DofBaseCfg,
    apply_robot_defaults,
)


@registry.envcfg("Newhex18DofJoystickFlat")
@dataclass
class Newhex18DofJoystickFlatCfg(Newhex18DofBaseCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(model_file=""))
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    sensor: JoystickSensor = field(default_factory=JoystickSensor)
    domain_rand: NewhexDomainRandConfig = field(default_factory=NewhexDomainRandConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)
    reward_config: RewardConfig | None = None


@registry.env("Newhex18DofJoystickFlat", sim_backend="mujoco")
@registry.env("Newhex18DofJoystickFlat", sim_backend="motrix")
class Newhex18DofWalkTask(NewhexWalkTask):
    _cfg: Newhex18DofJoystickFlatCfg

    def __init__(
        self, cfg: Newhex18DofJoystickFlatCfg, num_envs: int = 1, backend_type: str = "mujoco"
    ):
        apply_robot_defaults(cfg)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 69, "critic": 72}

    def get_foot_contact(self) -> np.ndarray:
        contacts = []
        for sensor_name in self._cfg.sensor.feet_force:
            values = np.asarray(self._backend.get_sensor_data(sensor_name), dtype=get_global_dtype())
            if values.ndim == 2 and values.shape[1] > 1:
                contacts.append(np.linalg.norm(values, axis=1))
            else:
                contacts.append(np.ravel(values))
        return np.stack(contacts, axis=1)
