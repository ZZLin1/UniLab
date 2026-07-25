#!/usr/bin/env python3
"""Drive a merged heterogeneous Newhex scene with each robot's own policy.

Implementation shape:
  - each selected robot runs in its own standard single-robot Motrix env
  - each env loads its matching RSL-RL PPO policy
  - a separate merged Motrix scene is used only for visualization
  - after every policy step, each single-robot qpos is copied into its segment
    of the merged scene data and rendered

This keeps the experiment isolated from normal UniLab eval/training entrypoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.scene import SceneCfg
from unilab.training.common import create_env, ensure_registries
from unilab.training.reward import resolve_reward_dict
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "newhex_hetero_motrix_grid"
DEFAULT_MANIFEST = EXPERIMENT_DIR / "selected_robots.csv"
DEFAULT_SCENE = EXPERIMENT_DIR / "generated" / "hetero_5_scene.xml"


@dataclass
class RobotSession:
    robot_name: str
    env: Any
    wrapped_env: RslRlVecEnvWrapper
    policy: Any
    obs: Any
    qpos_width: int


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--spacing-x", type=float, default=2.0)
    parser.add_argument("--spacing-y", type=float, default=2.0)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument(
        "--heading-yaw",
        type=float,
        default=0.0,
        help="World-frame target heading yaw in radians for direction-mode playback.",
    )
    parser.add_argument(
        "--heading-stiffness",
        type=float,
        default=0.5,
        help="P gain used to convert heading error into the commanded yaw rate.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--steps-per-render",
        type=int,
        default=1,
        help="Run this many policy/env steps before each render sync.",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Do not throttle playback with time.sleep(1 / fps).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--lookat", nargs=3, type=float, default=[4.0, 0.0, 0.8])
    parser.add_argument("--distance", type=float, default=9.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load envs/policies and merged Motrix data without opening a render window.",
    )
    return parser.parse_args(argv)


def _read_manifest(path: Path, count: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < count:
        raise ValueError(f"manifest has {len(rows)} rows, need {count}")
    return rows[:count]


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _select_device(raw: str | None) -> str:
    if raw:
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_cfg() -> Any:
    config_dir = str((REPO_ROOT / "conf" / "ppo").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="config", overrides=["task=newhex_joystick_flat/motrix"])


def _checkpoint_path(row: dict[str, str], checkpoint: str) -> Path:
    if checkpoint == "latest":
        return _resolve_repo_path(row["checkpoint"])
    run_dir = _resolve_repo_path(row["checkpoint"]).parent
    name = checkpoint if checkpoint.startswith("model_") else f"model_{checkpoint}"
    if not name.endswith(".pt"):
        name = f"{name}.pt"
    candidate = run_dir / name
    if not candidate.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {candidate}")
    return candidate


def _refresh_env_state(env: Any) -> None:
    if env.state is None:
        return
    env._state = env.update_state(env.state)


def _align_env_root_forward(env: Any) -> None:
    """Force the single-robot playback env to start facing world +X."""
    qpos = np.asarray(env._init_qpos, dtype=np.float64)[None, :].copy()
    qvel = np.asarray(env._init_qvel, dtype=np.float64)[None, :].copy()
    base_pos = np.asarray(env._backend.get_base_pos(), dtype=np.float64)
    dof_pos = np.asarray(env.get_dof_pos(), dtype=np.float64)

    qpos[:, 0:2] = 0.0
    qpos[:, 2] = base_pos[:, 2]
    qpos[:, 3:7] = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    qpos[:, 7 : 7 + dof_pos.shape[1]] = dof_pos
    qvel[:, :] = 0.0
    env._backend.set_state(np.asarray([0], dtype=np.int32), qpos, qvel)
    _refresh_env_state(env)


def _algo_train_cfg(cfg: Any) -> dict[str, Any]:
    raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    train_cfg = normalize_ppo_train_cfg(cast(dict[str, Any], raw))
    train_cfg.setdefault("runner", {})
    train_cfg["runner"]["logger"] = "none"
    train_cfg["logger"] = "none"
    return train_cfg


def _make_session(
    *,
    row: dict[str, str],
    cfg: Any,
    train_cfg: dict[str, Any],
    device: str,
    velocity: float,
    heading_yaw: float,
    heading_stiffness: float,
    checkpoint: str,
) -> RobotSession:
    from rsl_rl.runners import OnPolicyRunner

    scene_xml = _resolve_repo_path(row["scene_xml"])
    reward_config = resolve_reward_dict(cfg)
    env_override = {
        "reward_config": reward_config,
        "scene": SceneCfg(model_file=str(scene_xml)),
        "commands": {
            "fixed_command": [float(velocity), 0.0, 0.0],
            "heading_command": True,
            "heading_range": [float(heading_yaw), float(heading_yaw)],
            "heading_control_stiffness": float(heading_stiffness),
        },
    }
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=env_override,
        sim_backend="motrix",
        task_name="NewhexJoystickFlat",
    )
    env.set_autoreset(False)
    wrapped_env = RslRlVecEnvWrapper(env, device=device)
    runner = OnPolicyRunner(wrapped_env, copy.deepcopy(train_cfg), log_dir=None, device=device)
    ckpt = _checkpoint_path(row, checkpoint)
    runner.load(
        str(ckpt),
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
    )
    policy = runner.get_inference_policy(device=device)
    obs, _ = wrapped_env.reset()
    _align_env_root_forward(env)
    obs = wrapped_env.get_observations()
    qpos_width = 7 + int(env.get_dof_pos().shape[1])
    return RobotSession(
        robot_name=row["robot_name"],
        env=env,
        wrapped_env=wrapped_env,
        policy=policy,
        obs=obs,
        qpos_width=qpos_width,
    )


def _session_qpos_motrix(session: RobotSession, *, offset_xy: tuple[float, float]) -> np.ndarray:
    env = session.env
    backend = env._backend
    base_pos = np.asarray(backend.get_base_pos(), dtype=np.float64)
    base_quat = np.asarray(backend.get_base_quat(), dtype=np.float64)
    dof_pos = np.asarray(env.get_dof_pos(), dtype=np.float64)
    base_pos = base_pos.copy()
    base_pos[:, 0] += float(offset_xy[0])
    base_pos[:, 1] += float(offset_xy[1])
    qpos_mujoco = np.concatenate([base_pos, base_quat, dof_pos], axis=1)
    return np.asarray(backend._mujoco_qpos_to_motrix(qpos_mujoco), dtype=np.float64)[0]


def _merged_qpos(
    sessions: Sequence[RobotSession],
    *,
    cols: int,
    spacing_x: float,
    spacing_y: float,
) -> np.ndarray:
    parts = []
    for slot, session in enumerate(sessions):
        row = slot // cols
        col = slot % cols
        parts.append(
            _session_qpos_motrix(
                session,
                offset_xy=(col * spacing_x, -row * spacing_y),
            )
        )
    return np.concatenate(parts, axis=0)[None, :]


def _render_settings() -> Any:
    from motrixsim.render import RenderSettings

    settings = RenderSettings.quality()
    settings.enable_shadow = True
    return settings


def _step_sessions(sessions: Sequence[RobotSession]) -> None:
    for session in sessions:
        action = session.policy(session.obs)
        session.obs, _reward, _done, _info = session.wrapped_env.step(action)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    count = int(args.count)
    manifest_rows = _read_manifest(args.manifest.expanduser().resolve(), count)
    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"merged scene does not exist: {scene}")

    ensure_registries()
    cfg = _load_cfg()
    device = _select_device(args.device)
    train_cfg = _algo_train_cfg(cfg)
    print(f"[play-merged] device={device}", flush=True)

    sessions = []
    for index, row in enumerate(manifest_rows, start=1):
        print(f"[play-merged] loading {index}/{count}: {row['robot_name']}", flush=True)
        sessions.append(
            _make_session(
                row=row,
                cfg=cfg,
                train_cfg=train_cfg,
                device=device,
                velocity=float(args.velocity),
                heading_yaw=float(args.heading_yaw),
                heading_stiffness=float(args.heading_stiffness),
                checkpoint=str(args.checkpoint),
            )
        )

    import motrixsim as mtx
    from motrixsim.render import RenderApp

    model = mtx.load_model(str(scene))
    data = mtx.SceneData(model)
    qpos = _merged_qpos(
        sessions,
        cols=int(args.cols),
        spacing_x=float(args.spacing_x),
        spacing_y=float(args.spacing_y),
    )
    if qpos.shape[1] != int(model.num_dof_pos):
        raise RuntimeError(
            f"merged qpos width mismatch: sessions={qpos.shape[1]}, model={model.num_dof_pos}"
        )
    data.set_dof_pos(qpos[0], model)
    model.forward_kinematic(data)
    if args.check_only:
        print(
            f"[play-merged] check-only ok: sessions={len(sessions)} qpos_width={qpos.shape[1]}",
            flush=True,
        )
        for session in sessions:
            close = getattr(session.env, "close", None)
            if callable(close):
                close()
        return 0

    frame_dt = 1.0 / max(float(args.fps), 1e-6)
    steps_per_render = max(int(args.steps_per_render), 1)
    with RenderApp("WARN") as render:
        render.launch(model, render_settings=_render_settings())
        render.set_main_camera(None)
        render.system_camera.set_view(
            [float(value) for value in args.lookat],
            float(args.distance),
            float(args.elevation),
            float(args.azimuth),
        )
        print("[play-merged] close the Motrix window to exit", flush=True)
        try:
            with torch.inference_mode():
                while not render.is_closed:
                    for _ in range(steps_per_render):
                        _step_sessions(sessions)
                    qpos = _merged_qpos(
                        sessions,
                        cols=int(args.cols),
                        spacing_x=float(args.spacing_x),
                        spacing_y=float(args.spacing_y),
                    )
                    data.set_dof_pos(qpos[0], model)
                    model.forward_kinematic(data)
                    render.sync(data)
                    if not args.no_sleep:
                        time.sleep(frame_dt)
        finally:
            for session in sessions:
                close = getattr(session.env, "close", None)
                if callable(close):
                    close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
