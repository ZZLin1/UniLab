#!/usr/bin/env python3
"""Drive a merged heterogeneous Newhex scene with RF-touch policies.

Each selected robot runs in its own standard single-robot Motrix env and loads
its matching `NewhexRFTouch` PPO policy. A separate merged Motrix scene is used
only for visualization. Reset and target sampling are left to the standard
`NewhexRFTouch` env implementation.
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
DEFAULT_MANIFEST = EXPERIMENT_DIR / "selected_rf_touch_robots.csv"
DEFAULT_SCENE = EXPERIMENT_DIR / "generated" / "hetero_36_scene.xml"


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
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--spacing-x", type=float, default=2.0)
    parser.add_argument("--spacing-y", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--episode-seconds", type=float, default=3.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--lookat", nargs=3, type=float, default=[5.0, -5.0, 0.8])
    parser.add_argument("--distance", type=float, default=12.0)
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
        return compose(config_name="config", overrides=["task=newhex_rf_touch/motrix"])


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
    checkpoint: str,
    episode_seconds: float,
) -> RobotSession:
    from rsl_rl.runners import OnPolicyRunner

    scene_xml = _resolve_repo_path(row["scene_xml"])
    reward_config = resolve_reward_dict(cfg)
    env_override = {
        "reward_config": reward_config,
        "scene": SceneCfg(model_file=str(scene_xml)),
        "max_episode_seconds": float(episode_seconds),
    }
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=env_override,
        sim_backend="motrix",
        task_name="NewhexRFTouch",
    )
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


def _close_sessions(sessions: Sequence[RobotSession]) -> None:
    for session in sessions:
        close = getattr(session.env, "close", None)
        if callable(close):
            close()


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
    print(f"[play-rf-touch] device={device}", flush=True)

    sessions = []
    for index, row in enumerate(manifest_rows, start=1):
        print(f"[play-rf-touch] loading {index}/{count}: {row['robot_name']}", flush=True)
        sessions.append(
            _make_session(
                row=row,
                cfg=cfg,
                train_cfg=train_cfg,
                device=device,
                checkpoint=str(args.checkpoint),
                episode_seconds=float(args.episode_seconds),
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
            f"[play-rf-touch] check-only ok: sessions={len(sessions)} qpos_width={qpos.shape[1]}",
            flush=True,
        )
        _close_sessions(sessions)
        return 0

    frame_dt = 1.0 / max(float(args.fps), 1e-6)
    with RenderApp("WARN") as render:
        render.launch(model, render_settings=_render_settings())
        render.set_main_camera(None)
        render.system_camera.set_view(
            [float(value) for value in args.lookat],
            float(args.distance),
            float(args.elevation),
            float(args.azimuth),
        )
        print("[play-rf-touch] close the Motrix window to exit", flush=True)
        try:
            with torch.inference_mode():
                while not render.is_closed:
                    for session in sessions:
                        action = session.policy(session.obs)
                        session.obs, _reward, _done, _info = session.wrapped_env.step(action)
                    qpos = _merged_qpos(
                        sessions,
                        cols=int(args.cols),
                        spacing_x=float(args.spacing_x),
                        spacing_y=float(args.spacing_y),
                    )
                    data.set_dof_pos(qpos[0], model)
                    model.forward_kinematic(data)
                    render.sync(data)
                    time.sleep(frame_dt)
        finally:
            _close_sessions(sessions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
