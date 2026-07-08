#!/usr/bin/env python3
"""Experimental 5x10 Newhex heterogeneous Motrix grid launcher.

This file is intentionally isolated under experiments/ so it cannot affect
normal UniLab training, eval, task configs, or robot assets.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPO_ROOT / "src" / "unilab" / "assets" / "robots" / "newhex"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl_ppo" / "NewhexJoystickFlat"
DEFAULT_MANIFEST = (
    REPO_ROOT / "experiments" / "newhex_hetero_motrix_grid" / "selected_robots.csv"
)
DEFAULT_NAMES_CSV = REPO_ROOT / "experiments" / "high_fitness_offspring_names.csv"
ROBOT_RE = re.compile(r"g\d{3}_[op]\d{3}$")


@dataclass(frozen=True)
class RobotPolicy:
    robot_name: str
    generation_dir: str
    scene_xml: Path
    run_dir: Path
    checkpoint: Path

    @property
    def load_run(self) -> str:
        return f"{self.generation_dir}/{self.robot_name}"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50, help="Number of distinct robots.")
    parser.add_argument("--seed", type=int, default=0, help="Random sampling seed.")
    parser.add_argument("--rows", type=int, default=5, help="Formation rows.")
    parser.add_argument("--cols", type=int, default=10, help="Formation columns.")
    parser.add_argument("--velocity", type=float, default=1.0, help="Forward command in m/s.")
    parser.add_argument(
        "--mode",
        choices=("manifest", "commands", "sequential", "motrix-grid"),
        default="motrix-grid",
        help="What to run.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint iteration/name to load, or 'latest'.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV manifest path for the sampled robot-policy pairs.",
    )
    parser.add_argument(
        "--policy-log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="Policy log root to discover runs from.",
    )
    parser.add_argument(
        "--names-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing a robot-name column. When set, robots are "
            "selected in CSV order instead of sampled randomly. For the current "
            f"36 high-fitness set, use {DEFAULT_NAMES_CSV.relative_to(REPO_ROOT)}."
        ),
    )
    parser.add_argument(
        "--success-exit-code",
        type=int,
        action="append",
        default=None,
        help="Sequential eval exit code treated as completed. Defaults to 0 and 245.",
    )
    return parser.parse_args(argv)


def _checkpoint_iteration(path: Path) -> int:
    if not path.name.startswith("model_") or path.suffix != ".pt":
        return -1
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path | None:
    if checkpoint != "latest":
        name = checkpoint if checkpoint.startswith("model_") else f"model_{checkpoint}"
        if not name.endswith(".pt"):
            name = f"{name}.pt"
        candidate = run_dir / name
        return candidate if candidate.is_file() else None

    checkpoints = sorted(
        (path for path in run_dir.glob("model_*.pt") if path.is_file()),
        key=_checkpoint_iteration,
    )
    return checkpoints[-1] if checkpoints else None


def _discover_pairs(*, checkpoint: str, log_root: Path) -> list[RobotPolicy]:
    if not log_root.is_dir():
        raise NotADirectoryError(f"missing policy log root: {log_root}")
    if not ASSET_ROOT.is_dir():
        raise NotADirectoryError(f"missing Newhex asset root: {ASSET_ROOT}")

    pairs: list[RobotPolicy] = []
    for generation_dir in sorted(path for path in log_root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in generation_dir.iterdir() if path.is_dir()):
            robot_name = run_dir.name
            if ROBOT_RE.fullmatch(robot_name) is None:
                continue
            scene_xml = ASSET_ROOT / robot_name / "scene.xml"
            if not scene_xml.is_file():
                continue
            resolved_checkpoint = _resolve_checkpoint(run_dir, checkpoint)
            if resolved_checkpoint is None:
                continue
            pairs.append(
                RobotPolicy(
                    robot_name=robot_name,
                    generation_dir=generation_dir.name,
                    scene_xml=scene_xml,
                    run_dir=run_dir,
                    checkpoint=resolved_checkpoint,
                )
            )
    return pairs


def _sample_pairs(pairs: Sequence[RobotPolicy], *, count: int, seed: int) -> list[RobotPolicy]:
    if count <= 0:
        raise ValueError(f"--count must be positive, got {count}")
    if len(pairs) < count:
        raise ValueError(f"only found {len(pairs)} eligible pairs, need {count}")
    return random.Random(seed).sample(list(pairs), count)


def _read_names_csv(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"names CSV does not exist: {path}")
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"names CSV has no header: {path}")
        name_field = "name" if "name" in reader.fieldnames else reader.fieldnames[-1]
        names = [row[name_field].strip() for row in reader if row.get(name_field, "").strip()]

    if not names:
        raise ValueError(f"names CSV is empty: {path}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"names CSV contains duplicate robot names: {', '.join(duplicates)}")
    invalid = [name for name in names if ROBOT_RE.fullmatch(name) is None]
    if invalid:
        raise ValueError(f"names CSV contains invalid robot names: {', '.join(invalid)}")
    return names


def _select_named_pairs(pairs: Sequence[RobotPolicy], names: Sequence[str]) -> list[RobotPolicy]:
    by_name = {pair.robot_name: pair for pair in pairs}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError(
            "could not find policy+scene pairs for requested robots: " + ", ".join(missing)
        )
    return [by_name[name] for name in names]


def _write_manifest(
    manifest: Path,
    selected: Sequence[RobotPolicy],
    *,
    rows: int,
    cols: int,
) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "slot",
                "row",
                "col",
                "robot_name",
                "load_run",
                "scene_xml",
                "checkpoint",
            ),
        )
        writer.writeheader()
        for slot, pair in enumerate(selected):
            writer.writerow(
                {
                    "slot": slot,
                    "row": slot // cols,
                    "col": slot % cols,
                    "robot_name": pair.robot_name,
                    "load_run": pair.load_run,
                    "scene_xml": str(pair.scene_xml.relative_to(REPO_ROOT)),
                    "checkpoint": str(pair.checkpoint.relative_to(REPO_ROOT)),
                }
            )


def _eval_command(pair: RobotPolicy, *, velocity: float, checkpoint: str) -> list[str]:
    command = [
        "uv",
        "run",
        "eval",
        "--algo",
        "ppo",
        "--task",
        "newhex_joystick_flat",
        "--sim",
        "motrix",
        f"algo.load_run={pair.load_run}",
        "+play_profile.scene.enabled=true",
        f"+play_profile.scene.source_model_file={pair.scene_xml}",
        f"play_profile.env.commands.fixed_command=[{float(velocity)},0.0,0.0]",
    ]
    if checkpoint != "latest":
        command.append(f"algo.checkpoint={checkpoint}")
    return command


def _print_commands(selected: Sequence[RobotPolicy], *, velocity: float, checkpoint: str) -> None:
    for pair in selected:
        print(" ".join(_eval_command(pair, velocity=velocity, checkpoint=checkpoint)))


def _run_sequential(
    selected: Sequence[RobotPolicy],
    *,
    velocity: float,
    checkpoint: str,
    success_exit_codes: set[int],
) -> int:
    for index, pair in enumerate(selected, start=1):
        command = _eval_command(pair, velocity=velocity, checkpoint=checkpoint)
        print(
            f"[newhex-hetero-grid] sequential {index}/{len(selected)} "
            f"{pair.robot_name}: {' '.join(command)}",
            flush=True,
        )
        exit_code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        if exit_code not in success_exit_codes:
            print(
                f"[newhex-hetero-grid] eval failed for {pair.robot_name} "
                f"with exit code {exit_code}",
                flush=True,
            )
            return exit_code
        if exit_code != 0:
            print(
                f"[newhex-hetero-grid] treating exit code {exit_code} as completed "
                f"for {pair.robot_name}",
                flush=True,
            )
    return 0


def _run_motrix_grid(*, count: int, rows: int, cols: int) -> int:
    if rows * cols != count:
        raise ValueError(f"formation rows*cols must equal count, got {rows}*{cols}!={count}")
    raise SystemExit(
        "Strict heterogeneous Motrix 5x10 playback is not supported by the current "
        "UniLab Motrix backend contract. The renderer accepts one SceneModel with a "
        "batch dimension; it cannot currently render 50 different robot XMLs, each "
        "driven by a different policy, in one interactive window. This experiment "
        "therefore stops before producing a misleading visualization. The manifest, "
        "commands, and sequential modes are safe and functional."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    log_root = args.policy_log_root.expanduser().resolve()
    pairs = _discover_pairs(checkpoint=str(args.checkpoint), log_root=log_root)
    names_csv = args.names_csv.expanduser().resolve() if args.names_csv is not None else None
    if names_csv is not None:
        selected = _select_named_pairs(pairs, _read_names_csv(names_csv))
        selection_note = f"selected {len(selected)} named pairs from {names_csv}"
    else:
        selected = _sample_pairs(pairs, count=int(args.count), seed=int(args.seed))
        selection_note = f"sampled {len(selected)} / {len(pairs)} eligible pairs"

    manifest = args.manifest.expanduser().resolve()
    _write_manifest(manifest, selected, rows=int(args.rows), cols=int(args.cols))
    print(
        f"[newhex-hetero-grid] {selection_note}; manifest={manifest}",
        flush=True,
    )

    if args.mode == "manifest":
        return 0
    if args.mode == "commands":
        _print_commands(selected, velocity=float(args.velocity), checkpoint=str(args.checkpoint))
        return 0
    if args.mode == "sequential":
        return _run_sequential(
            selected,
            velocity=float(args.velocity),
            checkpoint=str(args.checkpoint),
            success_exit_codes=set(args.success_exit_code or [0, 245]),
        )
    return _run_motrix_grid(count=len(selected), rows=int(args.rows), cols=int(args.cols))


if __name__ == "__main__":
    raise SystemExit(main())
