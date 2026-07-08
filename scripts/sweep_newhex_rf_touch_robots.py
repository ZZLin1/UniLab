#!/usr/bin/env python3
"""Train Newhex RF touch PPO sequentially across robot asset variants.

Usage:
  uv run scripts/sweep_newhex_rf_touch_robots.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RF_TOUCH_CFG = REPO_ROOT / "src" / "unilab" / "envs" / "manipulation" / "newhex" / "rf_touch.py"
NEWHEX_ASSETS = REPO_ROOT / "src" / "unilab" / "assets" / "robots" / "newhex"
DEFAULT_CSV_PATH = REPO_ROOT / "newhex_rf_touch_sweep_results.csv"
TRAIN_COMMAND = ("uv", "run", "train", "--algo", "ppo", "--task", "newhex_rf_touch", "--sim", "motrix")
MODEL_FILE_RE = re.compile(
    r'(?P<prefix>ASSETS_ROOT_PATH\s*/\s*"robots"\s*/\s*"newhex"\s*/\s*")'
    r'(?P<robot>[^"]+)'
    r'(?P<suffix>"\s*/\s*"scene\.xml")'
)
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MEAN_REWARD_RES = (
    re.compile(rf"\bMean reward\s*[:=]\s*(?P<value>{FLOAT_RE})", re.IGNORECASE),
    re.compile(rf"\bfinal_mean_reward\b[^-+\d]*(?P<value>{FLOAT_RE})", re.IGNORECASE),
    re.compile(rf"\bmean_reward\b[^-+\d]*(?P<value>{FLOAT_RE})", re.IGNORECASE),
)
ITERATION_RES = (
    re.compile(r"\bLearning iteration\s+(?P<iteration>\d+)\s*/\s*(?P<total>\d+)", re.IGNORECASE),
    re.compile(r"\bit(?:eration)?\s*[:=]\s*(?P<iteration>\d+)\b", re.IGNORECASE),
)
CSV_FIELDS = ("robot_name", "iteration", "mean_reward", "exit_code")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="First numeric suffix to run.")
    parser.add_argument("--end", type=int, default=49, help="Last numeric suffix to run, inclusive.")
    parser.add_argument(
        "--name-template",
        default="g041_o{index:03d}",
        help="Robot directory template. Use {index} for the numeric suffix.",
    )
    parser.add_argument(
        "--rf-touch-cfg",
        type=Path,
        default=RF_TOUCH_CFG,
        help="Path to src/unilab/envs/manipulation/newhex/rf_touch.py.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=NEWHEX_ASSETS,
        help="Directory containing Newhex robot asset folders.",
    )
    parser.add_argument(
        "--conda-env",
        default="unilab",
        help="Require this active CONDA_DEFAULT_ENV before training. Use '' to skip the check.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="CSV file to append one row to after each training run.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue with the next robot if a training command exits non-zero.",
    )
    parser.add_argument(
        "--restore-original",
        action="store_true",
        help="Restore the original RF touch robot directory after the sweep finishes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned edits and training commands without changing files or training.",
    )
    return parser.parse_args(argv)


def _format_robot_name(template: str, index: int) -> str:
    try:
        robot_name = template.format(index=index)
    except Exception as exc:
        raise ValueError(f"invalid --name-template {template!r}: {exc}") from exc
    if Path(robot_name).name != robot_name:
        raise ValueError(f"robot name must be a single path component: {robot_name!r}")
    return robot_name


def _read_current_robot(rf_touch_cfg: Path) -> tuple[str, str]:
    text = rf_touch_cfg.read_text()
    matches = list(MODEL_FILE_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one Newhex scene.xml model_file in {rf_touch_cfg}, "
            f"found {len(matches)}"
        )
    return text, matches[0].group("robot")


def _write_robot(rf_touch_cfg: Path, robot_name: str) -> None:
    text, _ = _read_current_robot(rf_touch_cfg)

    def replace(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{robot_name}{match.group('suffix')}"

    rf_touch_cfg.write_text(MODEL_FILE_RE.sub(replace, text, count=1))


def _validate_assets(assets_root: Path, robot_name: str) -> None:
    scene_xml = assets_root / robot_name / "scene.xml"
    if not scene_xml.is_file():
        raise FileNotFoundError(f"missing robot scene.xml: {scene_xml}")


def _check_conda_env(expected: str) -> None:
    if not expected:
        return
    active = os.environ.get("CONDA_DEFAULT_ENV")
    if active != expected:
        raise RuntimeError(
            f"expected active conda env {expected!r}, got {active!r}; "
            f"run `conda activate {expected}` before launching this script"
        )


def _parse_training_line(line: str) -> tuple[int | None, float | None]:
    clean = ANSI_RE.sub("", line)
    iteration = None
    mean_reward = None

    for pattern in ITERATION_RES:
        match = pattern.search(clean)
        if match is not None:
            iteration = int(match.group("iteration"))
            break

    for pattern in MEAN_REWARD_RES:
        match = pattern.search(clean)
        if match is not None:
            mean_reward = float(match.group("value"))
            break

    return iteration, mean_reward


def _run_training(robot_name: str) -> tuple[int, int | None, float | None]:
    print(f"[newhex-rf-touch-sweep] training {robot_name}: {' '.join(TRAIN_COMMAND)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        TRAIN_COMMAND,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    last_iteration = None
    last_mean_reward = None
    for line in process.stdout:
        print(line, end="", flush=True)
        iteration, mean_reward = _parse_training_line(line)
        if iteration is not None:
            last_iteration = iteration
        if mean_reward is not None:
            last_mean_reward = mean_reward

    exit_code = process.wait()
    return exit_code, last_iteration, last_mean_reward


def _append_result(
    csv_path: Path,
    *,
    robot_name: str,
    iteration: int | None,
    mean_reward: float | None,
    exit_code: int,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "robot_name": robot_name,
                "iteration": "" if iteration is None else iteration,
                "mean_reward": "" if mean_reward is None else mean_reward,
                "exit_code": exit_code,
            }
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.start > args.end:
        raise ValueError(f"--start must be <= --end, got {args.start} > {args.end}")

    rf_touch_cfg = args.rf_touch_cfg.expanduser().resolve()
    assets_root = args.assets_root.expanduser().resolve()
    csv_path = args.csv_path.expanduser().resolve()
    _check_conda_env(args.conda_env)
    _, original_robot = _read_current_robot(rf_touch_cfg)

    failures: list[tuple[str, int]] = []
    try:
        for index in range(args.start, args.end + 1):
            robot_name = _format_robot_name(args.name_template, index)
            _validate_assets(assets_root, robot_name)
            print(f"[newhex-rf-touch-sweep] set RF touch robot: {robot_name}", flush=True)
            if not args.dry_run:
                _write_robot(rf_touch_cfg, robot_name)
                exit_code, iteration, mean_reward = _run_training(robot_name)
                _append_result(
                    csv_path,
                    robot_name=robot_name,
                    iteration=iteration,
                    mean_reward=mean_reward,
                    exit_code=exit_code,
                )
                if mean_reward is None:
                    print(
                        f"[newhex-rf-touch-sweep] warning: no Mean reward found in logs "
                        f"for {robot_name}",
                        flush=True,
                    )
                else:
                    print(
                        f"[newhex-rf-touch-sweep] recorded {robot_name}: "
                        f"iteration={iteration}, mean_reward={mean_reward}",
                        flush=True,
                    )
            else:
                print(
                    f"[newhex-rf-touch-sweep] dry-run command: {' '.join(TRAIN_COMMAND)}",
                    flush=True,
                )
                print(f"[newhex-rf-touch-sweep] dry-run csv path: {csv_path}", flush=True)
                exit_code = 0

            if exit_code != 0:
                failures.append((robot_name, exit_code))
                print(
                    f"[newhex-rf-touch-sweep] training failed for {robot_name} "
                    f"with exit code {exit_code}",
                    flush=True,
                )
                if not args.continue_on_failure:
                    return exit_code
    finally:
        if args.restore_original and not args.dry_run:
            print(
                f"[newhex-rf-touch-sweep] restoring original RF touch robot: {original_robot}",
                flush=True,
            )
            _write_robot(rf_touch_cfg, original_robot)

    if failures:
        print("[newhex-rf-touch-sweep] completed with failures:", flush=True)
        for robot_name, exit_code in failures:
            print(f"  {robot_name}: exit code {exit_code}", flush=True)
        return 1

    print("[newhex-rf-touch-sweep] completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
