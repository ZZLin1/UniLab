#!/usr/bin/env python3
"""Run Newhex joystick sweep, then RF touch sweep.

Usage:
  uv run scripts/sweep_newhex_joystick_then_rf_touch.py
  uv run scripts/sweep_newhex_joystick_then_rf_touch.py --dry-run --conda-env ''
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
JOYSTICK_SWEEP = REPO_ROOT / "scripts" / "sweep_newhex_joystick_robots.py"
RF_TOUCH_SWEEP = REPO_ROOT / "scripts" / "sweep_newhex_rf_touch_robots.py"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="Run RF touch even if the joystick sweep exits non-zero.",
    )
    parser.add_argument(
        "sweep_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to both child sweep scripts. Put them after `--`.",
    )
    return parser.parse_args(argv)


def _child_args(args: argparse.Namespace) -> list[str]:
    child_args = list(args.sweep_args)
    if child_args and child_args[0] == "--":
        child_args = child_args[1:]
    return child_args


def _run_sweep(name: str, script: Path, child_args: Sequence[str]) -> int:
    if not script.is_file():
        raise FileNotFoundError(f"missing {name} sweep script: {script}")

    command = ["uv", "run", str(script), *child_args]
    print(f"[newhex-sweep-all] running {name}: {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    child_args = _child_args(args)

    joystick_code = _run_sweep("joystick", JOYSTICK_SWEEP, child_args)
    if joystick_code != 0 and not args.continue_after_failure:
        print(
            f"[newhex-sweep-all] joystick sweep failed with exit code {joystick_code}; "
            "skipping RF touch",
            flush=True,
        )
        return joystick_code

    rf_touch_code = _run_sweep("RF touch", RF_TOUCH_SWEEP, child_args)
    if joystick_code != 0:
        return joystick_code
    return rf_touch_code


if __name__ == "__main__":
    raise SystemExit(main())
