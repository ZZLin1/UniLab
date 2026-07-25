#!/usr/bin/env python3
"""Import all offspring URDFs in one generation, one MuJoCo tuning session at a time.

Edit OFFSPRING_DIR below when moving from generation_014 to generation_015, etc.

Usage:
  uv run scripts/import_generation_offspring_urdfs.py
  uv run scripts/import_generation_offspring_urdfs.py --offspring-dir /path/to/generation_015/offspring
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

# Change this path by hand for the next generation, for example generation_n/offspring.
OFFSPRING_DIR = Path(
    "/home/zzl/Hexapod_Generator/Hexapod_Generator/experiments/nsga2_hexapod/"
    "generations/generation_060/offspring"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offspring-dir",
        type=Path,
        default=OFFSPRING_DIR,
        help="Directory whose child folders contain offspring URDF files.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional first robot name or URDF stem to process, useful for resuming.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print import commands without launching MuJoCo or writing assets.",
    )
    return parser.parse_args(argv)


def _find_urdfs(offspring_dir: Path) -> list[Path]:
    if not offspring_dir.is_dir():
        raise NotADirectoryError(f"offspring directory does not exist: {offspring_dir}")
    urdfs = sorted(path for path in offspring_dir.glob("*/*.urdf") if path.is_file())
    if not urdfs:
        raise FileNotFoundError(f"no URDF files found under child folders of {offspring_dir}")
    return urdfs


def _apply_start(urdfs: Sequence[Path], start: str | None) -> list[Path]:
    if start is None:
        return list(urdfs)
    selected = [path for path in urdfs if path.stem >= start]
    if not selected:
        raise ValueError(f"start={start!r} did not match any remaining URDF stems")
    return selected


def _import_urdf(urdf: Path, *, dry_run: bool) -> int:
    robot_name = urdf.stem
    command = ["uv", "run", "unilab-import-robot", str(urdf), robot_name]
    print(f"[import-generation] importing {robot_name}: {' '.join(command)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    offspring_dir = args.offspring_dir.expanduser().resolve()
    urdfs = _apply_start(_find_urdfs(offspring_dir), args.start)

    print(f"[import-generation] found {len(urdfs)} URDF files in {offspring_dir}", flush=True)
    for index, urdf in enumerate(urdfs, start=1):
        print(
            f"[import-generation] {index}/{len(urdfs)}: {urdf.relative_to(offspring_dir)}",
            flush=True,
        )
        exit_code = _import_urdf(urdf, dry_run=args.dry_run)
        if exit_code != 0:
            print(
                f"[import-generation] import failed for {urdf.stem} with exit code {exit_code}",
                flush=True,
            )
            return exit_code

    print("[import-generation] completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
