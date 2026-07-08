#!/usr/bin/env python3
"""Rename timestamp PPO run directories to generation individual names.

The expected layout is:
  logs/rsl_rl_ppo/<Task>/<generation>/<timestamp_run>

For each generation directory such as ``g003_o``, timestamp children are sorted
lexicographically and renamed to ``g003_o000`` ... ``g003_o049``.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (
    REPO_ROOT / "logs" / "rsl_rl_ppo" / "NewhexJoystickFlat",
    REPO_ROOT / "logs" / "rsl_rl_ppo" / "NewhexRFTouch",
)
DEFAULT_MANIFEST = REPO_ROOT / "logs" / "rsl_rl_ppo" / "policy_run_dir_rename_manifest.csv"
GEN_RE = re.compile(r"g(?P<generation>\d{3})_(?P<kind>[op])$")
ROBOT_RE = re.compile(r"g\d{3}_[op]\d{3}$")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_motrix$")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        dest="roots",
        help="Task log root to process. Defaults to joystick and RF touch roots.",
    )
    parser.add_argument(
        "--start-generation",
        type=int,
        default=3,
        help="First generation number to rename.",
    )
    parser.add_argument(
        "--end-generation",
        type=int,
        default=None,
        help="Last generation number to rename. Defaults to all generations found.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV manifest path for the rename plan.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename directories. Without this flag, only print and write the plan.",
    )
    return parser.parse_args(argv)


def _generation_allowed(name: str, *, start: int, end: int | None) -> bool:
    match = GEN_RE.fullmatch(name)
    if match is None:
        return False
    generation = int(match.group("generation"))
    if generation < start:
        return False
    return end is None or generation <= end


def _target_name(generation_name: str, index: int) -> str:
    return f"{generation_name}{index:03d}"


def _missing_indices(generation_dir: Path) -> list[int]:
    existing = {
        int(path.name[-3:])
        for path in generation_dir.iterdir()
        if path.is_dir() and ROBOT_RE.fullmatch(path.name)
    }
    return [index for index in range(50) if index not in existing]


def _build_plan(roots: Sequence[Path], *, start_generation: int, end_generation: int | None):
    plan: list[tuple[Path, Path]] = []
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(f"task log root does not exist: {root}")
        for generation_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if not _generation_allowed(
                generation_dir.name,
                start=start_generation,
                end=end_generation,
            ):
                continue
            timestamp_dirs = sorted(
                path
                for path in generation_dir.iterdir()
                if path.is_dir() and TIMESTAMP_RE.fullmatch(path.name)
            )
            if not timestamp_dirs:
                continue
            missing = _missing_indices(generation_dir)
            if len(timestamp_dirs) > len(missing):
                raise ValueError(
                    f"{generation_dir} has {len(timestamp_dirs)} timestamp dirs but only "
                    f"{len(missing)} available individual names"
                )
            for source, index in zip(timestamp_dirs, missing, strict=True):
                plan.append((source, generation_dir / _target_name(generation_dir.name, index)))
    return plan


def _check_plan(plan: Sequence[tuple[Path, Path]]) -> None:
    targets = [target for _, target in plan]
    duplicate_targets = sorted({target for target in targets if targets.count(target) > 1})
    if duplicate_targets:
        names = "\n".join(str(path) for path in duplicate_targets[:20])
        raise ValueError(f"duplicate rename targets in plan:\n{names}")
    existing_targets = sorted(target for target in targets if target.exists())
    if existing_targets:
        names = "\n".join(str(path) for path in existing_targets[:20])
        raise FileExistsError(f"rename targets already exist:\n{names}")


def _write_manifest(manifest: Path, plan: Sequence[tuple[Path, Path]]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("source", "target"))
        writer.writeheader()
        for source, target in plan:
            writer.writerow(
                {
                    "source": str(source.relative_to(REPO_ROOT)),
                    "target": str(target.relative_to(REPO_ROOT)),
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    roots = tuple(path.expanduser().resolve() for path in (args.roots or DEFAULT_ROOTS))
    manifest = args.manifest.expanduser().resolve()
    plan = _build_plan(
        roots,
        start_generation=args.start_generation,
        end_generation=args.end_generation,
    )
    _check_plan(plan)
    _write_manifest(manifest, plan)

    action = "Renaming" if args.apply else "Would rename"
    print(f"[rename-policy-runs] {action} {len(plan)} directories", flush=True)
    print(f"[rename-policy-runs] manifest: {manifest}", flush=True)
    for source, target in plan[:20]:
        print(f"  {source.relative_to(REPO_ROOT)} -> {target.name}", flush=True)
    if len(plan) > 20:
        print(f"  ... {len(plan) - 20} more", flush=True)

    if args.apply:
        for source, target in plan:
            source.rename(target)
        print("[rename-policy-runs] completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
