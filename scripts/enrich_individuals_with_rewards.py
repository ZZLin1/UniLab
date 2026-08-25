#!/usr/bin/env python3
"""Add final and exact-step TensorBoard rewards to an individuals CSV.

Each CSV row must contain a ``name`` such as ``g000_p000``. The matching run
directory is resolved as ``<log-root>/g000_p/g000_p000``. By default the input
CSV is replaced atomically after every run has been queried successfully.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from query_tensorboard_rewards import DEFAULT_STEP, query_run

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "scripts" / "unique_individuals_3050.csv"
# DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl_ppo" / "NewhexJoystickFlat"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl_ppo" / "NewhexRFTouch"
FINAL_REWARD_COLUMN = "Final mean reward"
INDIVIDUAL_NAME_RE = re.compile(r"(?P<generation>g\d{3}_[op])\d{3}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input CSV (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"Task training-log root (default: {DEFAULT_LOG_ROOT}).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help=f"Exact TensorBoard iteration for component rewards (default: {DEFAULT_STEP}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 1) + 4),
        help="Number of runs to read concurrently (default: up to 16).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write another CSV instead of replacing --csv.",
    )
    return parser.parse_args(argv)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            fieldnames = list(reader.fieldnames)
            rows = list(reader)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV does not exist: {path}") from exc
    if "name" not in fieldnames:
        raise ValueError(f"CSV must contain a 'name' column: {path}")
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return fieldnames, rows


def _run_dir(log_root: Path, name: str) -> Path:
    match = INDIVIDUAL_NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid individual name: {name!r}")
    return log_root / match.group("generation") / name


def _query_one(log_root: Path, name: str, step: int) -> dict[str, Any]:
    return query_run(_run_dir(log_root, name), step)


def _query_all(
    rows: Sequence[dict[str, str]], log_root: Path, step: int, workers: int
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")

    names = [row.get("name", "").strip() for row in rows]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        preview = ", ".join(repr(name) for name in duplicates[:10])
        raise ValueError(f"duplicate names in CSV: {preview}")

    reports: list[dict[str, Any] | None] = [None] * len(rows)
    # TensorBoard protobuf decoding is CPU-heavy, so processes provide materially
    # better throughput than threads when thousands of event files are queried.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_query_one, log_root, name, step): (index, name)
            for index, name in enumerate(names)
        }
        completed = 0
        for future in as_completed(futures):
            index, name = futures[future]
            try:
                reports[index] = future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"failed to query {name}: {exc}") from exc
            completed += 1
            if completed == len(rows) or completed % 100 == 0:
                print(f"Queried {completed}/{len(rows)} runs", flush=True)

    return [report for report in reports if report is not None]


def _reward_columns(reports: Sequence[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for report in reports:
        for reward in report["rewards"]:
            name = reward["name"]
            if name not in seen:
                seen.add(name)
                columns.append(name)
    return columns


def _format_number(value: Any) -> str:
    return "" if value is None else format(float(value), ".12g")


def _enrich_rows(
    rows: Sequence[dict[str, str]], reports: Sequence[dict[str, Any]], reward_columns: Sequence[str]
) -> tuple[list[dict[str, str]], int]:
    enriched: list[dict[str, str]] = []
    missing_values = 0
    for source_row, report in zip(rows, reports, strict=True):
        row = dict(source_row)
        final_reward = report["summary_final_mean_reward"]
        if final_reward is None:
            final_reward = report["mean_reward"]
        if final_reward is None:
            raise ValueError(f"no final mean reward for {source_row['name']}")
        row[FINAL_REWARD_COLUMN] = _format_number(final_reward)

        values: dict[str, str] = {}
        for reward in report["rewards"]:
            value = reward["value"]
            if value is None and reward["scale"] == 0:
                value = 0.0
            values[reward["name"]] = _format_number(value)
        for column in reward_columns:
            row[column] = values.get(column, "")
            if not row[column]:
                missing_values += 1
        enriched.append(row)
    return enriched, missing_values


def _write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    csv_path = args.csv.expanduser().resolve()
    log_root = args.log_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else csv_path

    fieldnames, rows = _read_csv(csv_path)
    print(
        f"Reading {len(rows)} runs at TensorBoard step {args.step} with {args.workers} workers",
        flush=True,
    )
    reports = _query_all(rows, log_root, args.step, args.workers)
    reward_columns = _reward_columns(reports)
    enriched_rows, missing_values = _enrich_rows(rows, reports, reward_columns)

    appended_columns = [
        column
        for column in (FINAL_REWARD_COLUMN, *reward_columns)
        if column not in fieldnames
    ]
    output_fieldnames = [*fieldnames, *appended_columns]
    _write_csv_atomic(output_path, output_fieldnames, enriched_rows)

    print(f"Wrote {len(enriched_rows)} rows to {output_path}")
    print(f"Reward columns ({len(reward_columns)}): {', '.join(reward_columns)}")
    if missing_values:
        print(
            f"Warning: left {missing_values} empty component cells because those rewards "
            "were absent or not logged at the requested step."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
