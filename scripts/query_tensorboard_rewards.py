#!/usr/bin/env python3
"""Query reward weights and exact-step values from a UniLab TensorBoard run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_STEP = 999
MEAN_REWARD_TAGS = ("Train/mean_reward", "reward/mean")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="Directory containing run_config.json and events.*"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help=f"Exact TensorBoard iteration to query (default: {DEFAULT_STEP}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text table.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required file does not exist: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _value_at_step(accumulator: EventAccumulator, tag: str, step: int) -> float | None:
    matches = [event.value for event in accumulator.Scalars(tag) if event.step == step]
    return float(matches[-1]) if matches else None


def query_run(run_dir: Path, step: int) -> dict[str, Any]:
    """Return configured reward scales and scalar values at one exact iteration."""
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run directory does not exist: {run_dir}")
    if not any(run_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"no TensorBoard event file found in: {run_dir}")

    config = _load_json(run_dir / "run_config.json")
    try:
        scales = config["config"]["reward"]["scales"]
    except KeyError as exc:
        raise KeyError(f"missing config.reward.scales in {run_dir / 'run_config.json'}") from exc
    if not isinstance(scales, dict):
        raise ValueError("config.reward.scales must be a JSON object")

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))

    rewards = []
    for name, scale in scales.items():
        tag = f"reward/{name}"
        value = _value_at_step(accumulator, tag, step) if tag in scalar_tags else None
        rewards.append(
            {
                "name": str(name),
                "scale": float(scale),
                "value": value,
                "logged": tag in scalar_tags,
            }
        )

    mean_reward_tag = next((tag for tag in MEAN_REWARD_TAGS if tag in scalar_tags), None)
    mean_reward = (
        _value_at_step(accumulator, mean_reward_tag, step) if mean_reward_tag is not None else None
    )
    reference_tag = mean_reward_tag or next(
        (f"reward/{item['name']}" for item in rewards if item["logged"]), None
    )
    available_steps = (
        sorted({event.step for event in accumulator.Scalars(reference_tag)})
        if reference_tag is not None
        else []
    )

    summary_path = run_dir / "run_summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    values = [item["value"] for item in rewards if item["value"] is not None]
    return {
        "run_dir": str(run_dir),
        "step": step,
        "available_step_min": available_steps[0] if available_steps else None,
        "available_step_max": available_steps[-1] if available_steps else None,
        "rewards": rewards,
        "component_sum": math.fsum(values) if values else None,
        "mean_reward_tag": mean_reward_tag,
        "mean_reward": mean_reward,
        "summary_final_mean_reward": summary.get("final_mean_reward"),
        "mean_episode_length": _value_at_step(accumulator, "Train/mean_episode_length", step)
        if "Train/mean_episode_length" in scalar_tags
        else None,
    }


def _format_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.9g}"


def _print_table(report: dict[str, Any]) -> None:
    print(f"Run:  {report['run_dir']}")
    print(f"Step: {report['step']}")
    if report["mean_reward"] is None:
        lower = report["available_step_min"]
        upper = report["available_step_max"]
        suffix = f"; available range: {lower}..{upper}" if lower is not None else ""
        print(f"Warning: no mean reward was logged at this exact step{suffix}")

    name_width = max(11, *(len(item["name"]) for item in report["rewards"]))
    print()
    print(f"{'reward':<{name_width}}  {'scale':>12}  {'value@step':>14}")
    print(f"{'-' * name_width}  {'-' * 12}  {'-' * 14}")
    for item in report["rewards"]:
        value = item["value"]
        if value is None and item["scale"] == 0:
            rendered_value = "0 (disabled)"
        elif value is None and not item["logged"]:
            rendered_value = "N/A (not logged)"
        else:
            rendered_value = _format_value(value)
        print(f"{item['name']:<{name_width}}  {item['scale']:>12.9g}  {rendered_value:>14}")

    print()
    print(f"Component sum (logged weighted means): {_format_value(report['component_sum'])}")
    tag = report["mean_reward_tag"] or "mean reward"
    print(f"Episode mean reward ({tag}): {_format_value(report['mean_reward'])}")
    if report["summary_final_mean_reward"] is not None:
        print(
            "Final mean reward (run_summary.json): "
            f"{_format_value(float(report['summary_final_mean_reward']))}"
        )
    if report["mean_episode_length"] is not None:
        print(f"Mean episode length: {_format_value(report['mean_episode_length'])}")
    print(
        "Note: component_sum is not the episode mean reward; reward/* scalars are "
        "instantaneous weighted component means before ctrl_dt."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = query_run(args.run_dir, args.step)
    except (FileNotFoundError, NotADirectoryError, KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# uv run scripts/query_tensorboard_rewards.py logs/rsl_rl_ppo/NewhexJoystickFlat/g000_p/g000_p000
