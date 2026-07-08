#!/usr/bin/env python3
"""Merge selected Newhex robot scene XMLs into one visual MuJoCo scene.

This is a cold-path utility for experimenting with heterogeneous visualization.
It does not edit source robot assets or policy logs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "newhex_hetero_motrix_grid"
DEFAULT_MANIFEST = EXPERIMENT_DIR / "selected_robots.csv"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "generated" / "hetero_scene.xml"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest CSV from experiments/newhex_hetero_motrix_grid/run.py.",
    )
    parser.add_argument("--count", type=int, default=5, help="Number of robots to merge.")
    parser.add_argument("--rows", type=int, default=1, help="Formation rows.")
    parser.add_argument("--cols", type=int, default=5, help="Formation columns.")
    parser.add_argument("--spacing-x", type=float, default=2.0, help="Spacing along x.")
    parser.add_argument("--spacing-y", type=float, default=2.0, help="Spacing along y.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output merged scene XML path.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Load the generated XML with MuJoCo after writing it.",
    )
    parser.add_argument(
        "--validate-motrix",
        action="store_true",
        help="Load the generated XML with Motrix after writing it.",
    )
    parser.add_argument(
        "--no-motrix-compat",
        action="store_true",
        help="Do not apply small XML cleanups needed by Motrix's parser.",
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


def _base_scene() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(
        """
<mujoco model="newhex_heterogeneous_grid">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <default>
    <default class="floor">
      <geom type="plane" size="0 0 0.05" material="groundplane"/>
    </default>
  </default>
  <visual>
    <global offwidth="3840" offheight="2160"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="10 10" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <geom name="floor" class="floor" size="0 0 0.05"/>
  </worldbody>
</mujoco>
"""
    )


def _strip_scene_items(spec: mujoco.MjSpec) -> None:
    """Remove global scene items from a child spec before attach."""
    for light in list(spec.lights):
        spec.delete(light)
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for key in list(spec.keys):
        spec.delete(key)
    for texture in list(spec.textures):
        if texture.name == "groundplane":
            spec.delete(texture)
    for material in list(spec.materials):
        if material.name == "groundplane":
            spec.delete(material)
    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor" or geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(geom)


def _attach_robot(
    parent: mujoco.MjSpec,
    *,
    scene_xml: Path,
    prefix: str,
    pos: tuple[float, float, float],
) -> None:
    child = mujoco.MjSpec.from_file(str(scene_xml))
    _strip_scene_items(child)
    frame = parent.worldbody.add_frame(pos=[float(pos[0]), float(pos[1]), float(pos[2])])
    parent.attach(child, prefix=prefix, frame=frame)


def _write_scene(spec: mujoco.MjSpec, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_xml(), encoding="utf-8")


def _apply_motrix_compat(output: Path) -> None:
    text = output.read_text(encoding="utf-8")
    text = text.replace('colorspace="auto"', 'colorspace="sRGB"')
    output.write_text(text, encoding="utf-8")


def _validate_scene(output: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(output))
    print(
        f"[merge-scene] validated {output}: nq={model.nq} nv={model.nv} nu={model.nu}",
        flush=True,
    )


def _validate_motrix_scene(output: Path) -> None:
    from unilab.base.backend.motrix.scene import materialize_motrix_scene

    model: Any = materialize_motrix_scene(model_file=str(output))
    print(
        "[merge-scene] validated with Motrix: "
        f"num_dof_pos={model.num_dof_pos} num_dof_vel={model.num_dof_vel} "
        f"num_actuators={model.num_actuators}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    count = int(args.count)
    rows = int(args.rows)
    cols = int(args.cols)
    if count <= 0:
        raise ValueError(f"--count must be positive, got {count}")
    if rows * cols < count:
        raise ValueError(f"rows*cols must fit count, got {rows}*{cols} < {count}")

    selected = _read_manifest(args.manifest.expanduser().resolve(), count)
    spec = _base_scene()
    for slot, row in enumerate(selected):
        grid_row = slot // cols
        grid_col = slot % cols
        x = grid_col * float(args.spacing_x)
        y = -grid_row * float(args.spacing_y)
        robot_name = row["robot_name"]
        scene_xml = _resolve_repo_path(row["scene_xml"])
        print(f"[merge-scene] attach {slot}: {robot_name} at ({x:.2f}, {y:.2f})", flush=True)
        _attach_robot(
            spec,
            scene_xml=scene_xml,
            prefix=f"{robot_name}/",
            pos=(x, y, 0.0),
        )

    output = args.output.expanduser().resolve()
    _write_scene(spec, output)
    if not args.no_motrix_compat:
        _apply_motrix_compat(output)
    print(f"[merge-scene] wrote {output}", flush=True)
    if args.validate:
        _validate_scene(output)
    if args.validate_motrix:
        _validate_motrix_scene(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
