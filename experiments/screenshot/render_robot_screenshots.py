#!/usr/bin/env python3
"""Render one static Newhex screenshot for each robot listed in a CSV.

The script is intentionally self-contained under ``experiments/screenshot`` so
it does not affect training, task configs, assets, or backend code.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import imageio.v3 as iio
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CSV = SCRIPT_DIR / "locomotion_gt100_manipulation_gt90_population_origin_names.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "robot_screenshots"
DEFAULT_ASSET_ROOT = REPO_ROOT / "src" / "unilab" / "assets" / "robots" / "newhex"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RobotRow:
    name: str
    final_name: str
    row_number: int


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV with name/final_name columns.")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--lookat", nargs=3, type=float, default=[0.05, 0.0, 0.35])
    parser.add_argument(
        "--distance",
        type=float,
        default=3.2,
        help="Minimum camera distance. Auto-camera may increase this to fit larger robots.",
    )
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument(
        "--fixed-camera",
        action="store_true",
        help="Use --lookat and --distance exactly instead of fitting the robot bounds.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.25,
        help="Auto-camera framing margin. Larger values leave more white space around the robot.",
    )
    parser.add_argument(
        "--fovy",
        type=float,
        default=28.0,
        help="Perspective camera field-of-view in degrees. Smaller values reduce lens distortion.",
    )
    parser.add_argument(
        "--orthographic",
        action="store_true",
        help=(
            "Use MuJoCo's orthographic free camera. The default long-lens perspective "
            "mode gives more predictable framing for these generated scenes."
        ),
    )
    parser.add_argument(
        "--background",
        choices=("white", "scene"),
        default="white",
        help="white hides floor/sky and post-processes the background; scene keeps the MJCF background.",
    )
    parser.add_argument(
        "--shadows",
        action="store_true",
        help="Enable renderer shadows. Shadows are disabled by default.",
    )
    parser.add_argument(
        "--gl-backend",
        choices=("egl", "glfw", "osmesa"),
        default="egl",
        help="MuJoCo GL backend. Use glfw if you want to rely on a display server.",
    )
    parser.add_argument(
        "--background-tolerance",
        type=float,
        default=18.0,
        help="RGB distance threshold for replacing the rendered background with white.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-render images that already exist in the output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render only the first N CSV rows. Useful for checking camera settings.",
    )
    return parser.parse_args(argv)


def _read_rows(csv_path: Path, limit: int | None) -> list[RobotRow]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")

    rows: list[RobotRow] = []
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"name", "final_name"}
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            final_name = (row.get("final_name") or "").strip()
            if not name or not final_name:
                continue
            rows.append(RobotRow(name=name, final_name=final_name, row_number=row_number))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _safe_png_name(value: str) -> str:
    stem = SAFE_FILENAME_RE.sub("_", value.strip()).strip("._")
    if not stem:
        raise ValueError(f"final_name={value!r} cannot be converted to a safe filename")
    return f"{stem}.png"


def _set_model_home_qpos(model: object, data: object) -> None:
    if int(model.nkey) > 0:
        data.qpos[:] = model.key_qpos[0]


def _robot_visual_bounds(model: object, data: object) -> tuple[np.ndarray, float]:
    import mujoco

    groups = np.asarray(model.geom_group)
    types = np.asarray(model.geom_type)
    visual_mask = (groups == 2) & (types != int(mujoco.mjtGeom.mjGEOM_PLANE))
    if not np.any(visual_mask):
        center = np.asarray(model.stat.center, dtype=np.float64)
        return center, max(float(model.stat.extent), 1.0)

    centers = np.asarray(data.geom_xpos[visual_mask], dtype=np.float64)
    sizes = np.asarray(model.geom_size[visual_mask], dtype=np.float64)
    radii = np.linalg.norm(sizes, axis=1)
    lower = np.min(centers - radii[:, None], axis=0)
    upper = np.max(centers + radii[:, None], axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.5 * float(np.linalg.norm(upper - lower))
    return center, max(radius, 0.5)


def _make_camera(
    args: argparse.Namespace,
    model: object,
    data: object,
) -> object:
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.orthographic = 1 if args.orthographic else 0
    if args.fixed_camera:
        lookat = np.asarray(args.lookat, dtype=np.float64)
        distance = float(args.distance)
    else:
        center, radius = _robot_visual_bounds(model, data)
        lookat = center
        if args.orthographic:
            distance = max(float(args.distance), radius * float(args.margin))
        else:
            distance = max(float(args.distance), radius * float(args.margin) * 2.4)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.elevation = float(args.elevation)
    camera.azimuth = float(args.azimuth)
    return camera


def _make_scene_option(*, background: str) -> object:
    import mujoco

    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    if background == "scene":
        option.geomgroup[0] = 1
    option.geomgroup[2] = 1
    option.sitegroup[:] = 0
    option.tendongroup[:] = 0
    option.actuatorgroup[:] = 0
    return option


def _configure_render_flags(scene: object, *, background: str, shadows: bool) -> None:
    import mujoco

    if not shadows:
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    if background == "white":
        for flag in (
            mujoco.mjtRndFlag.mjRND_REFLECTION,
            mujoco.mjtRndFlag.mjRND_FOG,
            mujoco.mjtRndFlag.mjRND_HAZE,
            mujoco.mjtRndFlag.mjRND_SKYBOX,
        ):
            scene.flags[flag] = 0
        return

    for flag in (
        mujoco.mjtRndFlag.mjRND_FOG,
        mujoco.mjtRndFlag.mjRND_HAZE,
    ):
        scene.flags[flag] = 0


def _replace_background_with_white(image: np.ndarray, tolerance: float) -> np.ndarray:
    if tolerance < 0.0:
        return image

    rgb = np.asarray(image[..., :3], dtype=np.uint8)
    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    )
    background = np.median(border.astype(np.float32), axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - background[None, None, :], axis=2)
    output = rgb.copy()
    output[dist <= float(tolerance)] = 255
    return output


def _render_robot(scene_xml: Path, output_png: Path, args: argparse.Namespace) -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model.vis.global_.orthographic = 1 if args.orthographic else 0
    if not args.orthographic:
        model.vis.global_.fovy = float(args.fovy)
    data = mujoco.MjData(model)
    _set_model_home_qpos(model, data)
    mujoco.mj_forward(model, data)

    camera = _make_camera(args, model, data)
    scene_option = _make_scene_option(background=str(args.background))
    renderer = mujoco.Renderer(model, width=int(args.width), height=int(args.height))
    try:
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        _configure_render_flags(
            renderer.scene,
            background=str(args.background),
            shadows=bool(args.shadows),
        )
        image = renderer.render()
    finally:
        renderer.close()

    if args.background == "white":
        image = _replace_background_with_white(image, float(args.background_tolerance))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_png, image)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", str(args.gl_backend))

    csv_path = args.csv.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = _read_rows(csv_path, args.limit)

    missing_models: list[str] = []
    failed_renders: list[str] = []
    rendered = 0
    skipped_existing = 0

    for index, row in enumerate(rows, start=1):
        scene_xml = asset_root / row.name / "scene.xml"
        output_png = output_dir / _safe_png_name(row.final_name)

        if not scene_xml.is_file():
            missing_models.append(row.name)
            continue
        if args.skip_existing and output_png.is_file():
            skipped_existing += 1
            continue

        try:
            _render_robot(scene_xml, output_png, args)
        except Exception as exc:  # noqa: BLE001 - report all per-robot failures at the end.
            failed_renders.append(f"{row.name} (row {row.row_number}): {exc}")
            continue

        rendered += 1
        print(
            f"[{index}/{len(rows)}] rendered {row.name} -> {output_png.relative_to(SCRIPT_DIR)}",
            flush=True,
        )

    print(
        f"Done. rendered={rendered} skipped_existing={skipped_existing} "
        f"missing_models={len(missing_models)} failed_renders={len(failed_renders)}",
        flush=True,
    )

    unsuccessful = missing_models + [item.split(":", 1)[0] for item in failed_renders]
    if unsuccessful:
        print("Robots that were not successfully rendered:", flush=True)
        for name in unsuccessful:
            print(f"  {name}", flush=True)
    else:
        print("All requested robots were successfully rendered.", flush=True)

    if failed_renders:
        print("Render failure details:", flush=True)
        for item in failed_renders:
            print(f"  {item}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
