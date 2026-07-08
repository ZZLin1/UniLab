#!/usr/bin/env python3
"""Open a static Motrix preview window for a merged heterogeneous scene."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "experiments" / "newhex_hetero_motrix_grid" / "generated" / "hetero_5_scene.xml"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="Merged scene XML.")
    parser.add_argument("--fps", type=float, default=60.0, help="Preview refresh rate.")
    parser.add_argument("--lookat", nargs=3, type=float, default=[4.0, 0.0, 0.8])
    parser.add_argument("--distance", type=float, default=9.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    return parser.parse_args(argv)


def _render_settings() -> Any:
    from motrixsim.render import RenderSettings

    settings = RenderSettings.quality()
    settings.enable_shadow = True
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"merged scene does not exist: {scene}")

    import motrixsim as mtx
    from motrixsim.render import RenderApp

    model = mtx.load_model(str(scene))
    data = mtx.SceneData(model)
    model.forward_kinematic(data)

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
        print("[preview-scene] close the Motrix window to exit", flush=True)
        while not render.is_closed:
            render.sync(data)
            time.sleep(frame_dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
