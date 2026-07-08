# Newhex Heterogeneous Motrix Grid

Experimental launcher for selecting 50 different Newhex robots and pairing each
robot with its own `NewhexJoystickFlat` PPO policy.

The requested target is one Motrix interactive window containing a heterogeneous
5x10 formation: 50 different robot XMLs, each driven by its matching controller,
moving forward with a fixed `vx=1.0m/s` command.

## Usage

Generate and inspect the sampled set:

```bash
uv run experiments/newhex_hetero_motrix_grid/run.py --mode manifest --count 50 --seed 0
```

Print standard eval commands for the sampled robots:

```bash
uv run experiments/newhex_hetero_motrix_grid/run.py --mode commands --count 50 --seed 0
```

Run standard Motrix eval one robot at a time:

```bash
uv run experiments/newhex_hetero_motrix_grid/run.py --mode sequential --count 50 --seed 0
```

Try the strict heterogeneous 5x10 Motrix grid:

```bash
uv run experiments/newhex_hetero_motrix_grid/run.py --mode motrix-grid --count 50 --seed 0
```

## Current Backend Boundary

The current UniLab Motrix backend renders one `SceneModel` batched across env
slots. That supports a 5x10 grid of the same robot model, but not 50 different
robot XMLs with 50 different policies in one renderer.

For safety, `--mode motrix-grid` fails fast until a backend-owned heterogeneous
playback API exists. The manifest and sequential modes are fully functional and
do not edit any task config, robot XML, checkpoint, or env file.



uv run experiments/newhex_hetero_motrix_grid/play_merged_rf_touch_policies.py \
  --manifest experiments/newhex_hetero_motrix_grid/top36_manipulation.csv \
  --count 36 \
  --cols 6 \
  --scene experiments/newhex_hetero_motrix_grid/generated/top36_manipulation_scene.xml \
  --episode-seconds 3.0 \
  --lookat 5 -5 0.8 \
  --distance 12

uv run experiments/newhex_hetero_motrix_grid/play_merged_policies.py \
  --manifest experiments/newhex_hetero_motrix_grid/top36_locomotion.csv \
  --count 36 \
  --cols 6 \
  --scene experiments/newhex_hetero_motrix_grid/generated/top36_locomotion_scene.xml \
  --lookat 5 -5 0.8 \
  --distance 12
