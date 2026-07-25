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

# 生成Manifest
## 生成 移动任务 Manifest
```bash
uv run experiments/newhex_hetero_motrix_grid/run.py \
  --mode manifest \
  --names-csv experiments/top36_locomotion_offspring_names.csv \
  --rows 6 \
  --cols 6 \
  --manifest experiments/newhex_hetero_motrix_grid/top36_locomotion.csv \
  --policy-log-root logs/rsl_rl_ppo/NewhexJoystickFlat
```
## 生成 操作任务 Manifest
```bash
uv run experiments/newhex_hetero_motrix_grid/run.py \
  --mode manifest \
  --names-csv experiments/top36_manipulation_offspring_names.csv \
  --rows 6 \
  --cols 6 \
  --manifest experiments/newhex_hetero_motrix_grid/top36_manipulation.csv \
  --policy-log-root logs/rsl_rl_ppo/NewhexRFTouch
```

# 合并场景
## 合并 移动任务 场景
```bash
uv run experiments/newhex_hetero_motrix_grid/merge_scene.py \
  --manifest experiments/newhex_hetero_motrix_grid/top36_locomotion.csv \
  --count 36 \
  --rows 6 \
  --cols 6 \
  --output experiments/newhex_hetero_motrix_grid/generated/top36_locomotion_scene.xml \
  --validate \
  --validate-motrix
```
## 合并 操作任务 场景
```bash
uv run experiments/newhex_hetero_motrix_grid/merge_scene.py \
  --manifest experiments/newhex_hetero_motrix_grid/top36_manipulation.csv \
  --count 36 \
  --rows 6 \
  --cols 6 \
  --output experiments/newhex_hetero_motrix_grid/generated/top36_manipulation_scene.xml \
  --validate \
  --validate-motrix
```

# 静态预览场景
```bash
uv run experiments/newhex_hetero_motrix_grid/preview_scene.py \
  --scene experiments/newhex_hetero_motrix_grid/generated/top36_locomotion_scene.xml \
  --lookat 5 -5 0.8 \
  --distance 12
```

# top36播放策略
## 最大36个 移动适应度 个体播放策略
```bash
uv run experiments/newhex_hetero_motrix_grid/play_merged_policies.py \
  --manifest experiments/newhex_hetero_motrix_grid/top36_locomotion.csv \
  --count 36 \
  --cols 6 \
  --scene experiments/newhex_hetero_motrix_grid/generated/top36_locomotion_scene.xml \
  --lookat 5 -5 0.8 \
  --distance 12 \
  --velocity 1.5 \
  --heading-yaw 0.0 \
  --steps-per-render 2 \
  --no-sleep
```
## 最大36个 操作适应度 个体播放策略
```bash
uv run experiments/newhex_hetero_motrix_grid/play_merged_rf_touch_policies.py \
  --manifest experiments/newhex_hetero_motrix_grid/top36_manipulation.csv \
  --count 36 \
  --cols 6 \
  --scene experiments/newhex_hetero_motrix_grid/generated/top36_manipulation_scene.xml \
  --episode-seconds 3.0 \
  --lookat 5 -5 0.8 \
  --distance 12
``` 
