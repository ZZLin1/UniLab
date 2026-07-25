<h1 align="center"> UniLab </h1>

<h3 align="center">
面向超越 GPU 主导范式的机器人 RL 异构架构
</h3>

<p align="center">语言：简体中文 | <a href="README.md">English</a></p>

<p align="center">
  <a href="https://unilabsim.github.io"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.30313"><img src="https://img.shields.io/badge/arxiv-2605.30313-red" alt="arXiv"></a>
  <a href="https://unilabsim.github.io/paper/"><img src="https://img.shields.io/badge/paper-UniLab-orange" alt="Paper"></a>
  <a href="https://unilabsim.github.io/UniLab-doc/"><img src="https://img.shields.io/badge/docs-UniLab--doc-blue" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

<p align="center">
  <img src="docs/sphinx/source/_static/assets/teaser.jpg" alt="UniLab 预告图" width="95%">
</p>

<p align="center"><em>无需 GPU 仿真后端即可训练机器人 RL。预告图由 MotrixSim 渲染。</em></p>

从下面的 `快速演示` 开始运行主训练命令。推荐使用 `uv` 安装；Conda 和 pip 用户目前也应继续遵循 `uv` 工作流。平台相关说明与当前边界见 [安装指南](https://unilabsim.github.io/UniLab-doc/zh_CN/1-getting_started/2-installation.html)。

## ✨ 亮点

```
┌───────────────────┐                            ┌─────────────────────────┐
│  CPU Physics Sim  │   Unified Shared Memory    │   GPU Policy Training   │
│   MuJoCo/Motrix   │ ─────────────────────────▶ │     PPO / SAC / TD3     │
│ Multithread Step  │    SharedReplayBuffer      │ CUDA / MPS / ROCm / XPU │
└───────────────────┘                            └─────────────────────────┘
```

- **异构 RL 运行时：** CPU 并行仿真通过共享内存流式传输 transition，而策略学习运行在 GPU 加速器上。
- **两套物理后端：** MuJoCoUni 和 MotrixSim 通过后端专用适配器和任务 owner 配置接入。
- **统一训练 CLI：** `uv run train` 和 `uv run eval` 覆盖 PPO、MLX PPO、APPO、SAC、TD3 和 FlashSAC；额外的 HORA 与 HIM-PPO 路径以脚本级工作流文档化。
- **配置拥有的任务：** Hydra owner YAML 会同时选择 task、reward、backend 和 algorithm；后端切换通过 `task=<task>/<backend>` 表达。
- **跨平台安装路径：** 仓库覆盖 Linux CUDA、Linux ROCm、Linux XPU，以及 Apple Silicon / macOS 的安装流程。

## 🚀 快速演示

```bash
# 0. 如果还没有安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. 克隆仓库
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. 安装依赖
# 请按你的平台选择对应的安装命令。

# Linux CUDA 或 macOS
make setup-motrix
# 不使用 shell completion 设置时：uv sync --extra motrix
# 如果没有安装 `make`：uv sync --extra motrix && uv run --no-sync unilab-complete install

# Linux AMD / ROCm
# make sync-rocm

# Linux Intel Arc / iGPU
# make sync-xpu

# 3. 预训练 checkpoint 回放（首次运行会从 Hugging Face 下载）
uv run demo dance
```

可用的 demo 名称：`teaser`、`dance`、`wallflip`、`boxtracking`、`locomani`、`inhandgrasp`。
完整的命令与参数请参阅 [统一 CLI](https://unilabsim.github.io/UniLab-doc/zh_CN/2-user_guide/1-training/1-cli_reference.html) 页面。

> 中国大陆用户：动作、场景和 demo checkpoint 首次运行时会从 Hugging Face 拉取。如果 `huggingface.co`
> 无法访问，请在运行 demo 命令前先将客户端切到社区镜像：
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

用于训练与评估：

```bash
uv run train --algo appo --task go2_joystick_flat --sim motrix

uv run eval --algo appo --task go2_joystick_flat --sim motrix --load-run -1

# Linux / 服务器环境下的 Motrix 无头视频导出
uv run eval --algo appo --task go2_joystick_flat --sim motrix --load-run -1 --render-mode record
```

这会路由到 `go2_joystick_flat/motrix` 任务 owner 配置，并保持后端选择显式化。

在 macOS / MacBook 上，UniLab CLI 在需要时会通过 `mxpython` 路由 Motrix 交互式回放。Motrix 默认使用交互式回放；要导出无头视频请使用 `--render-mode record`，要跳过回放请使用 `--render-mode none`。更细的脚本级命令请参阅 [训练指南](https://unilabsim.github.io/UniLab-doc/zh_CN/2-user_guide/1-training/0-index.html)。

## 🏃 示例运行

```bash
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

```bash
uv run train --algo sac --task g1_motion_tracking --sim motrix
```

```bash
uv run train --algo appo --task sharpa_inhand --sim mujoco --profile hora
```

> Grasp cache 首次训练时会自动从 Hugging Face (`unilabsim/unilab-caches`) 下载到 `src/unilab/assets/caches/`，无需手动操作；如需为自定义 scale 重新生成（较慢）：
>
> ```bash
> bash scripts/sharpa_collect_grasps.sh 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
> ```

```bash
uv run train --algo ppo --task go2_arm_manip_loco --sim motrix
uv run eval --algo ppo --task go2_arm_manip_loco --sim motrix --load-run -1
```

```bash
uv run train --algo ppo --task sharpa_inhand --sim mujoco --profile hora
```

使用 `uv run train` 进行训练，使用 `uv run eval` 进行检查点回放，`uv run demo` 用于本地 demo 预设。这些命令可以明确指定算法、任务和后端。

更多训练命令、脚本级入口、算法矩阵、续训流程以及 W&B 细节请参阅 [训练指南](https://unilabsim.github.io/UniLab-doc/zh_CN/2-user_guide/1-training/0-index.html)。

## 📚 文档

请使用已发布的 [UniLab 文档](https://unilabsim.github.io/UniLab-doc/)；中文文档入口见 [中文文档索引](https://unilabsim.github.io/UniLab-doc/zh_CN/0-index.html)。高信号入口如下：

- [快速上手](https://unilabsim.github.io/UniLab-doc/zh_CN/1-getting_started/0-index.html)：安装、Docker 运行时、依赖配置和首次运行命令
- [训练指南](https://unilabsim.github.io/UniLab-doc/zh_CN/2-user_guide/1-training/0-index.html)：训练、回放、续训流程、Hydra override 和 W&B
- [仿真后端](https://unilabsim.github.io/UniLab-doc/zh_CN/2-user_guide/3-backends/0-index.html)：生成的 MuJoCo / Motrix 支持矩阵
- [开发者指南](https://unilabsim.github.io/UniLab-doc/zh_CN/4-developer_guide/0-index.html)：契约、分层与验证边界
- [ADR 索引](https://unilabsim.github.io/UniLab-doc/adr/ADR-0000-index.html)：已采纳的架构决策

## 💬 社群交流

| 微信群 | 小助手微信 |
| :---: | :---: |
| <img src="docs/sphinx/source/_static/assets/unilab-wechat-group.jpg" alt="UniLab 微信群二维码" width="220"> | <img src="docs/sphinx/source/_static/assets/unilab-wechat-assistant.jpg" alt="UniLab 小助手微信二维码" width="150"> |
| 扫码加入 UniLab 微信群。 | 如果微信群已满，请添加小助手微信，并备注 `unilab交流`。 |

## 🧾 引用

### UniLab

```bibtex
@article{jia2026unilab,
  title         = {UniLab: A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms},
  author        = {Yufei Jia and Zhanxiang Cao and Mingrui Yu and Heng Zhang and Shenyu Chen and Dixuan Jiang and Meng Li and Xiaofan Li and Yiyang Liu and Junzhe Wu and Zheng Li and XiLin Fang and Tingyu Cui and Shengcheng Fu and Haoyang Li and Anqi Wang and Zifan Wang and Dongjie Zhu and Chenyu Cao and Zhenbiao Huang and Ziang Zheng and Jie Lu and Xin Ma and Zhengyang Wei and Xiang Zhao and Tianyue Zhan and Ye He and Yuxiang Chen and Yizhou Jiang and Yue Li and Haizhou Ge and Yuhang Dong and Fan Jia and Ziheng Zhang and Meng Zhang and Xiwa Deng and Zhixing Chen and Hanyang Shao and Chenxin Dong and Yixuan Li and Yizhi Chen and Bokui Chen and Kaifeng Zhang and Hanqing Cui and Yusen Qin and Ruqi Huang and Lei Han and Tiancai Wang and Xiang Li and Yue Gao and Guyue Zhou},
  journal       = {arXiv preprint arXiv:2605.30313},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.30313}
}
```

### 物理后端

```bibtex
@article{jia2026mujocouni,
  title  = {MuJoCoUni: Persistent Batched Runtime Primitives for MuJoCo},
  author = {Jia, Yufei and Wu, Junzhe},
  journal = {arXiv preprint arXiv:2605.24922},
  year   = {2026}
}

@software{motrixsim2026,
  title  = {MotrixSim: A Physics Simulation Engine for Robotics and Embodied AI},
  author = {{Motphys Team}},
  year   = {2026},
  url    = {https://motrixsim.readthedocs.io/},
  note   = {Python binary package}
}
```


## 多机器人 eval

本节用于在一个 Motrix 窗口中可视化多个不同 Newhex 机器人。当前实现放在
`experiments/newhex_hetero_motrix_grid/` 下，不会修改正常训练、eval、任务配置或机器人
XML。流程分为三步：

1. 生成机器人-策略清单 manifest。
2. 把这些机器人的 `scene.xml` 合并成一个可视化场景。
3. 用每个机器人自己的 PPO 策略驱动合并场景。

默认示例使用 `experiments/high_fitness_offspring_names.csv` 中列出的 36 个机器人，站成
`6 x 6` 方阵。CSV 至少需要包含一列 `name`，内容形如 `g006_o007`。

### 前置检查

请确认以下文件/目录已经存在：

```bash
ls experiments/high_fitness_offspring_names.csv
ls logs/rsl_rl_ppo/NewhexJoystickFlat
ls logs/rsl_rl_ppo/NewhexRFTouch
ls src/unilab/assets/robots/newhex
```

每个机器人需要同时满足：

- `src/unilab/assets/robots/newhex/<robot_name>/scene.xml` 存在。
- 对应任务的策略目录存在，例如
  `logs/rsl_rl_ppo/NewhexJoystickFlat/g006_o/g006_o007/model_*.pt`。

如果策略目录仍是日期名，先运行重命名脚本：

```bash
uv run scripts/rename_policy_run_dirs.py --start-generation 40 --apply
```

`--start-generation` 按实际需要修改。

### 生成 Joystick Manifest

```bash
uv run experiments/newhex_hetero_motrix_grid/run.py \
  --mode manifest \
  --names-csv experiments/high_fitness_offspring_names.csv \
  --rows 6 \
  --cols 6 \
  --manifest experiments/newhex_hetero_motrix_grid/selected_robots.csv \
  --policy-log-root logs/rsl_rl_ppo/NewhexJoystickFlat
```

输出文件：

```text
experiments/newhex_hetero_motrix_grid/selected_robots.csv
```

### 生成 RF Touch Manifest

```bash
uv run experiments/newhex_hetero_motrix_grid/run.py \
  --mode manifest \
  --names-csv experiments/high_fitness_offspring_names.csv \
  --rows 6 \
  --cols 6 \
  --manifest experiments/newhex_hetero_motrix_grid/selected_rf_touch_robots.csv \
  --policy-log-root logs/rsl_rl_ppo/NewhexRFTouch
```

输出文件：

```text
experiments/newhex_hetero_motrix_grid/selected_rf_touch_robots.csv
```

### 生成合并场景

Joystick 和 RF Touch 使用同一批机器人 XML，所以只需要一个合并场景：

```bash
uv run experiments/newhex_hetero_motrix_grid/merge_scene.py \
  --manifest experiments/newhex_hetero_motrix_grid/selected_robots.csv \
  --count 36 \
  --rows 6 \
  --cols 6 \
  --output experiments/newhex_hetero_motrix_grid/generated/hetero_36_scene.xml \
  --validate \
  --validate-motrix
```

输出文件：

```text
experiments/newhex_hetero_motrix_grid/generated/hetero_36_scene.xml
```

`--validate` 会用 MuJoCo 加载生成的 XML；`--validate-motrix` 会用 Motrix 加载生成的
XML。二者都通过后再运行策略可视化。

### 静态预览场景

这一步只检查 36 个机器人是否都站在正确位置，不加载策略：

```bash
uv run experiments/newhex_hetero_motrix_grid/preview_scene.py \
  --scene experiments/newhex_hetero_motrix_grid/generated/hetero_36_scene.xml \
  --lookat 5 -5 0.8 \
  --distance 12
```

关闭 Motrix 窗口即可退出。

### Joystick 可视化

先做一次不打开窗口的加载检查：

```bash
uv run experiments/newhex_hetero_motrix_grid/play_merged_policies.py \
  --manifest experiments/newhex_hetero_motrix_grid/selected_robots.csv \
  --count 36 \
  --cols 6 \
  --scene experiments/newhex_hetero_motrix_grid/generated/hetero_36_scene.xml \
  --check-only
```

通过后打开 Motrix 可视化窗口：

```bash
uv run experiments/newhex_hetero_motrix_grid/play_merged_policies.py \
  --manifest experiments/newhex_hetero_motrix_grid/selected_robots.csv \
  --count 36 \
  --cols 6 \
  --scene experiments/newhex_hetero_motrix_grid/generated/hetero_36_scene.xml \
  --velocity 1.0 \
  --lookat 5 -5 0.8 \
  --distance 12
```

说明：

- 每个机器人加载自己的 `NewhexJoystickFlat` PPO 策略。
- `--velocity 1.0` 表示前向速度命令为 `1m/s`。
- 脚本会把每个机器人的初始朝向对齐到同一世界方向，避免 reset yaw 随机导致方阵朝不同方向移动。

### RF Touch 可视化

RF Touch 使用原始 `NewhexRFTouch` env 的 reset/target sampling 规则。脚本只覆盖 episode
最长时间，默认 `--episode-seconds 3.0`，即每 3 秒左右按任务 reset 规则重新采样目标。

先做一次不打开窗口的加载检查：

```bash
uv run experiments/newhex_hetero_motrix_grid/play_merged_rf_touch_policies.py \
  --manifest experiments/newhex_hetero_motrix_grid/selected_rf_touch_robots.csv \
  --count 36 \
  --cols 6 \
  --scene experiments/newhex_hetero_motrix_grid/generated/hetero_36_scene.xml \
  --check-only
```

通过后打开 Motrix 可视化窗口：

```bash
uv run experiments/newhex_hetero_motrix_grid/play_merged_policies.py \
  --manifest experiments/newhex_hetero_motrix_grid/top36_locomotion.csv \
  --count 36 \
  --cols 6 \
  --scene experiments/newhex_hetero_motrix_grid/generated/top36_locomotion_scene.xml \
  --lookat 5 -5 0.8 \
  --distance 12
```

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

说明：

- 每个机器人加载自己的 `NewhexRFTouch` PPO 策略。
- 目标点采样范围来自 `conf/ppo/task/newhex_rf_touch/motrix.yaml` 中的
  `env.target_sampling`，也就是实际训练配置。
- 如果想改变每个 episode 的最长时间，可以修改 `--episode-seconds`。

### 少量机器人快速测试

如果只想先测试 5 个机器人，可以复用同一个流程，把 `--count/--rows/--cols/output` 改小：

```bash
uv run experiments/newhex_hetero_motrix_grid/merge_scene.py \
  --manifest experiments/newhex_hetero_motrix_grid/selected_robots.csv \
  --count 5 \
  --rows 1 \
  --cols 5 \
  --output experiments/newhex_hetero_motrix_grid/generated/hetero_5_scene.xml \
  --validate \
  --validate-motrix
```

```bash
uv run experiments/newhex_hetero_motrix_grid/play_merged_policies.py \
  --manifest experiments/newhex_hetero_motrix_grid/selected_robots.csv \
  --count 5 \
  --cols 5 \
  --scene experiments/newhex_hetero_motrix_grid/generated/hetero_5_scene.xml
```

### 常见问题

- `manifest does not exist`：先运行对应的 `run.py --mode manifest` 命令。
- `merged scene does not exist`：先运行 `merge_scene.py` 生成 `hetero_36_scene.xml`。
- `could not find policy+scene pairs`：检查机器人 XML 是否存在，策略目录是否已经重命名为
  `gNNN_o/gNNN_oXXX`。
- Motrix 窗口关闭后程序退出是正常行为。




### Work for LYF

urdf导出xml, 需要在mujoco中调整kryframe的高度
```bash
uv run unilab-import-robot <‘urdf路径’> newhex
```
在mujoco页面下截个图

记录urdf名称与newhex名称

修改`src/unilab/envs/locomotion/newhex/joystick.py`第74行路径中的newhex

训练
```bash
uv run train --algo ppo --task newhex_joystick_flat --sim motrix
uv run train --algo ppo --task newhex_rf_touch --sim motrix
```

uv run scripts/import_generation_offspring_urdfs.py

更改两个sweep的序号

uv run scripts/sweep_newhex_joystick_then_rf_touch.py


uv run scripts/rename_policy_run_dirs.py --start-generation 48 --apply
