# 默认：白底、无阴影、自动居中、长焦透视减少畸变
uv run experiments/screenshot/render_robot_screenshots.py

# 增加 MJCF 场景背景
uv run experiments/screenshot/render_robot_screenshots.py --background scene

# 增加影子
uv run experiments/screenshot/render_robot_screenshots.py --shadows

# 背景 + 影子
uv run experiments/screenshot/render_robot_screenshots.py --background scene --shadows

# 更远/更松一点
uv run experiments/screenshot/render_robot_screenshots.py --distance 3.8 --margin 1.4

# 更小透视畸变，FOV 越小越像长焦
uv run experiments/screenshot/render_robot_screenshots.py --fovy 22