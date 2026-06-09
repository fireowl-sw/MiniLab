# MiniLab

MiniLab is a lightweight simulation and testing platform for robotic hand environments. It provides simulation utilities, models, and environments using MuJoCo and Gymnasium.

## 开发历程与操作日志 (Development & Operation Log)

### 步骤 1：项目初始化与 Git 设置
- **发生时间**：2026-06-09 15:10:45 (CST)
- **目的作用**：初始化 Git 仓库和 uv 依赖管理环境，导入外部机器手（Sharpa）仿真模型资源。
- **执行的命令**：
  ```bash
  # 初始化 Git 仓库，设置主分支为 main
  git init -b main

  # 初始化 uv 项目结构（library 模式）
  uv init --lib

  # 锁定 Python 版本为 3.13
  uv python pin 3.13

  # 添加核心依赖包
  uv add numpy torch mujoco gymnasium

  # 同步项目环境
  uv sync

  # 创建资源目录并拷贝机器手（Sharpa）模型文件
  mkdir -p ./assets/robots/sharpa_wave/
  cp /Users/fireowl/Documents/auto_ws/robot_ws/UniLab/src/unilab/assets/robots/sharpa_wave/scene.xml ./assets/robots/sharpa_wave/scene.xml
  cp /Users/fireowl/Documents/auto_ws/robot_ws/UniLab/src/unilab/assets/robots/sharpa_wave/right_sharpa_wave.xml ./assets/robots/sharpa_wave/right_sharpa_wave.xml
  cp -r /Users/fireowl/Documents/auto_ws/robot_ws/UniLab/src/unilab/assets/robots/sharpa_wave/meshes ./assets/robots/sharpa_wave/
  ```
- **产生的文件**：
  - **项目配置文件**：
    - `pyproject.toml` (项目元数据与依赖定义，设置 `requires-python = ">=3.13"`)
    - `.python-version` (指定使用的 Python 版本为 3.13)
    - `uv.lock` (依赖锁文件)
  - **机器手模型文件**：
    - [scene.xml](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/assets/robots/sharpa_wave/scene.xml) (仿真场景定义文件)
    - [right_sharpa_wave.xml](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/assets/robots/sharpa_wave/right_sharpa_wave.xml) (右手 Sharpa 机器人结构定义文件)
    - [meshes/](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/assets/robots/sharpa_wave/meshes/) (机器手网格模型 STL 文件目录)
