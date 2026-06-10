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

### 步骤 2：手写第一个 MuJoCo 加载测试脚本（`tests/01_test_mujoco.py`）
- **发生时间**：2026-06-09 15:21:05 (CST)
- **目的作用**：验证我们能否利用原生 MuJoCo API 成功加载和读取 Sharpa 机器手的物理描述文件，提取关键控制维度。
- **代码原理简述**：使用 `mujoco.MjModel.from_xml_path` 加载描述模型静态结构 and 属性的 `MjModel`；使用 `mujoco.MjData` 构建存储仿真运行状态与动力学变量的 `MjData`。
- **执行的命令**：
  ```bash
  .venv/bin/python tests/01_test_mujoco.py
  ```
- **产生的文件**：
  - [01_test_mujoco.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/tests/01_test_mujoco.py) (首个 MuJoCo 加载测试脚本)

### 步骤 3：手写机器手 3D 交互式可视化与正弦波控制（`tests/02_test_mujoco_visual.py`）
- **发生时间**：2026-06-09 15:29:10 (CST)
- **目的作用**：新建可视化测试脚本，通过注册 MuJoCo 的控制回调函数（`mujoco.set_mjcb_control`）进行 22 维致动器的正弦波控制，并利用 `mujoco.viewer.launch` 启动管理式 3D 可视化窗口。该方法保证渲染主循环在主线程运行，完美兼容 macOS 环境。
- **执行的命令**：
  ```bash
  .venv/bin/python tests/02_test_mujoco_visual.py
  ```
- **产生的文件**：
  - [02_test_mujoco_visual.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/tests/02_test_mujoco_visual.py) (带渲染和正弦波控制的测试脚本)

### 步骤 4：手写单进程 Gymnasium 环境封装（`tests/03_test_mujoco_gym.py`）
- **发生时间**：2026-06-09 15:46:50 (CST)
- **目的作用**：将底层的 MuJoCo 仿真封装为标准的强化学习 Gymnasium 接口，定义 22 维控制空间与 44 维关节状态观测空间，完成单进程 RL 闭环封装。
- **执行的命令**：
  ```bash
  .venv/bin/python tests/03_test_mujoco_gym.py
  ```
- **产生的文件**：
  - [03_test_mujoco_gym.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/tests/03_test_mujoco_gym.py) (Gymnasium 环境封装及调试入口脚本)

### 步骤 5：手写极简多进程共享内存通信测试（`tests/04_test_mujoco_shared.py`）
- **发生时间**：2026-06-09 16:24:20 (CST)
- **目的作用**：使用共享 Tensor 和 mp.Event 实现主进程与环境子进程之间的零拷贝数据传输与同步，体验异构 RL 计算的并发底座。
- **执行的命令**：
  ```bash
  .venv/bin/python tests/04_test_mujoco_shared.py
  ```
- **产生的文件**：
  - [04_test_mujoco_shared.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/tests/04_test_mujoco_shared.py) (多进程共享内存通信与握手测试脚本)

### 步骤 6：手写向量化多环境与批量共享内存测试（`tests/05_test_mujoco_vector_shared.py`）
- **发生时间**：2026-06-09 17:17:30 (CST)
- **目的作用**：将共享内存数据通道从单环境（1D Tensor）扩展为多环境批处理（2D Tensor），在子进程中运行并行环境采样，实现真正的“批量异构并行”。
- **执行的命令**：
  ```bash
  .venv/bin/python tests/05_test_mujoco_vector_shared.py
  ```
- **产生的文件**：
  - [05_test_mujoco_vector_shared.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/tests/05_test_mujoco_vector_shared.py) (批量共享内存多环境通信与同步测试脚本)

### 步骤 7：手写 PPO 策略网络与轨迹收集测试（`tests/06_test_ppo_collect.py`）
- **发生时间**：2026-06-09 18:00:00 (CST)
- **目的作用**：定义 Actor-Critic 对角高斯策略网络，主进程通过批量共享内存与仿真子进程进行交互，并维护 32 步 Rollout Buffer 收集采样轨迹。利用 GAE 进行优势估计，计算并反向传播 PPO 损失函数，打通神经网络计算图。
- **执行的命令**：
  ```bash
  .venv/bin/python tests/06_test_ppo_collect.py
  ```
- **产生的文件**：
  - [06_test_ppo_collect.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/tests/06_test_ppo_collect.py) (PPO 策略推理、轨迹收集与反向传播测试脚本)

### 步骤 8：重构模块化项目包与 PPO 策略训练流水线（`scripts/train.py` 等）
- **发生时间**：2026-06-10 16:48:50 (CST)
- **目的作用**：将前期零散测试代码重构整理为符合标准项目结构的模块化包（如 `envs`、`ipc`、`algos` 包），并编写首个完整 PPO 强化学习训练流水线 `scripts/train.py`，实现多环境并行轨迹采集与策略参数自我演进。
- **执行的命令**：
  ```bash
  .venv/bin/python scripts/train.py
  ```
- **产生的文件**：
  - [sharpa_env.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/src/minilab/envs/sharpa_env.py) (标准环境封装)
  - [vector_env.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/src/minilab/ipc/vector_env.py) (多进程向量化环境 IPC 封装)
  - [ppo.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/src/minilab/algos/ppo.py) (ActorCritic 神经网络定义)
  - [train.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/scripts/train.py) (PPO 策略训练主入口程序)



