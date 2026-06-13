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

### 步骤 9：引入配置管理器 Hydra 并重构训练脚本
- **发生时间**：2026-06-10 16:58:30 (CST)
- **目的作用**：为项目引入 Facebook 的配置管理工具 Hydra，将算法和环境的超参数抽离到外部 YAML 文件（如 `conf/config.yaml`、`conf/env/sharpa.yaml`、`conf/algo/ppo.yaml`）中，使超参数更加集中且支持在命令行进行无代码级覆盖。
- **执行的命令**：
  ```bash
  .venv/bin/python scripts/train.py algo.lr=0.0001 total_updates=10
  ```
- **产生的文件**：
  - [config.yaml](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/conf/config.yaml) (Hydra 主配置文件)
  - [sharpa.yaml](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/conf/env/sharpa.yaml) (Sharpa 环境配置文件)
  - [ppo.yaml](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/conf/algo/ppo.yaml) (PPO 算法配置文件)

### 步骤 10：手写 PPO 控制策略的 3D 渲染与评估脚本（`scripts/eval.py`）
- **发生时间**：2026-06-10 17:51:30 (CST)
- **目的作用**：编写 PPO 控制策略的 3D 渲染与评估脚本，载入训练好的神经网络权重，并在 MuJoCo GUI 窗口中以确定性策略（动作均值）控制 Sharpa 机器手，直观观测模型控制器的物理表现。
- **执行的命令**：
  ```bash
  .venv/bin/python scripts/eval.py
  ```
- **产生的文件**：
  - [eval.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/scripts/eval.py) (PPO 策略 3D 评估与渲染脚本)

### 步骤 11：手写手内操作（In-hand Manipulation） Gymnasium 环境
- **发生时间**：2026-06-11 17:21:00 (CST)
- **目的作用**：重构环境库代码以支持手内圆柱体操作任务。在 reset 时将手部初始化为捏紧姿态，物体置于掌心上方；扩展状态观测空间至 60 维，包含手部关节状态、物体 3D 姿态与速度以及目标旋转轴；在 step 中计算包含 Z 轴旋转速度、位移惩罚、锚点对齐及动作惩罚的复合奖励函数，并实现低于阈值时判定坠落的 episode 终止逻辑。
- **执行的命令**：
  ```bash
  .venv/bin/python tests/07_test_inhand_gym.py
  ```
- **产生的文件**：
  - [sharpa_env.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/src/minilab/envs/sharpa_env.py) (重构后的手内操作 Gymnasium 环境类)
  - [07_test_inhand_gym.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/tests/07_test_inhand_gym.py) (单进程手内操作环境物理与维度测试脚本)

### 步骤 12：实现 C++ 原生并行 VectorEnv 与验证
- **发生时间**：2026-06-11 17:32:00 (CST)
- **目的作用**：废弃原基于 Python multiprocessing 的多进程共享内存方案，重构并升级为基于 MuJoCo C++ 原生多线程并行仿真底座的 BatchEnvPool。该方案实现了全向量化的 reset、step 计算、状态提取（兼容并修复了 mjSTATE_FULLPHYSICS 的 time 偏移问题）以及自动重置（Auto-reset）机制，从而彻底避免了 Python GIL 限制和跨进程拷贝开销。
- **执行的命令**：
  ```bash
  .venv/bin/python tests/08_test_inhand_vector.py
  ```
- **产生的文件**：
  - [vector_env.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/src/minilab/ipc/vector_env.py) (重构后的 C++ BatchEnvPool 向量化并行环境类)
  - [08_test_inhand_vector.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/tests/08_test_inhand_vector.py) (C++ 并行向量环境功能与自动重置验证脚本)

### 步骤 13：PPO 实机训练与 3D 物理效果评估
- **发生时间**：2026-06-11 17:54:00 (CST)
- **目的作用**：修复 `scripts/eval.py` 使其支持 60 维状态观测与默认捏球姿态初始对齐。调整配置参数 `total_updates: 1000` 并通过 C++ 原生并行 `BatchEnvPool` 环境底座启动 PPO 强化学习策略训练。训练吞吐率由多进程的约 3000 FPS 暴涨至 **约 10000 FPS**，仅用约 100 秒即完成了 100 万步物理交互；平均奖励稳定上升至正数区间并收敛，成功保存了 `ppo_sharpa.pt` 权重。拉起评估窗口后，可直观观测到机器手成功实现捏球姿态，并尝试在手内对红色圆柱体实施旋转操作。
- **执行的命令**：
  ```bash
  .venv/bin/python scripts/train.py
  .venv/bin/python scripts/eval.py
  ```
- **产生的文件**：
  - [eval.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/scripts/eval.py) (重构后的手内操作策略 3D 渲染与评估脚本)

### 步骤 14：手握姿态随机化、PD 虚拟力矩功惩罚、控制限收缩与 5000 轮训练视频诊断
- **发生时间**：2026-06-13 10:37:00 (CST)
- **目的作用**：
  1. **资产拷贝与依赖升级**：拷贝 `sharpa_grasp_linspace_1.npy` 数据集资产，并在 `pyproject.toml` 添加 `imageio` 与 `imageio-ffmpeg`，实现视频离屏录制环境。
  2. **初态随机化**：在 Reset 和 Auto-reset 时从抓握数据集中均匀随机采样初始关节与物体状态，提升策略泛化性。
  3. **力矩与功奖励重构**：从模型中提取 PD 控制器增益参数，在物理步进后计算虚拟控制力矩与物理功，作为惩罚项引入复合奖励。
  4. **控制限收缩**：将控制角度的目标指令剪切范围缩小至最大范围的 0.9，防止自锁。
  5. **5000轮训练与离屏录屏**：完成 5000 轮（512 万步）的完整训练，模型稳定收敛至正奖励值；通过 `eval.py record=True` 成功生成 20 FPS 离屏诊断视频 `eval_run.mp4`，验证机器手在真实随机抓握初态下的极佳手内旋转操纵表现。
  6. **交互式 GLFW 被动窗口重构**：修改 `scripts/eval.py` 中的交互式 GLFW 模式，使用 `mujoco.viewer.launch_passive` 替换原有的控制回调函数注册方式。这解决了窗口启动时初始状态被重置为张开手掌导致物体坠落的问题，并添加了 Backspace 键重置时重新从抓持数据集中加载随机初态的逻辑。
- **执行的命令**：
  ```bash
  # 运行 PPO 训练
  .venv/bin/python scripts/train.py total_updates=5000
  # 生成离屏评估录像
  .venv/bin/python scripts/eval.py record=True
  # 启动交互式 3D 仿真窗口 (在 macOS 上以 mjpython 启动)
  .venv/bin/mjpython scripts/eval.py
  ```
- **产生的文件**：
  - [sharpa_env.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/src/minilab/envs/sharpa_env.py) (重构后的 PD 力矩奖励与 0.9 限位 Gymnasium 环境)
  - [vector_env.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/src/minilab/ipc/vector_env.py) (重构后的 C++ BatchEnvPool 并行抓握状态采样向量化环境)
  - [eval.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/scripts/eval.py) (改用 launch_passive 控制循环与支持 Backspace 随机重置的评估脚本)
  - `eval_run.mp4` (生成的 100 步物理控制旋转诊断视频)

### 步骤 15：指尖精细操纵约束与高度动态终止判定重构
- **发生时间**：2026-06-13 11:28:40 (CST)
- **目的作用**：约束圆柱体必须在指尖进行搓动操纵，防止其掉入手掌心包裹。
- **物理/数学原理解析**：
  在之前的方案中，坠落终止高度设为固定的绝对下限 $0.51906$（相比初始高度 $0.61906$ 宽限了 $10\text{cm}$）。这导致 PPO 策略偏向于选择“偷懒”的掌心包裹策略（Power Grasp），即将物体下滑到掌心区域，靠手掌的包夹和阻挡来防止物体坠落，同时尝试搓动。这种姿态阻碍了指尖的精细搓动（Precision Grasp）并容易形成挤压自锁。
  本重构将终止判定条件改为动态的局部范围限制：以环境每次 reset 时初始物体高度 $z_0$ 为基准，仅允许物体在 $\pm 2\text{cm}$ 的高度区间内微幅浮动（$\text{reset\_height\_lower} = z_0 - 0.02$，$\text{reset\_height\_upper} = z_0 + 0.02$）。一旦超出此动态区间即判定为坠落（terminated = True）并扣除大额惩罚。这迫使策略放弃让物体滑落到掌心的行为，在仅能使用指尖进行捏合的同时实现了高效的精细操作。
  在 1000 轮的快速验证中，机器手尚未充分学会“Finger Gaiting”（手指交替离合步态），导致旋转后半段关节锁死。扩展到 5000 轮完整收敛训练后，机器手在 5 秒的过渡期之后，成功通过关节交替松开和微调，实现了圆柱体在指尖处持续、平稳、无锁死的 15 秒以上长程旋转操纵。
- **执行的命令**：
  ```bash
  # 运行 5000 轮完整指尖操纵策略训练 (约 8 分钟)
  .venv/bin/python scripts/train.py total_updates=5000
  # 生成 15 秒 (300步) 离屏评估视频
  .venv/bin/python scripts/eval.py record=True
  ```
- **产生的文件**：
  - [sharpa_env.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/src/minilab/envs/sharpa_env.py) (重构后引入单环境动态高度上下限判定)
  - [vector_env.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/src/minilab/ipc/vector_env.py) (重构后引入向量化多环境独立高度上下限更新与判定)
  - [eval.py](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab/scripts/eval.py) (评估帧数调整为 300 步)
  - `eval_run.mp4` (生成的 15 秒指尖精细搓动旋转视频)
