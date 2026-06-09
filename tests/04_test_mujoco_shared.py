import os
import sys
import time
import importlib.util
import torch
import torch.multiprocessing as mp
import numpy as np

# 动态载入同级目录下的 03_test_mujoco_gym.py 文件中的 SharpaHandGymEnv
# 因为文件名以数字开头且执行目录可能不在 sys.path 中，使用基于文件路径的 importlib.util 导入最稳健。
curr_dir = os.path.dirname(os.path.abspath(__file__))
gym_file_path = os.path.join(curr_dir, "03_test_mujoco_gym.py")

spec = importlib.util.spec_from_file_location("mujoco_gym_env", gym_file_path)
gym_module = importlib.util.module_from_spec(spec)
sys.modules["mujoco_gym_env"] = gym_module
spec.loader.exec_module(gym_module)
SharpaHandGymEnv = gym_module.SharpaHandGymEnv

def worker_fn(shared_obs, shared_action, shared_reward, action_ready, data_ready, stop_event):
    """
    子进程工作函数：实例化仿真环境并运行物理步进。
    通过共享内存读取动作并将物理仿真产生的状态和奖励写回，使用 Event 进行同步。
    """
    print("[子进程] 正在初始化仿真环境...")
    env = SharpaHandGymEnv()
    
    # 重置环境
    obs, info = env.reset()
    
    # 写入初始观测到共享内存中
    shared_obs.copy_(torch.from_numpy(obs))
    
    # 初始数据准备完毕，通知主进程
    data_ready.set()
    
    while not stop_event.is_set():
        # 1. 等待主进程发出 action_ready 信号 (超时时间设为 0.1秒以便检测 stop_event)
        is_ready = action_ready.wait(timeout=0.1)
        if not is_ready:
            continue
        
        # 收到动作就绪信号，清除该信号以便下一次等待
        action_ready.clear()
        
        # 2. 从共享内存中读取动作数据并转成 numpy
        action = shared_action.numpy()
        
        # 3. 在环境中执行 step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 4. 将最新观测和奖励写回共享内存
        # copy_ 方法用于原地修改共享内存 Tensor 中的数值，确保主进程能直接看到最新变化
        shared_obs.copy_(torch.from_numpy(obs))
        shared_reward[0] = reward
        
        # 5. 通知主进程：物理仿真和数据写回完成，可以读取数据了
        data_ready.set()
        
    print("[子进程] 工作循环退出。")


if __name__ == "__main__":
    # 使用 spawn 启动模式创建子进程上下文
    ctx = mp.get_context("spawn")
    
    print("[主进程] 分配 CPU 共享内存中的 Tensor...")
    # share_memory_() 将 Tensor 移动到共享内存段，多个进程可以对其进行零拷贝的读写操作
    shared_obs = torch.zeros(44).share_memory_()      # 44 维观测
    shared_action = torch.zeros(22).share_memory_()   # 22 维动作
    shared_reward = torch.zeros(1).share_memory_()    # 1 维奖励值
    
    print("[主进程] 创建同步事件...")
    action_ready = ctx.Event()  # 主进程通知子进程的事件
    data_ready = ctx.Event()    # 子进程通知主进程的事件
    stop_event = ctx.Event()    # 安全停止子进程的事件
    
    # 创建并启动子进程
    process = ctx.Process(
        target=worker_fn,
        args=(shared_obs, shared_action, shared_reward, action_ready, data_ready, stop_event)
    )
    process.start()
    
    # 运行 5 个 step 来测试共享内存的数据交互
    print("[主进程] 开始主循环交互调试 (5步)...")
    for step in range(1, 6):
        # 1. 等待子进程将数据写回共享内存
        data_ready.wait()
        data_ready.clear()  # 清除信号，准备下一次等待
        
        # 2. 从共享内存中安全读取数据（零拷贝）
        # 对共享内存中 Tensor 的读取是实时的
        obs_numpy = shared_obs.numpy()
        reward_val = shared_reward[0].item()
        
        print(f"\n--- [主进程] Step {step} 收到数据 ---")
        print(f"  Obs (前 3 关节角度):  {obs_numpy[:3]}")
        print(f"  Obs (前 3 关节速度):  {obs_numpy[22:25]}")
        print(f"  Reward (奖励分值):   {reward_val:.6f}")
        
        # 3. 随机生成下一步动作并写入共享动作空间
        action = np.random.uniform(-1.0, 1.0, size=(22,)).astype(np.float32)
        shared_action.copy_(torch.from_numpy(action))
        
        # 4. 通知子进程：动作写入完成，可以执行仿真物理步进了
        action_ready.set()
        
    # 安全清理与关闭子进程
    print("\n[主进程] 交互测试结束，正在安全关闭子进程...")
    stop_event.set()
    process.join()
    print("[主进程] 子进程已成功关闭，测试完成！")
