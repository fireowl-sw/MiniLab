import os
import sys
import time
import importlib.util
import torch
import torch.multiprocessing as mp
import numpy as np

# 动态载入同级目录下的 03_test_mujoco_gym.py 中的 SharpaHandGymEnv
curr_dir = os.path.dirname(os.path.abspath(__file__))
gym_file_path = os.path.join(curr_dir, "03_test_mujoco_gym.py")

spec = importlib.util.spec_from_file_location("mujoco_gym_env", gym_file_path)
gym_module = importlib.util.module_from_spec(spec)
sys.modules["mujoco_gym_env"] = gym_module
spec.loader.exec_module(gym_module)
SharpaHandGymEnv = gym_module.SharpaHandGymEnv

def worker_fn(shared_obs, shared_action, shared_reward, shared_done, action_ready, data_ready, stop_event):
    """
    子进程工作函数：实例化 4 个独立的仿真环境，并在循环中批量步进。
    使用共享的 2D/1D Tensor 和主进程高效交互。
    """
    num_envs = shared_obs.shape[0]
    print(f"[子进程] 正在初始化 {num_envs} 个并行仿真环境...")
    envs = [SharpaHandGymEnv() for _ in range(num_envs)]
    
    # 1. 批量重置所有环境，并填充共享观测内存
    for i in range(num_envs):
        obs, info = envs[i].reset()
        shared_obs[i].copy_(torch.from_numpy(obs))
        shared_reward[i] = 0.0
        shared_done[i] = False
        
    print("[子进程] 所有环境初始化 Reset 完成。")
    # 初始化数据就绪，通知主进程
    data_ready.set()
    
    while not stop_event.is_set():
        # 等待主进程写入批量动作
        is_ready = action_ready.wait(timeout=0.1)
        if not is_ready:
            continue
            
        action_ready.clear()
        
        # 将共享内存动作转为 NumPy，方便环境读取
        actions_np = shared_action.numpy()
        
        # 2. 循环对这 4 个环境分别执行 step
        for i in range(num_envs):
            env = envs[i]
            # 获取对应的动作切片 (22维)
            action = actions_np[i]
            
            # 如果环境之前已经 done 了，我们在此处做一个自动重置 (Auto-reset)
            if shared_done[i]:
                obs, info = env.reset()
                reward = 0.0
                terminated = False
                truncated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                
            # 原地更新共享内存中对应行/对应索引的数值
            shared_obs[i].copy_(torch.from_numpy(obs))
            shared_reward[i] = reward
            shared_done[i] = terminated or truncated
            
        # 3. 批量更新完毕，通知主进程读取数据
        data_ready.set()
        
    print("[子进程] 批量环境工作循环安全退出。")


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    num_envs = 4
    
    print(f"[主进程] 分配并行数 = {num_envs} 的批量共享内存 Tensor (2D/1D)...")
    # shape 分别为 (4, 44) 和 (4, 22)，这形成了批处理数据块 (Batch)
    shared_obs = torch.zeros((num_envs, 44)).share_memory_()
    shared_action = torch.zeros((num_envs, 22)).share_memory_()
    shared_reward = torch.zeros(num_envs).share_memory_()
    shared_done = torch.zeros(num_envs, dtype=torch.bool).share_memory_()
    
    print("[主进程] 创建同步事件...")
    action_ready = ctx.Event()
    data_ready = ctx.Event()
    stop_event = ctx.Event()
    
    # 启动批量仿真子进程
    process = ctx.Process(
        target=worker_fn,
        args=(shared_obs, shared_action, shared_reward, shared_done, action_ready, data_ready, stop_event)
    )
    process.start()
    
    print("[主进程] 开始批量多环境交互调试 (5步)...")
    for step in range(1, 6):
        # 1. 等待批量数据就绪
        data_ready.wait()
        data_ready.clear()
        
        # 2. 从共享内存中获取批量状态数据
        obs_batch = shared_obs.numpy()
        reward_batch = shared_reward.numpy()
        done_batch = shared_done.numpy()
        
        print(f"\n--- [主进程] Step {step} 收到批量数据 (Batch Shape: {obs_batch.shape}) ---")
        # 打印第一个环境 (Env 0) 和最后一个环境 (Env 3) 的数据进行比对，以证明它们独立并行且不互干扰
        print(f"  [Env 0] 前 3 关节角度: {obs_batch[0, :3]} | 奖励: {reward_batch[0]:.6f} | Done: {done_batch[0]}")
        print(f"  [Env 3] 前 3 关节角度: {obs_batch[3, :3]} | 奖励: {reward_batch[3]:.6f} | Done: {done_batch[3]}")
        
        # 3. 随机生成下一步的批量动作并写入共享内存 (4, 22)
        actions = np.random.uniform(-1.0, 1.0, size=(num_envs, 22)).astype(np.float32)
        shared_action.copy_(torch.from_numpy(actions))
        
        # 4. 触发信号，通知子进程进行物理步进
        action_ready.set()
        
    print("\n[主进程] 调试测试结束，安全关闭子进程...")
    stop_event.set()
    process.join()
    print("[主进程] 批量子进程已安全退出，测试成功！")
