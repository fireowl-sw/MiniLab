import os
import sys
import torch
import numpy as np
import time  # 💡 导入时间库

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from minilab.ipc.vector_env import BatchVectorEnv

def main():
    print("Initializing C++ BatchEnvPool Vector Environment...")
    num_envs = 8  # 💡 可以设置多一些环境测试极限速度
    envs = BatchVectorEnv(num_envs=num_envs, obs_dim=60, action_dim=22)
    envs.reset()
    
    steps = 1000  # 💡 增加步数，让时间测量更准确
    print(f"\n--- Stepping Environment ({steps} steps) ---")
    
    start_time = time.time()  # 💡 记录开始时间
    for step in range(1, steps + 1):
        actions = torch.randn(num_envs, 22) * 0.5
        obs, rewards, dones = envs.step(actions)
    end_time = time.time()  # 💡 记录结束时间
    
    # 💡 计算并打印 FPS
    total_time = end_time - start_time
    total_frames = steps * num_envs
    fps = total_frames / total_time
    
    print(f"\n耗时: {total_time:.4f} 秒")
    print(f"总帧数: {total_frames} 帧")
    print(f"测试 FPS: {fps:.2f}")
    
    envs.close()

if __name__ == "__main__":
    main()
