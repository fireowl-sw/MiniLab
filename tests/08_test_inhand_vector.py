import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from minilab.ipc.vector_env import BatchVectorEnv

def main():
    print("Initializing C++ BatchEnvPool Vector Environment...")
    # 初始化并行数为 4 的并行向量环境
    envs = BatchVectorEnv(num_envs=4, obs_dim=60, action_dim=22)
    
    print("\n--- Resetting Environment ---")
    obs = envs.reset()
    print(f"Observation batch shape: {obs.shape} (Expected: [4, 60])")
    print(f"Observation dtype: {obs.dtype} (Expected: torch.float32)")
    
    # 随机选择动作执行 30 步，观察物理反应和自动重置 (Auto-reset)
    print("\n--- Stepping Environment (30 steps) ---")
    for step in range(1, 31):
        # 产生随机动作抖动
        actions = torch.randn(4, 22) * 0.5
        obs, rewards, dones = envs.step(actions)
        
        # 统计当前 Step 中有几个环境触发了坠落重置
        num_resets = dones.sum().item()
        
        # 提取第 0 个环境的物体高度、奖励和 Done 标志
        env0_obj_z = obs[0, 46].item()
        env0_reward = rewards[0].item()
        env0_done = dones[0].item()
        
        print(f"Step {step:02d} | Resets in batch: {num_resets} | Env 0 Obj Z: {env0_obj_z:.4f} | Env 0 Reward: {env0_reward:7.3f} | Env 0 Done: {env0_done}")
        
    envs.close()
    print("\nBatch Vector Env test completed successfully!")

if __name__ == "__main__":
    main()
