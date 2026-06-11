import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from minilab.envs.sharpa_env import SharpaHandGymEnv

def main():
    print("Initializing In-Hand Manipulation Environment...")
    env = SharpaHandGymEnv()
    obs, info = env.reset()
    
    print(f"Observation dimension successfully expanded: {obs.shape[0]} (Expected: 60)")
    print(f"Initial object position: {obs[44:47]} (Expected: {env.object_pos_anchor})")
    
    # 随机选择动作执行 20 步，观测物体的物理变化与坠落触发判定
    for step in range(1, 21):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        obj_pos = obs[44:47]
        obj_angvel = obs[54:57]
        print(f"Step {step:02d} | Reward: {reward:7.3f} | Obj Pos: {obj_pos} | Obj AngVel: {obj_angvel} | Terminated: {terminated}")
        if terminated:
            print("Object dropped! Resetting...")
            env.reset()
            
if __name__ == "__main__":
    main()
