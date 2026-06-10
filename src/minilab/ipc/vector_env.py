import torch
import torch.multiprocessing as mp
import numpy as np
from minilab.envs.sharpa_env import SharpaHandGymEnv

def worker_fn(shared_obs, shared_action, shared_reward, shared_done, action_ready, data_ready, stop_event):
    num_envs = shared_obs.shape[0]
    envs = [SharpaHandGymEnv() for _ in range(num_envs)]
    
    # 批量重置所有环境
    for i in range(num_envs):
        obs, _ = envs[i].reset()
        shared_obs[i].copy_(torch.from_numpy(obs))
        shared_reward[i] = 0.0
        shared_done[i] = False
        
    data_ready.set()
    
    while not stop_event.is_set():
        is_ready = action_ready.wait(timeout=0.1)
        if not is_ready:
            continue
        action_ready.clear()
        
        actions_np = shared_action.numpy()
        for i in range(num_envs):
            env = envs[i]
            action = actions_np[i]
            
            if shared_done[i]:
                obs, _ = env.reset()
                reward = 0.0
                terminated = False
                truncated = False
            else:
                obs, reward, terminated, truncated, _ = env.step(action)
                
            shared_obs[i].copy_(torch.from_numpy(obs))
            shared_reward[i] = reward
            shared_done[i] = terminated or truncated
            
        data_ready.set()

class SharedMemoryVectorEnv:
    def __init__(self, num_envs=4, obs_dim=44, action_dim=22):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        ctx = mp.get_context("spawn")
        self.shared_obs = torch.zeros((num_envs, obs_dim)).share_memory_()
        self.shared_action = torch.zeros((num_envs, action_dim)).share_memory_()
        self.shared_reward = torch.zeros(num_envs).share_memory_()
        self.shared_done = torch.zeros(num_envs, dtype=torch.bool).share_memory_()
        
        self.action_ready = ctx.Event()
        self.data_ready = ctx.Event()
        self.stop_event = ctx.Event()
        
        self.process = ctx.Process(
            target=worker_fn,
            args=(
                self.shared_obs,
                self.shared_action,
                self.shared_reward,
                self.shared_done,
                self.action_ready,
                self.data_ready,
                self.stop_event
            )
        )
        self.process.start()
        
        # 等待子进程就绪
        self.data_ready.wait()
        self.data_ready.clear()

    def reset(self):
        return self.shared_obs.clone()

    def step(self, actions):
        """
        actions: torch.Tensor, Shape (num_envs, action_dim)
        """
        self.shared_action.copy_(actions)
        self.action_ready.set()
        
        self.data_ready.wait()
        self.data_ready.clear()
        
        return self.shared_obs.clone(), self.shared_reward.clone(), self.shared_done.clone()

    def close(self):
        self.stop_event.set()
        self.process.join()
