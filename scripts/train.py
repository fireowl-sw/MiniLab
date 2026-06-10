import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
# 将 src 目录添加到模块查找路径中
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from minilab.ipc.vector_env import SharedMemoryVectorEnv
from minilab.algos.ppo import ActorCritic

def train():
    num_envs = 4
    rollout_steps = 128
    total_updates = 30  # 极简训练验证
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    ppo_epochs = 4
    mini_batch_size = 32
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    device = torch.device("cpu")
    print(f"[训练进程] 使用计算设备: {device}")
    
    # 1. 实例化环境和网络
    envs = SharedMemoryVectorEnv(num_envs=num_envs)
    agent = ActorCritic().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    
    # 2. 轨迹缓冲区分配
    obs_buffer = torch.zeros((rollout_steps, num_envs, 44)).to(device)
    action_buffer = torch.zeros((rollout_steps, num_envs, 22)).to(device)
    log_prob_buffer = torch.zeros((rollout_steps, num_envs)).to(device)
    reward_buffer = torch.zeros((rollout_steps, num_envs)).to(device)
    done_buffer = torch.zeros((rollout_steps, num_envs)).to(device)
    value_buffer = torch.zeros((rollout_steps, num_envs)).to(device)
    
    # 3. 初始化状态
    next_obs = envs.reset().to(device)
    next_done = torch.zeros(num_envs).to(device)
    
    start_time = time.time()
    print("[训练进程] 开始进行 PPO 强化学习策略训练循环...")
    
    for update in range(1, total_updates + 1):
        episode_rewards = []
        # 3.1 轨迹收集阶段
        for step in range(rollout_steps):
            obs_buffer[step] = next_obs
            done_buffer[step] = next_done
            
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_obs)
                value_buffer[step] = value.squeeze(-1)
                
            action_buffer[step] = action
            log_prob_buffer[step] = log_prob
            
            # 环境步进
            next_obs_raw, reward_raw, done_raw = envs.step(action.cpu())
            
            next_obs = next_obs_raw.to(device)
            next_done = done_raw.to(device).float()
            reward_buffer[step] = reward_raw.to(device)
            
            episode_rewards.append(reward_raw.mean().item())
            
        # 3.2 优势估计 (GAE)
        advantages = torch.zeros((rollout_steps, num_envs)).to(device)
        last_gae_lam = 0
        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze(-1)
            
        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - done_buffer[t + 1]
                nextvalues = value_buffer[t + 1]
            delta = reward_buffer[t] + gamma * nextvalues * nextnonterminal - value_buffer[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * nextnonterminal * last_gae_lam
        returns = advantages + value_buffer
        
        # 展平数据块
        b_obs = obs_buffer.reshape(-1, 44)
        b_actions = action_buffer.reshape(-1, 22)
        b_logprobs = log_prob_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_buffer.reshape(-1)
        
        # 3.3 策略优化 PPO Epochs
        batch_size = rollout_steps * num_envs
        indices = np.arange(batch_size)
        
        pg_losses, v_losses, ent_losses = [], [], []
        
        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]
                
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )
                
                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_idx]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((new_value.squeeze(-1) - b_returns[mb_idx]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
                
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy_loss.item())
                
        avg_reward = np.mean(episode_rewards)
        fps = int((rollout_steps * num_envs) / (time.time() - start_time))
        start_time = time.time()
        
        if update % 5 == 0 or update == 1:
            print(f"Update {update:02d}/{total_updates:02d} | "
                  f"Mean Reward: {avg_reward:.4f} | "
                  f"Policy Loss: {np.mean(pg_losses):.4f} | "
                  f"Value Loss: {np.mean(v_losses):.4f} | "
                  f"Entropy: {np.mean(ent_losses):.4f} | "
                  f"FPS: {fps}")
            
    # 4. 训练结束，保存模型
    os.makedirs("weights", exist_ok=True)
    model_path = "weights/ppo_sharpa.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"[训练进程] 训练完成！权重已成功保存至 {model_path}")
    
    envs.close()
    print("[训练进程] 向量化仿真进程已安全退出。")

if __name__ == "__main__":
    train()
