import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=44, action_dim=22):
        super().__init__()
        # 1. 独立的 Actor 决策网络 (从状态直接映射到动作均值)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        
        # 2. 独立的 Critic 估值网络 (从状态直接映射到状态价值 V)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 3. 将 log_std 初始值从 0.0 降低到 -0.5 (限制初始乱晃幅度，使探索更加聚焦和安全)
        self.actor_logstd = nn.Parameter(torch.fill(torch.zeros(action_dim), -0.5))

    def get_value(self, x):
        """Critic 前向传播"""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """前向传播计算动作分布、值及 Log 概率"""
        action_mean = self.actor(x)
        # 限制 log_std 在 [-2.0, -0.5] 之间，防止探索发散，维持手部的精细微调动作
        log_std = torch.clamp(self.actor_logstd, -2.0, -0.5)
        action_std = log_std.exp()
        
        # 构造高斯分布
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(x)
        
        return action, log_prob, entropy, value
