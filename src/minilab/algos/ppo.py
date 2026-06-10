import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=44, action_dim=22):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(64, 1)

    def get_value(self, x):
        hidden = self.shared(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.shared(x)
        action_mean = self.actor_mean(hidden)
        action_std = self.actor_logstd.exp()
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(hidden)
        
        return action, log_prob, entropy, value
