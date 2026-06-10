import os
import sys
import time
import importlib.util
import torch
import torch.nn as nn
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

# ==========================================
# 1. 定义 PPO Actor-Critic 网络
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=44, action_dim=22):
        super().__init__()
        # 共享特征提取器
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # Actor 头部：输出连续动作的均值 (mean)
        self.actor_mean = nn.Linear(64, action_dim)
        # 可学习的对数标准差参数 (log_std)，代表高斯分布的方差探索幅度
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        # Critic 头部：输出 1 维状态价值 (Value)
        self.critic = nn.Linear(64, 1)

    def get_value(self, x):
        """仅提取状态价值"""
        hidden = self.shared(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        """核心前向传播：计算动作分布、状态价值以及采样动作"""
        hidden = self.shared(x)
        action_mean = self.actor_mean(hidden)
        # 对数标准差转为标准差
        action_std = self.actor_logstd.exp()
        # 构建多元对角高斯分布
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            # 随机采样动作
            action = dist.sample()
            
        # 计算动作的对数概率 (在 22 个维度上求和)
        log_prob = dist.log_prob(action).sum(dim=-1)
        # 计算动作分布熵，鼓励探索
        entropy = dist.entropy().sum(dim=-1)
        # 估计状态价值
        value = self.critic(hidden)
        
        return action, log_prob, entropy, value


# ==========================================
# 2. 定义环境子进程的工作函数
# ==========================================
def worker_fn(shared_obs, shared_action, shared_reward, shared_done, action_ready, data_ready, stop_event):
    """
    子进程工作函数：管理多并行环境并与主进程同步
    """
    num_envs = shared_obs.shape[0]
    print(f"[子进程] 正在初始化 {num_envs} 个并行仿真环境...")
    envs = [SharpaHandGymEnv() for _ in range(num_envs)]
    
    # 批量重置环境并写入初始观测
    for i in range(num_envs):
        obs, info = envs[i].reset()
        shared_obs[i].copy_(torch.from_numpy(obs))
        shared_reward[i] = 0.0
        shared_done[i] = False
        
    print("[子进程] 所有环境初始化 Reset 完成。")
    data_ready.set()
    
    while not stop_event.is_set():
        # 等待动作就绪
        is_ready = action_ready.wait(timeout=0.1)
        if not is_ready:
            continue
            
        action_ready.clear()
        
        # 读取共享内存动作
        actions_np = shared_action.numpy()
        
        # 对各个环境循环执行 step
        for i in range(num_envs):
            env = envs[i]
            action = actions_np[i]
            
            if shared_done[i]:
                # 自动重置 done 的环境
                obs, info = env.reset()
                reward = 0.0
                terminated = False
                truncated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                
            shared_obs[i].copy_(torch.from_numpy(obs))
            shared_reward[i] = reward
            shared_done[i] = terminated or truncated
            
        # 数据写入完成，通知主进程
        data_ready.set()
        
    print("[子进程] 工作循环退出。")


# ==========================================
# 3. 主进程与 Rollout Buffer 收集
# ==========================================
if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    num_envs = 4
    rollout_steps = 32
    
    print("[主进程] 分配共享内存中用于多进程通信的 Tensor...")
    shared_obs = torch.zeros((num_envs, 44)).share_memory_()
    shared_action = torch.zeros((num_envs, 22)).share_memory_()
    shared_reward = torch.zeros(num_envs).share_memory_()
    shared_done = torch.zeros(num_envs, dtype=torch.bool).share_memory_()
    
    print("[主进程] 创建同步事件...")
    action_ready = ctx.Event()
    data_ready = ctx.Event()
    stop_event = ctx.Event()
    
    # 初始化 PPO 神经网络模型与优化器
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # 启动环境子进程
    process = ctx.Process(
        target=worker_fn,
        args=(shared_obs, shared_action, shared_reward, shared_done, action_ready, data_ready, stop_event)
    )
    process.start()
    
    # 分配 Rollout Buffer (轨迹收集缓冲区)
    obs_buffer = torch.zeros((rollout_steps, num_envs, 44))
    action_buffer = torch.zeros((rollout_steps, num_envs, 22))
    log_prob_buffer = torch.zeros((rollout_steps, num_envs))
    reward_buffer = torch.zeros((rollout_steps, num_envs))
    done_buffer = torch.zeros((rollout_steps, num_envs))
    value_buffer = torch.zeros((rollout_steps, num_envs))
    
    # 等待子进程完成初始 Reset
    data_ready.wait()
    data_ready.clear()
    
    # 主进程维护的当前步状态
    next_obs = shared_obs.clone()
    next_done = torch.zeros(num_envs)
    
    print(f"[主进程] 开始收集长度为 {rollout_steps} 步的轨迹数据 (Batch Size = {num_envs})...")
    
    # 轨迹采集主循环
    for step in range(rollout_steps):
        obs_buffer[step] = next_obs
        done_buffer[step] = next_done
        
        # 1. 神经网络前向传播得到动作决策 (不计算梯度以提高采样速度)
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(next_obs)
            value_buffer[step] = value.squeeze(-1)
            
        action_buffer[step] = action
        log_prob_buffer[step] = log_prob
        
        # 2. 将动作写进共享内存并通知子进程推进仿真
        shared_action.copy_(action)
        action_ready.set()
        
        # 3. 等待子进程计算完毕
        data_ready.wait()
        data_ready.clear()
        
        # 4. 获取下一时刻观测、奖励和 done 标记
        reward_buffer[step] = shared_reward.clone()
        next_done = shared_done.clone().float()
        next_obs = shared_obs.clone()
        
        if (step + 1) % 8 == 0:
            print(f"  已收集 {step + 1} / {rollout_steps} 步数据...")

    print("[主进程] 数据收集完毕。开始 PPO GAE 优势估计与 Loss 计算...")
    
    # ==========================================
    # 4. 优势估计 (GAE 计算) 与 PPO 损失校验
    # ==========================================
    advantages = torch.zeros((rollout_steps, num_envs))
    last_gae_lam = 0
    
    with torch.no_grad():
        next_value = model.get_value(next_obs).squeeze(-1)
        
    # 逆向递归计算 GAE 优势值
    for t in reversed(range(rollout_steps)):
        if t == rollout_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - done_buffer[t + 1]
            nextvalues = value_buffer[t + 1]
        # 时间差分偏差 (TD Error)
        delta = reward_buffer[t] + 0.99 * nextvalues * nextnonterminal - value_buffer[t]
        advantages[t] = last_gae_lam = delta + 0.99 * 0.95 * nextnonterminal * last_gae_lam
        
    returns = advantages + value_buffer

    # 展平数据为 Batch，方便批量计算 Loss
    b_obs = obs_buffer.reshape(-1, 44)
    b_actions = action_buffer.reshape(-1, 22)
    b_logprobs = log_prob_buffer.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = value_buffer.reshape(-1)

    # 重新进行计算图前向传播 (开启梯度)
    _, new_logprobs, entropy, new_value = model.get_action_and_value(b_obs, b_actions)
    
    # 计算概率比例 Ratio
    logratio = new_logprobs - b_logprobs
    ratio = logratio.exp()
    
    # 优势归一化
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    
    # Policy Loss (Actor 损失：带 Clipping 的代理损失)
    pg_loss1 = -b_advantages * ratio
    pg_loss2 = -b_advantages * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    
    # Value Loss (Critic 损失：均方误差 MSE)
    v_loss = 0.5 * ((new_value.squeeze(-1) - b_returns) ** 2).mean()
    
    # Entropy Loss (分布熵损失，用于增加探索性)
    entropy_loss = entropy.mean()
    
    # 总 PPO 损失函数
    loss = pg_loss + 0.5 * v_loss - 0.01 * entropy_loss
    
    # 梯度反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 5. 计算并提取神经网络第一层 Linear 权重的梯度范数
    # 以此证明 PyTorch 自动微分计算图完全打通且形状无误
    grad_norm = 0.0
    for param in model.shared[0].parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    print("--------------------------------------------------")
    print("PPO Loss and Backpropagation Verification:")
    print(f"  Policy Loss (Actor):       {pg_loss.item():.6f}")
    print(f"  Value Loss (Critic):       {v_loss.item():.6f}")
    print(f"  Entropy Loss:              {entropy_loss.item():.6f}")
    print(f"  Total PPO Loss:            {loss.item():.6f}")
    print(f"  Actor-Critic Grad Norm:    {grad_norm:.6f}")
    print("--------------------------------------------------")
    
    # 清理并关闭环境子进程
    print("[主进程] 调试结束，正在安全关闭子进程...")
    stop_event.set()
    process.join()
    print("[主进程] 批量子进程已安全退出，测试成功！")
