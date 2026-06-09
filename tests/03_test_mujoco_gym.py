import os
import numpy as np
import gymnasium as gym
import mujoco

class SharpaHandGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # 1. 获取模型 XML 文件的绝对路径
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../assets/robots/sharpa_wave/scene.xml")
        )
        
        # 2. 加载 MuJoCo 模型与数据
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 3. 记录机器手受控关节的数量 (22维)
        self.num_joints = 22  # Sharpa 机器手的 22 维关节/电机控制
        
        # 4. 定义动作空间 (Action Space)：22 维，范围在 [-1.0, 1.0] 之间
        # 动作代表每个关节目标角度的变化增量
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32
        )
        
        # 5. 定义观测空间 (Observation Space)：44 维
        # 前 22 维为关节角度位置 (qpos)，后 22 维为关节角速度 (qvel)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_joints * 2,),
            dtype=np.float32
        )
        
        # 记录初始位置，用于计算奖励函数
        self.init_qpos = np.zeros(self.num_joints, dtype=np.float32)

    def _get_obs(self):
        # 提取前 22 个受控关节的位置和速度
        qpos = self.data.qpos[:self.num_joints].copy()
        qvel = self.data.qvel[:self.num_joints].copy()
        # 拼接成 44 维观测向量
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def reset(self, seed=None, options=None):
        # 1. 随机数种子初始化
        super().reset(seed=seed)
        
        # 2. 重置 MuJoCo 仿真状态与数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 3. 记录重置时的初始关节位置，以便作为基准计算奖励
        self.init_qpos = self.data.qpos[:self.num_joints].copy()
        
        # 4. 获取初始观测并返回
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # 确保动作值在指定区间 [-1.0, 1.0] 内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 1. 计算控制指令并发射给致动器
        # 新的目标角度 = 当前关节角度 + action / 24.0 (增量式位置控制)
        current_qpos = self.data.qpos[:self.num_joints]
        target_ctrl = current_qpos + action / 24.0
        self.data.ctrl[:self.num_joints] = target_ctrl
        
        # 2. 调用 MuJoCo 引擎进行物理仿真步进
        # 这将推进时间步长，并在后台触发致动器的力矩计算
        mujoco.mj_step(self.model, self.data)
        
        # 3. 步进后提取最新的观测
        observation = self._get_obs()
        
        # 4. 计算奖励函数：关节角度偏离初始位置的负二次惩罚项
        diff = self.data.qpos[:self.num_joints] - self.init_qpos
        reward = float(-np.sum(np.square(diff)))
        
        # 5. 设定终止与超时标志 (本测试中默认不终止)
        terminated = False
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    print("Testing SharpaHandGymEnv...")
    env = SharpaHandGymEnv()
    
    # 重置环境
    obs, info = env.reset(seed=42)
    print(f"Initial Observation Shape: {obs.shape}")
    print(f"Initial Observation Sample (first 5 joint angles):\n {obs[:5]}")
    print("--------------------------------------------------")
    
    # 模拟运行 10 个 Step
    for step_idx in range(1, 11):
        # 随机采样动作
        action = env.action_space.sample()
        
        # 执行一步
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step_idx}:")
        print(f"  Action (sample first 3): {action[:3]}")
        print(f"  Observation Shape:       {obs.shape}")
        print(f"  Reward:                  {reward:.6f}")
        print(f"  First 3 joint positions: {obs[:3]}")
        print(f"  First 3 joint velocities:{obs[22:25]}")
        print("--------------------------------------------------")
    
    print("Test run completed successfully!")
