import os
import numpy as np
import gymnasium as gym
import mujoco

# 稳定的初始手内抓取关节姿态 (角度值，转成弧度后控制手部处于捏紧圆柱体姿态)
SOURCE_DEFAULT_HAND_JOINT_POS_DEG = (
    95.12771, -3.11244, 14.81626, -1.03493, 12.23986,
    65.21091, 6.1133, 15.58495, 5.90325, 31.74149,
    -0.95812, 41.88173, 12.844, 31.72383, 9.84458,
    35.22366, 18.02839, 10.9712, 68.30895, 7.99151,
    5.89626, 5.89875
)

class SharpaHandGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # 1. 载入模型 XML
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../assets/robots/sharpa_wave/scene.xml")
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_joints = 22
        
        # 2. 保存捏紧圆柱体的默认关节位置 (弧度)
        self.default_angles = np.deg2rad(np.asarray(SOURCE_DEFAULT_HAND_JOINT_POS_DEG, dtype=np.float32))
        
        # 3. 设定默认物体中心锚点 (对应 XML 中的物体初始高度 z=0.61906)
        self.object_pos_anchor = np.array([-0.09559, -0.00517, 0.61906], dtype=np.float32)
        # 目标旋转轴（设为绕 Z 轴旋转）
        self.rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # 4. 定义 22 维动作空间
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        
        # 5. 定义 60 维状态观测空间：
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32
        )

    def _get_obs(self):
        qpos = self.data.qpos[:self.num_joints].copy()
        qvel = self.data.qvel[:self.num_joints].copy()
        
        # 提取物体坐标与旋转四元数
        object_pos = self.data.qpos[self.num_joints:self.num_joints+3].copy()
        object_quat = self.data.qpos[self.num_joints+3:self.num_joints+7].copy()
        
        # 提取物体的线速度与角速度
        object_linvel = self.data.qvel[self.num_joints:self.num_joints+3].copy()
        object_angvel = self.data.qvel[self.num_joints+3:self.num_joints+6].copy()
        
        # 拼接成 60 维观测向量
        obs = np.concatenate([
            qpos, qvel,
            object_pos, object_quat,
            object_linvel, object_angvel,
            self.rot_axis
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # 将手部关节和执行器初始化为捏紧姿态
        self.data.qpos[:self.num_joints] = self.default_angles
        self.data.ctrl[:self.num_joints] = self.default_angles
        
        # 将圆柱体位置重置到掌心上方，姿态重置为单位四元数
        self.data.qpos[self.num_joints:self.num_joints+3] = self.object_pos_anchor
        self.data.qpos[self.num_joints+3:self.num_joints+7] = np.array([1, 0, 0, 0])
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 1. 计算 PD 增量控制目标
        current_qpos = self.data.qpos[:self.num_joints]
        target_ctrl = current_qpos + action / 24.0
        self.data.ctrl[:self.num_joints] = target_ctrl
        
        # 2. 物理步进
        mujoco.mj_step(self.model, self.data)
        
        # 3. 提取观测
        obs = self._get_obs()
        
        object_pos = self.data.qpos[self.num_joints:self.num_joints+3]
        object_linvel = self.data.qvel[self.num_joints:self.num_joints+3]
        object_angvel = self.data.qvel[self.num_joints+3:self.num_joints+6]
        
        # 4. 计算复合奖励
        # 4.1 沿 Z 轴的旋转速度奖励
        rotate_reward = np.clip(np.sum(object_angvel * self.rot_axis), -1.0, 1.0)
        
        # 4.2 物体移动位移惩罚
        linvel_penalty = -1.0 * np.sum(np.square(object_linvel))
        
        # 4.3 物体偏离中心锚点的距离倒数奖励
        dist_to_anchor = np.linalg.norm(object_pos - self.object_pos_anchor)
        pos_holding_reward = 1.0 / (dist_to_anchor + 0.001)
        
        # 4.4 动作惩罚
        action_penalty = -0.01 * np.sum(np.square(action))
        
        reward = float(
            1.5 * rotate_reward + 
            0.1 * linvel_penalty + 
            0.5 * pos_holding_reward + 
            action_penalty
        )
        
        # 5. 坠落终止判定 (低于锚点高度 10 厘米判定为坠落)
        terminated = False
        if object_pos[2] < (self.object_pos_anchor[2] - 0.1):
            terminated = True
            reward -= 10.0  # 坠落惩罚
            
        truncated = False
        return obs, reward, terminated, truncated, {}
