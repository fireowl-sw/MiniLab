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
        
        # 初始化上一时刻虚拟控制目标
        self.prev_targets = self.default_angles.copy()

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
        
        # 重置上一时刻虚拟控制目标
        self.prev_targets = self.default_angles.copy()
        
        # 将圆柱体位置重置到掌心上方，姿态重置为单位四元数
        self.data.qpos[self.num_joints:self.num_joints+3] = self.object_pos_anchor
        self.data.qpos[self.num_joints+3:self.num_joints+7] = np.array([1, 0, 0, 0])
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 1. 采用 UniLab 虚拟参考轨迹控制：增量加在 prev_targets 上，防止控制延迟
        target_ctrl = self.prev_targets + action / 24.0
        
        ctrl_min = self.model.actuator_ctrlrange[:22, 0]
        ctrl_max = self.model.actuator_ctrlrange[:22, 1]
        self.prev_targets = np.clip(target_ctrl, ctrl_min, ctrl_max)
        self.data.ctrl[:self.num_joints] = self.prev_targets
        
        # 2. 物理步进 12 次
        for _ in range(12):
            mujoco.mj_step(self.model, self.data)
        
        # 3. 提取观测
        obs = self._get_obs()
        
        qpos = self.data.qpos[:self.num_joints]
        object_pos = self.data.qpos[self.num_joints:self.num_joints+3]
        object_linvel = self.data.qvel[self.num_joints:self.num_joints+3]
        object_angvel = self.data.qvel[self.num_joints+3:self.num_joints+6]
        
        # 4. 计算复合奖励
        # 4.1 对称绝对值旋转速度奖励 (权重 5.0，clip 0~1.0)
        rotate_reward = 5.0 * np.clip(np.abs(object_angvel[2]), 0.0, 1.0)
        
        # 4.2 物体偏离中心锚点的距离惩罚 (改为常数梯度平滑惩罚，消除挤压自锁)
        dist_to_anchor = np.linalg.norm(object_pos - self.object_pos_anchor)
        dist_penalty = -10.0 * dist_to_anchor
        
        # 4.3 关节姿态偏差惩罚 (对齐 UniLab，限制大拇指过度下压与手指被动撑开，权重 -0.4)
        pose_diff_penalty = -0.4 * np.sum(np.square(qpos - self.default_angles))
        
        # 4.4 物体移动速度惩罚 (权重 -0.3)
        linvel_penalty = -0.3 * np.sum(np.abs(object_linvel))
        
        # 4.5 动作惩罚
        action_penalty = -0.01 * np.sum(np.square(action))
        
        reward = float(
            rotate_reward + 
            dist_penalty + 
            pose_diff_penalty + 
            linvel_penalty + 
            action_penalty
        )
        
        # 5. 坠落终止判定 (低于锚点高度 10 厘米判定为坠落)
        terminated = False
        if object_pos[2] < (self.object_pos_anchor[2] - 0.1):
            terminated = True
            reward -= 10.0  # 坠落惩罚
            
        truncated = False
        return obs, reward, terminated, truncated, {}
