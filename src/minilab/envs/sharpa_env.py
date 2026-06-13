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
        
        # 4. 载入抓握数据集
        dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../assets/robots/sharpa_wave/sharpa_grasp_linspace_1.npy")
        )
        self.dataset = np.load(dataset_path)
        
        # 5. 提取 actuator PD gains
        self.kp = self.model.actuator_gainprm[:self.num_joints, 0]
        self.kd = -self.model.actuator_biasprm[:self.num_joints, 2]
        
        # 6. 定义 0.9 缩放的控制范围上限与下限
        self.ctrl_min = self.model.actuator_ctrlrange[:self.num_joints, 0] * 0.9
        self.ctrl_max = self.model.actuator_ctrlrange[:self.num_joints, 1] * 0.9
        
        # 7. 定义动作空间与状态空间
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
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
        
        # 从数据集中均匀采样一行
        idx = np.random.randint(0, len(self.dataset))
        S = self.dataset[idx]
        
        # 设置手部关节角与控制目标为采样的初始状态
        self.data.qpos[:self.num_joints] = S[:self.num_joints]
        self.data.ctrl[:self.num_joints] = S[:self.num_joints]
        self.prev_targets = S[:self.num_joints].copy()
        
        # 设置物体初始位置与四元数姿态
        self.data.qpos[self.num_joints:self.num_joints+3] = S[self.num_joints:self.num_joints+3]
        self.data.qpos[self.num_joints+3:self.num_joints+7] = S[self.num_joints+3:self.num_joints+7]
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 1. 采用 UniLab 虚拟参考轨迹控制：增量加在 prev_targets 上，防止控制延迟，并使用 0.9 缩放范围裁剪
        target_ctrl = self.prev_targets + action / 24.0
        self.prev_targets = np.clip(target_ctrl, self.ctrl_min, self.ctrl_max)
        self.data.ctrl[:self.num_joints] = self.prev_targets
        
        # 2. 物理步进 12 次
        for _ in range(12):
            mujoco.mj_step(self.model, self.data)
        
        # 3. 提取观测
        obs = self._get_obs()
        
        qpos = self.data.qpos[:self.num_joints]
        qvel = self.data.qvel[:self.num_joints]
        
        # 提取物体观测
        x_obj = self.data.qpos[self.num_joints:self.num_joints+3]
        v_obj = self.data.qvel[self.num_joints:self.num_joints+3]
        w_obj = self.data.qvel[self.num_joints+3:self.num_joints+6]
        
        # 4. 计算虚拟关节力矩
        torque_virtual = self.kp * (self.prev_targets - qpos) - self.kd * qvel
        
        # 5. 计算六项复合奖励
        # 5.1 rotate (旋转奖励)
        rotate_reward = np.clip(np.sum(w_obj * self.rot_axis), -0.5, 0.5) * 2.5
        # 5.2 obj_linvel (物体平移惩罚)
        linvel_penalty = np.sum(np.abs(v_obj)) * -0.3
        # 5.3 pose_diff (关节偏离惩罚)
        pose_diff_penalty = np.sum(np.square(qpos - self.default_angles)) * -0.4
        # 5.4 torque (控制力矩惩罚)
        torque_penalty = np.sum(np.square(torque_virtual)) * -0.1
        # 5.5 work (物理功惩罚)
        work_penalty = (np.sum(torque_virtual * qvel)) ** 2 * -0.5
        # 5.6 object_pos (位置锚定奖励)
        dist_to_anchor = np.linalg.norm(x_obj - self.object_pos_anchor)
        object_pos_reward = (1.0 / (dist_to_anchor + 0.001)) * 0.003
        
        total_reward = float(
            rotate_reward + 
            linvel_penalty + 
            pose_diff_penalty + 
            torque_penalty + 
            work_penalty + 
            object_pos_reward
        )
        
        # 缩放总奖励为 0.05
        reward = total_reward * 0.05
        
        # 6. 坠落终止判定与惩罚 (低于高度阈值 0.51906 判定为坠落并终止)
        terminated = False
        if x_obj[2] < 0.51906:
            terminated = True
            reward -= 10.0
            
        truncated = False
        return obs, reward, terminated, truncated, {}
