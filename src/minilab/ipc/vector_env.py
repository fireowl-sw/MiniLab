import os
import numpy as np
import torch
import mujoco
from mujoco import batch_env

class BatchVectorEnv:
    def __init__(self, num_envs=8, obs_dim=60, action_dim=22):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 1. 载入模型 XML 并创建 model 实例
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../assets/robots/sharpa_wave/scene.xml")
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.nq = self.model.nq
        self.nv = self.model.nv
        
        # 2. 保存捏紧圆柱体的默认关节位置 (弧度) 和初始参数
        from minilab.envs.sharpa_env import SOURCE_DEFAULT_HAND_JOINT_POS_DEG
        self.default_angles = np.deg2rad(np.asarray(SOURCE_DEFAULT_HAND_JOINT_POS_DEG, dtype=np.float64))
        self.object_pos_anchor = np.array([-0.09559, -0.00517, 0.61906], dtype=np.float64)
        self.rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        
        # 3. 载入抓握数据集
        dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../assets/robots/sharpa_wave/sharpa_grasp_linspace_1.npy")
        )
        self.dataset = np.load(dataset_path)
        
        # 4. 提取 actuator PD gains
        self.kp = self.model.actuator_gainprm[:22, 0]
        self.kd = -self.model.actuator_biasprm[:22, 2]
        
        # 5. 定义 0.9 缩放的控制范围上限与下限
        self.ctrl_min = self.model.actuator_ctrlrange[:22, 0] * 0.9
        self.ctrl_max = self.model.actuator_ctrlrange[:22, 1] * 0.9
        
        # 6. 创建 C++ BatchEnvPool (使用 4 个 C++ 线程进行多线程并行加速物理计算)
        self.pool = batch_env.BatchEnvPool(self.model, nbatch=self.num_envs, nthread=4)
        self.nstate = self.pool.nstate
        
        # 7. 用于在 Python 端辅助计算 mjSTATE_FULLPHYSICS 的临时数据容器
        self.tmp_data = mujoco.MjData(self.model)
        
        # 8. 初始化保存环境当前 State 状态的批处理数组
        self.states = np.zeros((self.num_envs, self.nstate), dtype=np.float64)
        
        # 初始化上一时刻虚拟控制目标 (对于每个平行环境)
        self.prev_targets = np.tile(self.default_angles, (self.num_envs, 1))
        
        # 记录每个环境独立的指尖高度限制范围
        self.reset_height_lower = np.zeros(self.num_envs, dtype=np.float64)
        self.reset_height_upper = np.zeros(self.num_envs, dtype=np.float64)

    def _generate_perturbed_states(self, n):
        """
        从数据集中随机采样 n 个抓握姿态，并将其转换成完整的 MjState 数组形式
        """
        states = np.zeros((n, self.nstate), dtype=np.float64)
        prev_targets = np.zeros((n, 22), dtype=np.float64)
        z_0s = np.zeros(n, dtype=np.float64)
        
        # 批量随机选择索引
        indices = np.random.randint(0, len(self.dataset), size=n)
        for i, idx in enumerate(indices):
            S = self.dataset[idx]
            
            # 恢复 qpos, ctrl 以及物体的位置和姿态四元数
            self.tmp_data.qpos[:22] = S[:22]
            self.tmp_data.ctrl[:22] = S[:22]
            self.tmp_data.qpos[22:25] = S[22:25]
            self.tmp_data.qpos[25:29] = S[25:29]
            
            # 重置速度为 0
            self.tmp_data.qvel[:] = 0.0
            
            mujoco.mj_forward(self.model, self.tmp_data)
            
            # 序列化为 C++ 状态向量
            mujoco.mj_getState(self.model, self.tmp_data, states[i], int(mujoco.mjtState.mjSTATE_FULLPHYSICS))
            prev_targets[i] = S[:22]
            z_0s[i] = S[24]
            
        return states, prev_targets, z_0s

    def reset(self):
        # 批量生成所有环境的随机初始状态
        initial_states, prev_targets, z_0s = self._generate_perturbed_states(self.num_envs)
        
        # 使用 pool.reset 批量初始化 C++ thread datas
        env_ids = np.arange(self.num_envs, dtype=np.int32)
        reset_states, _ = self.pool.reset(env_ids, initial_states)
        self.states = reset_states.copy()
        
        # 重置虚拟控制目标 prev_targets
        self.prev_targets = prev_targets.copy()
        
        # 保存动态上下限
        self.reset_height_lower = z_0s - 0.02
        self.reset_height_upper = z_0s + 0.02
        
        # 计算批量观测并返回 torch.Tensor
        obs = self._get_obs(self.states)
        return torch.from_numpy(obs).float()

    def _get_obs(self, states):
        """
        根据批量状态 states (shape: (nbatch, nstate)) 构建 60 维观测向量 (shape: (nbatch, obs_dim))
        """
        # states[:, 0] 是时间 (time)
        # states[:, 1 : 1 + nq] 是 qpos
        # states[:, 1 + nq : 1 + nq + nv] 是 qvel
        qpos_batch = states[:, 1 : 1 + self.nq]
        qvel_batch = states[:, 1 + self.nq : 1 + self.nq + self.nv]
        
        # 提取各个特征分量
        hand_qpos = qpos_batch[:, :22]
        object_pos = qpos_batch[:, 22:25]
        object_quat = qpos_batch[:, 25:29]
        
        hand_qvel = qvel_batch[:, :22]
        object_linvel = qvel_batch[:, 22:25]
        object_angvel = qvel_batch[:, 25:28]
        
        # 重复 rot_axis
        rot_axis_batch = np.tile(self.rot_axis, (states.shape[0], 1))
        
        # 拼接观测向量 (60 维)
        obs = np.concatenate([
            hand_qpos, hand_qvel,
            object_pos, object_quat,
            object_linvel, object_angvel,
            rot_axis_batch
        ], axis=1).astype(np.float32)
        
        return obs

    def step(self, actions):
        """
        actions: torch.Tensor 或 np.ndarray，形状为 (num_envs, 22)
        """
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy().astype(np.float64)
        else:
            actions_np = np.asarray(actions, dtype=np.float64)
            
        actions_np = np.clip(actions_np, -1.0, 1.0)
        
        # 1. 计算 PD 增量控制目标
        target_ctrl = self.prev_targets + actions_np / 24.0
        
        # 裁剪 ctrl 范围
        self.prev_targets = np.clip(target_ctrl, self.ctrl_min, self.ctrl_max)
        
        # 2. 构造 control trajectory: shape (nbatch, nstep, nu) = (num_envs, 12, 22)
        control = np.repeat(self.prev_targets[:, None, :], 12, axis=1)
        
        # 3. 物理并行步进
        next_states = self.pool.step(self.states, nstep=12, control=control)
        
        # 4. 计算奖励与 done (terminated/truncated)
        qpos_batch = next_states[:, 1 : 1 + self.nq]
        qvel_batch = next_states[:, 1 + self.nq : 1 + self.nq + self.nv]
        
        hand_qpos = qpos_batch[:, :22]
        hand_qvel = qvel_batch[:, :22]
        object_pos = qpos_batch[:, 22:25]
        object_linvel = qvel_batch[:, 22:25]
        object_angvel = qvel_batch[:, 25:28]
        
        # 计算虚拟扭矩
        torque_virtual = self.kp[None, :] * (self.prev_targets - hand_qpos) - self.kd[None, :] * hand_qvel
        
        # 4.1 rotate (旋转奖励)
        rotate_reward = np.clip(np.sum(object_angvel * self.rot_axis[None, :], axis=1), -0.5, 0.5) * 2.5
        
        # 4.2 obj_linvel (物体移动速度惩罚)
        linvel_penalty = np.sum(np.abs(object_linvel), axis=1) * -0.3
        
        # 4.3 pose_diff (关节姿态偏差惩罚)
        pose_diff_penalty = np.sum(np.square(hand_qpos - self.default_angles[None, :]), axis=1) * -0.4
        
        # 4.4 torque (力矩惩罚)
        torque_penalty = np.sum(np.square(torque_virtual), axis=1) * -0.1
        
        # 4.5 work (物理功惩罚)
        work_penalty = (np.sum(torque_virtual * hand_qvel, axis=1)) ** 2 * -0.5
        
        # 4.6 object_pos (位置锚定奖励)
        dist_to_anchor = np.linalg.norm(object_pos - self.object_pos_anchor[None, :], axis=1)
        object_pos_reward = (1.0 / (dist_to_anchor + 0.001)) * 0.003
        
        total_rewards = rotate_reward + linvel_penalty + pose_diff_penalty + torque_penalty + work_penalty + object_pos_reward
        rewards = total_rewards * 0.05
        
        # 5. 坠落与指尖偏离终止判定 (低于或高于动态高度限制判定为终止)
        dones = (object_pos[:, 2] < self.reset_height_lower) | (object_pos[:, 2] > self.reset_height_upper)
        # 坠落惩罚
        rewards[dones] -= 10.0
        
        # 6. Auto-reset 自动重置已完成/坠落的环境
        reset_indices = np.where(dones)[0]
        if len(reset_indices) > 0:
            reset_init_states, reset_prev_targets, reset_z_0s = self._generate_perturbed_states(len(reset_indices))
            reset_states, _ = self.pool.reset(reset_indices, reset_init_states)
            next_states[reset_indices] = reset_states
            # 对重置的环境重置虚拟控制目标 prev_targets 与动态上下限
            self.prev_targets[reset_indices] = reset_prev_targets
            self.reset_height_lower[reset_indices] = reset_z_0s - 0.02
            self.reset_height_upper[reset_indices] = reset_z_0s + 0.02
            
        # 7. 更新内部状态并计算观测
        self.states = next_states.copy()
        obs = self._get_obs(self.states)
        
        return (
            torch.from_numpy(obs).float(),
            torch.from_numpy(rewards).float(),
            torch.from_numpy(dones).bool()
        )

    def close(self):
        if hasattr(self, "pool") and self.pool is not None:
            self.pool.close()

# 别名兼容，支持已有的训练脚本导入
SharedMemoryVectorEnv = BatchVectorEnv
