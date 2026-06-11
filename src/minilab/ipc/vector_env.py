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
        
        # 2. 捏紧圆柱体的默认关节位置 (弧度) 和初始参数
        from minilab.envs.sharpa_env import SOURCE_DEFAULT_HAND_JOINT_POS_DEG
        self.default_angles = np.deg2rad(np.asarray(SOURCE_DEFAULT_HAND_JOINT_POS_DEG, dtype=np.float64))
        self.object_pos_anchor = np.array([-0.09559, -0.00517, 0.61906], dtype=np.float64)
        self.rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        
        # 3. 创建 C++ BatchEnvPool (使用 4 个 C++ 线程进行多线程并行加速物理计算)
        self.pool = batch_env.BatchEnvPool(self.model, nbatch=self.num_envs, nthread=4)
        
        # 4. 初始化单个环境的初始状态向量 (nq + nv) 用于 Auto-reset
        tmp_data = mujoco.MjData(self.model)
        tmp_data.qpos[:22] = self.default_angles
        tmp_data.qpos[22:25] = self.object_pos_anchor
        tmp_data.qpos[25:29] = np.array([1.0, 0.0, 0.0, 0.0]) # 自由物体的 qpos 是 position (3) + quaternion (4)
        mujoco.mj_forward(self.model, tmp_data)
        
        self.nstate = self.pool.nstate
        self.init_state_single = np.zeros(self.nstate, dtype=np.float64)
        mujoco.mj_getState(self.model, tmp_data, self.init_state_single, int(mujoco.mjtState.mjSTATE_FULLPHYSICS))
        
        # 5. 初始化保存环境当前 State 状态的批处理数组
        self.states = np.tile(self.init_state_single, (self.num_envs, 1))
        
        # 初始化上一时刻虚拟控制目标 (对于每个平行环境)
        self.prev_targets = np.tile(self.default_angles, (self.num_envs, 1))

    def reset(self):
        # 批量重置所有环境的状态
        env_ids = np.arange(self.num_envs, dtype=np.int32)
        initial_states = np.tile(self.init_state_single, (self.num_envs, 1))
        
        # 使用 pool.reset 批量初始化 C++ thread datas
        reset_states, _ = self.pool.reset(env_ids, initial_states)
        self.states = reset_states.copy()
        
        # 重置虚拟控制目标 prev_targets
        self.prev_targets = np.tile(self.default_angles, (self.num_envs, 1))
        
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
        
        # 1. 计算 PD 增量控制目标 (从当前的 hand_qpos 计算，qpos 位于 states 的 1 : 1 + nq 范围)
        qpos_batch = self.states[:, 1 : 1 + self.nq]
        current_qpos = qpos_batch[:, :22]
        target_ctrl = self.prev_targets + actions_np / 24.0
        
        # 裁剪 ctrl 范围
        ctrl_min = self.model.actuator_ctrlrange[:22, 0]
        ctrl_max = self.model.actuator_ctrlrange[:22, 1]
        self.prev_targets = np.clip(target_ctrl, ctrl_min, ctrl_max)
        
        # 2. 构造 control trajectory: shape (nbatch, nstep, nu) = (num_envs, 12, 22)
        control = np.repeat(self.prev_targets[:, None, :], 12, axis=1)
        
        # 3. 物理并行步进
        next_states = self.pool.step(self.states, nstep=12, control=control)
        
        # 4. 计算奖励与 done (terminated/truncated)
        qpos_batch = next_states[:, 1 : 1 + self.nq]
        qvel_batch = next_states[:, 1 + self.nq : 1 + self.nq + self.nv]
        
        hand_qpos = qpos_batch[:, :22]
        object_pos = qpos_batch[:, 22:25]
        object_linvel = qvel_batch[:, 22:25]
        object_angvel = qvel_batch[:, 25:28]
        
        # 4.1 对称绝对值旋转速度奖励 (权重 5.0，clip 0~1.0)
        rotate_reward = 5.0 * np.clip(np.abs(object_angvel[:, 2]), 0.0, 1.0)
        
        # 4.2 物体偏离中心锚点的距离惩罚 (常数梯度，消除自锁)
        dist_to_anchor = np.linalg.norm(object_pos - self.object_pos_anchor, axis=1)
        dist_penalty = -10.0 * dist_to_anchor
        
        # 4.3 关节姿态偏差惩罚 (权重 -0.4)
        pose_diff_penalty = -0.4 * np.sum(np.square(hand_qpos - self.default_angles), axis=1)
        
        # 4.4 物体移动速度惩罚 (权重 -0.3)
        linvel_penalty = -0.3 * np.sum(np.abs(object_linvel), axis=1)
        
        # 4.5 动作惩罚
        action_penalty = -0.01 * np.sum(np.square(actions_np), axis=1)
        
        rewards = rotate_reward + dist_penalty + pose_diff_penalty + linvel_penalty + action_penalty
        
        # 5. 坠落终止判定 (低于锚点高度 10 厘米判定为坠落)
        dones = object_pos[:, 2] < (self.object_pos_anchor[2] - 0.1)
        # 坠落惩罚
        rewards[dones] -= 10.0
        
        # 6. Auto-reset 自动重置已完成/坠落的环境
        reset_indices = np.where(dones)[0]
        if len(reset_indices) > 0:
            reset_init_states = np.tile(self.init_state_single, (len(reset_indices), 1))
            reset_states, _ = self.pool.reset(reset_indices, reset_init_states)
            next_states[reset_indices] = reset_states
            # 对重置的环境重置虚拟控制目标 prev_targets
            self.prev_targets[reset_indices] = self.default_angles.copy()
            
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
