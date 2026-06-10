import os
import numpy as np
import gymnasium as gym
import mujoco

class SharpaHandGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # 精确定位物理资产相对于当前文件的位置 (从 src/minilab/envs/ 回退三级到项目根目录)
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../assets/robots/sharpa_wave/scene.xml")
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_joints = 22
        
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_joints * 2,), dtype=np.float32
        )
        self.init_qpos = np.zeros(self.num_joints, dtype=np.float32)

    def _get_obs(self):
        qpos = self.data.qpos[:self.num_joints].copy()
        qvel = self.data.qvel[:self.num_joints].copy()
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.init_qpos = self.data.qpos[:self.num_joints].copy()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        current_qpos = self.data.qpos[:self.num_joints]
        target_ctrl = current_qpos + action / 24.0
        self.data.ctrl[:self.num_joints] = target_ctrl
        
        mujoco.mj_step(self.model, self.data)
        observation = self._get_obs()
        
        diff = self.data.qpos[:self.num_joints] - self.init_qpos
        reward = float(-np.sum(np.square(diff)))
        
        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, {}
