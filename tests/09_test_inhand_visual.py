import os
import sys
import numpy as np
import mujoco
import mujoco.viewer

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from minilab.envs.sharpa_env import SharpaHandGymEnv

def main():
    print("Initializing 3D Interactive Viewer for In-Hand Manipulation...")
    # 1. 初始化我们升级后的手内操作环境
    env = SharpaHandGymEnv()
    env.reset()
    
    # 2. 定义控制回调函数 (施加微小的随机扰动动作，观察物体的物理响应)
    def controller_cb(model, data):
        # 产生微小的随机抖动动作，防止关节瞬间散开
        action = np.random.uniform(-0.1, 0.1, size=(22,))
        current_qpos = data.qpos[:22]
        target_ctrl = current_qpos + action / 24.0
        
        # 限制在物理范围之内
        ctrl_limits_low = model.actuator_ctrlrange[:22, 0]
        ctrl_limits_high = model.actuator_ctrlrange[:22, 1]
        data.ctrl[:22] = np.clip(target_ctrl, ctrl_limits_low, ctrl_limits_high)
        
    # 3. 注册控制回调函数并启动图形窗口
    mujoco.set_mjcb_control(controller_cb)
    
    print("Starting 3D visualizer. You should see the Sharpa hand holding a red cylinder.")
    print("Close the window to exit.")
    
    # 启动 macOS 兼容的 3D 管理式窗口
    mujoco.viewer.launch(env.model, env.data)
    
    # 4. 清理回调
    mujoco.set_mjcb_control(None)
    print("Visualizer closed.")

if __name__ == "__main__":
    main()
