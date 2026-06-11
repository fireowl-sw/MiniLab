import os
import sys
import torch
import numpy as np
import mujoco
import mujoco.viewer
import hydra
from omegaconf import DictConfig

# 将 src 目录添加到模块查找路径中
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from minilab.algos.ppo import ActorCritic

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def eval(cfg: DictConfig):
    # 1. 确定计算设备
    device = torch.device(cfg.device)
    print(f"[评估进程] 使用设备: {device}")
    
    # 2. 构造策略网络并加载权重
    obs_dim = cfg.env.obs_dim
    action_dim = cfg.env.action_dim
    agent = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    
    model_path = os.path.join(cfg.weights_dir, "ppo_sharpa.pt")
    if not os.path.exists(model_path):
        print(f"[错误] 找不到权重文件: {model_path}，请先运行 scripts/train.py 完成训练。")
        return
        
    # 加载权重
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    print(f"[评估进程] 成功载入策略权重: {model_path}")
    
    # 3. 加载 MuJoCo 物理场景文件
    xml_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/robots/sharpa_wave/scene.xml")
    )
    print(f"[评估进程] 正在加载物理场景: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 初始化手部姿势与物体位置，匹配训练重置状态
    from minilab.envs.sharpa_env import SOURCE_DEFAULT_HAND_JOINT_POS_DEG
    default_angles = np.deg2rad(np.asarray(SOURCE_DEFAULT_HAND_JOINT_POS_DEG, dtype=np.float32))
    object_pos_anchor = np.array([-0.09559, -0.00517, 0.61906], dtype=np.float32)
    rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    data.qpos[:action_dim] = default_angles
    data.ctrl[:action_dim] = default_angles
    data.qpos[action_dim:action_dim+3] = object_pos_anchor
    data.qpos[action_dim+3:action_dim+7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    
    # 4. 编写物理控制回调函数 (注入训练好的策略)
    def controller_cb(model, data):
        # 提取当前状态的关节角度位置与速度
        qpos = data.qpos[:action_dim].copy()
        qvel = data.qvel[:action_dim].copy()
        
        # 提取物体坐标与旋转四元数
        object_pos = data.qpos[action_dim:action_dim+3].copy()
        object_quat = data.qpos[action_dim+3:action_dim+7].copy()
        
        # 提取物体的线速度与角速度
        object_linvel = data.qvel[action_dim:action_dim+3].copy()
        object_angvel = data.qvel[action_dim+3:action_dim+6].copy()
        
        # 拼接成 60 维观测向量
        obs = np.concatenate([
            qpos, qvel,
            object_pos, object_quat,
            object_linvel, object_angvel,
            rot_axis
        ]).astype(np.float32)
        
        # 转换为 PyTorch Tensor 输入给策略网络
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 获取确定性动作均值 (Evaluation 时不进行高斯随机采样)
            action_mean = agent.actor(obs_tensor).squeeze(0).cpu().numpy()
            
        # 动作幅度限制在 [-1.0, 1.0] 内，计算物理增量位置控制
        action_clipped = np.clip(action_mean, -1.0, 1.0)
        current_qpos = data.qpos[:action_dim]
        target_ctrl = current_qpos + action_clipped / 24.0
        data.ctrl[:action_dim] = target_ctrl
        
    # 5. 注册控制回调函数并启动可视化渲染窗口
    mujoco.set_mjcb_control(controller_cb)
    print("[评估进程] 正在启动 3D 可视化仿真窗口 (在 macOS 上以主线程运行)...")
    
    # 这一步会挂起当前线程，启动渲染窗口并计算物理步进，直到关闭窗口
    mujoco.viewer.launch(model, data)
    
    # 清理回调函数
    mujoco.set_mjcb_control(None)
    print("[评估进程] 仿真窗口已关闭，评估结束。")

if __name__ == "__main__":
    eval()
