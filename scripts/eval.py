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
    
    # 从数据集中加载抓握姿态做初始化，匹配训练的随机化初始化
    dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../assets/robots/sharpa_wave/sharpa_grasp_linspace_1.npy")
    )
    dataset = np.load(dataset_path)
    idx = np.random.randint(0, len(dataset))
    S = dataset[idx]
    
    data.qpos[:action_dim] = S[:action_dim]
    data.ctrl[:action_dim] = S[:action_dim]
    prev_targets = S[:action_dim].copy()
    data.qpos[action_dim:action_dim+3] = S[action_dim:action_dim+3]
    data.qpos[action_dim+3:action_dim+7] = S[action_dim+3:action_dim+7]
    mujoco.mj_forward(model, data)
    
    rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ctrl_min = model.actuator_ctrlrange[:action_dim, 0] * 0.9
    ctrl_max = model.actuator_ctrlrange[:action_dim, 1] * 0.9
    prev_targets = np.clip(prev_targets, ctrl_min, ctrl_max)
    
    if cfg.get("record", False):
        import imageio
        video_path = "eval_run.mp4"
        print(f"[评估进程] 正在启动离屏录像，视频将保存至: {video_path}")
        
        # 创建渲染器
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = []
        
        for step in range(100):
            # 1. 提取当前状态并推理动作
            qpos = data.qpos[:action_dim].copy()
            qvel = data.qvel[:action_dim].copy()
            object_pos = data.qpos[action_dim:action_dim+3].copy()
            object_quat = data.qpos[action_dim+3:action_dim+7].copy()
            object_linvel = data.qvel[action_dim:action_dim+3].copy()
            object_angvel = data.qvel[action_dim+3:action_dim+6].copy()
            
            obs = np.concatenate([
                qpos, qvel,
                object_pos, object_quat,
                object_linvel, object_angvel,
                rot_axis
            ]).astype(np.float32)
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean = agent.actor(obs_tensor).squeeze(0).cpu().numpy()
                
            action_clipped = np.clip(action_mean, -1.0, 1.0)
            target_ctrl = prev_targets + action_clipped / 24.0
            prev_targets = np.clip(target_ctrl, ctrl_min, ctrl_max)
            data.ctrl[:action_dim] = prev_targets
            
            # 2. 物理步进 12 次 (20Hz控制频率)
            for _ in range(12):
                mujoco.mj_step(model, data)
                
            # 3. 渲染画面帧
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
            
        imageio.mimsave(video_path, frames, fps=20)
        print(f"[评估进程] 离屏视频录制完成，已成功保存至: {video_path}")
        renderer.close()
    else:
        # 交互式 GLFW 模式
        step_counter = 0
        
        def controller_cb(model, data):
            nonlocal step_counter, prev_targets
            if step_counter % 12 == 0:
                qpos = data.qpos[:action_dim].copy()
                qvel = data.qvel[:action_dim].copy()
                
                object_pos = data.qpos[action_dim:action_dim+3].copy()
                object_quat = data.qpos[action_dim+3:action_dim+7].copy()
                
                object_linvel = data.qvel[action_dim:action_dim+3].copy()
                object_angvel = data.qvel[action_dim+3:action_dim+6].copy()
                
                obs = np.concatenate([
                    qpos, qvel,
                    object_pos, object_quat,
                    object_linvel, object_angvel,
                    rot_axis
                ]).astype(np.float32)
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action_mean = agent.actor(obs_tensor).squeeze(0).cpu().numpy()
                    
                action_clipped = np.clip(action_mean, -1.0, 1.0)
                target_ctrl = prev_targets + action_clipped / 24.0
                prev_targets = np.clip(target_ctrl, ctrl_min, ctrl_max)
                
            data.ctrl[:action_dim] = prev_targets
            step_counter += 1
            
        mujoco.set_mjcb_control(controller_cb)
        print("[评估进程] 正在启动 3D 可视化仿真窗口 (在 macOS 上以主线程运行)...")
        mujoco.viewer.launch(model, data)
        mujoco.set_mjcb_control(None)
        print("[评估进程] 仿真窗口已关闭，评估结束。")

if __name__ == "__main__":
    eval()
