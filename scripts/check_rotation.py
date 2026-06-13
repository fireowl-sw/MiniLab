import os
import sys
import numpy as np
import torch
import mujoco

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from minilab.algos.ppo import ActorCritic

def get_yaw(q):
    # q is [w, x, y, z]
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def main():
    device = torch.device("cpu")
    obs_dim = 60
    action_dim = 22
    
    agent = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../weights/ppo_sharpa.pt"))
    if not os.path.exists(model_path):
        print(f"Error: Weights not found at {model_path}")
        return
        
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    print(f"Loaded weights from {model_path}")
    
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/robots/sharpa_wave/scene.xml"))
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/robots/sharpa_wave/sharpa_grasp_linspace_1.npy"))
    dataset = np.load(dataset_path)
    # Use the same seed or first item for reproducible check
    np.random.seed(42)
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
    
    yaws = []
    heights = []
    ang_vels = []
    
    initial_yaw = get_yaw(data.qpos[action_dim+3:action_dim+7])
    prev_yaw = initial_yaw
    total_rotation = 0.0
    
    print("Starting simulation analysis...")
    print(f"Initial Object Pos: {data.qpos[action_dim:action_dim+3]}")
    
    for step in range(300):
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
            action_mean = agent.actor(obs_tensor).squeeze(0).numpy()
            
        action_clipped = np.clip(action_mean, -1.0, 1.0)
        target_ctrl = prev_targets + action_clipped / 24.0
        prev_targets = np.clip(target_ctrl, ctrl_min, ctrl_max)
        data.ctrl[:action_dim] = prev_targets
        
        for _ in range(12):
            mujoco.mj_step(model, data)
            
        # Record yaw rotation
        current_quat = data.qpos[action_dim+3:action_dim+7].copy()
        current_yaw = get_yaw(current_quat)
        
        # Unwrap yaw angle to compute continuous rotation
        diff = current_yaw - prev_yaw
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        total_rotation += diff
        prev_yaw = current_yaw
        
        yaws.append(total_rotation)
        heights.append(data.qpos[action_dim+2])
        ang_vels.append(data.qvel[action_dim+5]) # w_z is index 5 in qvel (0-2: hand joint velocities? No, qvel has size 22 + 6 = 28. Object qvel is 22 to 27. index 22-24: linvel, index 25-27: angvel. w_z is index 27.)
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Height: {data.qpos[action_dim+2]:.4f} | Total Rot (deg): {np.degrees(total_rotation):.2f} | w_z: {data.qvel[action_dim+5]:.4f}")
            
    print("\n--- Simulation Summary ---")
    print(f"Final Height: {heights[-1]:.4f}")
    print(f"Total Rotation (degrees): {np.degrees(total_rotation):.2f}")
    
    # Check if stuck / frozen in last 5 seconds (steps 200-300)
    rotation_last_100 = np.degrees(np.abs(yaws[-1] - yaws[200]))
    avg_angvel_last_100 = np.mean(np.abs(ang_vels[200:]))
    print(f"Rotation in last 5 seconds (steps 200-300): {rotation_last_100:.2f} degrees")
    print(f"Avg w_z in last 5 seconds: {avg_angvel_last_100:.4f}")
    
    # Check if object fell
    has_fallen = any(h < 0.60 for h in heights)
    
    print("\n--- CRITICAL VERIFICATION RESULTS ---")
    if has_fallen:
        print("RESULT: FAILURE - Cylinder FELL / DROOPED into the palm during the evaluation (Height went below 0.60m).")
    elif rotation_last_100 < 5.0 and avg_angvel_last_100 < 0.1:
        print("RESULT: FAILURE - Cylinder is STUCK / FROZEN in the hand towards the end of the evaluation.")
    else:
        print("RESULT: SUCCESS - Cylinder continues to rotate at fingertip height without falling or locking!")

if __name__ == "__main__":
    main()
