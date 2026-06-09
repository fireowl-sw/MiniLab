import os
import numpy as np
import mujoco
import mujoco.viewer

# 使用绝对路径定位机器人 XML 模型文件
xml_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../assets/robots/sharpa_wave/scene.xml")
)

print(f"Loading XML from: {xml_path}")

# 加载模型静态数据 (MjModel) 与动态数据 (MjData)
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 1. 定义控制回调函数 (Controller Callback)
# 每次 MuJoCo 在物理步进中计算动力学时，都会自动调用这个函数来获取控制输入。
def controller_cb(model, data):
    # 计算正弦控制信号 (幅值为 0.5，角频率为 2.0 rad/s)
    # 利用 data.time (仿真时间) 计算正弦值，以确保控制信号与仿真时钟保持同步
    sin_val = np.sin(data.time * 2.0) * 0.5
    
    # 将控制信号赋给所有致动器 (Actuators)
    # data.ctrl 是一维数组，大小为 model.nu (22)。使用 [:] 将标量值广播赋予所有 22 维致动器接口。
    data.ctrl[:] = sin_val

# 2. 注册控制回调函数到 MuJoCo 物理引擎中
mujoco.set_mjcb_control(controller_cb)

print("Starting managed viewer (runs on main thread, works natively on macOS)...")
# 3. 启动管理式可视化窗口
# mujoco.viewer.launch 会在主进程的主线程上直接启动图形窗口，并接管主循环。
# 此时它会自动处理物理步进(mj_step)、同步渲染(sync)、帧率限制(time.sleep)以及在步进前调用我们注册的 controller_cb。
# 这样能够完美绕过 macOS 底层对多线程 GUI 渲染的限制，不需要使用 mjpython。
mujoco.viewer.launch(model, data)

# 4. 退出后清除控制回调函数，避免对后续仿真任务产生残留影响
mujoco.set_mjcb_control(None)
print("Viewer closed.")
