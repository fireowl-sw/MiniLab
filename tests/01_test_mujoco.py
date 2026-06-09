import os
import mujoco

# 使用绝对路径定位机器人 XML 模型文件
# scene.xml 包含了右手机器人模型 (right_sharpa_wave.xml) 并定义了基础仿真环境（如天空盒、地面等）
xml_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../assets/robots/sharpa_wave/scene.xml")
)

print(f"Loading XML from: {xml_path}")

# 加载模型静态数据 (MjModel)
# MjModel 包含了仿真场景中的各种静态物理参数（例如几何形状、质量、关节类型、致动器限制等），这些参数在仿真运行期间通常是不会发生改变的。
model = mujoco.MjModel.from_xml_path(xml_path)

# 创建物理状态动态数据 (MjData)
# MjData 用于存储仿真的所有动态变量和中间计算结果（例如关节的位置、速度、受力情况以及接触点等），随着仿真时间的推进而实时更新。
data = mujoco.MjData(model)

# 打印提取的关键控制维度
print("----------------------------------------")
print("MuJoCo Model Loaded Successfully!")
print(f"Number of joints (关节数)   [model.njnt]: {model.njnt}")
print(f"Degrees of freedom (自由度) [model.nv]:   {model.nv}")
print(f"Number of actuators (致动器) [model.nu]:   {model.nu}")
print("----------------------------------------")
