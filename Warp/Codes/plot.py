import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 第一组控制点的坐标
control_points_1 = np.array([
    [-7.8444, -63.0149],
    [-10.3686, -68.6130],
    [-13.8925, -68.5081],
    [-9.0425, -76.4874],
    [-9.1979, -73.9129],
    [-10.4356, -72.0595],
    [-8.0110, -70.8319],
    [-7.0098, -78.6004],
    [-10.6902, -68.8064]
])

# 第二组控制点的坐标
control_points_2 = np.array([
    [-7.7128, -60.9169],
    [-8.2973, -62.1008],
    [-9.2867, -63.2453],
    [-9.6091, -65.3365],
    [-9.6417, -67.3499],
    [-9.6821, -68.5187],
    [-9.5204, -69.5098],
    [-9.2608, -71.7582],
    [-9.4467, -71.8809]
])

# 时间序列（假设两组数据有相同的时间序列）
time_sequence = np.arange(1, len(control_points_1) + 1)

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 第一组数据: x->时间, y->原始y, z->原始x
x1 = time_sequence
y1 = control_points_1[:, 1]
z1 = control_points_1[:, 0]

# 第二组数据: x->时间, y->原始y, z->原始x
x2 = time_sequence
y2 = control_points_2[:, 1]
z2 = control_points_2[:, 0]

# 绘制第一组控制点的运动轨迹 (使用蓝色，标记点缩小)
ax.plot(x1, y1, z1, marker='o', color='b', label="Original Trajectory", markersize=4)

# 绘制第二组控制点的运动轨迹 (使用红色，标记点缩小)
ax.plot(x2, y2, z2, marker='^', color='r', label="Smoothed Trajectory", markersize=4)

# 标题和标签
ax.set_xlabel('Time')
ax.set_ylabel('Y')
ax.set_zlabel('X')

# 显示图例
ax.legend()

# 保存到本地指定路径
output_path = '/home/sunleyao/sly/UDIS2-main/UDIS2++-experiment-Trajectory_visualization/Warp/control_points_motion_two_sets.png'  # 指定服务器上的路径
plt.savefig(output_path, dpi=300)

print(f"图像已保存至 {output_path}")
