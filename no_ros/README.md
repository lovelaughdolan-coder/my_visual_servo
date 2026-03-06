# YOLO + IBVS Standalone (No-ROS)

本项目提供了一个在无 ROS 环境下运行的基于 YOLO 的机械臂视觉伺服 (IBVS) 控制系统。

## 1. 环境准备

建议在专用的虚拟环境中安装依赖：

```bash
pip install opencv-python numpy ultralytics pyrealsense2 scipy
```

## 2. 核心脚本

| 脚本名 | 作用 | 说明 |
| :--- | :--- | :--- |
| `select_target.py` | **目标选择工具** | 预览相机流，手动通过数字键选择目标，并将其坐标(u,v)和深度/面积保存到 `target_config.json` |
| `main_depth_ibvs.py` | **深度反馈 IBVS** | 使用 RealSense D2C 深度信息作为控制反馈（更精确），推荐使用 |
| `main_no_ros_ibvs.py` | **面积反馈 IBVS** | 使用 BBox 面积估计距离（适用于无深度相机或简单场景） |

## 3. 使用流程

### 第一步：采集目标参数
运行选择工具，将机械臂手动移至你认为理想的“标称位”，然后记录该位置目标的像素参数：
```bash
python3 select_target.py
```
- **数字键 0-9**: 切换选中的目标 ID。
- **c**: 确认保存到 `target_config.json`。
- **q**: 退出。

### 第二步：开启视觉伺服
运行控制器，它将自动加载上一步保存的参数作为收敛目标：
```bash
# 推荐版 (基于深度)
python3 main_depth_ibvs.py

# 备份版 (基于面积)
python3 main_no_ros_ibvs.py
```
- **s**: 开启控制（机械臂开始运动）。
- **x**: 紧急停止运动。
- **q**: 退出并断开机器人连接。

## 4. 常见问题配置
在脚本开头的 `用户配置区` 可以修改以下参数：
- `ROBOT_HOST`: Elfin 机械臂 IP。
- `YOLO_MODEL_PATH`: YOLO 模型路径。
- `LAMBDA_XY / LAMBDA_Z`: 控制增益，如果抖动剧烈请调小。
- `DEAD_ZONE_PX`: 死区大小，增加此值可减少到达目标点后的微小抖动。
