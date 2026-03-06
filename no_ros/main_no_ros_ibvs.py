#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 Python 无 ROS 版本的基于 YOLO 的视觉伺服 (IBVS)

核心依赖:
  pip install opencv-python numpy ultralytics pyrealsense2 scipy

功能:
  1. 使用 RealSense (pyrealsense2) 读取 RGB 流并自动获取内参
  2. 运行 YOLO 检测/跟踪，OpenCV 窗口实时显示
  3. 按 's' 开启视觉伺服 → 按 'x' 暂停 → 按 'q' 退出

坐标变换链 (与 ROS TF 等价):
  R_base_cam = R_base_tcp(动态, 从机械臂实时读取) × R_tcp_cam(静态, 手眼标定)
"""

import cv2
import numpy as np
import time
import socket
import json
import os
from scipy.spatial.transform import Rotation
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    print("警告: 未检测到 pyrealsense2，将使用标准 OpenCV 摄像头")
    HAS_REALSENSE = False


# ==========================================
# 用户配置区 (发给同事后，重点修改这里)
# ==========================================

# 1. 机器人配置
ROBOT_HOST = '192.168.1.11'
ROBOT_PORT = 10003

# 2. YOLO 模型配置
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', '2-26merged.pt')
TARGET_CLASS_ID = 0

# 3. 视觉伺服 (IBVS) 参数
TARGET_U = 640.0              # 画面中心 u (1280x720 → 640)
TARGET_V = 360.0              # 画面中心 v (1280x720 → 360)
TARGET_AREA = 10000.0         # 期望包围框面积
LAMBDA_XY = 0.5               # XY 轴增益
LAMBDA_Z = 0.12               # Z 轴增益
MAX_LINEAR_VEL = 80.0         # mm/s 限幅
DEFAULT_DEPTH = 0.4           # 假定深度 m
DEAD_ZONE_PX = 5.0            # 像素死区
DEAD_ZONE_AREA_RATIO = 0.15   # 面积死区

# 4. 相机内参 (无 RealSense 时的回退值)
FX_DEFAULT = 607.0
FY_DEFAULT = 606.0

# 5. 静态外参: R_tcp_cam_optical
#    = R(elfin_tcp_link → camera_link) × R(camera_link → camera_color_optical_frame)
#    来源: launch 文件中的手眼标定四元数 + RealSense 标准光学帧旋转
R_TCP_CAM_OPTICAL = np.array([
    [-0.69781061,  0.71615162, -0.01368220],
    [-0.71628053, -0.69763898,  0.01555798],
    [ 0.00159663,  0.02065682,  0.99978535]
])

# ==========================================

# 自动加载 select_target.py 保存的配置
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'target_config.json')
if os.path.exists(_CONFIG_FILE):
    with open(_CONFIG_FILE) as _f:
        _cfg = json.load(_f)
    TARGET_U = _cfg.get('target_u', TARGET_U)
    TARGET_V = _cfg.get('target_v', TARGET_V)
    if _cfg.get('target_area') is not None:
        TARGET_AREA = _cfg['target_area']
    print(f"📌 从 target_config.json 加载目标: u={TARGET_U}, v={TARGET_V}, area={TARGET_AREA}")


class RobotController:
    """机械臂 Socket 通信封装"""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.is_connected = False

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.host, self.port))
            self.is_connected = True
            print(f"✅ 连接机器人成功 ({self.host}:{self.port})")
            self.send_cmd('GrpReset,0,;')
            time.sleep(0.5)
            self.send_cmd('GrpEnable,0,;')
            time.sleep(1.0)
            return True
        except Exception as e:
            print(f"❌ 连接机器人失败: {e}")
            self.is_connected = False
            return False

    def send_cmd(self, cmd):
        if not self.is_connected:
            return ""
        try:
            self.sock.sendall(cmd.encode('utf-8'))
            return self.sock.recv(4096).decode('utf-8').strip()
        except:
            return ""

    def send_speed_l(self, v_mm, run_time=0.3):
        cmd = (f'SpeedL,0,{v_mm[0]:.2f},{v_mm[1]:.2f},{v_mm[2]:.2f},'
               f'0.00,0.00,0.00,300,80,{run_time},;')
        self.send_cmd(cmd)

    def read_tcp_rotation(self):
        """读取当前 TCP 姿态，返回 R_base_tcp (3x3 旋转矩阵)"""
        resp = self.send_cmd('ReadActPos,0,;')
        parts = resp.split(',')
        if len(parts) >= 14 and parts[1] == 'OK':
            # parts[11..13] = rx, ry, rz (度, Euler ZYX)
            rx = float(parts[11])
            ry = float(parts[12])
            rz = float(parts[13])
            R_base_tcp = Rotation.from_euler(
                'ZYX', [rz, ry, rx], degrees=True
            ).as_matrix()
            return R_base_tcp
        return None

    def get_R_base_cam(self):
        """动态计算 R_base_cam = R_base_tcp × R_tcp_cam_optical"""
        R_base_tcp = self.read_tcp_rotation()
        if R_base_tcp is None:
            return None
        return R_base_tcp @ R_TCP_CAM_OPTICAL

    def stop(self):
        if self.is_connected:
            self.send_cmd('GrpStop,0,;')
            self.sock.close()
            self.is_connected = False


def main():
    print("=======================================")
    print(" YOLO + IBVS 纯 Python 脱机测试程序")
    print("=======================================")

    # --- 初始化机器人 ---
    robot = RobotController(ROBOT_HOST, ROBOT_PORT)
    robot.connect()

    # 验证动态外参
    R_test = robot.get_R_base_cam()
    if R_test is not None:
        print(f"✅ 动态外参 R_base_cam 计算成功")
    else:
        print("⚠️ 无法读取机器人姿态，外参将不可用")

    # --- 初始化 YOLO ---
    print(f"Loading YOLO model {YOLO_MODEL_PATH}...")
    model = YOLO(YOLO_MODEL_PATH)
    print("Pre-warming model with FP16 (half=True)...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, imgsz=640, half=True, verbose=False)
    print("YOLO loaded.")

    # --- 初始化相机 ---
    if HAS_REALSENSE:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        intrinsics = (profile.get_stream(rs.stream.color)
                      .as_video_stream_profile().get_intrinsics())
        fx, fy = intrinsics.fx, intrinsics.fy
        print(f"RealSense 内参: fx={fx:.2f}, fy={fy:.2f}")
    else:
        cap = cv2.VideoCapture(0)
        fx, fy = FX_DEFAULT, FY_DEFAULT

    ibvs_active = False
    ema_alpha = 0.3
    filtered_u = filtered_v = filtered_area = None

    try:
        while True:
            t0 = time.time()

            # --- 1. 取图 ---
            if HAS_REALSENSE:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            # --- 2. YOLO 推理 ---
            results = model.track(
                frame, persist=True, tracker='bytetrack.yaml',
                conf=0.25, iou=0.45, half=True, imgsz=640,
                classes=[TARGET_CLASS_ID], verbose=False,
            )
            annotated = results[0].plot()

            # --- 3. 提取目标 ---
            boxes = results[0].boxes
            target_found = False
            raw_u = raw_v = raw_area = 0.0

            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()

                # 筛选目标类别，取 ID 最小的
                candidates = [(ids[i], i) for i in range(len(ids))
                              if cls_ids[i] == TARGET_CLASS_ID]
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    _, idx = candidates[0]
                    x1, y1, x2, y2 = xyxy[idx]
                    raw_u = float((x1 + x2) / 2)
                    raw_v = float((y1 + y2) / 2)
                    raw_area = float((x2 - x1) * (y2 - y1))
                    target_found = True
                    cv2.drawMarker(annotated, (int(raw_u), int(raw_v)),
                                   (0, 255, 0), cv2.MARKER_CROSS, 30, 2)

            # --- 4. IBVS 控制 ---
            if target_found:
                if filtered_u is None:
                    filtered_u, filtered_v, filtered_area = raw_u, raw_v, raw_area
                else:
                    filtered_u = ema_alpha * raw_u + (1 - ema_alpha) * filtered_u
                    filtered_v = ema_alpha * raw_v + (1 - ema_alpha) * filtered_v
                    filtered_area = ema_alpha * raw_area + (1 - ema_alpha) * filtered_area

                if ibvs_active:
                    err_u = filtered_u - TARGET_U
                    err_v = filtered_v - TARGET_V
                    err_area = filtered_area - TARGET_AREA
                    center_err = np.sqrt(err_u**2 + err_v**2)
                    area_ratio = abs(err_area) / TARGET_AREA

                    # 死区
                    if center_err < DEAD_ZONE_PX and area_ratio < DEAD_ZONE_AREA_RATIO:
                        print(f"✅ 到位! Err=[{err_u:.0f},{err_v:.0f}]px Area={filtered_area:.0f}")
                        robot.send_speed_l([0, 0, 0], run_time=0.3)
                    else:
                        Z = DEFAULT_DEPTH

                        # 比例控制律 (符号与 ROS 版 ibvs_yolo_controller 一致)
                        vx_cam = LAMBDA_XY * (err_u / fx) * Z
                        vy_cam = LAMBDA_XY * (err_v / fy) * Z

                        z_gain = LAMBDA_Z
                        if area_ratio < 0.3:
                            z_gain *= (area_ratio / 0.3)
                        vz_cam = -z_gain * (err_area / TARGET_AREA) * Z

                        v_cam = np.array([vx_cam, vy_cam, vz_cam])

                        # 动态坐标变换: 相机系 → 基座系
                        R_base_cam = robot.get_R_base_cam()
                        if R_base_cam is not None:
                            v_base_mm = (R_base_cam @ v_cam) * 1000.0

                            # 限幅
                            norm = np.linalg.norm(v_base_mm)
                            if norm > MAX_LINEAR_VEL:
                                v_base_mm = v_base_mm / norm * MAX_LINEAR_VEL

                            print(f"🏃 Err=[{err_u:.0f},{err_v:.0f}] "
                                  f"Area={filtered_area:.0f} "
                                  f"Vel=[{v_base_mm[0]:.0f},{v_base_mm[1]:.0f},"
                                  f"{v_base_mm[2]:.0f}]mm/s")
                            robot.send_speed_l(v_base_mm, run_time=0.15)
                        else:
                            print("⚠️ 无法获取机器人姿态")
                            robot.send_speed_l([0, 0, 0])
            else:
                filtered_u = None
                if ibvs_active:
                    robot.send_speed_l([0, 0, 0], run_time=0.2)
                    print("⚠️ 未检测到目标")

            # --- 5. 可视化 ---
            dt = max(time.time() - t0, 1e-6)
            status = "IBVS ACTIVE" if ibvs_active else "Standby [s]=start"
            color = (0, 0, 255) if ibvs_active else (255, 255, 0)
            cv2.putText(annotated, f"FPS:{1/dt:.0f} | {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # 画目标十字
            cv2.drawMarker(annotated, (int(TARGET_U), int(TARGET_V)),
                           (0, 255, 0), cv2.MARKER_TILTED_CROSS, 30, 1)
            cv2.imshow("No-ROS IBVS", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not ibvs_active:
                print("➡️ 开启 IBVS!")
                ibvs_active = True
            elif key == ord('x') and ibvs_active:
                print("🛑 暂停 IBVS!")
                ibvs_active = False
                robot.send_speed_l([0, 0, 0])

    finally:
        robot.stop()
        if HAS_REALSENSE:
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
