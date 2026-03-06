#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 Python IBVS + 深度点云距离控制 (无 ROS)

与 main_no_ros_ibvs.py 的区别:
  - RealSense 开启 Depth-to-Color 对齐 (D2C)
  - Z 轴控制不再用"包围框面积"估计，而是直接读取 YOLO mask/bbox 区域内
    的对齐深度图，取中位数作为真实距离 (米)
  - Z 轴控制律变为：vz = λ_z * (Z_current - Z_target)
    太远则前进，太近则后退

核心依赖:
  pip install opencv-python numpy ultralytics pyrealsense2 scipy
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
    print("❌ 此版本必须使用 pyrealsense2 以获取深度数据!")
    HAS_REALSENSE = False


# ==========================================
# 用户配置区
# ==========================================

# 1. 机器人
ROBOT_HOST = '192.168.1.11'
ROBOT_PORT = 10003

# 2. YOLO
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', '2-26merged.pt')
TARGET_CLASS_ID = 0

# 3. IBVS 参数
TARGET_U = 640.0              # 画面中心 u (1280x720 → 640)
TARGET_V = 360.0              # 画面中心 v (1280x720 → 360)
TARGET_DEPTH = 0.40           # 期望距离 (米), 机械臂末端到目标的距离
LAMBDA_XY = 0.3               # XY 轴增益 (降低以减少超调)
LAMBDA_Z = 0.3                # Z 轴增益 (降低以减少超调)
MAX_LINEAR_VEL = 80.0         # mm/s
DEAD_ZONE_PX = 2.0            # 像素死区 (缩小以提升精度)
DEAD_ZONE_DEPTH = 0.01       # 深度死区 (米), 10mm 以内视为到位

# 4. 深度滤波
DEPTH_MIN = 0.10              # 有效深度下限 (米)
DEPTH_MAX = 1.50              # 有效深度上限 (米)
DEPTH_EMA_ALPHA = 0.3         # 深度 EMA 滤波系数

# 5. 相机内参 (无 RealSense 时回退)
FX_DEFAULT = 607.0
FY_DEFAULT = 606.0

# 6. 静态外参 R_tcp_cam_optical
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
    if _cfg.get('target_depth') is not None:
        TARGET_DEPTH = _cfg['target_depth']
    print(f"📌 从 target_config.json 加载目标: u={TARGET_U}, v={TARGET_V}, depth={TARGET_DEPTH}")


class RobotController:
    """机械臂 Socket 通信"""

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
        resp = self.send_cmd('ReadActPos,0,;')
        parts = resp.split(',')
        if len(parts) >= 14 and parts[1] == 'OK':
            rx, ry, rz = float(parts[11]), float(parts[12]), float(parts[13])
            return Rotation.from_euler('ZYX', [rz, ry, rx], degrees=True).as_matrix()
        return None

    def get_R_base_cam(self):
        R_base_tcp = self.read_tcp_rotation()
        if R_base_tcp is None:
            return None
        return R_base_tcp @ R_TCP_CAM_OPTICAL

    def stop(self):
        if self.is_connected:
            self.send_cmd('GrpStop,0,;')
            self.sock.close()
            self.is_connected = False


def extract_roi_depth(depth_image, mask=None, bbox=None):
    """从对齐后的深度图中提取 ROI 区域的中位数深度 (米)

    Args:
        depth_image: (H, W) uint16 深度图, 单位毫米
        mask: (H, W) bool 掩码 (来自 YOLO 分割), 优先使用
        bbox: (x1, y1, x2, y2) 包围框, mask 不可用时使用

    Returns:
        depth_m: 中位数深度 (米), 如果无效则返回 None
    """
    if mask is not None:
        roi_depths = depth_image[mask]
    elif bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_image.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi_depths = depth_image[y1:y2, x1:x2].flatten()
    else:
        return None

    # 过滤无效值 (0 表示没有深度)
    valid = roi_depths[roi_depths > 0]

    # 转米并过滤范围
    valid_m = valid.astype(np.float32) / 1000.0
    valid_m = valid_m[(valid_m > DEPTH_MIN) & (valid_m < DEPTH_MAX)]

    if len(valid_m) < 10:
        return None

    return float(np.median(valid_m))


def main():
    if not HAS_REALSENSE:
        print("此版本需要 pyrealsense2, 请安装后重试")
        return

    print("=======================================")
    print(" YOLO + IBVS + 深度控制 (纯 Python)")
    print("=======================================")

    # --- 机器人 ---
    robot = RobotController(ROBOT_HOST, ROBOT_PORT)
    robot.connect()

    R_test = robot.get_R_base_cam()
    if R_test is not None:
        print("✅ 动态外参计算成功")

    # --- YOLO ---
    print(f"Loading YOLO: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    print("Pre-warming (FP16)...")
    model(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, half=True, verbose=False)
    print("YOLO ready.")

    # --- RealSense (D2C 对齐) ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)

    # 创建对齐器: depth → color
    align = rs.align(rs.stream.color)

    # 深度时域滤波 (减少帧间深度抖动)
    temporal_filter = rs.temporal_filter()

    intrinsics = (profile.get_stream(rs.stream.color)
                  .as_video_stream_profile().get_intrinsics())
    fx, fy = intrinsics.fx, intrinsics.fy
    print(f"RealSense 内参: fx={fx:.2f}, fy={fy:.2f}")
    print(f"分辨率: 1280x720, 深度时域滤波已开启")
    print(f"深度对齐 (D2C) 已开启")

    ibvs_active = False
    ema_alpha = 0.3
    filtered_u = filtered_v = filtered_depth = None

    try:
        while True:
            t0 = time.time()

            # --- 1. 取图 (对齐) ---
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 深度时域滤波
            depth_frame = temporal_filter.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # uint16, mm

            # --- 2. YOLO 推理 ---
            results = model.track(
                color_image, persist=True, tracker='bytetrack.yaml',
                conf=0.25, iou=0.45, half=True, imgsz=640,
                classes=[TARGET_CLASS_ID], verbose=False,
            )
            annotated = results[0].plot()

            # --- 3. 提取目标 + 深度 ---
            boxes = results[0].boxes
            masks = results[0].masks  # 分割模型才有
            target_found = False
            raw_u = raw_v = 0.0
            raw_depth = None

            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()

                candidates = [(ids[i], i) for i in range(len(ids))
                              if cls_ids[i] == TARGET_CLASS_ID]
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    _, idx = candidates[0]
                    x1, y1, x2, y2 = xyxy[idx]
                    raw_u = float((x1 + x2) / 2)
                    raw_v = float((y1 + y2) / 2)
                    target_found = True

                    # 提取深度: 优先用 mask, 否则用 bbox
                    mask_2d = None
                    if masks is not None and idx < len(masks):
                        mask_data = masks[idx].data.cpu().numpy()[0]  # (H', W')
                        # mask 可能和原图尺寸不同, resize
                        if mask_data.shape != depth_image.shape:
                            mask_data = cv2.resize(
                                mask_data, (depth_image.shape[1], depth_image.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        mask_2d = mask_data > 0.5

                    raw_depth = extract_roi_depth(
                        depth_image,
                        mask=mask_2d,
                        bbox=(x1, y1, x2, y2)
                    )

                    # 画标记
                    cv2.drawMarker(annotated, (int(raw_u), int(raw_v)),
                                   (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                    if raw_depth is not None:
                        cv2.putText(annotated, f"Z={raw_depth:.3f}m",
                                    (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- 4. IBVS 控制 ---
            if target_found:
                if filtered_u is None:
                    filtered_u, filtered_v = raw_u, raw_v
                    filtered_depth = raw_depth
                else:
                    filtered_u = ema_alpha * raw_u + (1 - ema_alpha) * filtered_u
                    filtered_v = ema_alpha * raw_v + (1 - ema_alpha) * filtered_v
                    if raw_depth is not None:
                        if filtered_depth is not None:
                            filtered_depth = (DEPTH_EMA_ALPHA * raw_depth +
                                              (1 - DEPTH_EMA_ALPHA) * filtered_depth)
                        else:
                            filtered_depth = raw_depth

                if ibvs_active:
                    err_u = filtered_u - TARGET_U
                    err_v = filtered_v - TARGET_V
                    center_err = np.sqrt(err_u**2 + err_v**2)

                    # Z 误差
                    if filtered_depth is not None:
                        err_depth = filtered_depth - TARGET_DEPTH
                        depth_ok = abs(err_depth) < DEAD_ZONE_DEPTH
                    else:
                        err_depth = 0.0
                        depth_ok = True  # 没有深度就不控制 Z

                    # 死区
                    if center_err < DEAD_ZONE_PX and depth_ok:
                        z_str = f"Z={filtered_depth:.3f}m" if filtered_depth else "Z=N/A"
                        print(f"✅ 到位! Err=[{err_u:.0f},{err_v:.0f}]px {z_str}")
                        robot.send_speed_l([0, 0, 0], run_time=0.3)
                    else:
                        # 用实测深度 (如果有), 否则回退到默认
                        Z = filtered_depth if filtered_depth else 0.4

                        # XY: 比例控制律
                        vx_cam = LAMBDA_XY * (err_u / fx) * Z
                        vy_cam = LAMBDA_XY * (err_v / fy) * Z

                        # Z: 直接用深度误差控制
                        # err_depth > 0 (太远) → 前进 (vz > 0)
                        # err_depth < 0 (太近) → 后退 (vz < 0)
                        vz_cam = LAMBDA_Z * err_depth

                        v_cam = np.array([vx_cam, vy_cam, vz_cam])

                        # 动态坐标变换
                        R_base_cam = robot.get_R_base_cam()
                        if R_base_cam is not None:
                            v_base_mm = (R_base_cam @ v_cam) * 1000.0
                            norm = np.linalg.norm(v_base_mm)
                            if norm > MAX_LINEAR_VEL:
                                v_base_mm = v_base_mm / norm * MAX_LINEAR_VEL

                            z_str = f"Z={filtered_depth:.3f}" if filtered_depth else "N/A"
                            print(f"🏃 Err=[{err_u:.0f},{err_v:.0f}] "
                                  f"{z_str}(tgt:{TARGET_DEPTH:.2f}) "
                                  f"Vel=[{v_base_mm[0]:.0f},{v_base_mm[1]:.0f},"
                                  f"{v_base_mm[2]:.0f}]mm/s")
                            robot.send_speed_l(v_base_mm, run_time=0.15)
                        else:
                            robot.send_speed_l([0, 0, 0])
            else:
                filtered_u = filtered_v = filtered_depth = None
                if ibvs_active:
                    robot.send_speed_l([0, 0, 0], run_time=0.2)
                    print("⚠️ 未检测到目标")

            # --- 5. 可视化 ---
            dt = max(time.time() - t0, 1e-6)
            status = "IBVS+Depth ACTIVE" if ibvs_active else "Standby [s]=start"
            color = (0, 0, 255) if ibvs_active else (255, 255, 0)
            cv2.putText(annotated, f"FPS:{1/dt:.0f} | {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.drawMarker(annotated, (int(TARGET_U), int(TARGET_V)),
                           (0, 255, 0), cv2.MARKER_TILTED_CROSS, 30, 1)

            # 深度图伪彩色可视化 (右下角小图)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            mini_h, mini_w = 120, 160
            depth_mini = cv2.resize(depth_colormap, (mini_w, mini_h))
            annotated[-mini_h:, -mini_w:] = depth_mini

            cv2.imshow("IBVS + Depth", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not ibvs_active:
                print("➡️ 开启 IBVS+Depth!")
                ibvs_active = True
            elif key == ord('x') and ibvs_active:
                print("🛑 暂停!")
                ibvs_active = False
                robot.send_speed_l([0, 0, 0])

    finally:
        robot.stop()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
