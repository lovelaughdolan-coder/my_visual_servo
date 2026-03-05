#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBVS 视觉伺服控制器 (Eye-in-Hand)
使用 SpeedL (笛卡尔速度伺服) 实现图像空间的视觉伺服

当前版本：ArUco 4 角点作为视觉特征 (8 个特征值)
未来版本：可替换为 YOLO 检测结果 (u,v)

IBVS 核心公式：
  e = s - s*                    (图像特征误差)
  v_cam = -λ * pinv(Lx) * e    (相机坐标系速度)
  v_base = R_base_cam * v_cam   (基座坐标系速度 → SpeedL)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point, TwistStamped
from visualization_msgs.msg import Marker
import tf2_ros
import numpy as np
import socket
import time
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation


class IBVSControllerEyeInHand(Node):
    """IBVS 视觉伺服控制器 (Eye-in-Hand, SpeedL)"""

    def __init__(self):
        super().__init__('ibvs_controller_eye_in_hand')

        # ========================= 参数 =========================
        self.declare_parameter('robot_host', '192.168.0.10')
        self.declare_parameter('robot_port', 10003)
        self.declare_parameter('lambda_gain', 0.5)         # IBVS 增益 λ
        self.declare_parameter('max_linear_vel', 80.0)     # mm/s
        self.declare_parameter('max_angular_vel', 15.0)    # deg/s
        self.declare_parameter('control_rate', 10.0)       # Hz

        # 目标特征参数：标记在图像中的目标位置
        self.declare_parameter('target_u', 320.0)          # 目标中心 u (像素)
        self.declare_parameter('target_v', 240.0)          # 目标中心 v (像素)
        self.declare_parameter('target_area', 25000.0)     # 目标面积 (像素²)，控制 Z 方向距离

        # ArUco 参数
        self.declare_parameter('aruco_dict', 'DICT_5X5_1000')
        self.declare_parameter('marker_size', 0.15)        # m

        # TF 帧
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('base_frame', 'elfin_base_link')

        # 相机话题
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')

        # 深度 Z 估计方式
        self.declare_parameter('default_depth', 0.4)       # m, 默认深度（无法估计时使用）
        self.declare_parameter('ema_alpha', 0.5)             # EMA 滤波系数 (0~1, 越小越平滑)
        self.declare_parameter('dead_zone_px', 3.0)          # 死区: 中心误差 < 此值时视为到位
        self.declare_parameter('dead_zone_area_ratio', 0.05)  # 死区: 面积误差比 < 此值时视为到位

        # 获取参数
        self.robot_host = self.get_parameter('robot_host').value
        self.robot_port = self.get_parameter('robot_port').value
        self.lambda_gain = self.get_parameter('lambda_gain').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.control_rate = self.get_parameter('control_rate').value
        self.target_u = self.get_parameter('target_u').value
        self.target_v = self.get_parameter('target_v').value
        self.target_area = self.get_parameter('target_area').value
        self.marker_size = self.get_parameter('marker_size').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.default_depth = self.get_parameter('default_depth').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.dead_zone_px = self.get_parameter('dead_zone_px').value
        self.dead_zone_area_ratio = self.get_parameter('dead_zone_area_ratio').value

        # ========================= ArUco 检测器 =========================
        dict_name = self.get_parameter('aruco_dict').value
        aruco_dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_5X5_100)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # ========================= 状态 =========================
        self.running = False
        self.current_pose = None        # [x,y,z,rx,ry,rz] mm/deg
        self.intrinsic_mat = None       # 3x3 相机内参矩阵
        self.distortion = None          # 畸变系数
        self.latest_corners = None      # 最新检测到的 ArUco 角点
        self.filtered_corners = None    # EMA 滤波后的角点
        self.latest_depth_z = None      # 从 solvePnP 估计的深度 Z
        self.latest_stamp = None        # 最新检测时间戳
        self._no_marker_count = 0
        self._debug_count = 0  # 调试计数
        self._arrived = False           # 是否已到位（用于减少日志）

        # Socket
        self.sock = None

        # CV Bridge
        self.bridge = CvBridge()

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ========================= ROS 发布器 =========================
        self.vel_cmd_pub = self.create_publisher(TwistStamped, '/ibvs/velocity_cmd', 10)
        self.vel_marker_pub = self.create_publisher(Marker, '/ibvs/velocity_arrow', 10)
        self.debug_image_pub = self.create_publisher(Image, '/ibvs/debug_image', 10)

        # ========================= ROS 订阅器 =========================
        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)

        self.get_logger().info('=== IBVS Controller (Eye-in-Hand, SpeedL) ===')
        self.get_logger().info(f'Robot: {self.robot_host}:{self.robot_port}')
        self.get_logger().info(f'Lambda: {self.lambda_gain}')
        self.get_logger().info(f'Target: u={self.target_u}, v={self.target_v}, area={self.target_area}')
        self.get_logger().info(f'Camera frame: {self.camera_frame}')
        self.get_logger().info(f'Image topic: {self.image_topic}')

        # 连接并初始化
        self.connect_and_init()

        # 控制定时器
        if self.running:
            self.control_timer = self.create_timer(
                1.0 / self.control_rate, self.control_loop)

    # =================================================================
    #  Socket 通信 (与 PBVS 相同)
    # =================================================================
    def send_cmd(self, cmd):
        """发送命令并接收响应"""
        try:
            self.sock.sendall(cmd.encode('utf-8'))
            resp = self.sock.recv(4096).decode('utf-8').strip()
            return resp
        except Exception as e:
            self.get_logger().error(f'发送失败: {e}')
            return ''

    def connect_and_init(self):
        """连接并初始化"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.robot_host, self.robot_port))
            self.get_logger().info('Socket 连接成功')

            resp = self.send_cmd('GrpReset,0,;')
            self.get_logger().info(f'复位: {resp}')
            time.sleep(0.5)

            resp = self.send_cmd('GrpEnable,0,;')
            self.get_logger().info(f'使能: {resp}')
            time.sleep(1.0)

            resp = self.send_cmd('ReadRobotState,0,;')
            self.get_logger().info(f'机器人状态: {resp}')

            self.update_current_pose()
            if self.current_pose is not None:
                self.running = True
                self.get_logger().info('初始化完成，等待相机数据...')
            else:
                self.get_logger().error('读取初始位置失败')

        except Exception as e:
            self.get_logger().error(f'初始化失败: {e}')
            self.running = False

    def update_current_pose(self):
        """更新当前位姿"""
        resp = self.send_cmd('ReadActPos,0,;')
        parts = resp.split(',')
        if len(parts) >= 14 and parts[1] == 'OK':
            self.current_pose = np.array([
                float(parts[8]), float(parts[9]), float(parts[10]),
                float(parts[11]), float(parts[12]), float(parts[13])
            ])
            return True
        return False

    def send_speed_l(self, linear_vel, angular_vel=[0, 0, 0],
                     linear_acc=300, angular_acc=80, run_time=0.5):
        """发送 SpeedL"""
        cmd = f'SpeedL,0,{linear_vel[0]:.2f},{linear_vel[1]:.2f},{linear_vel[2]:.2f},'
        cmd += f'{angular_vel[0]:.2f},{angular_vel[1]:.2f},{angular_vel[2]:.2f},'
        cmd += f'{linear_acc},{angular_acc},{run_time},;'
        try:
            self.sock.sendall(cmd.encode('utf-8'))
            resp = self.sock.recv(4096).decode('utf-8').strip()
            # 调试：打印完整命令和响应
            if self._debug_count < 5 or self._debug_count % 50 == 0:
                self.get_logger().info(f'[DEBUG] CMD: {cmd}')
                self.get_logger().info(f'[DEBUG] RSP: {resp}')
            self._debug_count += 1
            if 'Fail' in resp:
                self.get_logger().warn(f'SpeedL FAIL: {resp}')
                return False
            return True
        except Exception as e:
            self.get_logger().error(f'SpeedL 异常: {e}')
            return False

    # =================================================================
    #  相机回调
    # =================================================================
    def camera_info_callback(self, msg):
        """接收相机内参"""
        if self.intrinsic_mat is None:
            self.intrinsic_mat = np.array(msg.k).reshape(3, 3)
            self.distortion = np.array(msg.d)
            self.get_logger().info(
                f'相机内参已接收: fx={self.intrinsic_mat[0,0]:.1f}, '
                f'fy={self.intrinsic_mat[1,1]:.1f}, '
                f'cx={self.intrinsic_mat[0,2]:.1f}, cy={self.intrinsic_mat[1,2]:.1f}')

    def image_callback(self, msg):
        """接收图像，检测 ArUco 角点"""
        if self.intrinsic_mat is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'图像转换失败: {e}')
            return

        # ArUco 检测
        corners, ids, _ = self.aruco_detector.detectMarkers(cv_image)

        if ids is not None and len(ids) > 0:
            # 取第一个标记的 4 个角点
            raw_corners = corners[0][0]  # shape: (4, 2)，4个角点的 (u,v)
            self.latest_stamp = self.get_clock().now()

            # EMA 滤波平滑角点，减少像素级抖动
            if self.filtered_corners is None:
                self.filtered_corners = raw_corners.copy()
            else:
                self.filtered_corners = (self.ema_alpha * raw_corners +
                                         (1 - self.ema_alpha) * self.filtered_corners)
            self.latest_corners = self.filtered_corners

            # 用 solvePnP 估计深度 Z
            self._estimate_depth(corners[0], cv_image)

            # 发布调试图像
            self._publish_debug_image(cv_image, corners, ids)

    def _estimate_depth(self, corner, image):
        """通过 solvePnP 估计标记中心的深度 Z"""
        # ArUco 标记在物体坐标系中的 3D 角点
        half = self.marker_size / 2.0
        obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float64)

        img_points = corner.reshape(4, 2).astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points,
            self.intrinsic_mat, self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if success:
            self.latest_depth_z = float(tvec[2][0])  # Z 分量 (m)

    def _publish_debug_image(self, image, corners, ids):
        """发布带标注的调试图像"""
        debug = image.copy()

        # 画 ArUco 边框
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)

        # 画目标十字线
        tu, tv = int(self.target_u), int(self.target_v)
        cv2.drawMarker(debug, (tu, tv), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)

        # 画当前中心
        if self.latest_corners is not None:
            center = np.mean(self.latest_corners, axis=0).astype(int)
            cv2.circle(debug, tuple(center), 8, (0, 0, 255), -1)
            cv2.line(debug, (tu, tv), tuple(center), (255, 255, 0), 2)

        # 画目标区域框（基于目标面积）
        half_side = int(np.sqrt(self.target_area) / 2)
        cv2.rectangle(debug,
                      (tu - half_side, tv - half_side),
                      (tu + half_side, tv + half_side),
                      (0, 255, 0), 2)

        try:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
        except Exception:
            pass

    # =================================================================
    #  IBVS 核心：交互矩阵 + 控制律
    # =================================================================
    def compute_interaction_matrix(self, corners_px, Z):
        """构建经典 IBVS 交互矩阵 Lx (8×6)

        对每个特征点 (u,v)，先转换为归一化坐标 (x,y)：
            x = (u - cx) / fx
            y = (v - cy) / fy

        交互矩阵每行：
            Lx_point = [ -1/Z   0    x/Z   xy    -(1+x²)   y  ]
                       [  0   -1/Z   y/Z  (1+y²)   -xy     -x  ]

        Args:
            corners_px: (N, 2) 像素坐标
            Z: 深度 (m)

        Returns:
            Lx: (2N, 6) 交互矩阵
        """
        fx = self.intrinsic_mat[0, 0]
        fy = self.intrinsic_mat[1, 1]
        cx = self.intrinsic_mat[0, 2]
        cy = self.intrinsic_mat[1, 2]

        Lx = []
        for u, v in corners_px:
            # 归一化图像坐标
            x = (u - cx) / fx
            y = (v - cy) / fy

            Lx.append([
                -1.0/Z,   0.0,     x/Z,    x*y,      -(1+x*x),  y
            ])
            Lx.append([
                0.0,      -1.0/Z,  y/Z,    (1+y*y),  -x*y,      -x
            ])

        return np.array(Lx)

    def compute_target_corners(self):
        """根据目标中心和目标面积计算目标角点像素坐标

        假设标记在目标位置时是正方形（无旋转），4 角点均匀分布。

        Returns:
            (4, 2) 目标角点像素坐标
        """
        half_side = np.sqrt(self.target_area) / 2.0
        tu, tv = self.target_u, self.target_v
        return np.array([
            [tu - half_side, tv - half_side],  # 左上
            [tu + half_side, tv - half_side],  # 右上
            [tu + half_side, tv + half_side],  # 右下
            [tu - half_side, tv + half_side],  # 左下
        ])

    def camera_vel_to_base_vel(self, v_cam, w_cam):
        """将相机坐标系的速度转换到基座坐标系

        通过 TF2 获取相机→基座的旋转，然后：
            v_base = R_base_cam * v_cam
            w_base = R_base_cam * w_cam

        注意：SpeedL 的线速度单位是 mm/s，角速度是 deg/s

        Args:
            v_cam: (3,) 线速度 m/s（相机系）
            w_cam: (3,) 角速度 rad/s（相机系）

        Returns:
            linear_vel_mm: (3,) 线速度 mm/s（基座系）
            angular_vel_deg: (3,) 角速度 deg/s（基座系）
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame,
                rclpy.time.Time()
            )
            q = trans.transform.rotation
            R_base_cam = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            # 旋转速度到基座系
            v_base = R_base_cam @ v_cam  # m/s
            w_base = R_base_cam @ w_cam  # rad/s

            # 单位转换
            linear_vel_mm = v_base * 1000.0          # m/s → mm/s
            angular_vel_deg = np.degrees(w_base)     # rad/s → deg/s

            return linear_vel_mm, angular_vel_deg

        except Exception as e:
            self.get_logger().warn(f'TF 变换失败: {e}')
            return None, None

    # =================================================================
    #  主控制循环
    # =================================================================
    def control_loop(self):
        """IBVS 主控制循环"""
        if not self.running:
            return

        if not self.update_current_pose():
            return

        # 检查是否有有效的角点检测
        if self.latest_corners is None or self.latest_stamp is None:
            self._no_marker_count += 1
            if self._no_marker_count % 10 == 1:
                self.get_logger().info('等待 ArUco 标记...')
            self.send_speed_l([0, 0, 0], run_time=0.3)
            return

        # 检查时效性
        age = (self.get_clock().now() - self.latest_stamp).nanoseconds / 1e9
        if age > 1.0:
            self._no_marker_count += 1
            if self._no_marker_count % 10 == 1:
                self.get_logger().info(f'标记数据过期 ({age:.1f}s)')
            self.send_speed_l([0, 0, 0], run_time=0.3)
            return

        self._no_marker_count = 0

        # ===== 当前特征 s =====
        s = self.latest_corners.flatten()  # (8,)：[u0,v0, u1,v1, u2,v2, u3,v3]

        # ===== 目标特征 s* =====
        s_star = self.compute_target_corners().flatten()  # (8,)

        # ===== 误差 e = s - s* =====
        e = s - s_star  # (8,)

        # ===== 深度 Z =====
        Z = self.latest_depth_z if self.latest_depth_z is not None else self.default_depth
        Z = max(Z, 0.05)  # 安全下限

        # ===== 构建交互矩阵 Lx (8×6) =====
        Lx = self.compute_interaction_matrix(self.latest_corners, Z)

        # ===== 控制律: v_cam = -λ * pinv(Lx) * e =====
        # e 的单位是像素，需要转换为归一化坐标
        fx = self.intrinsic_mat[0, 0]
        fy = self.intrinsic_mat[1, 1]
        # 将像素误差归一化
        e_normalized = np.zeros_like(e)
        for i in range(4):
            e_normalized[2*i]   = e[2*i]   / fx  # u 方向
            e_normalized[2*i+1] = e[2*i+1] / fy  # v 方向

        # 伪逆
        Lx_pinv = np.linalg.pinv(Lx)  # (6, 8)

        # 相机坐标系下的速度 [vx, vy, vz, wx, wy, wz]
        vel_cam = -self.lambda_gain * Lx_pinv @ e_normalized  # (6,)

        v_cam = vel_cam[:3]  # 线速度 m/s
        w_cam = vel_cam[3:]  # 角速度 rad/s

        # ===== 坐标变换：相机系 → 基座系 =====
        result = self.camera_vel_to_base_vel(v_cam, w_cam)
        if result[0] is None:
            self.send_speed_l([0, 0, 0], run_time=0.3)
            return

        linear_vel_mm, angular_vel_deg = result

        # ===== 速度限幅 =====
        lin_norm = np.linalg.norm(linear_vel_mm)
        if lin_norm > self.max_linear_vel:
            linear_vel_mm = linear_vel_mm / lin_norm * self.max_linear_vel

        ang_norm = np.linalg.norm(angular_vel_deg)
        if ang_norm > self.max_angular_vel:
            angular_vel_deg = angular_vel_deg / ang_norm * self.max_angular_vel

        # ===== 误差统计 =====
        center = np.mean(self.latest_corners, axis=0)
        center_err_u = center[0] - self.target_u
        center_err_v = center[1] - self.target_v
        current_area = cv2.contourArea(self.latest_corners.astype(np.float32))
        area_err = current_area - self.target_area
        center_err_norm = np.sqrt(center_err_u**2 + center_err_v**2)
        area_ratio = abs(area_err) / self.target_area

        # ===== 死区判断：误差足够小时停止 =====
        if center_err_norm < self.dead_zone_px and area_ratio < self.dead_zone_area_ratio:
            if not self._arrived:
                self.get_logger().info(
                    f'\u2705 到位! Center:[{center_err_u:.1f},{center_err_v:.1f}]px '
                    f'Area:{current_area:.0f} Z:{Z:.3f}m')
                self._arrived = True
            self.send_speed_l([0, 0, 0], run_time=0.3)
            return

        self._arrived = False

        self.get_logger().info(
            f'Center Err: [{center_err_u:.0f},{center_err_v:.0f}]px '
            f'Area: {current_area:.0f} (tgt:{self.target_area:.0f}) '
            f'Z:{Z:.3f}m '
            f'Vel: [{linear_vel_mm[0]:.0f},{linear_vel_mm[1]:.0f},{linear_vel_mm[2]:.0f}]mm/s '
            f'AngVel: [{angular_vel_deg[0]:.1f},{angular_vel_deg[1]:.1f},{angular_vel_deg[2]:.1f}]deg/s'
        )

        # ===== 发布可视化 =====
        self._publish_velocity_viz(linear_vel_mm)

        # ===== 发送速度 =====
        run_time = 1.2 / self.control_rate
        self.send_speed_l(linear_vel_mm, angular_vel=angular_vel_deg, run_time=run_time)

    def _publish_velocity_viz(self, linear_vel_mm):
        """发布速度可视化"""
        if self.current_pose is None:
            return

        stamp = self.get_clock().now().to_msg()
        current_pos_m = self.current_pose[:3] / 1000.0

        vel_msg = TwistStamped()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = self.base_frame
        vel_msg.twist.linear.x = linear_vel_mm[0] / 1000.0
        vel_msg.twist.linear.y = linear_vel_mm[1] / 1000.0
        vel_msg.twist.linear.z = linear_vel_mm[2] / 1000.0
        self.vel_cmd_pub.publish(vel_msg)

        # 速度箭头
        vel_arrow = Marker()
        vel_arrow.header.stamp = stamp
        vel_arrow.header.frame_id = self.base_frame
        vel_arrow.ns = 'ibvs_velocity'
        vel_arrow.id = 0
        vel_arrow.type = Marker.ARROW
        vel_arrow.action = Marker.ADD
        vel_arrow.points = []
        p1 = Point()
        p1.x, p1.y, p1.z = current_pos_m[0], current_pos_m[1], current_pos_m[2]
        vel_arrow.points.append(p1)
        p2 = Point()
        scale = 10.0
        linear_vel_m = linear_vel_mm / 1000.0
        p2.x = current_pos_m[0] + linear_vel_m[0] * scale
        p2.y = current_pos_m[1] + linear_vel_m[1] * scale
        p2.z = current_pos_m[2] + linear_vel_m[2] * scale
        vel_arrow.points.append(p2)
        vel_arrow.scale.x = 0.02
        vel_arrow.scale.y = 0.04
        vel_arrow.scale.z = 0.05
        vel_arrow.color.r = 0.0
        vel_arrow.color.g = 1.0
        vel_arrow.color.b = 0.0
        vel_arrow.color.a = 1.0
        self.vel_marker_pub.publish(vel_arrow)

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.sendall('GrpStop,0,;'.encode('utf-8'))
                self.sock.close()
            except:
                pass


def main(args=None):
    rclpy.init(args=args)
    controller = IBVSControllerEyeInHand()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('手动终止')
    finally:
        controller.stop()
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
