#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBVS 视觉伺服控制器 — YOLO 版 (Eye-in-Hand)
仅使用 YOLO 检测的 (u, v, bbox_area) 控制 XYZ 三个平移自由度。

订阅: /yolo/target_pixel (geometry_msgs/PointStamped)
  point.x = u, point.y = v, point.z = bbox_area

控制律（简化比例控制）：
  vx_cam = -λ * (u - u*) / fx * Z
  vy_cam = -λ * (v - v*) / fy * Z
  vz_cam = -λ_z * (area - area*) / area* * Z
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped, Point, TwistStamped
from visualization_msgs.msg import Marker
import tf2_ros
import numpy as np
import socket
import time
from scipy.spatial.transform import Rotation


class IBVSYoloController(Node):
    """IBVS YOLO 控制器 (Eye-in-Hand, SpeedL, 仅 XYZ)"""

    def __init__(self):
        super().__init__('ibvs_yolo_controller')

        # ========================= 参数 =========================
        self.declare_parameter('robot_host', '192.168.0.10')
        self.declare_parameter('robot_port', 10003)
        self.declare_parameter('lambda_gain', 0.5)
        self.declare_parameter('lambda_z_gain', 0.12)
        self.declare_parameter('max_linear_vel', 80.0)      # mm/s
        self.declare_parameter('control_rate', 10.0)         # Hz

        self.declare_parameter('target_u', 320.0)
        self.declare_parameter('target_v', 240.0)
        self.declare_parameter('target_area', 5000.0)

        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('base_frame', 'elfin_base_link')

        self.declare_parameter('target_pixel_topic', '/yolo/target_pixel')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')

        self.declare_parameter('default_depth', 0.4)
        self.declare_parameter('ema_alpha', 0.3)
        self.declare_parameter('dead_zone_px', 5.0)
        self.declare_parameter('dead_zone_area_ratio', 0.15)

        # 获取参数
        self.robot_host = self.get_parameter('robot_host').value
        self.robot_port = self.get_parameter('robot_port').value
        self.lambda_gain = self.get_parameter('lambda_gain').value
        self.lambda_z_gain = self.get_parameter('lambda_z_gain').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.control_rate = self.get_parameter('control_rate').value
        self.target_u = self.get_parameter('target_u').value
        self.target_v = self.get_parameter('target_v').value
        self.target_area = self.get_parameter('target_area').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.default_depth = self.get_parameter('default_depth').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.dead_zone_px = self.get_parameter('dead_zone_px').value
        self.dead_zone_area_ratio = self.get_parameter('dead_zone_area_ratio').value

        # ========================= 状态 =========================
        self.running = False
        self.current_pose = None
        self.intrinsic_mat = None
        self.filtered_u = None
        self.filtered_v = None
        self.filtered_area = None
        self.latest_stamp = None
        self._no_target_count = 0
        self._debug_count = 0
        self._arrived = False

        # Socket
        self.sock = None

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ========================= ROS =========================
        self.vel_cmd_pub = self.create_publisher(
            TwistStamped, '/ibvs_yolo/velocity_cmd', 10)
        self.vel_marker_pub = self.create_publisher(
            Marker, '/ibvs_yolo/velocity_arrow', 10)

        target_topic = self.get_parameter('target_pixel_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value

        self.info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.target_sub = self.create_subscription(
            PointStamped, target_topic, self.target_callback, 10)

        self.get_logger().info('=== IBVS YOLO Controller (Eye-in-Hand, XYZ) ===')
        self.get_logger().info(f'Robot: {self.robot_host}:{self.robot_port}')
        self.get_logger().info(f'Lambda XY: {self.lambda_gain}, Z: {self.lambda_z_gain}')
        self.get_logger().info(
            f'Target: u={self.target_u}, v={self.target_v}, area={self.target_area}')
        self.get_logger().info(f'Subscribe: {target_topic}')

        self.connect_and_init()

        if self.running:
            self.control_timer = self.create_timer(
                1.0 / self.control_rate, self.control_loop)

    # =================================================================
    #  Socket 通信
    # =================================================================
    def send_cmd(self, cmd):
        try:
            self.sock.sendall(cmd.encode('utf-8'))
            return self.sock.recv(4096).decode('utf-8').strip()
        except Exception as e:
            self.get_logger().error(f'发送失败: {e}')
            return ''

    def connect_and_init(self):
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

            self.update_current_pose()
            if self.current_pose is not None:
                self.running = True
                self.get_logger().info('初始化完成，等待 YOLO 目标...')
            else:
                self.get_logger().error('读取初始位置失败')
        except Exception as e:
            self.get_logger().error(f'初始化失败: {e}')
            self.running = False

    def update_current_pose(self):
        resp = self.send_cmd('ReadActPos,0,;')
        parts = resp.split(',')
        if len(parts) >= 14 and parts[1] == 'OK':
            self.current_pose = np.array([
                float(parts[8]), float(parts[9]), float(parts[10]),
                float(parts[11]), float(parts[12]), float(parts[13])
            ])
            return True
        return False

    def send_speed_l(self, linear_vel, angular_vel=(0, 0, 0),
                     linear_acc=300, angular_acc=80, run_time=0.5):
        cmd = (f'SpeedL,0,{linear_vel[0]:.2f},{linear_vel[1]:.2f},{linear_vel[2]:.2f},'
               f'{angular_vel[0]:.2f},{angular_vel[1]:.2f},{angular_vel[2]:.2f},'
               f'{linear_acc},{angular_acc},{run_time},;')
        try:
            self.sock.sendall(cmd.encode('utf-8'))
            resp = self.sock.recv(4096).decode('utf-8').strip()
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
    #  回调
    # =================================================================
    def camera_info_callback(self, msg):
        if self.intrinsic_mat is None:
            self.intrinsic_mat = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f'相机内参: fx={self.intrinsic_mat[0,0]:.1f} '
                f'fy={self.intrinsic_mat[1,1]:.1f} '
                f'cx={self.intrinsic_mat[0,2]:.1f} cy={self.intrinsic_mat[1,2]:.1f}')

    def target_callback(self, msg):
        """接收 YOLO 目标: (u, v, bbox_area)"""
        raw_u = msg.point.x
        raw_v = msg.point.y
        raw_area = msg.point.z

        if raw_area <= 0:
            return

        self.latest_stamp = self.get_clock().now()

        # EMA 滤波
        a = self.ema_alpha
        if self.filtered_u is None:
            self.filtered_u = raw_u
            self.filtered_v = raw_v
            self.filtered_area = raw_area
        else:
            self.filtered_u = a * raw_u + (1 - a) * self.filtered_u
            self.filtered_v = a * raw_v + (1 - a) * self.filtered_v
            self.filtered_area = a * raw_area + (1 - a) * self.filtered_area

    # =================================================================
    #  坐标变换
    # =================================================================
    def camera_vel_to_base_vel(self, v_cam):
        """相机系线速度 → 基座系线速度 (mm/s)"""
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rclpy.time.Time())
            q = trans.transform.rotation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            return (R @ v_cam) * 1000.0  # m/s → mm/s
        except Exception as e:
            self.get_logger().warn(f'TF 变换失败: {e}')
            return None

    # =================================================================
    #  主控制循环
    # =================================================================
    def control_loop(self):
        if not self.running:
            return

        if not self.update_current_pose():
            return

        # 有效目标检查
        if self.filtered_u is None or self.latest_stamp is None:
            self._no_target_count += 1
            if self._no_target_count % 10 == 1:
                self.get_logger().info('等待 YOLO 目标...')
            self.send_speed_l([0, 0, 0], run_time=0.3)
            return

        age = (self.get_clock().now() - self.latest_stamp).nanoseconds / 1e9
        if age > 1.0:
            self._no_target_count += 1
            if self._no_target_count % 10 == 1:
                self.get_logger().info(f'目标过期 ({age:.1f}s)')
            self.send_speed_l([0, 0, 0], run_time=0.3)
            return

        self._no_target_count = 0

        if self.intrinsic_mat is None:
            self.get_logger().info('等待相机内参...')
            return

        fx = self.intrinsic_mat[0, 0]
        fy = self.intrinsic_mat[1, 1]

        # ===== 误差 =====
        err_u = self.filtered_u - self.target_u
        err_v = self.filtered_v - self.target_v
        err_area = self.filtered_area - self.target_area

        center_err = np.sqrt(err_u**2 + err_v**2)
        area_ratio = abs(err_area) / self.target_area

        # ===== 死区 =====
        if center_err < self.dead_zone_px and area_ratio < self.dead_zone_area_ratio:
            if not self._arrived:
                self.get_logger().info(
                    f'✅ 到位! Err:[{err_u:.1f},{err_v:.1f}]px '
                    f'Area:{self.filtered_area:.0f} (tgt:{self.target_area:.0f})')
                self._arrived = True
            self.send_speed_l([0, 0, 0], run_time=0.3)
            return

        self._arrived = False

        # ===== 深度 =====
        Z = self.default_depth

        # ===== 比例控制律 =====
        # 相机光学系: X→右, Y→下, Z→前（光轴方向）
        # XY: 目标在右(err_u>0) → 相机右移(vx>0) → 正号
        # Z:  目标太近(err_area>0) → 相机后退(vz<0) → 负号
        vx_cam = self.lambda_gain * (err_u / fx) * Z
        vy_cam = self.lambda_gain * (err_v / fy) * Z

        # Z 轴渐进减速：误差 < 30% 时按比例降低增益，防止超调
        z_gain = self.lambda_z_gain
        if area_ratio < 0.30:
            z_gain *= (area_ratio / 0.30)
        vz_cam = -z_gain * (err_area / self.target_area) * Z
        v_cam = np.array([vx_cam, vy_cam, vz_cam])

        # ===== 坐标变换 =====
        linear_vel_mm = self.camera_vel_to_base_vel(v_cam)
        if linear_vel_mm is None:
            self.send_speed_l([0, 0, 0], run_time=0.3)
            return

        # ===== 限幅 =====
        norm = np.linalg.norm(linear_vel_mm)
        if norm > self.max_linear_vel:
            linear_vel_mm = linear_vel_mm / norm * self.max_linear_vel

        # ===== 日志 =====
        self.get_logger().info(
            f'Err: [{err_u:.0f},{err_v:.0f}]px '
            f'Area: {self.filtered_area:.0f} (tgt:{self.target_area:.0f}) '
            f'Vel: [{linear_vel_mm[0]:.0f},{linear_vel_mm[1]:.0f},'
            f'{linear_vel_mm[2]:.0f}]mm/s')

        # ===== 可视化 =====
        self._publish_velocity_viz(linear_vel_mm)

        # ===== 发送 =====
        run_time = 1.2 / self.control_rate
        self.send_speed_l(linear_vel_mm, run_time=run_time)

    # =================================================================
    #  可视化
    # =================================================================
    def _publish_velocity_viz(self, linear_vel_mm):
        if self.current_pose is None:
            return

        stamp = self.get_clock().now().to_msg()
        pos_m = self.current_pose[:3] / 1000.0

        vel_msg = TwistStamped()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = self.base_frame
        vel_msg.twist.linear.x = linear_vel_mm[0] / 1000.0
        vel_msg.twist.linear.y = linear_vel_mm[1] / 1000.0
        vel_msg.twist.linear.z = linear_vel_mm[2] / 1000.0
        self.vel_cmd_pub.publish(vel_msg)

        arrow = Marker()
        arrow.header.stamp = stamp
        arrow.header.frame_id = self.base_frame
        arrow.ns = 'ibvs_yolo_vel'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.points = []

        p1 = Point(x=pos_m[0], y=pos_m[1], z=pos_m[2])
        arrow.points.append(p1)

        vel_m = linear_vel_mm / 1000.0
        sc = 10.0
        p2 = Point(
            x=pos_m[0] + vel_m[0] * sc,
            y=pos_m[1] + vel_m[1] * sc,
            z=pos_m[2] + vel_m[2] * sc)
        arrow.points.append(p2)

        arrow.scale.x = 0.02
        arrow.scale.y = 0.04
        arrow.scale.z = 0.05
        arrow.color.r = 1.0
        arrow.color.g = 0.5
        arrow.color.b = 0.0
        arrow.color.a = 1.0
        self.vel_marker_pub.publish(arrow)

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.sendall('GrpStop,0,;'.encode('utf-8'))
                self.sock.close()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = IBVSYoloController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('手动终止')
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
