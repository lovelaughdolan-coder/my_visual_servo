#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PBVS 视觉伺服控制器 (Eye-to-Hand)
使用 SpeedL (笛卡尔速度伺服) 实现直接的速度控制
带 RViz 可视化功能
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped, TwistStamped
from visualization_msgs.msg import Marker
import tf2_ros
import numpy as np
import socket
import time


class PBVSController(Node):
    """PBVS 视觉伺服控制器 (SpeedL 模式) - 带 RViz 可视化"""

    def __init__(self):
        super().__init__('pbvs_controller')

        # 参数
        self.declare_parameter('robot_host', '192.168.0.10')
        self.declare_parameter('robot_port', 10003)
        self.declare_parameter('kp_linear', 1.0)
        self.declare_parameter('target_offset_x', 0.0)
        self.declare_parameter('target_offset_y', 0.0)
        self.declare_parameter('target_offset_z', 0.4)
        self.declare_parameter('max_linear_vel', 200.0)  # mm/s
        self.declare_parameter('kp_angular', 0.5)  # 姿态增益
        self.declare_parameter('max_angular_vel', 30.0)  # deg/s
        self.declare_parameter('control_rate', 10.0)  # Hz
        self.declare_parameter('marker_timeout', 1.0)
        self.declare_parameter('marker_max_distance', 2.0)  # m, 过滤距离太远的误识别
        self.declare_parameter('marker_max_jump', 0.5)  # m, 过滤跳变太大的检测

        # 获取参数
        self.robot_host = self.get_parameter('robot_host').value
        self.robot_port = self.get_parameter('robot_port').value
        self.kp_linear = self.get_parameter('kp_linear').value
        self.target_offset = np.array([
            self.get_parameter('target_offset_x').value,
            self.get_parameter('target_offset_y').value,
            self.get_parameter('target_offset_z').value
        ])
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.control_rate = self.get_parameter('control_rate').value
        self.marker_timeout = self.get_parameter('marker_timeout').value
        self.marker_max_distance = self.get_parameter('marker_max_distance').value
        self.marker_max_jump = self.get_parameter('marker_max_jump').value

        # 状态
        self.marker_pose = None
        self.marker_stamp = None
        self.running = False
        self.current_pose = None  # [x,y,z,rx,ry,rz] in mm/deg
        self.last_valid_marker_pos = None  # 上一次有效的标记位置 (m)

        # Socket
        self.sock = None

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ===== ROS 发布器 (可视化) =====
        # 末端位置 (PoseArray - 便于在 RViz 中用 Axes 显示)
        self.ee_pose_pub = self.create_publisher(
            PoseArray, '/pbvs/end_effector_pose', 10
        )
        # 目标位置 (PoseArray)
        self.target_pose_pub = self.create_publisher(
            PoseArray, '/pbvs/target_pose', 10
        )
        # 速度指令 (TwistStamped)
        self.vel_cmd_pub = self.create_publisher(
            TwistStamped, '/pbvs/velocity_cmd', 10
        )
        # 速度箭头 (Marker)
        self.vel_marker_pub = self.create_publisher(
            Marker, '/pbvs/velocity_arrow', 10
        )
        # 误差向量 (Marker)
        self.error_marker_pub = self.create_publisher(
            Marker, '/pbvs/error_arrow', 10
        )

        # 订阅 ArUco
        self.aruco_sub = self.create_subscription(
            PoseArray, '/aruco/poses', self.aruco_callback, 10
        )

        self.get_logger().info('=== PBVS Controller (SpeedL + RViz) ===')
        self.get_logger().info(f'Robot: {self.robot_host}:{self.robot_port}')
        self.get_logger().info(f'Target offset: {self.target_offset} m')
        self.get_logger().info(f'Max linear vel: {self.max_linear_vel} mm/s')
        self.get_logger().info(f'Control rate: {self.control_rate} Hz')
        self.get_logger().info('Publishing to: /pbvs/end_effector_pose, /pbvs/target_pose, /pbvs/velocity_cmd')

        # 连接并初始化
        self.connect_and_init()

        # 控制定时器
        if self.running:
            self.control_timer = self.create_timer(
                1.0 / self.control_rate, self.control_loop
            )

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
                self.get_logger().info('初始化完成')
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

    def aruco_callback(self, msg):
        if len(msg.poses) > 0:
            self.marker_pose = msg.poses[0]
            self.marker_stamp = self.get_clock().now()

    def get_marker_in_base(self):
        """获取标记在基座坐标系的完整位姿
        返回: (position, rotation_matrix) 或 None
        - position: [x, y, z] in m
        - rotation_matrix: 3x3 旋转矩阵 (ArUco 在基座系的朝向)
        """
        if self.marker_pose is None:
            return None

        now = self.get_clock().now()
        if self.marker_stamp is None:
            return None
        age = (now - self.marker_stamp).nanoseconds / 1e9
        if age > self.marker_timeout:
            return None

        try:
            # 获取 camera_color_optical_frame 到 elfin_base_link 的变换
            trans = self.tf_buffer.lookup_transform(
                'elfin_base_link', 'camera_color_optical_frame',
                rclpy.time.Time()
            )
            
            from scipy.spatial.transform import Rotation
            
            # 相机到基座的变换
            t_cam = trans.transform.translation
            q_cam = trans.transform.rotation
            R_cam = Rotation.from_quat([q_cam.x, q_cam.y, q_cam.z, q_cam.w]).as_matrix()
            T_base_cam = np.eye(4)
            T_base_cam[:3, :3] = R_cam
            T_base_cam[:3, 3] = [t_cam.x, t_cam.y, t_cam.z]
            
            # ArUco 在相机坐标系的位姿
            marker_pos_cam = np.array([
                self.marker_pose.position.x,
                self.marker_pose.position.y,
                self.marker_pose.position.z
            ])
            marker_quat_cam = [
                self.marker_pose.orientation.x,
                self.marker_pose.orientation.y,
                self.marker_pose.orientation.z,
                self.marker_pose.orientation.w
            ]
            R_marker_cam = Rotation.from_quat(marker_quat_cam).as_matrix()
            
            # 构建 ArUco 在相机系的齐次变换矩阵
            T_cam_marker = np.eye(4)
            T_cam_marker[:3, :3] = R_marker_cam
            T_cam_marker[:3, 3] = marker_pos_cam
            
            # ArUco 在基座系的变换
            T_base_marker = T_base_cam @ T_cam_marker
            
            # 提取位置和旋转
            marker_pos_base = T_base_marker[:3, 3]
            marker_rot_base = T_base_marker[:3, :3]
            
            return marker_pos_base, marker_rot_base
        except Exception as e:
            self.get_logger().warn(f'TF 变换失败: {e}')
            return None

    def send_speed_l(self, linear_vel, angular_vel=[0,0,0], linear_acc=500, angular_acc=100, run_time=0.5):
        """发送 SpeedL"""
        cmd = f'SpeedL,0,{linear_vel[0]:.2f},{linear_vel[1]:.2f},{linear_vel[2]:.2f},'
        cmd += f'{angular_vel[0]:.2f},{angular_vel[1]:.2f},{angular_vel[2]:.2f},'
        cmd += f'{linear_acc},{angular_acc},{run_time},;'

        try:
            self.sock.sendall(cmd.encode('utf-8'))
            resp = self.sock.recv(4096).decode('utf-8').strip()
            if 'Fail' in resp:
                self.get_logger().warn(f'SpeedL: {resp}')
                return False
            return True
        except Exception as e:
            self.get_logger().error(f'SpeedL 异常: {e}')
            return False

    def publish_visualization(self, current_pos_mm, current_rpy_deg, target_pos_mm, linear_vel_mm):
        """发布 RViz 可视化话题
        current_pos_mm: [x, y, z] in mm
        current_rpy_deg: [rx, ry, rz] in degrees (Elfin 欧拉角)
        target_pos_mm: [x, y, z] in mm
        linear_vel_mm: [vx, vy, vz] in mm/s
        """
        from scipy.spatial.transform import Rotation
        stamp = self.get_clock().now().to_msg()
        
        # 转换为米
        current_pos_m = current_pos_mm / 1000.0
        target_pos_m = target_pos_mm / 1000.0
        linear_vel_m = linear_vel_mm / 1000.0
        
        # 欧拉角转四元数 (Elfin 使用 XYZ 外旋欧拉角，即 'xyz' intrinsic = 'XYZ' extrinsic)
        # 注意: scipy 的 'xyz' 是 intrinsic (内旋)，等价于 ZYX extrinsic
        # Elfin 文档通常使用 ZYX 外旋 (Rz->Ry->Rx)，对应 scipy 的 'zyx' extrinsic = 'xyz' intrinsic
        # 但最安全的方式是尝试 'XYZ' extrinsic (等价于 'zyx' intrinsic)
        try:
            rot = Rotation.from_euler('ZYX', [current_rpy_deg[2], current_rpy_deg[1], current_rpy_deg[0]], degrees=True)
            quat = rot.as_quat()  # [x, y, z, w]
        except Exception as e:
            self.get_logger().warn(f'欧拉角转换失败: {e}')
            quat = [0, 0, 0, 1]
        
        # === 1. 末端位置 (PoseArray) ===
        from geometry_msgs.msg import Pose
        ee_msg = PoseArray()
        ee_msg.header.stamp = stamp
        ee_msg.header.frame_id = 'elfin_base_link'
        ee_pose = Pose()
        ee_pose.position.x = current_pos_m[0]
        ee_pose.position.y = current_pos_m[1]
        ee_pose.position.z = current_pos_m[2]
        ee_pose.orientation.x = quat[0]
        ee_pose.orientation.y = quat[1]
        ee_pose.orientation.z = quat[2]
        ee_pose.orientation.w = quat[3]
        ee_msg.poses.append(ee_pose)
        self.ee_pose_pub.publish(ee_msg)
        
        # === 2. 目标位置 (PoseArray) ===
        target_msg = PoseArray()
        target_msg.header.stamp = stamp
        target_msg.header.frame_id = 'elfin_base_link'
        target_pose = Pose()
        target_pose.position.x = target_pos_m[0]
        target_pose.position.y = target_pos_m[1]
        target_pose.position.z = target_pos_m[2]
        target_pose.orientation.w = 1.0  # 目标暂时用默认姿态
        target_msg.poses.append(target_pose)
        self.target_pose_pub.publish(target_msg)
        
        # === 3. 速度指令 (TwistStamped) ===
        vel_msg = TwistStamped()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = 'elfin_base_link'
        vel_msg.twist.linear.x = linear_vel_m[0]
        vel_msg.twist.linear.y = linear_vel_m[1]
        vel_msg.twist.linear.z = linear_vel_m[2]
        self.vel_cmd_pub.publish(vel_msg)
        
        # === 4. 速度箭头 (Marker - 蓝色) ===
        vel_arrow = Marker()
        vel_arrow.header.stamp = stamp
        vel_arrow.header.frame_id = 'elfin_base_link'
        vel_arrow.ns = 'velocity'
        vel_arrow.id = 0
        vel_arrow.type = Marker.ARROW
        vel_arrow.action = Marker.ADD
        # 起点: 末端位置
        vel_arrow.points = []
        from geometry_msgs.msg import Point
        p1 = Point()
        p1.x, p1.y, p1.z = current_pos_m[0], current_pos_m[1], current_pos_m[2]
        vel_arrow.points.append(p1)
        # 终点: 末端 + 速度向量 (放大10倍便于观察)
        p2 = Point()
        scale = 10.0  # 放大系数
        p2.x = current_pos_m[0] + linear_vel_m[0] * scale
        p2.y = current_pos_m[1] + linear_vel_m[1] * scale
        p2.z = current_pos_m[2] + linear_vel_m[2] * scale
        vel_arrow.points.append(p2)
        # 样式
        vel_arrow.scale.x = 0.02  # 箭头杆粗细
        vel_arrow.scale.y = 0.04  # 箭头头粗细
        vel_arrow.scale.z = 0.05
        vel_arrow.color.r = 0.0
        vel_arrow.color.g = 0.0
        vel_arrow.color.b = 1.0
        vel_arrow.color.a = 1.0
        self.vel_marker_pub.publish(vel_arrow)
        
        # === 5. 误差向量 (Marker - 红色) ===
        error_arrow = Marker()
        error_arrow.header.stamp = stamp
        error_arrow.header.frame_id = 'elfin_base_link'
        error_arrow.ns = 'error'
        error_arrow.id = 0
        error_arrow.type = Marker.ARROW
        error_arrow.action = Marker.ADD
        # 起点: 末端位置
        error_arrow.points = []
        error_arrow.points.append(p1)
        # 终点: 目标位置
        p3 = Point()
        p3.x, p3.y, p3.z = target_pos_m[0], target_pos_m[1], target_pos_m[2]
        error_arrow.points.append(p3)
        # 样式
        error_arrow.scale.x = 0.015
        error_arrow.scale.y = 0.03
        error_arrow.scale.z = 0.04
        error_arrow.color.r = 1.0
        error_arrow.color.g = 0.0
        error_arrow.color.b = 0.0
        error_arrow.color.a = 0.8
        self.error_marker_pub.publish(error_arrow)

    def control_loop(self):
        """主控制循环"""
        if not self.running:
            return

        if not self.update_current_pose():
            return

        current_pos = self.current_pose[:3]  # mm, [x, y, z]
        current_rpy = self.current_pose[3:6]  # deg, [rx, ry, rz]

        marker_result = self.get_marker_in_base()
        
        if marker_result is not None:
            marker_pos, marker_rot = marker_result  # marker_pos in m, marker_rot is 3x3 matrix
            
            # === 过滤误识别 ===
            # 1. 距离过滤：过滤掉距离机器人太远的检测
            marker_distance = np.linalg.norm(marker_pos)
            if marker_distance > self.marker_max_distance:
                self.get_logger().warn(f'标记距离过远 ({marker_distance:.2f}m > {self.marker_max_distance}m)，已过滤')
                return
            
            # 2. 跳变过滤：过滤掉跳变太大的检测
            if self.last_valid_marker_pos is not None:
                jump_distance = np.linalg.norm(marker_pos - self.last_valid_marker_pos)
                if jump_distance > self.marker_max_jump:
                    self.get_logger().warn(f'标记跳变过大 ({jump_distance:.3f}m > {self.marker_max_jump}m)，已过滤')
                    return
            
            # 更新最后有效位置
            self.last_valid_marker_pos = marker_pos.copy()
            
            # 在 ArUco 局部坐标系中定义偏移量 (m)
            # offset_local = [x_local, y_local, z_local]
            # ArUco 的 Z 轴指向标记表面的法线方向
            offset_local = np.array([
                self.target_offset[0],  # ArUco X 方向偏移
                self.target_offset[1],  # ArUco Y 方向偏移  
                self.target_offset[2]   # ArUco Z 方向偏移 (法线方向)
            ])
            
            # 将偏移量从 ArUco 局部坐标系转换到基座坐标系
            offset_base = marker_rot @ offset_local
            
            # 目标位置 = 标记位置 + 旋转后的偏移量 (m -> mm)
            target_pos = (marker_pos + offset_base) * 1000.0
            
            # === 位置控制 ===
            error_pos = target_pos - current_pos
            linear_vel = self.kp_linear * error_pos
            vel_norm = np.linalg.norm(linear_vel)
            if vel_norm > self.max_linear_vel:
                linear_vel = linear_vel / vel_norm * self.max_linear_vel
            
            # === 姿态控制 ===
            # 目标姿态：末端 Z 轴对准 ArUco 的 -Z 方向（即朝向标记表面）
            # ArUco 的 Z 轴指向相机，我们希望末端 Z 轴指向 ArUco（即 -ArUco_Z）
            from scipy.spatial.transform import Rotation
            
            # 当前末端姿态（从欧拉角转换）
            current_rot = Rotation.from_euler('ZYX', [current_rpy[2], current_rpy[1], current_rpy[0]], degrees=True)
            current_rot_matrix = current_rot.as_matrix()
            
            # 目标姿态：末端 Z 轴对准 ArUco 的 -Z 方向
            # marker_rot 的第三列是 ArUco 的 Z 轴方向
            aruco_z = marker_rot[:, 2]  # ArUco Z 轴（指向相机）
            target_z = -aruco_z  # 末端 Z 轴应该指向 ArUco（相反方向）
            
            # 构建目标旋转矩阵（使用 Z 轴和保持 X/Y 尽量不变的方法）
            # 使用 Rodrigues 公式或简单的轴角方法
            current_z = current_rot_matrix[:, 2]  # 当前末端 Z 轴
            
            # 计算姿态误差（用轴角表示）
            # 误差 = 从当前姿态到目标方向的旋转
            # 使用叉积得到旋转轴，点积得到角度
            cross = np.cross(current_z, target_z)
            dot = np.dot(current_z, target_z)
            
            # 归一化
            cross_norm = np.linalg.norm(cross)
            if cross_norm > 1e-6:
                axis = cross / cross_norm
                angle = np.arctan2(cross_norm, dot)  # rad
                
                # 角速度 = 增益 * 角度 * 轴 (转换为 deg/s)
                angular_vel = self.kp_angular * np.degrees(angle) * axis
                
                # 限制角速度
                angular_vel_norm = np.linalg.norm(angular_vel)
                if angular_vel_norm > self.max_angular_vel:
                    angular_vel = angular_vel / angular_vel_norm * self.max_angular_vel
            else:
                # 已经对齐，不需要旋转
                angular_vel = np.array([0.0, 0.0, 0.0])
                angle = 0.0
            
            # 调试日志
            self.get_logger().info(
                f'Pos Err: [{error_pos[0]:.0f},{error_pos[1]:.0f},{error_pos[2]:.0f}] '
                f'Ang Err: {np.degrees(angle):.1f}deg '
                f'AngVel: [{angular_vel[0]:.1f},{angular_vel[1]:.1f},{angular_vel[2]:.1f}]'
            )
            
            # 发布可视化 (传入欧拉角)
            self.publish_visualization(current_pos, current_rpy, target_pos, linear_vel)
            
            # 发送速度（线速度 + 角速度）
            run_time = 1.2 / self.control_rate
            self.send_speed_l(linear_vel, angular_vel=angular_vel, run_time=run_time)
            
        else:
            if not hasattr(self, '_no_marker_count'):
                self._no_marker_count = 0
            self._no_marker_count += 1
            if self._no_marker_count % 5 == 1:
                self.get_logger().info('等待 ArUco 标记...')
            self.send_speed_l([0, 0, 0], run_time=0.3)

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
    controller = PBVSController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Stopping...')
    finally:
        controller.stop()
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
