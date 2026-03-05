"""
眼在手上（Eye-in-Hand）手眼标定程序

相机安装在机械臂末端，ArUco 标记固定在外部。
求解：camera_color_optical_frame → elfin_end_link（末端坐标系）的变换
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from threading import Thread, Event
import cv2
import numpy as np
import yaml
import socket
from geometry_msgs.msg import PoseArray, Pose
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import random
import itertools


def pose_to_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """将位置和四元数转换为 4x4 齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    T[:3, 3] = position
    return T


def matrix_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将 4x4 齐次变换矩阵转换为位置和四元数"""
    position = T[:3, 3]
    quaternion = Rotation.from_matrix(T[:3, :3]).as_quat()
    return position, quaternion


def rpy_to_matrix(rpy_deg: np.ndarray) -> np.ndarray:
    """将欧拉角（度）转换为旋转矩阵，使用固定轴 XYZ (等价于 ZYX 内旋)"""
    return Rotation.from_euler('ZYX', [rpy_deg[2], rpy_deg[1], rpy_deg[0]], degrees=True).as_matrix()


def _compute_marker_consistency(T_cam2end: np.ndarray,
                                T_base2end_list: List[np.ndarray],
                                T_cam2marker_list: List[np.ndarray]) -> List[float]:
    """
    计算标定结果的精度指标：所有样本推算出的标记在基座系中的位置一致性。
    T_base2marker = T_base2end @ T_end2cam @ T_cam2marker (应恒定)
    返回每个样本相对于中位数位置的误差（米）。
    """
    T_end2cam = np.linalg.inv(T_cam2end)
    marker_positions = []
    for i in range(len(T_base2end_list)):
        T_base2marker = T_base2end_list[i] @ T_end2cam @ T_cam2marker_list[i]
        marker_positions.append(T_base2marker[:3, 3])

    marker_positions = np.array(marker_positions)
    # 使用中位数作为参考（比第一个点更鲁棒）
    ref_pos = np.median(marker_positions, axis=0)
    errors = [np.linalg.norm(p - ref_pos) for p in marker_positions]
    return errors


def _opencv_solve(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list, method) -> np.ndarray:
    """调用 OpenCV calibrateHandEye 并返回 T_cam2end"""
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g2b_list, t_g2b_list,
        R_t2c_list, t_t2c_list,
        method=method
    )
    T_cam2end = np.eye(4)
    T_cam2end[:3, :3] = R_cam2gripper
    T_cam2end[:3, 3] = t_cam2gripper.reshape(3)
    return T_cam2end


def _refine_with_optimization(T_cam2end_init: np.ndarray,
                               T_base2end_list: List[np.ndarray],
                               T_cam2marker_list: List[np.ndarray]) -> np.ndarray:
    """
    使用 scipy.optimize.least_squares 对标定结果进行非线性精化。
    目标：最小化所有样本推算出的标记位置的方差。
    参数化：旋转用 Rodrigues 向量 (3 DOF) + 平移 (3 DOF) = 6 DOF
    """
    # 初始参数化
    R_init = T_cam2end_init[:3, :3]
    rvec_init = Rotation.from_matrix(R_init).as_rotvec()
    t_init = T_cam2end_init[:3, 3]
    x0 = np.concatenate([rvec_init, t_init])

    def residuals(x):
        rvec = x[:3]
        tvec = x[3:6]
        R = Rotation.from_rotvec(rvec).as_matrix()
        T_cam2end = np.eye(4)
        T_cam2end[:3, :3] = R
        T_cam2end[:3, 3] = tvec
        T_end2cam = np.linalg.inv(T_cam2end)

        marker_positions = []
        for i in range(len(T_base2end_list)):
            T_base2marker = T_base2end_list[i] @ T_end2cam @ T_cam2marker_list[i]
            marker_positions.append(T_base2marker[:3, 3])

        marker_positions = np.array(marker_positions)
        centroid = np.mean(marker_positions, axis=0)
        # 残差 = 每个标记位置到质心的偏差 (展平)
        return (marker_positions - centroid).ravel()

    result = least_squares(residuals, x0, method='lm')

    T_refined = np.eye(4)
    T_refined[:3, :3] = Rotation.from_rotvec(result.x[:3]).as_matrix()
    T_refined[:3, 3] = result.x[3:6]
    return T_refined


def solve_hand_eye_all_methods(T_base2end_list: List[np.ndarray],
                                T_cam2marker_list: List[np.ndarray],
                                logger=None) -> np.ndarray:
    """
    综合手眼标定求解器 (参考 easy_handeye / ethz-asl / tingelst 开源实现)

    改进：
    1. 跑全部 5 种 OpenCV 方法 + 对比选优
    2. RANSAC 异常值剔除
    3. scipy 非线性优化精化

    返回: T_cam2end (4x4 齐次变换矩阵)
    """
    n = len(T_base2end_list)
    if n < 3:
        raise ValueError("至少需要 3 组数据")

    def log(msg):
        if logger:
            logger.info(msg)

    # 准备 OpenCV 输入
    R_g2b_list = [T[:3, :3] for T in T_base2end_list]
    t_g2b_list = [T[:3, 3].reshape(3, 1) for T in T_base2end_list]
    R_t2c_list = [T[:3, :3] for T in T_cam2marker_list]
    t_t2c_list = [T[:3, 3].reshape(3, 1) for T in T_cam2marker_list]

    # ====== 第 1 步：跑全部 5 种 OpenCV 方法 ======
    methods = {
        'TSAI': cv2.CALIB_HAND_EYE_TSAI,
        'PARK': cv2.CALIB_HAND_EYE_PARK,
        'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = {}
    log("\n--- 5 种 OpenCV 方法对比 ---")
    for name, method in methods.items():
        try:
            T = _opencv_solve(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list, method)
            errors = _compute_marker_consistency(T, T_base2end_list, T_cam2marker_list)
            mean_err = np.mean(errors)
            max_err = np.max(errors)
            results[name] = {'T': T, 'mean_err': mean_err, 'max_err': max_err}
            log(f"  {name:12s}: 平均误差 {mean_err*1000:.2f} mm, 最大 {max_err*1000:.2f} mm, "
                f"平移 [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}]")
        except Exception as e:
            log(f"  {name:12s}: 失败 ({e})")

    if not results:
        raise RuntimeError("所有方法均失败")

    # 选择平均误差最小的方法
    best_name = min(results, key=lambda k: results[k]['mean_err'])
    best_T = results[best_name]['T']
    log(f"\n✅ 最佳初始方法: {best_name} (平均误差 {results[best_name]['mean_err']*1000:.2f} mm)")

    # ====== 第 2 步：RANSAC 异常值剔除 ======
    if n >= 8:
        log(f"\n--- RANSAC 异常值剔除 (共 {n} 个样本) ---")
        best_ransac_err = float('inf')
        best_ransac_T = best_T
        best_inliers = list(range(n))

        # RANSAC 参数
        sample_size = max(6, int(n * 0.6))  # 每次采样 60% 的数据
        n_iterations = min(100, max(20, n * 3))  # 迭代次数
        inlier_threshold = 0.015  # 15mm 内视为内点

        for _ in range(n_iterations):
            # 随机采样
            indices = random.sample(range(n), sample_size)
            sub_R_g2b = [R_g2b_list[i] for i in indices]
            sub_t_g2b = [t_g2b_list[i] for i in indices]
            sub_R_t2c = [R_t2c_list[i] for i in indices]
            sub_t_t2c = [t_t2c_list[i] for i in indices]

            try:
                T_test = _opencv_solve(sub_R_g2b, sub_t_g2b, sub_R_t2c, sub_t_t2c,
                                       methods[best_name])
            except Exception:
                continue

            # 用全集计算内点
            all_errors = _compute_marker_consistency(T_test, T_base2end_list, T_cam2marker_list)
            inliers = [i for i, e in enumerate(all_errors) if e < inlier_threshold]

            if len(inliers) >= 6:
                inlier_mean_err = np.mean([all_errors[i] for i in inliers])
                # 优先选内点多的，其次选误差小的
                score = -len(inliers) * 1000 + inlier_mean_err
                if score < best_ransac_err:
                    best_ransac_err = score
                    best_ransac_T = T_test
                    best_inliers = inliers

        # 用内点重新标定
        if len(best_inliers) < n:
            outliers = set(range(n)) - set(best_inliers)
            log(f"  剔除异常值: {sorted(outliers)}")
            log(f"  保留内点: {len(best_inliers)}/{n}")

            inlier_R_g2b = [R_g2b_list[i] for i in best_inliers]
            inlier_t_g2b = [t_g2b_list[i] for i in best_inliers]
            inlier_R_t2c = [R_t2c_list[i] for i in best_inliers]
            inlier_t_t2c = [t_t2c_list[i] for i in best_inliers]
            inlier_base2end = [T_base2end_list[i] for i in best_inliers]
            inlier_cam2marker = [T_cam2marker_list[i] for i in best_inliers]

            try:
                best_T = _opencv_solve(inlier_R_g2b, inlier_t_g2b,
                                       inlier_R_t2c, inlier_t_t2c,
                                       methods[best_name])
                err_after = _compute_marker_consistency(best_T, inlier_base2end, inlier_cam2marker)
                log(f"  RANSAC 后误差: 平均 {np.mean(err_after)*1000:.2f} mm, "
                    f"最大 {np.max(err_after)*1000:.2f} mm")
                # 更新列表用于后续精化
                T_base2end_list = inlier_base2end
                T_cam2marker_list = inlier_cam2marker
            except Exception:
                log("  RANSAC 重标定失败，使用全集结果")
        else:
            log(f"  未发现明显异常值 (所有样本误差 < {inlier_threshold*1000:.0f} mm)")

    # ====== 第 3 步：非线性优化精化 ======
    log("\n--- 非线性优化精化 ---")
    err_before = _compute_marker_consistency(best_T, T_base2end_list, T_cam2marker_list)
    log(f"  精化前: 平均 {np.mean(err_before)*1000:.2f} mm")

    try:
        T_refined = _refine_with_optimization(best_T, T_base2end_list, T_cam2marker_list)
        err_after = _compute_marker_consistency(T_refined, T_base2end_list, T_cam2marker_list)
        log(f"  精化后: 平均 {np.mean(err_after)*1000:.2f} mm")

        # 只在精化后确实更好时才使用
        if np.mean(err_after) < np.mean(err_before):
            best_T = T_refined
            log(f"  ✅ 采用精化结果 (改善 {(np.mean(err_before)-np.mean(err_after))*1000:.2f} mm)")
        else:
            log(f"  ⚠️ 精化未改善，保留原始结果")
    except Exception as e:
        log(f"  精化失败: {e}，保留原始结果")

    return best_T


class EyeInHandCalibrator(Node):
    """眼在手上手眼标定节点"""
    
    def __init__(self):
        super().__init__('eye_in_hand_calibrator')
        
        # 0. 声明并获取参数
        self.declare_parameter('server_host', '192.168.1.11')
        self.declare_parameter('robot_id', 1)
        self.declare_parameter('camera_name', 'camera')
        
        self.server_host = self.get_parameter('server_host').get_parameter_value().string_value
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        
        # 如果是机器人2且IP是默认值，尝试自动调整为 192.168.1.10
        if self.robot_id == 2 and self.server_host == '192.168.1.11':
            self.server_host = '192.168.1.10'
            self.get_logger().info(f"自动将机器人 2 的 IP 设置为: {self.server_host}")

        # 配置参数
        self.min_points = 10  # 最少采集点数
        self.collected = 0
        self.calib_data = []  # 存储: (T_base2end, T_cam2marker)
        self.exit_flag = Event()
        
        # ArUco 订阅（与 launch 文件中的 aruco 节点输出话题一致）
        self.aruco_pose: Optional[Pose] = None
        self.aruco_sub = self.create_subscription(
            PoseArray, 'aruco/poses', self.aruco_cb, 10
        )
        
        # 机械臂 Socket
        self.robot_socket = None
        self.init_robot_socket()
        
        # 启动交互线程
        self.input_thread = Thread(target=self.user_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
        
        # 日志
        self.get_logger().info(f"===== 眼在手上（Eye-in-Hand）标定程序启动 (Robot {self.robot_id}) =====")
        self.get_logger().info(f"机器人 IP: {self.server_host}, 订阅话题: aruco/poses")
        self.get_logger().info("相机安装在机械臂末端，ArUco 标记固定在外部")
        self.get_logger().info(f"要求：采集 {self.min_points} 个不同姿态的数据点")
        self.get_logger().info("请在示教器上开启零力示教，手动拖动机械臂采集数据")

    def init_robot_socket(self) -> bool:
        """初始化机械臂 Socket"""
        if self.robot_socket:
            try:
                self.robot_socket.close()
            except:
                pass
        
        try:
            self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.robot_socket.settimeout(10)
            self.robot_socket.connect((self.server_host, 10003))
            
            # 尝试发送使能命令
            self.robot_socket.sendall("GrpEnable,0,;".encode('utf-8'))
            resp = self.robot_socket.recv(1024).decode('utf-8').strip()
            self.get_logger().info(f"机械臂连接成功 ({self.server_host}:10003)：{resp}")
            
            # 验证连接：尝试读取位姿
            import time
            time.sleep(0.3)
            self.robot_socket.sendall("ReadActPos,0,;".encode('utf-8'))
            test_resp = self.robot_socket.recv(1024).decode('utf-8').strip()
            if "OK" in test_resp:
                self.get_logger().info(f"通信验证成功：{test_resp[:60]}...")
            else:
                self.get_logger().warn(f"通信验证响应异常：{test_resp}")
            
            return True
        except Exception as e:
            self.get_logger().error(f"Socket 连接失败 ({self.server_host}:{10003}): {e}")
            self.robot_socket = None
            return False

    def send_command(self, cmd: str) -> str:
        """发送命令并返回响应，包含简单的重连逻辑"""
        for retry in range(2):
            if not self.robot_socket:
                if not self.init_robot_socket():
                    return ""
            
            try:
                self.robot_socket.sendall(cmd.encode('utf-8'))
                resp = self.robot_socket.recv(1024).decode('utf-8').strip()
                if not resp:
                    self.get_logger().warn(f"命令 {cmd.split(',')[0]} 返回空响应，重试...")
                    import time
                    time.sleep(0.5)
                    continue
                return resp
            except (socket.error, BrokenPipeError) as e:
                self.get_logger().warn(f"Socket 错误 ({e})，正在尝试重连...")
                self.robot_socket = None
                continue
            except Exception as e:
                self.get_logger().error(f"命令发送失败: {e}")
                return ""
        return ""


    def aruco_cb(self, msg: PoseArray):
        """缓存 ArUco 检测结果"""
        if len(msg.poses) > 0:
            self.aruco_pose = msg.poses[0]

    def get_marker_pose_in_camera(self) -> Optional[np.ndarray]:
        """
        获取 ArUco 标记在相机坐标系中的位姿
        返回: 4x4 齐次变换矩阵 T_cam2marker
        """
        if self.aruco_pose is None:
            self.get_logger().error("未检测到 ArUco 标记！")
            return None
        
        position = np.array([
            self.aruco_pose.position.x,
            self.aruco_pose.position.y,
            self.aruco_pose.position.z
        ])
        quaternion = np.array([
            self.aruco_pose.orientation.x,
            self.aruco_pose.orientation.y,
            self.aruco_pose.orientation.z,
            self.aruco_pose.orientation.w
        ])
        
        return pose_to_matrix(position, quaternion)

    def get_end_effector_pose(self) -> Optional[np.ndarray]:
        """
        获取末端执行器在基座坐标系中的位姿
        返回: 4x4 齐次变换矩阵 T_base2end
        """
        if not self.robot_socket:
            return None
        try:
            # 直接使用 socket 通信（与眼在手外标定保持一致）
            self.robot_socket.sendall("ReadActPos,0,;".encode('utf-8'))
            resp = self.robot_socket.recv(1024).decode('utf-8').strip()
            parts = resp.split(',')
            
            # 解析返回数据: ReadActPos,OK,j1,j2,j3,j4,j5,j6,x,y,z,rx,ry,rz,...
            if len(parts) < 14 or "OK" not in parts:
                self.get_logger().error(f"读取位姿失败: {resp}")
                return None
            
            # 提取笛卡尔位姿 (mm -> m, deg)
            x = float(parts[8]) / 1000.0
            y = float(parts[9]) / 1000.0
            z = float(parts[10]) / 1000.0
            rx = float(parts[11])
            ry = float(parts[12])
            rz = float(parts[13])
            
            # 构建变换矩阵
            T = np.eye(4)
            T[:3, :3] = rpy_to_matrix(np.array([rx, ry, rz]))
            T[:3, 3] = [x, y, z]
            
            return T
            
        except Exception as e:
            self.get_logger().error(f"获取末端位姿失败：{e}")
            return None

    def collect_point(self):
        """采集一组数据点"""
        # 1. 获取末端在基座系的位姿
        T_base2end = self.get_end_effector_pose()
        if T_base2end is None:
            return
        
        # 2. 获取标记在相机系的位姿
        T_cam2marker = self.get_marker_pose_in_camera()
        if T_cam2marker is None:
            return
        
        # 3. 保存数据
        self.calib_data.append((T_base2end.copy(), T_cam2marker.copy()))
        self.collected += 1
        
        pos = T_base2end[:3, 3]
        self.get_logger().info(f"采集第 {self.collected} 个点：")
        self.get_logger().info(f"  末端位置 (m): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        self.get_logger().info(f"  标记距离 (m): {T_cam2marker[2, 3]:.4f}")
        
        # 采集足够点后提示
        if self.collected >= self.min_points:
            self.get_logger().info(f"\n✅ 已采集 {self.min_points} 个点，可以输入 'c' 进行标定")

    def calibrate(self):
        """执行手眼标定"""
        if self.collected < 3:
            self.get_logger().error("至少需要 3 组数据才能标定！")
            return
        
        self.get_logger().info("\n===== 开始手眼标定 (Eye-in-Hand) =====")
        
        # 提取数据
        T_base2end_list = [d[0] for d in self.calib_data]
        T_cam2marker_list = [d[1] for d in self.calib_data]
        
        try:
            # 求解 T_cam2end (相机到末端的变换) - 使用改进的综合求解器
            T_cam2end = solve_hand_eye_all_methods(T_base2end_list, T_cam2marker_list, logger=self.get_logger())
            
            # 计算 T_end2cam (末端到相机的变换，用于发布 TF)
            T_end2cam = np.linalg.inv(T_cam2end)
            
            # 转换为位置和四元数
            pos_cam2end, quat_cam2end = matrix_to_pose(T_cam2end)
            pos_end2cam, quat_end2cam = matrix_to_pose(T_end2cam)
            
            # 输出结果
            self.get_logger().info("\n===== 标定结果 =====")
            self.get_logger().info(f"变换: camera_color_optical_frame → elfin_end_link")
            self.get_logger().info(f"T_cam2end:\n{T_cam2end}")
            self.get_logger().info(f"平移 (m): [{pos_cam2end[0]}, {pos_cam2end[1]}, {pos_cam2end[2]}]")
            self.get_logger().info(f"四元数 [x,y,z,w]: [{quat_cam2end[0]}, {quat_cam2end[1]}, {quat_cam2end[2]}, {quat_cam2end[3]}]")
            
            self.get_logger().info(f"\n反向变换: elfin_end_link → camera_color_optical_frame")
            self.get_logger().info(f"T_end2cam:\n{T_end2cam}")
            self.get_logger().info(f"平移 (m): [{pos_end2cam[0]}, {pos_end2cam[1]}, {pos_end2cam[2]}]")
            self.get_logger().info(f"四元数 [x,y,z,w]: [{quat_end2cam[0]}, {quat_end2cam[1]}, {quat_end2cam[2]}, {quat_end2cam[3]}]")
            
            # 验证精度 (使用改进的中位数参考方法)
            errors = _compute_marker_consistency(T_cam2end, T_base2end_list, T_cam2marker_list)
            self.get_logger().info(f"\n最终验证误差: 平均 {np.mean(errors)*1000:.2f} mm, 最大 {np.max(errors)*1000:.2f} mm")
            
            # 保存结果（不做 round，保留完整精度）
            result = {
                "calibration_type": "eye_in_hand",
                "transform": "camera_color_optical_frame → elfin_end_link",
                "T_cam2end": {
                    "matrix": T_cam2end.tolist(),
                    "translation_m": pos_cam2end.tolist(),
                    "quaternion_xyzw": quat_cam2end.tolist()
                },
                "T_end2cam": {
                    "matrix": T_end2cam.tolist(),
                    "translation_m": pos_end2cam.tolist(),
                    "quaternion_xyzw": quat_end2cam.tolist()
                },
                "validation": {
                    "mean_error_m": float(np.mean(errors)),
                    "max_error_m": float(np.max(errors)),
                    "sample_count": self.collected
                }
            }
            
            output_file = f"eye_in_hand_calib_result_robot{self.robot_id}.yaml"
            with open(output_file, "w") as f:
                yaml.dump(result, f, indent=4, default_flow_style=False)
            self.get_logger().info(f"\n✅ 结果已保存至 {output_file}")
            
        except Exception as e:
            self.get_logger().error(f"标定失败: {e}")
            import traceback
            traceback.print_exc()

    def validate_calibration(self, T_cam2end: np.ndarray, 
                            T_base2end_list: List[np.ndarray],
                            T_cam2marker_list: List[np.ndarray]) -> List[float]:
        """
        验证标定精度
        原理：T_base2marker = T_base2end @ T_end2cam @ T_cam2marker
             所有数据点计算出的 T_base2marker 应该一致（标记固定在世界系）
        """
        T_end2cam = np.linalg.inv(T_cam2end)
        T_base2marker_list = []
        
        for i in range(len(T_base2end_list)):
            T_base2marker = T_base2end_list[i] @ T_end2cam @ T_cam2marker_list[i]
            T_base2marker_list.append(T_base2marker)
        
        # 计算标记位置的一致性（以第一个为参考）
        ref_pos = T_base2marker_list[0][:3, 3]
        errors = []
        for T in T_base2marker_list[1:]:
            error = np.linalg.norm(T[:3, 3] - ref_pos)
            errors.append(error)
        
        return errors if errors else [0.0]

    def user_loop(self):
        """用户交互线程"""
        while not self.exit_flag.is_set():
            prompt = f"\n已采集 {self.collected}/{self.min_points} 个点\n"
            prompt += "输入: y=采集, c=标定, q=退出: "
            try:
                inp = input(prompt).strip().lower()
                if inp == 'q':
                    self.get_logger().info("退出程序")
                    if self.robot_socket:
                        self.robot_socket.close()
                    self.exit_flag.set()
                    break
                elif inp == 'y':
                    self.collect_point()
                elif inp == 'c':
                    self.calibrate()
                else:
                    self.get_logger().warn("请输入 y/c/q")
            except EOFError:
                break


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    calibrator = EyeInHandCalibrator()
    executor.add_node(calibrator)
    
    try:
        while not calibrator.exit_flag.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        calibrator.get_logger().info("手动终止")
    finally:
        calibrator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
