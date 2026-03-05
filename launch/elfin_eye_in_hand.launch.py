"""
眼在手上可视化启动文件

同时启动：
1. RealSense 相机 (rs_launch.py + 眼在手标定 TF)
2. ArUco 位姿检测
3. 机械臂姿态可视化 (elfin_force_visualizer)
4. RViz2 (使用独立的 eye_in_hand_visualizer.rviz)
"""
import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    TimerAction,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ========================= 参数声明 =========================
    declare_robot_id = DeclareLaunchArgument(
        'robot_id', default_value='1',
        description='机器人ID: 1 或 2'
    )

    robot_id = LaunchConfiguration('robot_id')

    # 根据 robot_id 自动配置
    server_host = PythonExpression([
        "'192.168.1.11' if int(", robot_id, ") == 1 else '192.168.1.10'"
    ])
    camera_name = 'camera'

    # ========================= 1. 相机 (眼在手上) =========================
    realsense_share_dir = get_package_share_directory('realsense2_camera')
    rs_launch_path = os.path.join(realsense_share_dir, 'launch', 'rs_launch.py')
    camera_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_path),
        launch_arguments={
            'camera_name': camera_name,
            'camera_namespace': '',
            'enable_color': 'true',
            'enable_depth': 'true',
            'align_depth.enable': 'true',
            'rgb_camera.color_profile': '1280x720x30',
            'depth_module.depth_profile': '1280x720x30',
        }.items()
    )

    # ===== 眼在手标定 TF: elfin_end_link → camera_link =====
    # 标定值来自 rs_robot1.launch.py (T_end2link)
    def create_hand_eye_tf(context):
        child_frame = f'{camera_name}_link'
        return [
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='realsense_handeye_tf_publisher',
                arguments=[
                    '--x', '-0.04945240755722853',
                    '--y', '0.07260164957032905',
                    '--z', '0.01523068207767902',
                    '--qx', '0.26958026588320116',
                    '--qy', '0.661554901728171',
                    '--qz', '0.26303272288844876',
                    '--qw', '-0.6484484396894337',
                    '--frame-id', 'elfin_tcp_link',
                    '--child-frame-id', child_frame,
                ],
                output='screen',
            )
        ]

    # ========================= 2. ArUco 位姿检测 =========================
    aruco_share_dir = get_package_share_directory('aruco_pose_estimation')
    aruco_launch_path = os.path.join(
        aruco_share_dir,
        'launch',
        'aruco_pose_estimation.launch.py'
    )
    aruco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(aruco_launch_path),
        launch_arguments={
            'camera_name': camera_name,
            'use_realsense': 'false',  # 相机已由本 launch 启动，不要重复启动
        }.items()
    )

    # ========================= 3. 机械臂姿态可视化 =========================
    force_visualizer = Node(
        package='elfin_sdk_control_c',
        executable='elfin_force_visualizer',
        name='elfin_force_visualizer',
        output='screen',
        parameters=[{
            'server_host': server_host,
            'robot_id': PythonExpression(['int(', robot_id, ')']),
        }]
    )

    # ========================= 4. RViz2 =========================
    pkg_share = get_package_share_directory('elfin_sdk_control_c')
    rviz_config = os.path.join(pkg_share, 'rviz', 'eye_in_hand_visualizer.rviz')

    rviz_node = TimerAction(
        period=2.0,  # 延迟 2 秒，等待 TF 就绪
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config],
                output='screen',
            )
        ]
    )

    # ========================= 5. PBVS 控制器 (可选) =========================
    declare_enable_pbvs = DeclareLaunchArgument(
        'enable_pbvs', default_value='false',
        description='是否启动 PBVS 视觉伺服控制器'
    )
    declare_target_offset_z = DeclareLaunchArgument(
        'target_offset_z', default_value='0.15',
        description='目标偏移量 Z 方向 (m)'
    )

    enable_pbvs = LaunchConfiguration('enable_pbvs')
    target_offset_z = LaunchConfiguration('target_offset_z')

    from launch.conditions import IfCondition
    pbvs_node = Node(
        package='v5_control',
        executable='pbvs_controller_eye_in_hand',
        name='pbvs_controller_eye_in_hand',
        output='screen',
        condition=IfCondition(enable_pbvs),
        parameters=[{
            'robot_host': server_host,
            'robot_port': 10003,
            'target_offset_z': target_offset_z,
        }]
    )

    # ========================= 6. IBVS 控制器 (可选) =========================
    declare_enable_ibvs = DeclareLaunchArgument(
        'enable_ibvs', default_value='false',
        description='是否启动 IBVS 视觉伺服控制器'
    )

    enable_ibvs = LaunchConfiguration('enable_ibvs')

    ibvs_node = Node(
        package='v5_control',
        executable='ibvs_controller_eye_in_hand',
        name='ibvs_controller_eye_in_hand',
        output='screen',
        condition=IfCondition(enable_ibvs),
        parameters=[{
            'robot_host': server_host,
            'robot_port': 10003,
        }]
    )

    # ========================= 组装 =========================
    return LaunchDescription([
        declare_robot_id,
        declare_enable_pbvs,
        declare_target_offset_z,
        declare_enable_ibvs,
        camera_node,
        OpaqueFunction(function=create_hand_eye_tf),
        aruco_launch,
        force_visualizer,
        rviz_node,
        pbvs_node,
        ibvs_node,
    ])

