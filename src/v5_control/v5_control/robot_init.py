#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blinx/Elfin 机械臂初始化脚本
直接使用 CCClient，简洁清晰
"""

import time
import argparse
from .CCClient import CCClient


def run_init_sequence(host: str = '192.168.0.10', speed_percent: float = 0.9,
                      waypoint0_joint: list = None, waypoint1_joint: list = None,
                      move_to_waypoints: bool = True):
    """
    执行初始化序列
    
    Args:
        host: 机械臂 IP 地址
        speed_percent: 速度百分比 (0-1)
        waypoint0_joint: 位置0 关节角度
        waypoint1_joint: 位置1 关节角度  
        move_to_waypoints: 是否执行移动到预设位置
    """
    # 默认位置 (与 elfin_init.cpp 一致)
    if waypoint0_joint is None:
        waypoint0_joint = [-114.0, 8.0, 120.0, 0.0, -20.0, 0.0]
    if waypoint1_joint is None:
        waypoint1_joint = [0.0, -46.0, 105.0, 0.0, -130.0, 0.0]
    
    waypoint0_pose = [-58.5, -700.5, 547.89, 48.6, -55.0, 15.5]
    waypoint1_pose = [-58.5, -700.5, 547.89, 48.6, -55.0, 15.5]

    # 创建客户端并连接
    cps = CCClient()
    print(f"正在连接到 {host}:10003...")
    cps.connectTCPSocket(host)
    print("连接成功！")

    try:
        # 1. 复位
        print("\n=== 步骤1: 复位 ===")
        ret = cps.HRIF_GrpReset()
        print(f"响应: {ret}")
        time.sleep(1.0)

        # 2. 使能
        print("\n=== 步骤2: 使能 ===")
        ret = cps.HRIF_GrpEnable()
        print(f"响应: {ret}")
        time.sleep(1.0)

        # 3. 设置速度
        print(f"\n=== 步骤3: 设置速度 ({speed_percent * 100}%) ===")
        ret = cps.SetOverride(speed_percent)
        print(f"响应: {ret}")
        time.sleep(0.5)

        # 4. 读取当前位置
        print("\n=== 步骤4: 读取当前位置 ===")
        pos = cps.Pose_ReadJoint()
        if pos and len(pos) >= 12:
            print(f"关节角度: {pos[:6]}")
            print(f"笛卡尔坐标: {pos[6:12]}")

        # 5. 移动到预设位置
        if move_to_waypoints:
            print("\n=== 步骤5: 移动到位置0 ===")
            print(f"  关节: {waypoint0_joint}")
            ret = cps.WayPoint(0, waypoint0_pose, waypoint0_joint, 
                               'TCP', 'Base', 100, 120, 0, 1, 0, 0, 0, 0)
            print(f"响应: {ret}")
            time.sleep(2.0)

            print("\n=== 步骤6: 移动到位置1 ===")
            print(f"  关节: {waypoint1_joint}")
            ret = cps.WayPoint(0, waypoint1_pose, waypoint1_joint,
                               'TCP', 'Base', 100, 120, 0, 1, 0, 0, 0, 0)
            print(f"响应: {ret}")
            time.sleep(1.0)

        print("\n========== 初始化完成 ==========")
        return True

    except Exception as e:
        print(f"初始化出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("关闭连接...")
        cps.closeTCPSocket()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='Blinx/Elfin 机械臂初始化')
    parser.add_argument('--host', type=str, default='192.168.0.10',
                        help='机械臂 IP (默认: 192.168.0.10)')
    parser.add_argument('--speed', type=float, default=0.9,
                        help='速度 0-1 (默认: 0.9)')
    parser.add_argument('--no-move', action='store_true',
                        help='不执行移动')
    
    args = parser.parse_args()
    run_init_sequence(host=args.host, speed_percent=args.speed, 
                      move_to_waypoints=not args.no_move)


if __name__ == '__main__':
    main()
