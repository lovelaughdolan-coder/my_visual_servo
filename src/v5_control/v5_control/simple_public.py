#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版公共配置类
用于 robot_init.py 初始化脚本
"""

class SimplePublic:
    """简化的公共配置类，只包含必要的变量"""
    
    def __init__(self, host: str = '192.168.0.10'):
        """
        初始化
        
        Args:
            host: 机械臂 IP 地址
        """
        # 机械臂配置
        self.host = host
        
        # 实时数据（由 BlinxControl 的后台线程更新）
        self.robot_now_coord = []   # 实时笛卡尔坐标
        self.robot_now_angle = []   # 实时关节角度
        self.robot_now_TCP_coord = []  # TCP坐标
        
        # 状态标志
        self.robot_state_flag = []  # 状态标志列表
        self.robot_state = []       # 状态文本列表
        self.robot_controller_state = ''  # 控制器状态
        self.robot_state_inform = ''  # 状态机信息
        self.robot_speed = 25       # 速度百分比
        self.stop_flag = False      # 停止标志
        
        # 状态机信息映射 (来自原 PublicClass，扩展更多状态码)
        self.robot_state_info = {
            '0': '初始化',
            '1': '待机',
            '2': '运行中',
            '3': '暂停',
            '4': '停止',
            '5': '错误',
            '6': '急停',
            '33': '就绪',  # ReadCurFSM 可能返回的状态
            '34': '运动中',
            '35': '暂停中',
        }
