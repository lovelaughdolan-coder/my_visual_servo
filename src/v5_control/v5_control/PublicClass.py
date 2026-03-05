import configparser
import re
import numpy as np
import os

class Blinx_Public():
    def __init__(self, config_path: str = None):
        self.config = configparser.ConfigParser()
        
        # 配置文件路径：优先使用传入的路径，否则使用同目录下的 config.ini
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        
        # 读取配置文件
        config_exists = os.path.exists(config_path)
        if config_exists:
            self.config.read(config_path, encoding='utf-8')

        # region AGV变量 (使用安全读取方式)
        self.agv_ip = self._get_config('AGV_Config', 'agv_ip', '192.168.192.5')
        self.agv_navigation_port = int(self._get_config('AGV_Config', 'agv_navigation_port', '19206'))
        self.agv_push = self.config.get('AGV_Config', 'agv_push')   # agv实时数据获取接口
        self.receive_push = True
        self.current_station = "LM1"  # AGV站点数据
        self.blocked = None  # 赋值AGV是否被阻挡
        self.battery_level = None  # 机器人电量
        self.charging = None  # 机器人是否在充电
        self.DI = None  # DI数据
        self.DO = None  # DO数据
        self.brake = None  # 是否抱闸
        self.is_stop = None  # 机器人底盘是否禁止
        self.fatals = None  # 严重报警
        self.errors = None  # 报警
        self.warnings = None  # 警告
        self.x = None  # x坐标
        self.y = None  # y坐标
        self.angle = None  # 角度坐标
        self.emergency = None  # 急停状态
        self.agv_statu = "ready"   # AGV状态
        self.target_station = ""   # 用户下发站点
        self.agv_arrived = "no"   # AGV到达状态
        self.statu = True   # 实时数据制订状态
        # endregion

        # region 机械臂变量
        self.host = self.config.get('ipconfig', 'ip')   # 获取配置文件ip地址和port端口
        self.navigation_angle = list(eval(self.config.get('Robot_Config', 'navigation_angle')))   # 导航角度
        self.light_switch_coord = list(eval(self.config.get('Robot_Config', 'light_switch_coord')))   # 照明开关检测位置
        self.stored_energy_coord = list(eval(self.config.get('Robot_Config', 'stored_energy_coord')))   # 储能开关检测位置
        self.restoration_coord = list(eval(self.config.get('Robot_Config', 'restoration_coord')))   # 复位按钮检测位置
        self.charged_displayer_coord = list(eval(self.config.get('Robot_Config', 'charged_displayer_coord')))   # 带电显示器检测位置
        self.longRange_locally_coord = list(eval(self.config.get('Robot_Config', 'longRange_locally_coord')))   # 远程就地检测位置
        self.longRange_locally_coord2 = list(eval(self.config.get('Robot_Config', 'longRange_locally_coord2')))  # 远程就地检测位置
        self.divideShut_brake_coord = list(eval(self.config.get('Robot_Config', 'divideShut_brake_coord')))   # 分合闸检测位置
        self.breaker_coord = list(eval(self.config.get('Robot_Config', 'breaker_coord')))   # 断路器识别位置
        self.grounding_distance = list(eval(self.config.get('Robot_Config', 'grounding_distance')))   # 接地刀闸检测位置
        self.CK_device_coord = list(eval(self.config.get('Robot_Config', 'CK_device_coord')))   # 测控装置

        self.robot_now_coord = []   # 实时角度
        self.robot_now_angle = []   # 实时坐标
        self.robot_now_TCP_coord = []  # TCP坐标

        self.light_switch_m = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'light_switch_m'))   # 照明开关标定矩阵
        self.restoration_m = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'restoration_m'))   # 复位按钮标定矩阵
        self.stored_energy_m = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'stored_energy_m'))   # 储能按钮标定矩阵
        self.longRange_locally_m = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'longRange_locally_m'))   # 远程就地标定矩阵
        self.longRange_locally_m2 = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'longRange_locally_m2'))  # 远程就地标定矩阵
        self.divideShut_brake_m = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'divideShut_brake_m'))   # 分合闸标定矩阵
        self.charged_displayer_m = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'charged_displayer_m'))   # 验电显示器标定矩阵
        self.CK_device_m = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'CK_device_m'))  # 测控装置标定矩阵
        self.grounding_m = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'grounding_m'))   # 接地刀闸标定矩阵
        self.breaker_m = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'breaker_m'))   # 接地刀闸标定矩阵
        self.grounding_distance_m = self.blinx_strMatrix_to_Matrix(self.config.get('Robot_Config', 'grounding_distance_m'))   # 三坐标标定矩阵

        # endregion

        # region 云台相机变量
        self.visible_light_ip = self.config.get('Camera_Config', 'visible_light_ip')   # 可见光IP
        self.visible_light_port = self.config.get('Camera_Config', 'visible_light_port')   # 可见光端口
        self.visible_light_user = self.config.get('Camera_Config', 'visible_light_user')   # 可见光账户
        self.visible_light_pwd = self.config.get('Camera_Config', 'visible_light_pwd')   # 可见光密码

        self.thermal_imagery_ip = self.config.get('Camera_Config', 'thermal_imagery_ip')   # 热成像IP
        self.thermal_imagery_port = self.config.get('Camera_Config', 'thermal_imagery_port')   # 热成像端口
        self.thermal_imagery_user = self.config.get('Camera_Config', 'thermal_imagery_user')   # 热成像账户
        self.thermal_imagery_pwd = self.config.get('Camera_Config', 'thermal_imagery_pwd')   # 热成像密码
        # endregion

        # region 放电检测仪变量
        self.ele_serial_port = "COM5"   # 放电检测仪串口号
        self.ele_baud_rate = 9600   # 放电检测仪波特率
        # endregion

        # region 旋转轴变量
        self.mot_serial_port1 = "COM2"   # 旋转轴机器人末端串口号
        self.mot_baud_rate = 115200   # 旋转轴波特率
        self.mot_serial_port2 = "COM3"   # 旋转轴三坐标末端串口号
        # endregion

        # region 三坐标雷赛控制板变量
        self._CardID = 0  # 卡号
        self.dcurrent_speed = float()
        self.str_dcurrent_speed = str
        self.dunitPos = float()
        self.enPos = float()
        self.Axis_State = "停止中"
        # endregion
        self.ele_state = False

    # 矩阵转换
    def blinx_strMatrix_to_Matrix(self, strM):  # 字符串转2*3矩阵
        out = strM.replace('[', '').replace(']', '')  # 去掉中括号
        str = out.replace('\n', ' ')
        res = re.sub(' +', ' ', str)  # 去掉一个或多个空格
        # print("out:", out)
        dlist = res.strip(' ').split(' ')  # 转换成一个list
        listresult = []
        for i in range(0, len(dlist)):
            listresult.append(float(dlist[i]))  # 将字符串list转为float型的list
        # print("listresult:", listresult)
        darr = np.array(listresult)  # 将list转换为array
        # print("darr:", darr)
        resultM = darr.reshape(2, 3)  # 将array转换为2维(2,3)的矩阵
        # print("result:", resultM)
        return resultM