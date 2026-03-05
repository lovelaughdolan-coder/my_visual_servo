import time

import numpy as np
from .CCClient import CCClient
import threading

class BlinxControl:
    # region 初始化
    def __init__(self, public_class):
        print("机械臂初始化")
        self.public_class = public_class
        # # 构造机器人库的类
        self.cps = CCClient()
        # # 连接到 机器人的 IP 地址
        self.cps.connectTCPSocket(self.public_class.host)
        self.cps.HRIF_Connect2Controller()
        self.state = self.cps.readRobotState()
        print(self.state)
        self.power_status = int(self.state[11])
        print(self.power_status)
        self.cps.HRIF_ConnectToBox()

        self.cps.HRIF_Electrify()

        self.public_class.stop_flag = False
        # if not self.power_status:
        #     self.cps.HRIF_OSCmd(1)
        #     self.cps.HRIF_ConnectToBox()
        #     self.cps.HRIF_GrpReset()
        #     self.cps.HRIF_Electrify()

        # 设置初始速度百分比
        # self.blinx_speed(25)
        # 使能
        #self.cps.HRIF_GrpEnable()
        # 开启线程实时读取角度与坐标值
        #time.sleep(1)
        # 开线程，防止堵塞
        self.robot_thread = threading.Thread(target=self.blinx_robot_value_get)
        # # 初始化线程锁
        # self.lock = threading.Lock()

        self.robot_thread.daemon = True
        self.robot_thread.start()


    # endregion

    # region 获取当前机械臂角度和位置信息
    def blinx_init(self):
        try:
            # 获取当前角度和坐标
            ret = self.cps.Pose_ReadJoint()
            # print(1, ret)
            if ret and len(ret) > 8:
                if len(self.public_class.robot_now_angle) > 0:
                    self.public_class.robot_now_angle[0] = ret[0]
                    self.public_class.robot_now_angle[1] = ret[1]
                    self.public_class.robot_now_angle[2] = ret[2]
                    self.public_class.robot_now_angle[3] = ret[3]
                    self.public_class.robot_now_angle[4] = ret[4]
                    self.public_class.robot_now_angle[5] = ret[5]
                else:
                    self.public_class.robot_now_angle.append(ret[0])
                    self.public_class.robot_now_angle.append(ret[1])
                    self.public_class.robot_now_angle.append(ret[2])
                    self.public_class.robot_now_angle.append(ret[3])
                    self.public_class.robot_now_angle.append(ret[4])
                    self.public_class.robot_now_angle.append(ret[5])
                if len(self.public_class.robot_now_coord) > 0:
                    self.public_class.robot_now_coord[0] = ret[6]
                    self.public_class.robot_now_coord[1] = ret[7]
                    self.public_class.robot_now_coord[2] = ret[8]
                    self.public_class.robot_now_coord[3] = ret[9]
                    self.public_class.robot_now_coord[4] = ret[10]
                    self.public_class.robot_now_coord[5] = ret[11]
                else:
                    self.public_class.robot_now_coord.append(ret[6])
                    self.public_class.robot_now_coord.append(ret[7])
                    self.public_class.robot_now_coord.append(ret[8])
                    self.public_class.robot_now_coord.append(ret[9])
                    self.public_class.robot_now_coord.append(ret[10])
                    self.public_class.robot_now_coord.append(ret[11])
                #print(self.public_class.robot_now_coord)
        except Exception as e:
            print("实时数据获取错误："+str(e))
    # endregion

    # region 设置速度百分比
    def blinx_speed(self, speed):
        if 1 <= speed < 100:
            ret = self.cps.SetOverride(speed / 100)
            if ret[1] == "OK" and ret[0] == "SetOverride":
                return True
            else:
                print("速度控制失败", ret)

    # endregion

    # region 六个轴角度同时控制
    def blinx_move_joint_all(self, value1, value2, value3, value4, value5, value6):
        self.J1_angle = value1
        self.J2_angle = value2
        self.J3_angle = value3
        self.J4_angle = value4
        self.J5_angle = value5
        self.J6_angle = value6
        joints = [self.J1_angle, self.J2_angle, self.J3_angle, self.J4_angle, self.J5_angle, self.J6_angle]
        ret = self.cps.moveJ(joints)
        if ret[1] == "OK" and ret[0] == "moveJ":
            return True
        else:
            print("角度控制失败", ret)


    # endregion
    # region 六个轴坐标同时控制
    def blinx_move_coordinate_all(self, coord_list):
        self.X_axis = coord_list[0]
        self.Y_axis = coord_list[1]
        self.Z_axis = coord_list[2]
        self.Rx_axis = coord_list[3]
        self.Ry_axis = coord_list[4]
        self.Rz_axis = coord_list[5]
        coordinates = [self.X_axis, self.Y_axis, self.Z_axis, self.Rx_axis, self.Ry_axis, self.Rz_axis]
        ret = self.cps.moveL(coordinates)
        if ret[1] == "OK" and ret[0] == "MoveL":
            return True
        else:
            print("坐标控制失败", ret)
            return False

    # endregion

    # region 路点运动
    def blinx_move_waypoint(self, coord_list, joint_list, tcpname, move_type):
        """
            coord_list:空间坐标
            joint_list:关节坐标
            move_type:
                1：直线运动
                0：关节运动
            """
        # 直线运动
        if move_type:
            ret = self.cps.WayPoint(1, coord_list, joint_list, tcpname, 'Base', 400, 400, 0, 1, 0, 0, 0, 0)
            # print(ret)
            if ret[1] == "OK" and ret[0] == "WayPoint":
                return True
            else:
                print("坐标控制失败", ret)
                return False
        # 关节运动
        else:
            ret = self.cps.WayPoint(0, coord_list, joint_list, tcpname, 'Base', 100, 300, 0, 1, 0, 0, 0, 0)
            if ret[1] == "OK" and ret[0] == "WayPoint":
                return True
            else:
                print("角度控制失败", ret)
                return False
    # endregion
    # region 路点相对运动
    def blinx_move_waypoint_rel(self, move_type, Axis0, Axis1, Axis2, Axis3, Axis4, Axis5, tcpname):
        """
            move_type:运动类型
                0：关节相对运动
                1：直线相对运动
            Axis0~5: 轴的增量
            tcpname: TCP名称
            ps：x左减右加
                y前加后减
                z上加下减
        """
        # 创建一个列表来存储轴掩码
        AxisMask = [0, 0, 0, 0, 0, 0]

        # 检查每个轴的增量是否非零，并相应地设置轴掩码
        for i, axis in enumerate([Axis0, Axis1, Axis2, Axis3, Axis4, Axis5]):
            if axis != 0:
                AxisMask[i] = 1
        # 判断运动类型
        if move_type:
            speed = 400
        else:
            speed = 100

        ret = self.cps.WayPointRel(move_type, 0, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 1,
                                   AxisMask, Axis0, Axis1, Axis2, Axis3, Axis4, Axis5, tcpname, "Base", speed, 300,
                                   0, 1, 0, 0, 0, 0)
        if ret[1] == "OK" and ret[0] == "WayPointRel":
            return True
        else:
            print("相对运动控制失败", ret)
            return False

    # endregion

    # region连接控制器电箱
    def blinx_Connect_box(self):
        ret = self.cps.HRIF_ConnectToBox()
        if ret[1] == "OK" and ret[0] == "ConnectToBox":
            return True
        else:
            print("连接控制器电箱失败", ret)
            return False
    # endregion

    # 连接控制器
    def blinx_connect2controller(self):
        ret = self.cps.HRIF_Connect2Controller()
        if ret[1] == "OK" and ret[0] == "StartMaster":
            return True
        else:
            print("连接控制器失败", ret)
            return False

    def blinx_disconnect2controller(self):
        ret = self.cps.HRIF_Disonnect2Controller()
        if ret[1] == "OK" and ret[0] == "CloseMaster":
            return True
        else:
            print("断开控制器失败", ret)
            return False

    def blinx_read_controller_state(self):
        ret = self.cps.HRIF_ReadControllerState()
        if ret[1] == "OK" and ret[0] == "ReadControllerState":
            if ret[2]:
                return '已连接'
            else:
                return '未连接'
        else:
            print("读取控制器状态失败", ret)
            return False

    # 读取当前状态机
    def blinx_read_current_FSM(self):
        ret = self.cps.ReadCurFSM()
        # print(ret)
        if ret[1] == "OK" and ret[0] == "ReadCurFSM":
            info = self.public_class.robot_state_info[str(ret[2])]
            return info
        else:
            print("读取状态机失败", ret)
            return False

    # region 机器人使能控制
    # 使能
    def blinx_enable_robot(self):
        ret = self.cps.HRIF_GrpEnable()
        if ret[1] == "OK" and ret[0] == "GrpEnable":
            return True
        else:
            print("使能失败", ret)
            return False

    # 下使能
    def blinx_disable_robot(self):
        ret = self.cps.HRIF_GrpDisable()
        if ret[1] == "OK" and ret[0] == "GrpDisable":
            return True
        else:
            print("下使能失败", ret)
            return False
    # endregion

    # region 机器人上电
    def blinx_power_on(self):
        ret = self.cps.HRIF_Electrify()
        if ret[1] == "OK" and ret[0] == "Electrify":
            ret2 = self.cps.HRIF_ConnectToBox()
            if ret2[1] == "OK" and ret[0] == "ConnectToBox":
                return True
            else:
                print("连接控制器电箱失败", ret)
                return False
        else:
            print("上电失败", ret)
            return False
    # endregion

    # region 机器人下电
    def blinx_power_off(self):
        ret = self.cps.HRIF_BlackOut()
        if ret[1] == "OK" and ret[0] == "BlackOut":
            return True
        else:
            print("下电失败", ret)
            return False

    # endregion

    # region 机器人停止
    def blinx_stop_robot(self):
        ret = self.cps.stop()
        if ret[1] == "OK" and ret[0] == "GrpStop":
            self.public_class.stop_flag = True
            return True
        else:
            print("停止失败", ret)
            return False

    # endregion

    # region 机器人复位
    def blinx_reset(self):
        ret = self.cps.HRIF_GrpReset()
        if ret[1] == "OK" and ret[0] == "GrpReset":
            return True
        else:
            print("复位失败", ret)
            return False
    # endregion

    # region 以名称设置坐标系
    def blinx_set_Tcp(self, TcpName):
        """
            TCP:默认工具坐标
            TCP_claw：夹爪工具坐标
            TCP_stick：按钮杆工具坐标
            TCP_breaker：断路器执行器工具坐标
        """
        ret = self.cps.HRIF_SetTCPByName(TcpName)
        if ret[1] == "OK" and ret[0] == "SetTCPByName":
            return True
        else:
            print("设置坐标系失败", ret)
            return False
    # endregion

    # region 空间相对运动

    def blinx_relative_move_coord(self, AxisId, Dir, Dis, ToolMotion):
        """
            AxisId：坐标轴编号
                0~5，对应空间坐标X~Rz
            Dir：方向
                0：负方向
                1：正方向
            Dis：相对运动距离
            ToolMotion：运动坐标类型
                0：按当前选择的用户坐标运动
                1：按Tool坐标运动
        """
        ret = self.cps.HRIF_MoveRelL(AxisId, Dir, Dis, ToolMotion)
        if ret[1] == "OK" and ret[0] == "MoveRelL":
            return True
        else:
            print("空间相对运动失败", ret)
            return False
    # endregion

    # region 关节相对运动
    def blinx_relative_move_joint(self, AxisId, Dir, Dis):
        """
            AxisId：坐标轴编号
                0~5，对应空间坐标J1~J6
            Dir：方向
                0：负方向
                1：正方向
            Dis：相对运动距离
        """
        ret = self.cps.HRIF_MoveRelJ(AxisId, Dir, Dis)
        if ret[1] == "OK" and ret[0] == "MoveRelJ":
            return True
        else:
            print("关节相对运动失败", ret)
            return False
    # endregion

    # region 当前动作是否完成
    # def blinx_move_completed(self):
    #     while True:
    #         ret = self.cps.waitMoveDone()
    #         if ret is not None:
    #             print("运动未完成")
    #         else:
    #             print("运动已完成")
    #             self.blinx_init()
    #             time.sleep(0.1)
    #             return True
    def blinx_move_completed(self):
        time.sleep(0.3)
        while True:
            # print(self.public_class.stop_flag)
            if not self.public_class.stop_flag:
                if self.public_class.robot_state_flag[0] == 1:
                    continue
                elif self.public_class.robot_state_flag[0] == 0:
                    return True
            else:
                self.blinx_disable_robot()
                break
    # endregion

    # region 是否开启Tool坐标系运动模式
    def blinx_set_tool_motion_mode(self, state):
        ret = self.cps.HRIF_SetToolMotion(state)
        if ret[1] == "OK" and ret[0] == "SetToolMotion":
            return True
        else:
            print("获取Tool坐标系运动模式失败", ret)
            return False
    # endregion

    # region 机器人状态读取
    def blinx_read_status(self):
        ret = self.cps.readRobotState()
        if ret[1] == "OK" and ret[0] == "ReadRobotState":
            # print(ret)
            Move_state = int(ret[2])  # 运动状态
            Enable_state = int(ret[3])  # 使能状态
            Error_state = int(ret[4])  # 错误状态
            Pause_state = int(ret[8])  # 暂停状态
            Emergence_state = int(ret[9])  # 急停状态
            Electirfy_state = int(ret[11])  # 上电状态
            Pos_state = int(ret[14])  # 位置状态
            status_flag = [Move_state, Enable_state, Error_state, Pause_state, Emergence_state, Electirfy_state,
                           Pos_state]
            # print(status_flag)
            if Move_state:
                Move_state = "运动中"
            else:
                Move_state = "未运动"
            if Enable_state:
                Enable_state = "已使能"
            else:
                Enable_state = "未使能"
            if Error_state:
                Error_state = "有错误"
            else:
                Error_state = "无错误"
            if Pause_state:
                Pause_state = "已暂停"
            else:
                Pause_state = "未暂停"
            if Emergence_state:
                Emergence_state = "已急停"
            else:
                Emergence_state = "未急停"
            if Electirfy_state:
                Electirfy_state = "已上电"
            else:
                Electirfy_state = "未上电"
            if Pos_state:
                Pos_state = "已到位"
            else:
                Pos_state = "未到位"
            status_list = [Move_state, Enable_state, Error_state, Pause_state, Emergence_state, Electirfy_state,
                           Pos_state]
            return status_flag, status_list
        else:
            print("读取机器人状态失败", ret)
            return None


    # endregion

    # region 标定
    """
    标定转换
    x = 图像X坐标
    y = 图像Y坐标
    z = 深度值
    m = 标定矩阵
    """
    def blinx_calibration_side(self, x, y, m):
        coordinate = np.dot(m, [int(x), int(y), 1])  # 仿射逆变换，得到坐标（x,y)
        kx = int(coordinate[0])
        ky = int(coordinate[1])
        return kx, ky

    # endregion
    def blinx_robot_value_get(self):
        while True:
            # with self.lock:  # 确保线程安全
            self.blinx_init()
            self.public_class.robot_state_flag, self.public_class.robot_state = self.blinx_read_status() or (False, [])
            # print(self.public_class.robot_state_flag, self.public_class.robot_state)
            self.public_class.robot_controller_state = self.blinx_read_controller_state()
            self.public_class.robot_state_inform = self.blinx_read_current_FSM()
            # self.blinx_speed(self.public_class.robot_speed)
            # print(self.public_class.robot_state_flag, self.public_class.robot_controller_state, self.public_class.robot_state_inform)
    # region 退出
    def blinx_close(self):
        self.cps.closeTCPSocket()
    # endregion


