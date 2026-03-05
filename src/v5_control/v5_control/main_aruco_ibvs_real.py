import cv2
import numpy as np
from orbbec_camera import CameraOrbbec
from ibvs_algorithm import ibvs_control_law

# 仿真开关 - 设置为True启用真实机械臂控制，False仅做视觉验证
ENABLE_REAL_ROBOT = True

# 1. ArUco参数
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# 2. 目标特征（以图像中心为中心，宽高为120像素的矩形四个角点）
w, h = 120, 120
cx, cy = 327.718391, 238.751548  # 奥比中光相机主点
TARGET_POS = np.array([
    [cx - w//2, cy - h//2],  # 左上
    [cx + w//2, cy - h//2],  # 右上
    [cx + w//2, cy + h//2],  # 右下
    [cx - w//2, cy + h//2],  # 左下
])

# 3. 奥比中光相机参数
fx, fy = 465.102960, 466.864844
cx, cy = 327.718391, 238.751548
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# 畸变参数
dist_coeff = np.array([0.022622, -0.046539, -0.001964, 0.000786, 0.000000])

# 图像尺寸
camera_u, camera_v = 480, 640  # 注意：这里u是高度，v是宽度

Z = 0.5  # 假定深度0.5m

# 4. TCP偏移参数（机械臂末端到相机的变换）
tcp_offset = [-0.0088286041, -0.0860948001, 0.0429052466, 0.0, 0.0, 0.0]

# 5. RTDE控制接口（仅在启用真实机械臂时初始化）
rtde_c = None
if ENABLE_REAL_ROBOT:
    try:
        from rtde_control import RTDEControlInterface
        ROBOT_IP = "192.168.1.100"  # 请修改为实际机械臂IP地址
        rtde_c = RTDEControlInterface(ROBOT_IP)
        print(f"已连接到真实机械臂: {ROBOT_IP}")
        
        # 设置TCP偏移
        rtde_c.setTcp(tcp_offset)
        print(f"已设置TCP偏移: {tcp_offset}")
        
    except Exception as e:
        print(f"连接真实机械臂失败: {e}")
        print("将以纯视觉模式运行")
        ENABLE_REAL_ROBOT = False
        rtde_c = None
else:
    print("以纯视觉模式运行（未启用真实机械臂）")


def main():
    # 初始化奥比中光相机
    try:
        cam = CameraOrbbec(width=camera_v, height=camera_u, fps=30)
        print("奥比中光相机初始化成功")
    except Exception as e:
        print(f"奥比中光相机初始化失败: {e}")
        return
    
    print("按q退出...")
    try:
        while True:
            color_frame, depth_frame = cam.get_frames()
            if color_frame is None:
                print("未获取到相机帧")
                continue
                
            img = cam.frame_to_bgr_image(color_frame)
            if img is None:
                print("图像转换失败")
                continue
                
            # 畸变校正
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, (w, h), 1, (w, h))
            img_undist = cv2.undistort(img, K, dist_coeff, None, newcameramtx)
            
            gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray)

            if ids is not None and len(corners) > 0:
                pts = corners[0][0]  # shape: (4,2)
                for pt in pts:
                    cv2.circle(img_undist, tuple(pt.astype(int)), 5, (0,255,0), -1)
                current_pts = [tuple(pt) for pt in pts]
                desired_pts = [tuple(pt) for pt in TARGET_POS]
                depths = [Z] * 4
                xi = ibvs_control_law(current_pts, desired_pts, depths, K)
                v = xi[:3]
                omega = xi[3:]
                
                # IBVS坐标系修正：反转Z方向速度，使其与PBVS逻辑一致
                v[2] = -v[2]  # 反转Z方向速度
                
                print(f"原始IBVS线速度 (vx, vy, vz): {xi[:3]}")
                print(f"修正后线速度 (vx, vy, vz): {v}")
                print(f"角速度 (wx, wy, wz): {omega}")
                
                # 仅在启用真实机械臂时发送速度指令
                if ENABLE_REAL_ROBOT and rtde_c is not None:
                    try:
                        speed = np.concatenate([v, omega])
                        acceleration = 0.1  # 真实机械臂使用较小的加速度
                        time = 0.1
                        rtde_c.speedL(speed.tolist(), acceleration, time)
                    except Exception as e:
                        print(f"发送速度指令失败: {e}")
                
                for i in range(4):
                    cv2.line(img_undist, tuple(pts[i].astype(int)), tuple(TARGET_POS[i].astype(int)), (0,0,255), 2)
            else:
                print("未检测到ArUco标记")

            cv2.imshow("ArUco IBVS Demo (Orbbec)", img_undist)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        if ENABLE_REAL_ROBOT and rtde_c is not None:
            try:
                rtde_c.stopScript()
                print("已断开真实机械臂连接")
            except:
                pass

if __name__ == "__main__":
    main() 