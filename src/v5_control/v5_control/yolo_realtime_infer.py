#!/usr/bin/env python3
"""
ROS2 + YOLO 实时推理节点
订阅奥比中光 Gemini 330 相机的彩色图像话题，使用 YOLO 模型进行实时检测/分割推理。
在 OpenCV 窗口中显示检测结果和实时 FPS。

用法:
    1. 启动相机: ros2 launch orbbec_camera gemini_330_series.launch.py
    2. 运行推理: python3 yolo_realtime_infer.py [--model yolo26n.pt] [--topic /camera/color/image_raw]
    
按 'q' 退出。
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloRealtimeNode(Node):
    def __init__(self, model_path, topic, imgsz, conf, iou, half, agnostic_nms, classes, show, show_masks, use_track, tracker, pub_topic, target_class):
        super().__init__('yolo_realtime_infer')
        self.get_logger().info(f'加载模型: {model_path}')
        self.model = YOLO(model_path)

        # 预热模型
        self.get_logger().info(f'模型预热中 (half={half})...')
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model(dummy, imgsz=imgsz, half=half, verbose=False)
        self.get_logger().info('模型预热完成！')

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.show = show
        self.show_masks = show_masks
        self.use_track = use_track
        self.tracker = tracker
        self.target_class = target_class
        self.bridge = CvBridge()

        # FPS 统计
        self.fps_window = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = time.time()

        # 目标像素坐标发布器
        self.pub_topic = pub_topic
        self.target_pub = self.create_publisher(PointStamped, pub_topic, 10)
        self.get_logger().info(f'发布目标像素坐标到: {pub_topic}')

        # 订阅相机话题
        self.get_logger().info(f'订阅话题: {topic}')
        self.sub = self.create_subscription(
            Image, topic, self.image_callback, 10
        )

    def image_callback(self, msg):
        t0 = time.time()

        # 将 ROS Image 转为 OpenCV 格式
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return

        # YOLO 推理（track 模式可平滑帧间抖动）
        common_args = dict(
            imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            half=self.half, agnostic_nms=self.agnostic_nms,
            classes=self.classes, verbose=False,
        )
        if self.use_track:
            results = self.model.track(frame, persist=True, tracker=self.tracker, **common_args)
        else:
            results = self.model(frame, **common_args)

        # 绘制结果 (支持控制是否绘制 masks)
        annotated = results[0].plot(masks=self.show_masks)

        # 提取最小 ID 目标并发布 (u, v)
        target_info = self._publish_target(results, annotated, msg.header.stamp)

        # 计算 FPS
        t1 = time.time()
        dt = t1 - t0
        self.fps_window.append(dt)
        fps_avg = len(self.fps_window) / sum(self.fps_window)
        self.frame_count += 1

        # 在画面上显示状态
        mode = f'Track({self.tracker})' if self.use_track else 'Detect'
        state_text = f'FPS: {fps_avg:.1f} | {mode} | Half: {self.half}'
        cv2.putText(annotated, state_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        target_str = f'ID:{target_info["id"]} ({target_info["u"]},{target_info["v"]})' if target_info else 'No target'
        cv2.putText(annotated, f'Infer: {dt*1000:.1f}ms | Target: {target_str}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 每秒在终端打印一次统计
        if self.frame_count % 30 == 0:
            elapsed = t1 - self.start_time
            overall_fps = self.frame_count / elapsed
            self.get_logger().info(
                f'帧数: {self.frame_count} | '
                f'平均FPS: {overall_fps:.1f} | '
                f'实时FPS: {fps_avg:.1f} | '
                f'推理时间: {dt*1000:.1f}ms'
            )

        if self.show:
            cv2.imshow('YOLO Realtime Inference', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('用户按下 q，退出...')
                cv2.destroyAllWindows()
                rclpy.shutdown()

    def _publish_target(self, results, annotated, stamp):
        """从跟踪结果中找目标类别里 ID 最小的物体，发布其 (u,v) 像素坐标"""
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        # 获取跟踪 ID
        if boxes.id is None:
            return None

        ids = boxes.id.cpu().numpy().astype(int)
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        # 筛选目标类别
        candidates = []
        for i in range(len(ids)):
            if cls_ids[i] == self.target_class:
                candidates.append((ids[i], i))

        if not candidates:
            return None

        # 找 ID 最小的
        candidates.sort(key=lambda x: x[0])
        min_id, idx = candidates[0]

        # 计算包围框中心 (u, v) 和面积
        x1, y1, x2, y2 = xyxy[idx]
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        area = float((x2 - x1) * (y2 - y1))

        # 发布 PointStamped (x=u, y=v, z=bbox_area)
        pt_msg = PointStamped()
        pt_msg.header.stamp = stamp
        pt_msg.header.frame_id = 'camera_color_optical_frame'
        pt_msg.point.x = float(u)
        pt_msg.point.y = float(v)
        pt_msg.point.z = area
        self.target_pub.publish(pt_msg)

        # 在画面上高亮该目标（蓝色十字 + 圆圈）
        color = (255, 100, 0)  # 蓝色
        cv2.drawMarker(annotated, (u, v), color, cv2.MARKER_CROSS, 30, 2)
        cv2.circle(annotated, (u, v), 20, color, 2)
        cv2.putText(annotated, f'T{min_id} A:{area:.0f}', (u + 25, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return {'id': min_id, 'u': u, 'v': v, 'area': area}


def main():
    parser = argparse.ArgumentParser(description='ROS2 YOLO 实时推理')
    parser.add_argument('--model', type=str, default='yolo26n.pt',
                        help='YOLO 模型路径')
    parser.add_argument('--topic', type=str, default='/camera/color/image_raw',
                        help='ROS2 图像话题')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理分辨率')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU 阈值')
    parser.add_argument('--half', action='store_true',
                        help='使用 FP16 半精度推理')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='跨类别 NMS，抑制不同类别间的重叠检测')
    parser.add_argument('--classes', nargs='+', type=int, default=None,
                        help='只检测指定类别，如 --classes 0 或 --classes 0 2')
    parser.add_argument('--no-track', action='store_true',
                        help='关闭跟踪模式（默认启用 ByteTrack 跟踪以减少闪烁）')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                        help='跟踪器配置 (bytetrack.yaml 或 botsort.yaml)')
    parser.add_argument('--no-masks', action='store_true',
                        help='显示分割结果时不绘制 mask (仅显示框)')
    parser.add_argument('--pub-topic', type=str, default='/yolo/target_pixel',
                        help='目标像素坐标发布话题')
    parser.add_argument('--target-class', type=int, default=0,
                        help='要跟踪的目标类别 ID（发布该类别中 ID 最小的物体的 u,v）')
    parser.add_argument('--no-show', action='store_true',
                        help='不显示 OpenCV 窗口')
    args = parser.parse_args()

    rclpy.init()
    node = YoloRealtimeNode(
        model_path=args.model,
        topic=args.topic,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        agnostic_nms=args.agnostic_nms,
        classes=args.classes,
        show=not args.no_show,
        show_masks=not args.no_masks,
        use_track=not args.no_track,
        tracker=args.tracker,
        pub_topic=args.pub_topic,
        target_class=args.target_class,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        total_time = time.time() - node.start_time
        avg_fps = node.frame_count / max(total_time, 1e-6)
        node.get_logger().info(f'总计处理 {node.frame_count} 帧, 平均 FPS: {avg_fps:.1f}')
        cv2.destroyAllWindows()
        node.destroy_node()


if __name__ == '__main__':
    main()
