#!/usr/bin/env python3
"""
YOLO 视频推理脚本
读取本地视频文件，使用 YOLO 模型进行实时检测/分割推理。
在 OpenCV 窗口中显示检测结果和实时 FPS，可选保存输出视频。

用法:
    python3 yolo_video_infer.py --model model/yolo26n-seg.pt --source video/xxx.mp4
    python3 yolo_video_infer.py --model model/2-13-mosaic.pt --source video/ --half --conf 0.5

按 'q' 退出，按 空格键 暂停/继续。
"""

import argparse
import glob
import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

import math


# ========= 旋钮角度检测 =========

def calculate_knob_angle(contour, center, radius):
    """
    计算菊花型旋钮角度（8个凸起，间隔45°）。
    找第一象限离Y轴最近的凸起，返回 0~45° 的相对角度。
    """
    cx, cy = center

    # 1. 转极坐标
    polar_data = []
    for point in contour:
        px, py = point[0]
        dx = px - cx
        dy = -(py - cy)  # 图像y轴向下 → 数学y轴向上
        dist = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        polar_data.append((angle, dist, (px, py)))

    # 2. 按8个扇区找凸起（距离最远的点）
    num_sectors = 8
    sector_size = 360 / num_sectors
    protrusions = []

    for i in range(num_sectors):
        sector_start = i * sector_size
        sector_end = (i + 1) * sector_size
        sector_points = [p for p in polar_data if sector_start <= p[0] < sector_end]
        if sector_points:
            farthest = max(sector_points, key=lambda x: x[1])
            if farthest[1] > radius * 0.65:
                protrusions.append(farthest)

    if not protrusions:
        return None

    # 3. 找第一象限内离 Y 轴 (90°) 最近的凸起
    first_quadrant = [p for p in protrusions if 0 <= p[0] <= 90]

    if first_quadrant:
        first_quadrant.sort(key=lambda x: abs(90 - x[0]))
        closest = first_quadrant[0]
        angle_to_y = 90 - closest[0]
    else:
        protrusions.sort(key=lambda x: abs(90 - x[0]))
        closest = protrusions[0]
        angle_to_y = 90 - closest[0]

    final_angle = abs(angle_to_y) % 45

    return {
        'final_angle': final_angle,
        'angle_to_y': abs(angle_to_y),
        'protrusion_point': closest[2],
        'all_protrusions': protrusions,
        'center': (int(cx), int(cy)),
        'radius': radius,
    }


def draw_knob_angle(frame, result):
    """在画面上可视化旋钮角度检测结果"""
    if result is None:
        return

    cx, cy = result['center']
    radius = result['radius']
    angle = result['final_angle']
    prot_pt = result['protrusion_point']
    r_vis = int(radius * 0.8)

    # 画所有凸起点 (绿色)
    for p in result['all_protrusions']:
        pt = (int(p[2][0]), int(p[2][1]))
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    # 画圆心
    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    # 画 Y 轴参考线 (白色虚线效果，向上)
    y_end = (cx, cy - r_vis)
    cv2.line(frame, (cx, cy), y_end, (255, 255, 255), 2)
    cv2.putText(frame, "Y", (cx + 5, cy - r_vis - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 画最近凸起连线 (黄色)
    cv2.line(frame, (cx, cy), (int(prot_pt[0]), int(prot_pt[1])), (0, 255, 255), 2)

    # 画角度弧线 (蓝色)
    # 在 OpenCV 中角度从 X 正方向开始顺时针, 但 ellipse 是逆时针
    # 我们需要从 Y 轴正方向(向上) 到凸起点之间画弧
    prot_angle_deg = math.degrees(math.atan2(-(prot_pt[1] - cy), prot_pt[0] - cx))
    start_angle = -90  # Y 轴向上在 OpenCV 角度系统中是 -90°
    end_angle = -prot_angle_deg
    # 确保弧线方向正确
    if end_angle < start_angle:
        start_angle, end_angle = end_angle, start_angle
    cv2.ellipse(frame, (cx, cy), (r_vis // 2, r_vis // 2),
                0, start_angle, end_angle, (255, 180, 0), 2)

    # 显示角度值 (大字)
    cv2.putText(frame, f"{angle:.1f} deg",
                (cx + r_vis // 2 + 5, cy - r_vis // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def process_knob_angles(frame, results, h, w, knob_class_id=0):
    """从 YOLO 分割结果中提取旋钮 mask，计算并可视化角度"""
    if results[0].masks is None or results[0].boxes is None:
        return []

    masks_data = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    angle_results = []
    for i, cls_id in enumerate(cls_ids):
        if cls_id != knob_class_id:
            continue

        # 获取该旋钮的 mask
        mask = masks_data[i]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_resized > 0.5).astype(np.uint8) * 255

        # 找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        # 取最大轮廓
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 100:
            continue

        # 最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        # 计算角度
        result = calculate_knob_angle(contour, (cx, cy), radius)
        if result:
            draw_knob_angle(frame, result)
            angle_results.append(result)

    return angle_results


# ========= 视频推理 =========

def infer_video(model, video_path, args):
    """对单个视频文件进行推理"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return

    # 视频信息
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{'='*60}")
    print(f"视频: {video_path}")
    print(f"分辨率: {w}x{h} | 帧率: {fps_video:.1f} | 总帧数: {total_frames}")
    print(f"{'='*60}")

    # 输出视频 writer
    writer = None
    if args.save:
        out_name = os.path.splitext(os.path.basename(video_path))[0] + "_result.mp4"
        out_path = os.path.join(os.path.dirname(video_path) or ".", out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_video, (w, h))
        print(f"输出保存到: {out_path}")

    # FPS 统计
    fps_window = deque(maxlen=30)
    frame_count = 0
    t_start = time.time()
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()

            # YOLO 推理
            results = model(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                half=args.half,
                agnostic_nms=args.agnostic_nms,
                classes=args.classes,
                verbose=False,
            )

            # 绘制结果
            show_masks = not args.no_masks
            annotated = results[0].plot(masks=show_masks)

            # 旋钮角度检测
            if args.knob_angle:
                knob_results = process_knob_angles(annotated, results, h, w, knob_class_id=args.knob_class)
                # 在 OSD 显示角度
                for ki, kr in enumerate(knob_results):
                    cv2.putText(annotated, f"Knob{ki}: {kr['final_angle']:.1f} deg",
                                (10, 90 + ki * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 提取二值 mask 并在独立窗口显示
            binary_mask_vis = None
            if args.show_binary_mask and results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy()  # (N, H, W)
                # 合并所有 mask 为一张二值图
                combined = np.zeros((h, w), dtype=np.uint8)
                for i, mask in enumerate(masks_data):
                    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    combined = np.maximum(combined, (resized > 0.5).astype(np.uint8) * 255)
                binary_mask_vis = combined

            # 计算 FPS
            t1 = time.time()
            dt = t1 - t0
            fps_window.append(dt)
            fps_avg = len(fps_window) / sum(fps_window)
            frame_count += 1

            # OSD 信息
            progress = f"{frame_count}/{total_frames}"
            cv2.putText(annotated, f"FPS: {fps_avg:.1f} | {progress}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Infer: {dt*1000:.1f}ms | Conf: {args.conf} | IoU: {args.iou}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 终端打印
            if frame_count % 30 == 0:
                elapsed = t1 - t_start
                overall_fps = frame_count / elapsed
                print(f"[{progress}] 平均FPS: {overall_fps:.1f} | 实时FPS: {fps_avg:.1f} | 推理: {dt*1000:.1f}ms")

            if writer:
                writer.write(annotated)

        if not args.no_show:
            cv2.imshow(f"YOLO - {os.path.basename(video_path)}", annotated)
            if binary_mask_vis is not None:
                cv2.imshow("Binary Mask", binary_mask_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户按下 q，退出...")
                break
            elif key == ord(' '):
                paused = not paused
                print("暂停" if paused else "继续")
        else:
            # 无显示模式也不用等待
            pass

    # 统计
    total_time = time.time() - t_start
    avg_fps = frame_count / max(total_time, 1e-6)
    print(f"\n完成！共处理 {frame_count} 帧, 耗时 {total_time:.1f}s, 平均 FPS: {avg_fps:.1f}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='YOLO 视频推理')
    parser.add_argument('--model', type=str, default='yolo26n.pt', help='YOLO 模型路径')
    parser.add_argument('--source', type=str, default='video/', help='视频文件或目录')
    parser.add_argument('--imgsz', type=int, default=640, help='推理分辨率')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--half', action='store_true', help='FP16 半精度推理')
    parser.add_argument('--agnostic-nms', action='store_true', help='跨类别 NMS')
    parser.add_argument('--classes', nargs='+', type=int, default=None,
                        help='只检测指定类别，如 --classes 0 或 --classes 0 2')
    parser.add_argument('--no-masks', action='store_true', help='不绘制 mask')
    parser.add_argument('--show-binary-mask', action='store_true', help='独立窗口显示二值 mask')
    parser.add_argument('--knob-angle', action='store_true', help='启用旋钮角度检测')
    parser.add_argument('--knob-class', type=int, default=0, help='旋钮类别 ID (默认 0)')
    parser.add_argument('--no-show', action='store_true', help='不显示窗口')
    parser.add_argument('--save', action='store_true', help='保存推理结果视频')
    args = parser.parse_args()

    # 加载模型
    print(f"加载模型: {args.model}")
    model = YOLO(args.model)

    # 预热
    print("模型预热中...")
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model(dummy, imgsz=args.imgsz, half=args.half, verbose=False)
    print("预热完成！")

    # 收集视频文件
    source = args.source
    video_exts = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
    if os.path.isdir(source):
        videos = sorted([os.path.join(source, f) for f in os.listdir(source)
                         if f.lower().endswith(video_exts)])
        if not videos:
            print(f"[错误] 目录 {source} 中没有找到视频文件")
            return
        print(f"找到 {len(videos)} 个视频文件")
    elif os.path.isfile(source):
        videos = [source]
    else:
        print(f"[错误] 路径不存在: {source}")
        return

    # 逐个处理
    for video_path in videos:
        infer_video(model, video_path, args)


if __name__ == '__main__':
    main()
