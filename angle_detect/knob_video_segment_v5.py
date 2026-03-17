#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旋钮角度实时检测 (YOLOv5 版本)

使用 torch.hub.load 加载 YOLOv5 模型，搭配 MobileSAM 分割。
默认打开普通摄像头 (cam 0)，可选 --realsense 或 --video。

首次运行需要联网下载 YOLOv5 仓库代码到 torch hub 缓存，之后离线可用。
如果遇到 GitHub API 限制，可以手动克隆：
  git clone https://github.com/ultralytics/yolov5 /tmp/yolov5
然后使用 --yolo_repo /tmp/yolov5 参数指定本地路径。
"""

import cv2
import torch
import numpy as np
import argparse
import math
import time
import os

from mobile_sam import sam_model_registry, SamPredictor

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


# ============================================================
#  YOLOv5 加载器
# ============================================================

def load_yolov5_model(weights_path, repo_path=None):
    """
    加载 YOLOv5 模型。
    - repo_path 为 None: 从 GitHub 下载 (首次需联网，之后缓存)
    - repo_path 非空: 从本地克隆的 yolov5 仓库加载
    """
    if repo_path and os.path.isdir(repo_path):
        print(f"Loading YOLOv5 from local repo: {repo_path}")
        model = torch.hub.load(repo_path, 'custom', path=weights_path, source='local')
    else:
        print("Loading YOLOv5 from GitHub (cached after first download)...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model


# ============================================================
#  旋钮角度算法
# ============================================================

def calculate_knob_angle(contour, center, radius):
    """
    计算菊花型旋钮角度（8个凸起，间隔45°）。
    找第一象限离Y轴最近的凸起，返回 0~45° 的相对角度。
    """
    cx, cy = center

    polar_data = []
    for point in contour:
        px, py = point[0]
        dx = px - cx
        dy = -(py - cy)
        dist = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        polar_data.append((angle, dist, (px, py)))

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


def draw_knob_angle(frame, result, knob_id=""):
    """在画面上可视化旋钮角度检测结果"""
    if result is None:
        return

    cx, cy = result['center']
    radius = result['radius']
    angle = result['final_angle']
    prot_pt = result['protrusion_point']
    r_vis = int(radius * 0.8)

    for p in result['all_protrusions']:
        pt = (int(p[2][0]), int(p[2][1]))
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    y_end = (cx, cy - r_vis)
    cv2.line(frame, (cx, cy), y_end, (255, 255, 255), 2)
    cv2.putText(frame, "Y", (cx + 5, cy - r_vis - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.line(frame, (cx, cy), (int(prot_pt[0]), int(prot_pt[1])), (0, 255, 255), 2)

    prot_angle_deg = math.degrees(math.atan2(-(prot_pt[1] - cy), prot_pt[0] - cx))
    start_angle = -90
    end_angle = -prot_angle_deg
    if end_angle < start_angle:
        start_angle, end_angle = end_angle, start_angle

    cv2.ellipse(frame, (cx, cy), (r_vis // 2, r_vis // 2),
                0, start_angle, end_angle, (255, 180, 0), 2)

    label = f"{knob_id} {angle:.1f} deg" if knob_id else f"{angle:.1f} deg"
    cv2.putText(frame, label,
                (cx + r_vis // 2 + 5, cy - r_vis // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def apply_mask(image, mask, color, alpha=0.5):
    """在原图上叠加透明 mask"""
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        colored_mask[:, :, c] = mask * color[c]
    mask_indices = mask > 0
    image[mask_indices] = (
        image[mask_indices] * (1 - alpha) + colored_mask[mask_indices] * alpha
    ).astype(np.uint8)
    return image


# ============================================================
#  主程序
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knob Angle Detection (YOLOv5 + MobileSAM)"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="视频文件路径（不指定则打开摄像头）")
    parser.add_argument("--realsense", action="store_true",
                        help="使用 RealSense 相机")
    parser.add_argument("--cam_id", type=int, default=0,
                        help="普通摄像头 ID (默认 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出视频路径（仅视频模式有效）")
    parser.add_argument("--sam_weights", type=str,
                        default="no_ros/model/mobile_sam.pt",
                        help="MobileSAM 权重路径")
    parser.add_argument("--yolo_weights", type=str,
                        default="no_ros/model/v5.pt",
                        help="YOLOv5 权重路径")
    parser.add_argument("--yolo_repo", type=str, default=None,
                        help="本地 yolov5 仓库路径（可选，避免联网）")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO 置信度阈值")
    parser.add_argument("--use-center", action="store_true",
                        help="使用画面中心代替 YOLO 检测")
    args = parser.parse_args()

    # ---- 模型初始化 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    yolo_model = None
    if not args.use_center:
        yolo_model = load_yolov5_model(args.yolo_weights, args.yolo_repo)
        yolo_model.conf = args.conf
        yolo_model.classes = [0]  # 只检测 class 0 (旋钮)
        print(f"YOLOv5 loaded. conf={args.conf}, classes=[0]")

    print(f"Loading MobileSAM from {args.sam_weights}...")
    mobile_sam = sam_model_registry["vit_t"](checkpoint=args.sam_weights)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

    # ---- 输入源初始化 ----
    pipeline = None
    align = None
    cap = None
    out_writer = None
    width, height = 0, 0

    if args.realsense:
        if not HAS_REALSENSE:
            print("Error: pyrealsense2 未安装")
            exit(1)
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(rs_config)
        align = rs.align(rs.stream.color)
        width, height = 1280, 720
        print("RealSense 1280x720 ready.")
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: 无法打开视频 {args.video}")
            exit(1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    else:
        cap = cv2.VideoCapture(args.cam_id)
        if not cap.isOpened():
            print(f"Error: 无法打开摄像头 {args.cam_id}")
            exit(1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {args.cam_id}: {width}x{height}")

    frame_count = 0
    t_start = time.time()
    print("开始处理… 按 q 退出")

    try:
        while True:
            # ---- 获取帧 ----
            if args.realsense:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---- YOLO 检测 ----
            target_boxes = []
            if args.use_center:
                # 无 YOLO，使用画面中心作为 SAM point prompt
                pass
            else:
                # YOLOv5 推理 — 返回格式是 pandas DataFrame 或 tensor
                # results.xyxy[0] → tensor: [x1, y1, x2, y2, conf, cls]
                results = yolo_model(frame)
                detections = results.xyxy[0].cpu().numpy()  # (N, 6)
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    target_boxes.append(np.array([x1, y1, x2, y2]))

            # ---- SAM 推理 ----
            has_targets = len(target_boxes) > 0 or args.use_center
            if has_targets:
                predictor.set_image(frame_rgb)

            angle_results = []

            if args.use_center:
                # 使用画面中心 point prompt
                points_np = np.array([[width // 2, height // 2]])
                labels_np = np.array([1])
                masks, scores, _ = predictor.predict(
                    point_coords=points_np,
                    point_labels=labels_np,
                    multimask_output=True,
                )
                best_mask = masks[np.argmax(scores)]
                rand_color = np.random.randint(0, 255, (3,)).tolist()
                apply_mask(frame, best_mask, rand_color, alpha=0.4)

                mask_uint8 = (best_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if len(contours) > 0:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 100:
                        (cx, cy), radius = cv2.minEnclosingCircle(c)
                        result = calculate_knob_angle(c, (cx, cy), radius)
                        if result:
                            draw_knob_angle(frame, result, knob_id="KNOB_0")
                            angle_results.append(f"K_0:{result['final_angle']:.1f}")
            else:
                # 每个 BBox → SAM Box Prompt → 旋钮角度
                for i, box_xyxy in enumerate(target_boxes):
                    masks, scores, _ = predictor.predict(
                        box=box_xyxy,
                        multimask_output=True,
                    )
                    best_mask = masks[np.argmax(scores)]

                    rand_color = np.random.randint(0, 255, (3,)).tolist()
                    apply_mask(frame, best_mask, rand_color, alpha=0.4)

                    mask_uint8 = (best_mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if len(contours) > 0:
                        c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(c) > 100:
                            (cx, cy), radius = cv2.minEnclosingCircle(c)
                            result = calculate_knob_angle(c, (cx, cy), radius)
                            if result:
                                draw_knob_angle(frame, result, knob_id=f"KNOB_{i}")
                                angle_results.append(
                                    f"K_{i}:{result['final_angle']:.1f}"
                                )

            # ---- HUD ----
            elapsed = max(time.time() - t_start, 1e-6)
            fps_current = (frame_count + 1) / elapsed
            n_knobs = len(target_boxes) if not args.use_center else 1
            info_str = f"FPS: {fps_current:.1f} | Knobs: {n_knobs}"
            cv2.putText(frame, info_str, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            if angle_results:
                cv2.putText(frame, " | ".join(angle_results), (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if out_writer:
                out_writer.write(frame)

            cv2.imshow("Knob Angle (YOLOv5)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户退出。")
                break

            frame_count += 1

    except Exception as e:
        import traceback
        traceback.print_exc()

    finally:
        if pipeline:
            pipeline.stop()
        if cap:
            cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()

    total_time = max(time.time() - t_start, 1)
    print(f"\n处理完成！共 {frame_count} 帧，平均 FPS: {frame_count / total_time:.2f}")
