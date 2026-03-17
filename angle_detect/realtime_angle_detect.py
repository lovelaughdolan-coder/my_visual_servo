#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense 实时闸刀+旋钮角度检测

两条并行管线，共享同一个 YOLO 推理结果 (class 0 = 旋钮):
  1. 闸刀管线: 上下旋钮中心点 X 聚类配对 → SAM Point Prompt → 凸包刀尖角度
  2. 旋钮管线: 每个 YOLO BBox → SAM Box Prompt → 极坐标8扇区旋钮角度

用法:
  # RealSense 实时
  python3 realtime_angle_detect.py

  # 视频文件回退
  python3 realtime_angle_detect.py --video path/to/video.mp4
"""

import cv2
import torch
import numpy as np
import argparse
import math
import time
import os

from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


# ============================================================
#  旋钮角度算法 (来自 knob_video_segment.py)
# ============================================================

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


def draw_knob_angle(frame, result, knob_id=""):
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

    # 画 Y 轴参考线 (白色，向上)
    y_end = (cx, cy - r_vis)
    cv2.line(frame, (cx, cy), y_end, (255, 255, 255), 2)
    cv2.putText(frame, "Y", (cx + 5, cy - r_vis - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 画最近凸起连线 (黄色)
    cv2.line(frame, (cx, cy), (int(prot_pt[0]), int(prot_pt[1])), (0, 255, 255), 2)

    # 画角度弧线 (蓝色)
    prot_angle_deg = math.degrees(math.atan2(-(prot_pt[1] - cy), prot_pt[0] - cx))
    start_angle = -90  # Y 轴向上在 OpenCV 角度系统中是 -90°
    end_angle = -prot_angle_deg
    if end_angle < start_angle:
        start_angle, end_angle = end_angle, start_angle

    cv2.ellipse(frame, (cx, cy), (r_vis // 2, r_vis // 2),
                0, start_angle, end_angle, (255, 180, 0), 2)

    # 显示角度值
    label = f"{knob_id} {angle:.1f} deg" if knob_id else f"{angle:.1f} deg"
    cv2.putText(frame, label,
                (cx + r_vis // 2 + 5, cy - r_vis // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


# ============================================================
#  通用工具函数
# ============================================================

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


def ray_cast(start, direction, mask_uint8, w_img, h_img):
    """从 start 沿 direction 投射，统计 mask 内的距离"""
    pt = start.copy()
    dist = 0
    while 0 <= int(pt[0]) < w_img and 0 <= int(pt[1]) < h_img:
        if mask_uint8[int(pt[1]), int(pt[0])] == 0:
            break
        pt += direction
        dist += 1
    return dist


# ============================================================
#  闸刀角度解算 (来自 video_segment.py)
# ============================================================

def compute_switch_angle(frame, mask, top_pt, pivot_pt, sw_id):
    """
    对一个闸刀 mask 做凸包分析，计算刀尖与基准轴的夹角。
    返回角度文本字符串。
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 画基准线
    cv2.line(frame, tuple(top_pt), tuple(pivot_pt), (255, 0, 255), 2)
    cv2.circle(frame, tuple(top_pt), 5, (0, 255, 0), -1)
    cv2.circle(frame, tuple(pivot_pt), 7, (0, 200, 255), -1)

    angle_text = f"SW{sw_id}: N/A"
    if len(contours) == 0:
        return angle_text

    c = max(contours, key=cv2.contourArea)
    ref_top = np.array(top_pt)
    pivot_pt_np = np.array(pivot_pt)

    hull = cv2.convexHull(c)
    hull_points = hull.reshape(-1, 2)

    if len(hull_points) < 4:
        return angle_text

    # 寻找刀尖
    score_pts = hull_points[:, 0] * 5.0 - hull_points[:, 1]
    leftbottom_idx = np.argmin(score_pts)
    tip_raw = hull_points[leftbottom_idx]

    n_hull = len(hull_points)
    edges = []
    for idx_e in range(n_hull):
        p1 = hull_points[idx_e]
        p2 = hull_points[(idx_e + 1) % n_hull]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        edges.append({'dx': dx, 'dy': dy, 'length': np.hypot(dx, dy)})

    slanted_candidates = [e for e in edges if e['dx'] < -5]
    if slanted_candidates:
        long_slanted = max(slanted_candidates, key=lambda x: x['length'])
        l_len = long_slanted['length']
        nx, ny = -long_slanted['dy'] / l_len, long_slanted['dx'] / l_len
    else:
        nx, ny = 0.5, 0.866

    # 偏移距离估算
    v_up_axis = ref_top - pivot_pt_np
    len_up_axis = np.linalg.norm(v_up_axis)
    offset_dist = 20.0
    if len_up_axis > 0:
        v_up_norm = v_up_axis / len_up_axis
        v_perp = np.array([-v_up_norm[1], v_up_norm[0]])
        h_img, w_img = mask_uint8.shape
        eval_pt = ref_top.copy().astype(float)

        axis_width = (ray_cast(eval_pt, v_perp, mask_uint8, w_img, h_img)
                      + ray_cast(eval_pt, -v_perp, mask_uint8, w_img, h_img))
        if axis_width > 0:
            offset_dist = (axis_width / 2.0) * 0.85

    tip_pt = np.array([
        float(tip_raw[0]) + nx * offset_dist,
        float(tip_raw[1]) + ny * offset_dist
    ], dtype=int)

    # 绘图
    cv2.circle(frame, tuple(tip_raw), 3, (0, 255, 255), -1)
    cv2.arrowedLine(frame, tuple(pivot_pt), tuple(tip_pt), (0, 165, 255), 3, tipLength=0.1)
    cv2.circle(frame, tuple(tip_pt), 4, (0, 165, 255), -1)

    # 物理夹角
    v_up = ref_top - pivot_pt_np
    v_slant = tip_pt - pivot_pt_np

    if np.linalg.norm(v_up) > 0 and np.linalg.norm(v_slant) > 0:
        dot = np.dot(v_up, v_slant)
        cos_angle = np.clip(dot / (np.linalg.norm(v_up) * np.linalg.norm(v_slant)), -1.0, 1.0)
        angle_between = np.degrees(np.arccos(cos_angle))
    else:
        angle_between = 0.0

    angle_text = f"SW{sw_id}: {angle_between:.1f} deg"
    cv2.putText(frame, angle_text, (pivot_pt[0] + 20, pivot_pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return angle_text


# ============================================================
#  主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RealSense / Video 实时闸刀+旋钮角度联合检测"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="视频文件路径（不指定则使用 RealSense）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出视频路径（仅 video 模式有效）")
    parser.add_argument("--sam_weights", type=str,
                        default="no_ros/model/mobile_sam.pt",
                        help="MobileSAM 权重路径")
    parser.add_argument("--yolo_weights", type=str,
                        default="no_ros/model/2-26merged.pt",
                        help="YOLO 权重路径")
    args = parser.parse_args()

    # ---- 硬件探测与模型初始化 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading YOLO from {args.yolo_weights}...")
    yolo_model = YOLO(args.yolo_weights)

    print(f"Loading MobileSAM from {args.sam_weights}...")
    mobile_sam = sam_model_registry["vit_t"](checkpoint=args.sam_weights)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

    # ---- 输入源初始化 ----
    use_realsense = (args.video is None)

    pipeline = None
    align = None
    cap = None
    out_writer = None

    if use_realsense:
        if not HAS_REALSENSE:
            print("Error: pyrealsense2 未安装，请指定 --video 参数")
            return
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(rs_config)
        align = rs.align(rs.stream.color)
        print("RealSense 1280x720 ready.")
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: 无法打开视频 {args.video}")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    X_THRESHOLD = 80  # 同一闸刀上下旋钮的 X 坐标容差
    frame_count = 0
    t_start = time.time()

    print("开始处理… 按 q 退出")

    try:
        while True:
            # ---- 获取帧 ----
            if use_realsense:
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

            # ==================================================
            #  YOLO：一次推理获取所有旋钮 (class 0)
            # ==================================================
            results = yolo_model.predict(frame, conf=0.25, classes=[0], verbose=False)

            yolo_detections = []  # {'pt': [cx,cy], 'box': xyxy}
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    xyxy = box.xyxy.cpu().numpy()[0]
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)
                    yolo_detections.append({'pt': [cx, cy], 'box': xyxy})

            # ==================================================
            #  管线 A：闸刀 (X 聚类配对 → Point Prompt)
            # ==================================================
            yolo_detections.sort(key=lambda d: d['pt'][0])

            clusters = []
            if yolo_detections:
                current_cluster = [yolo_detections[0]]
                for i in range(1, len(yolo_detections)):
                    avg_x = sum(p['pt'][0] for p in current_cluster) / len(current_cluster)
                    if abs(yolo_detections[i]['pt'][0] - avg_x) < X_THRESHOLD:
                        current_cluster.append(yolo_detections[i])
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [yolo_detections[i]]
                clusters.append(current_cluster)

            switch_groups = []
            for cluster in clusters:
                if len(cluster) == 2:
                    cluster.sort(key=lambda d: d['pt'][1])
                    top_pt = cluster[0]['pt']
                    pivot_pt = cluster[1]['pt']
                    switch_groups.append({
                        'top': top_pt, 'pivot': pivot_pt,
                        'id': len(switch_groups) + 1
                    })

            # ==================================================
            #  管线 B：旋钮 (每个 BBox → Box Prompt)
            # ==================================================
            knob_prompts = [d['box'] for d in yolo_detections]

            # ==================================================
            #  SAM 编码 (仅当有目标时)
            # ==================================================
            has_targets = len(switch_groups) > 0 or len(knob_prompts) > 0
            if has_targets:
                predictor.set_image(frame_rgb)

            switch_angle_results = []
            knob_angle_results = []

            # ---- 闸刀分割 + 角度 ----
            for sg in switch_groups:
                top_pt = sg['top']
                pivot_pt = sg['pivot']

                points_np = np.array([top_pt, pivot_pt])
                labels_np = np.array([1, 1])

                masks, scores, _ = predictor.predict(
                    point_coords=points_np,
                    point_labels=labels_np,
                    multimask_output=True,
                )
                best_mask = masks[np.argmax(scores)]

                # 叠加闸刀 mask (蓝色调)
                apply_mask(frame, best_mask, [255, 140, 50], alpha=0.35)

                angle_text = compute_switch_angle(
                    frame, best_mask, top_pt, pivot_pt, sg['id']
                )
                switch_angle_results.append(angle_text)

            # ---- 旋钮分割 + 角度 ----
            for ki, box_xyxy in enumerate(knob_prompts):
                masks, scores, _ = predictor.predict(
                    box=box_xyxy,
                    multimask_output=True,
                )
                best_mask = masks[np.argmax(scores)]

                # 叠加旋钮 mask (绿色调)
                apply_mask(frame, best_mask, [50, 255, 120], alpha=0.35)

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
                            draw_knob_angle(frame, result, knob_id=f"K{ki}")
                            knob_angle_results.append(
                                f"K{ki}:{result['final_angle']:.1f}"
                            )

            # ==================================================
            #  HUD 信息叠加
            # ==================================================
            elapsed = max(time.time() - t_start, 1e-6)
            fps_current = (frame_count + 1) / elapsed

            info_str = (f"FPS: {fps_current:.1f} | "
                        f"SW: {len(switch_groups)} | "
                        f"Knobs: {len(knob_prompts)}")
            cv2.putText(frame, info_str, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            all_angles = switch_angle_results + knob_angle_results
            if all_angles:
                cv2.putText(frame, " | ".join(all_angles), (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ---- 输出 ----
            if out_writer:
                out_writer.write(frame)

            cv2.imshow("Realtime Angle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户退出。")
                break

            frame_count += 1

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


if __name__ == "__main__":
    main()
