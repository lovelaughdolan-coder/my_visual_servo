#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标像素选择工具

功能:
  1. 打开 RealSense 相机 + YOLO 检测/追踪
  2. 画面上显示所有检测到的目标 ID 和中心坐标
  3. 按键盘数字键 (0-9) 选择对应 ID 的目标
  4. 选中后自动保存该目标的 (u, v, depth) 到 target_config.json
  5. main_depth_ibvs.py 启动时会自动读取这个配置文件

用法:
  python3 select_target.py
  → 按数字键选择 ID → 按 'c' 确认保存 → 按 'q' 退出
"""

import cv2
import numpy as np
import json
import os
import time
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

# ==========================================
# 配置
# ==========================================
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', '2-26merged.pt')
TARGET_CLASS_ID = 0
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'target_config.json')

# ==========================================


def main():
    print("=========================================")
    print(" 目标像素选择工具")
    print("=========================================")
    print(" 操作说明:")
    print("   数字键 0-9 : 选择对应 ID 的目标")
    print("   c          : 确认并保存当前选择")
    print("   r          : 重置选择")
    print("   q          : 退出")
    print("=========================================")

    # YOLO
    print(f"Loading: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    model(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, half=True, verbose=False)
    print("YOLO ready.")

    # RealSense
    if HAS_REALSENSE:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        temporal_filter = rs.temporal_filter()
        print("RealSense 1280x720 + D2C ready.")
    else:
        cap = cv2.VideoCapture(0)
        print("OpenCV camera (no depth)")

    selected_id = None       # 当前选中的 ID
    selected_u = None
    selected_v = None
    selected_depth = None
    selected_area = None
    confirmed = False

    try:
        while True:
            t0 = time.time()

            # 取图
            if HAS_REALSENSE:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                depth_frame = temporal_filter.process(depth_frame)
                frame = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                depth_image = None

            # YOLO
            results = model.track(
                frame, persist=True, tracker='bytetrack.yaml',
                conf=0.25, iou=0.45, half=True, imgsz=640,
                classes=[TARGET_CLASS_ID], verbose=False,
            )
            annotated = results[0].plot()

            # 提取所有目标
            boxes = results[0].boxes
            targets = {}  # id -> (u, v, depth, area, bbox)

            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()

                for i in range(len(ids)):
                    if cls_ids[i] == TARGET_CLASS_ID:
                        tid = int(ids[i])
                        x1, y1, x2, y2 = xyxy[i]
                        u = float((x1 + x2) / 2)
                        v = float((y1 + y2) / 2)
                        area = float((x2 - x1) * (y2 - y1))

                        # 深度
                        depth_m = None
                        if depth_image is not None:
                            cx, cy = int(u), int(v)
                            h, w = depth_image.shape
                            # 取中心 5x5 区域中位数
                            r = 5
                            y_lo, y_hi = max(0, cy - r), min(h, cy + r)
                            x_lo, x_hi = max(0, cx - r), min(w, cx + r)
                            roi = depth_image[y_lo:y_hi, x_lo:x_hi].flatten()
                            valid = roi[(roi > 100) & (roi < 1500)]  # 10cm~1.5m
                            if len(valid) > 0:
                                depth_m = float(np.median(valid)) / 1000.0

                        targets[tid] = (u, v, depth_m, area, (x1, y1, x2, y2))

            # 在画面上标注所有目标信息
            for tid, (u, v, d, a, bbox) in targets.items():
                x1, y1, x2, y2 = [int(c) for c in bbox]
                is_selected = (tid == selected_id)
                color = (0, 255, 0) if is_selected else (200, 200, 200)
                thickness = 3 if is_selected else 1

                # 画边框
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                # 画中心十字
                cv2.drawMarker(annotated, (int(u), int(v)), color,
                               cv2.MARKER_CROSS, 20, 2)

                # 标注文字
                d_str = f"Z={d:.3f}m" if d else "Z=N/A"
                label = f"ID:{tid} ({int(u)},{int(v)}) {d_str} A={a:.0f}"
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 如果被选中，额外高亮
                if is_selected:
                    selected_u = u
                    selected_v = v
                    selected_depth = d
                    selected_area = a
                    cv2.putText(annotated, ">>> SELECTED <<<",
                                (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

            # 状态栏
            dt = max(time.time() - t0, 1e-6)
            status_lines = [
                f"FPS:{1/dt:.0f} | Targets:{len(targets)}",
            ]
            if selected_id is not None:
                d_str = f"{selected_depth:.3f}m" if selected_depth else "N/A"
                status_lines.append(
                    f"Selected: ID={selected_id} "
                    f"u={selected_u:.0f} v={selected_v:.0f} depth={d_str}"
                )
                status_lines.append("Press 'c' to CONFIRM and save")
            else:
                status_lines.append("Press 0-9 to select target ID")

            if confirmed:
                status_lines.append(">>> SAVED to target_config.json <<<")

            for i, line in enumerate(status_lines):
                cv2.putText(annotated, line, (10, 30 + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Select Target", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # 数字键选择 ID
            if ord('0') <= key <= ord('9'):
                target_id = key - ord('0')
                if target_id in targets:
                    selected_id = target_id
                    confirmed = False
                    print(f"🎯 选中 ID={target_id}")
                else:
                    print(f"⚠️ ID={target_id} 当前不在画面中")

            # 确认保存
            if key == ord('c') and selected_id is not None and selected_u is not None:
                config = {
                    'target_id': selected_id,
                    'target_u': round(selected_u, 1),
                    'target_v': round(selected_v, 1),
                    'target_depth': round(selected_depth, 4) if selected_depth else None,
                    'target_area': round(selected_area, 1) if selected_area else None,
                    'note': f'Selected from YOLO ID={selected_id}',
                }
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                confirmed = True
                print(f"✅ 已保存到 {CONFIG_FILE}:")
                print(f"   u={config['target_u']}, v={config['target_v']}, "
                      f"depth={config['target_depth']}, area={config['target_area']}")

            # 重置
            if key == ord('r'):
                selected_id = None
                selected_u = selected_v = selected_depth = selected_area = None
                confirmed = False
                print("🔄 已重置选择")

    finally:
        if HAS_REALSENSE:
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
