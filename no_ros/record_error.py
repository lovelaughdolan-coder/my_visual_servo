#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBVS 精度记录工具

功能:
  1. 打开 RealSense 相机 + YOLO 检测
  2. 读取 target_config.json 中的目标参数 (u, v, area) 作为基准
  3. 连续采集若干帧取中位数，计算当前的 XY 像素误差和面积误差
  4. 将结果追加写入 experiment_log.json（支持多次实验）

用法:
  python3 record_error.py                         # 默认记录
  python3 record_error.py --tag "ibvs_run1"       # 带标签
  python3 record_error.py --tag "openloop_run1" --method pbvs  # 开环对照组
  python3 record_error.py --frames 30             # 采集 30 帧取中位数
"""

import cv2
import numpy as np
import json
import os
import time
import argparse
from datetime import datetime
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    print("⚠️ pyrealsense2 未安装，将使用 OpenCV 摄像头")

# ==========================================
# 配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', '2-26merged.pt')
TARGET_CLASS_ID = 0
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'target_config.json')
LOG_FILE = os.path.join(SCRIPT_DIR, 'experiment_log.json')


def load_target_config():
    """读取 select_target.py 保存的基准参数"""
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 未找到 {CONFIG_FILE}，请先运行 select_target.py")
        exit(1)
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    print(f"📌 基准参数: u={cfg['target_u']}, v={cfg['target_v']}, "
          f"area={cfg.get('target_area')}, depth={cfg.get('target_depth')}")
    return cfg


def detect_current_state(model, pipeline=None, cap=None, align=None,
                         temporal_filter=None, num_frames=20):
    """
    连续采集 num_frames 帧，对每帧做 YOLO 检测，
    取所有检测结果的中位数作为最终的 (u, v, area, depth)
    """
    all_u, all_v, all_area, all_depth = [], [], [], []

    print(f"📷 采集 {num_frames} 帧数据...")
    for i in range(num_frames):
        # 取图
        if HAS_REALSENSE and pipeline is not None:
            frames = pipeline.wait_for_frames()
            if align:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame:
                continue
            if temporal_filter and depth_frame:
                depth_frame = temporal_filter.process(depth_frame)
            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        else:
            ret, frame = cap.read()
            if not ret:
                continue
            depth_image = None

        # YOLO 检测 (不使用 track，避免 ID 跳动影响)
        results = model.predict(
            frame, conf=0.25, classes=[TARGET_CLASS_ID],
            verbose=False, half=True, imgsz=640
        )

        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            # 取置信度最高的检测
            confs = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs)
            xyxy = boxes.xyxy.cpu().numpy()[best_idx]

            x1, y1, x2, y2 = xyxy
            u = float((x1 + x2) / 2)
            v = float((y1 + y2) / 2)
            area = float((x2 - x1) * (y2 - y1))

            all_u.append(u)
            all_v.append(v)
            all_area.append(area)

            # 深度 (如果有的话)
            if depth_image is not None:
                cx, cy = int(u), int(v)
                h, w = depth_image.shape
                r = 5
                y_lo, y_hi = max(0, cy - r), min(h, cy + r)
                x_lo, x_hi = max(0, cx - r), min(w, cx + r)
                roi = depth_image[y_lo:y_hi, x_lo:x_hi].flatten()
                valid = roi[(roi > 100) & (roi < 1500)]
                if len(valid) > 0:
                    all_depth.append(float(np.median(valid)) / 1000.0)

        time.sleep(0.03)  # ~30fps

    if len(all_u) == 0:
        print("❌ 未检测到目标，请确认相机视野和 YOLO 模型")
        return None

    result = {
        'measured_u': float(np.median(all_u)),
        'measured_v': float(np.median(all_v)),
        'measured_area': float(np.median(all_area)),
        'measured_depth': float(np.median(all_depth)) if all_depth else None,
        'num_detections': len(all_u),
        'u_std': float(np.std(all_u)),
        'v_std': float(np.std(all_v)),
        'area_std': float(np.std(all_area)),
    }
    return result


def compute_errors(target_cfg, measurement):
    """计算各种误差"""
    err_u = measurement['measured_u'] - target_cfg['target_u']
    err_v = measurement['measured_v'] - target_cfg['target_v']
    err_xy = np.sqrt(err_u ** 2 + err_v ** 2)

    target_area = target_cfg.get('target_area')
    if target_area and target_area > 0:
        err_area_abs = measurement['measured_area'] - target_area
        err_area_ratio = err_area_abs / target_area * 100.0  # 百分比
    else:
        err_area_abs = None
        err_area_ratio = None

    target_depth = target_cfg.get('target_depth')
    if target_depth and measurement['measured_depth']:
        err_depth = measurement['measured_depth'] - target_depth
    else:
        err_depth = None

    return {
        'err_u': round(err_u, 2),
        'err_v': round(err_v, 2),
        'err_xy': round(err_xy, 2),
        'err_area_abs': round(err_area_abs, 1) if err_area_abs is not None else None,
        'err_area_ratio_pct': round(err_area_ratio, 2) if err_area_ratio is not None else None,
        'err_depth_m': round(err_depth, 4) if err_depth is not None else None,
    }


def append_to_log(record):
    """追加记录到 experiment_log.json"""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            log = json.load(f)
    else:
        log = []

    log.append(record)

    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"💾 已追加第 {len(log)} 条记录到 {LOG_FILE}")


def main():
    parser = argparse.ArgumentParser(description="IBVS 精度记录工具")
    parser.add_argument("--tag", type=str, default="", help="实验标签，如 ibvs_run1")
    parser.add_argument("--method", type=str, default="ibvs",
                        choices=["ibvs", "pbvs"],
                        help="控制方式: ibvs (闭环) / pbvs (开环)")
    parser.add_argument("--frames", type=int, default=20,
                        help="采集帧数，取中位数 (默认 20)")
    parser.add_argument("--log", type=str, default=None,
                        help="指定输出的 JSON 日志路径 (默认: experiment_log.json)")
    args = parser.parse_args()

    global LOG_FILE
    if args.log:
        LOG_FILE = args.log

    # 1. 加载基准
    target_cfg = load_target_config()

    # 2. YOLO
    print(f"Loading YOLO: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    model(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, half=True, verbose=False)
    print("YOLO ready.")

    # 3. 相机
    pipeline = None
    cap = None
    align = None
    temporal_filter = None

    if HAS_REALSENSE:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        temporal_filter = rs.temporal_filter()
        print("RealSense 1280x720 + D2C ready.")
        # 预热几帧
        for _ in range(10):
            pipeline.wait_for_frames()
    else:
        cap = cv2.VideoCapture(0)
        print("OpenCV camera ready.")

    # 4. 采集并计算
    try:
        measurement = detect_current_state(
            model, pipeline=pipeline, cap=cap,
            align=align, temporal_filter=temporal_filter,
            num_frames=args.frames
        )
    finally:
        if pipeline:
            pipeline.stop()
        if cap:
            cap.release()

    if measurement is None:
        print("❌ 采集失败")
        return

    # 5. 计算误差
    errors = compute_errors(target_cfg, measurement)

    # 6. 构建记录
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tag': args.tag,
        'method': args.method,
        'target': {
            'u': target_cfg['target_u'],
            'v': target_cfg['target_v'],
            'area': target_cfg.get('target_area'),
            'depth': target_cfg.get('target_depth'),
        },
        'measurement': measurement,
        'errors': errors,
    }

    # 打印结果
    print("\n" + "=" * 50)
    print(f"  实验记录  [{args.method.upper()}]  {args.tag}")
    print("=" * 50)
    print(f"  目标:  u={target_cfg['target_u']:.1f}  v={target_cfg['target_v']:.1f}  "
          f"area={target_cfg.get('target_area')}")
    print(f"  实测:  u={measurement['measured_u']:.1f}  v={measurement['measured_v']:.1f}  "
          f"area={measurement['measured_area']:.1f}")
    print(f"  ─────────────────────────────────")
    print(f"  XY像素误差:  err_u={errors['err_u']:.2f}  err_v={errors['err_v']:.2f}  "
          f"||err_xy||={errors['err_xy']:.2f} px")
    if errors['err_area_ratio_pct'] is not None:
        print(f"  面积误差:    {errors['err_area_abs']:.1f} px²  "
              f"({errors['err_area_ratio_pct']:.2f}%)")
    if errors['err_depth_m'] is not None:
        print(f"  深度误差:    {errors['err_depth_m']*1000:.1f} mm")
    print("=" * 50)

    # 7. 写入日志
    append_to_log(record)


if __name__ == "__main__":
    main()
