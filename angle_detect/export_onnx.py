#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultralytics PT -> ONNX 导出工具

示例:
  python3 angle_detect/export_onnx.py --weights no_ros/model/v5.pt --imgsz 640
  python3 angle_detect/export_onnx.py --weights no_ros/model/2-26merged.pt --imgsz 640
"""

import argparse
from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Ultralytics PT to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="PT 权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="导出输入尺寸")
    parser.add_argument("--half", action="store_true", help="启用半精度导出")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset 版本")
    parser.add_argument("--output", type=str, default=None, help="输出 ONNX 路径")
    args = parser.parse_args()

    model = YOLO(args.weights)
    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "half": args.half,
    }
    if args.opset is not None:
        export_kwargs["opset"] = args.opset

    onnx_path = model.export(**export_kwargs)

    if args.output:
        import os
        import shutil
        if os.path.abspath(onnx_path) != os.path.abspath(args.output):
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            shutil.copy2(onnx_path, args.output)
            print(f"ONNX saved to {args.output}")
        else:
            print(f"ONNX saved to {onnx_path}")
    else:
        print(f"ONNX saved to {onnx_path}")
