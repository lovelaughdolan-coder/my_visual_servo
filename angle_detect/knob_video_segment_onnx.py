#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旋钮角度实时检测 (YOLOv5/YOLOv8 ONNX 版本)

使用 onnxruntime 推理 YOLOv5/YOLOv8/YOLO26 ONNX 模型，搭配 MobileSAM 分割。
默认打开普通摄像头 (cam 0)，可选 --realsense 或 --video。
可用 --onnx_type 指定解析方式（默认 v5）。
"""

import cv2
import onnx
import onnxruntime as ort
import torch
import numpy as np
import argparse
import math
import time

from mobile_sam import sam_model_registry, SamPredictor

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


# ============================================================
#  YOLOv5/YOLOv8 ONNX 加载器
# ============================================================

class YoloV5Onnx:
    def __init__(self, onnx_path, half=False):
        self.half = half
        self.ratio = 1.0
        self.dw = 0.0
        self.dh = 0.0
        self.src_shape = None
        self.v8_boxes_in_original = False

        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("model error!")
        else:
            print("model success!")

        options = ort.SessionOptions()
        options.enable_profiling = False

        providers = [
            "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        ]
        self.onnx_session = ort.InferenceSession(
            onnx_path, sess_options=options, providers=providers
        )
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
        self.warm_up()

    def warm_up(self):
        for _ in range(3):
            input_numpy = np.empty(
                (1, 3, 640, 640), dtype=np.float16 if self.half else np.float32
            )
            input_feed = self.get_input_feed(input_numpy)
            _ = self.onnx_session.run(None, input_feed)[0]
        print("model warm up success!")

    def get_input_name(self):
        return [node.name for node in self.onnx_session.get_inputs()]

    def get_output_name(self):
        return [node.name for node in self.onnx_session.get_outputs()]

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]
        self.ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * self.ratio)), int(round(shape[0]) * self.ratio)
        self.dw = (new_shape[1] - new_unpad[0]) / 2
        self.dh = (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im

    def get_input_feed(self, image_numpy):
        return {name: image_numpy for name in self.input_name}

    def inference(self, image_bgr):
        self.src_shape = image_bgr.shape
        self.or_img = self.letterbox(image_bgr)
        img = self.or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = img.astype(dtype=np.half if self.half else np.float32)
        img /= 255.0
        img = img[None]
        input_feed = self.get_input_feed(img)
        start_time = time.time()
        pred = self.onnx_session.run(None, input_feed)[0]
        print("模型推理耗时:", time.time() - start_time)
        return pred

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def iou(self, a_box, b_box, isMin=False):
        if a_box.dtype == "float16" or b_box.dtype == "float16":
            a_box = a_box.astype(np.float32)
            b_box = b_box.astype(np.float32)
        a_box_area = (a_box[2] - a_box[0]) * (a_box[3] - a_box[1])
        b_box_area = (b_box[:, 2] - b_box[:, 0]) * (b_box[:, 3] - b_box[:, 1])

        xx1 = np.maximum(a_box[0], b_box[:, 0])
        yy1 = np.maximum(a_box[1], b_box[:, 1])
        xx2 = np.minimum(a_box[2], b_box[:, 2])
        yy2 = np.minimum(a_box[3], b_box[:, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        if isMin:
            ious = np.true_divide(inter, np.minimum(a_box_area, b_box_area))
        else:
            ious = np.true_divide(inter, (a_box_area + b_box_area - inter))
        return ious

    def nms(self, dets, thresh):
        if dets.shape[0] == 0:
            return np.array([])
        sort_index = dets[:, 4].argsort()[::-1]
        keep = []
        while sort_index.size > 0:
            keep.append(sort_index[0])
            box_a = dets[sort_index[0]]
            box_b = dets[sort_index[1:]]
            iou = self.iou(box_a, box_b)
            idx = np.where(iou <= thresh)[0]
            sort_index = sort_index[idx + 1]
        return keep

    def filter_box_v5(self, out_put, conf_threshold=0.25, iou_threshold=0.5, class_id=0):
        out_put = out_put[0]
        conf = out_put[:, 4] > conf_threshold
        box = out_put[conf == True]
        if box.shape[0] == 0:
            return []

        if box.shape[1] <= 5:
            return []

        cls_conf = box[..., 5:]
        cls = [int(np.argmax(cl)) for cl in cls_conf]
        all_cls = list(set(cls))

        output = []
        for curr_cls in all_cls:
            if curr_cls != class_id:
                continue
            curr_cls_box = []
            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])  # x y w h conf cls
            curr_cls_box = np.array(curr_cls_box)
            curr_cls_box = self.xywh2xyxy(curr_cls_box)
            idx = self.nms(curr_cls_box, iou_threshold)
            for k in idx:
                output.append(curr_cls_box[k])
        return output

    def filter_box_v8(self, out_put, conf_threshold=0.25, iou_threshold=0.5, class_id=0):
        pred = out_put[0]
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        if pred.shape[1] < 6:
            return []

        w_img = self.or_img.shape[1]
        h_img = self.or_img.shape[0]
        src_w = self.src_shape[1] if self.src_shape is not None else w_img
        src_h = self.src_shape[0] if self.src_shape is not None else h_img

        def try_denorm(coords, w, h):
            c = coords.copy()
            c[:, 0] *= w
            c[:, 2] *= w
            c[:, 1] *= h
            c[:, 3] *= h
            return c

        def xywh_to_xyxy(coords):
            xyxy = coords.copy()
            xyxy[:, 0] = coords[:, 0] - coords[:, 2] / 2
            xyxy[:, 1] = coords[:, 1] - coords[:, 3] / 2
            xyxy[:, 2] = coords[:, 0] + coords[:, 2] / 2
            xyxy[:, 3] = coords[:, 1] + coords[:, 3] / 2
            return xyxy

        def oob_ratio(xyxy, w, h):
            x1, y1, x2, y2 = xyxy.T
            invalid = (x2 < x1) | (y2 < y1) | (x1 < 0) | (y1 < 0) | (x2 > w) | (y2 > h)
            return np.mean(invalid)

        def score_candidate(coords, w, h, xywh=False):
            if xywh:
                coords = xywh_to_xyxy(coords)
            oob = oob_ratio(coords, w, h)
            x1, y1, x2, y2 = coords.T
            area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
            area_ratio = np.mean(area) / max(w * h, 1.0)
            return oob * 100 + area_ratio

        def select_best_coords(raw4):
            raw4 = raw4.copy()
            max_val = np.max(raw4)
            candidates = []
            if max_val <= 1.5:
                candidates.append((try_denorm(raw4, w_img, h_img), w_img, h_img, False, "denorm_lb_xyxy"))
                candidates.append((try_denorm(raw4, w_img, h_img), w_img, h_img, True, "denorm_lb_xywh"))
                candidates.append((try_denorm(raw4, src_w, src_h), src_w, src_h, False, "denorm_src_xyxy"))
                candidates.append((try_denorm(raw4, src_w, src_h), src_w, src_h, True, "denorm_src_xywh"))
            else:
                candidates.append((raw4, w_img, h_img, False, "lb_xyxy"))
                candidates.append((raw4, w_img, h_img, True, "lb_xywh"))
                candidates.append((raw4, src_w, src_h, False, "src_xyxy"))
                candidates.append((raw4, src_w, src_h, True, "src_xywh"))

            best = None
            best_score = None
            best_meta = None
            for coords, w, h, xywh, tag in candidates:
                xyxy = xywh_to_xyxy(coords) if xywh else coords
                score = score_candidate(xyxy, w, h, False)
                if best_score is None or score < best_score:
                    best_score = score
                    best = xyxy
                    best_meta = (w, h, tag)

            if best_meta is not None:
                self.v8_boxes_in_original = (best_meta[0] == src_w and best_meta[1] == src_h)
            return best

        # case: already NMS-ed output [x1,y1,x2,y2,score,cls]
        if pred.shape[1] == 6:
            boxes = pred
            raw4 = boxes[:, :4]
            coords = select_best_coords(raw4)
            boxes = np.concatenate([coords, boxes[:, 4:6]], axis=1)
            keep = boxes[:, 4] > conf_threshold
            boxes = boxes[keep]
            if boxes.shape[0] == 0:
                return []
            if class_id is not None:
                boxes = boxes[boxes[:, 5].astype(int) == class_id]
            if boxes.shape[0] == 0:
                return []
            boxes = boxes[:, :6]
            keep_idx = self.nms(boxes, iou_threshold)
            return [boxes[k] for k in keep_idx]

        # case: raw output [x,y,w,h,obj,cls...]
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        obj = pred[:, 4]
        cls_scores = pred[:, 5:]
        if cls_scores.shape[1] == 0:
            return []

        obj = sigmoid(obj)
        cls_scores = sigmoid(cls_scores)

        cls_id = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id] * obj

        keep = cls_conf > conf_threshold
        pred = pred[keep]
        cls_id = cls_id[keep]
        cls_conf = cls_conf[keep]
        if pred.shape[0] == 0:
            return []

        output = []
        curr_cls = class_id
        idxs = np.where(cls_id == curr_cls)[0]
        if idxs.size == 0:
            return []

        curr = pred[idxs]
        scores = cls_conf[idxs]
        coords = select_best_coords(curr[:, :4])
        boxes = np.concatenate(
            [coords, scores[:, None], np.full((idxs.size, 1), curr_cls)], axis=1
        )
        keep_idx = self.nms(boxes, iou_threshold)
        for k in keep_idx:
            output.append(boxes[k])
        return output

    def transform_coords(self, boxes, img_shape):
        transformed_boxes = []
        h, w = img_shape[:2]
        for box in boxes:
            x1 = int((box[0] - self.dw) / self.ratio)
            y1 = int((box[1] - self.dh) / self.ratio)
            x2 = int((box[2] - self.dw) / self.ratio)
            y2 = int((box[3] - self.dh) / self.ratio)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            transformed_boxes.append([x1, y1, x2, y2] + list(box[4:]))
        return np.array(transformed_boxes)


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
        description="Knob Angle Detection (YOLOv5/YOLOv8 ONNX + MobileSAM)"
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
                        default="no_ros/model/v5.onnx",
                        help="YOLO ONNX 权重路径")
    parser.add_argument("--onnx_type", type=str, default="v5", choices=["v5", "v8"],
                        help="ONNX 输出解析类型")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO 置信度阈值")
    parser.add_argument("--class_id", type=int, default=3,
                        help="指定检测类别 ID")
    parser.add_argument("--use-center", action="store_true",
                        help="使用画面中心代替 YOLO 检测")
    parser.add_argument("--no-letterbox", action="store_true",
                        help="v8 输出若已是原图坐标，则跳过 letterbox 反变换")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    yolo_model = None
    if not args.use_center:
        print(f"Loading YOLO ONNX from {args.yolo_weights} (type={args.onnx_type})...")
        yolo_model = YoloV5Onnx(args.yolo_weights)

    print(f"Loading MobileSAM from {args.sam_weights}...")
    mobile_sam = sam_model_registry["vit_t"](checkpoint=args.sam_weights)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

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

            target_boxes = []
            if args.use_center:
                pass
            else:
                pred = yolo_model.inference(frame)
                if args.onnx_type == "v8":
                    boxes = yolo_model.filter_box_v8(
                        pred, conf_threshold=args.conf, class_id=args.class_id
                    )
                else:
                    boxes = yolo_model.filter_box_v5(
                        pred, conf_threshold=args.conf, class_id=args.class_id
                    )
                if len(boxes) > 0:
                    if args.onnx_type == "v8" and yolo_model.v8_boxes_in_original:
                        boxes = np.array(boxes)
                    elif args.onnx_type == "v8" and args.no_letterbox:
                        boxes = np.array(boxes)
                    else:
                        boxes = yolo_model.transform_coords(boxes, frame.shape)
                else:
                    boxes = []
                for det in boxes:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) == args.class_id:
                        target_boxes.append(np.array([x1, y1, x2, y2]))

            has_targets = len(target_boxes) > 0 or args.use_center
            if has_targets:
                predictor.set_image(frame_rgb)

            angle_results = []

            if args.use_center:
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

            cv2.imshow("Knob Angle (ONNX)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户退出。")
                break

            frame_count += 1

    except Exception:
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
