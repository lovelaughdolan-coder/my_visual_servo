import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor
import math
import time
import os

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

def calculate_knob_angle(contour, center, radius):
    """
    源自 yolo_video_infer:
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

    # 画 Y 轴参考线 (白色虚线效果，向上)
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

    # 显示角度值 (大字)
    label = f"{knob_id} {angle:.1f} deg" if knob_id else f"{angle:.1f} deg"
    cv2.putText(frame, label,
                (cx + r_vis // 2 + 5, cy - r_vis // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def apply_mask(image, mask, color, alpha=0.5):
    """通过权重合成在原图上叠加透明 mask"""
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        colored_mask[:, :, c] = mask * color[c]
    
    mask_indices = mask > 0
    image[mask_indices] = (image[mask_indices] * (1 - alpha) + colored_mask[mask_indices] * alpha).astype(np.uint8)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single/Multi Knob Video/Camera Angle Detection using MobileSAM")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--realsense", action="store_true", help="Use RealSense camera")
    parser.add_argument("--cam_id", type=int, default=0, help="Standard OpenCV camera ID")
    parser.add_argument("--output", type=str, default="knob_result.mp4", help="Path to output video file (only for video mode)")
    parser.add_argument("--sam_weights", type=str, default="no_ros/model/mobile_sam.pt", help="Path to SAM weights")
    parser.add_argument("--yolo_weights", type=str, default="no_ros/model/2-26merged.pt", help="Path to YOLO weights")
    parser.add_argument("--use-center", action="store_true", help="Use video center point directly instead of YOLO")
    parser.add_argument("--play", action="store_true", help="Play video while processing (only for video mode)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    yolo_model = None
    if not args.use_center:
        print(f"Loading YOLO model from {args.yolo_weights}...")
        yolo_model = YOLO(args.yolo_weights)
    else:
        print("YOLO disabled (--use-center is active). Foreground points will be the screen center.")
    
    print(f"Loading MobileSAM from {args.sam_weights}...")
    model_type = "vit_t"
    mobile_sam = sam_model_registry[model_type](checkpoint=args.sam_weights)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

    # ---- 输入源探测 ----
    pipeline = None
    align = None
    cap = None
    width, height = 0, 0
    fps = 30
    total_frames = -1

    if args.realsense:
        if not HAS_REALSENSE:
            print("Error: pyrealsense2 is not installed.")
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
            print(f"Error: Could not open video {args.video}")
            exit(1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
    else:
        # 默认普通摄像头
        cap = cv2.VideoCapture(args.cam_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.cam_id}")
            exit(1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {args.cam_id} ready at {width}x{height}")

    out = None
    if args.video and args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"Starting processing. Press 'q' to quit.")
    pbar = None
    if args.video:
        pbar = tqdm(total=total_frames)
    
    frame_count = 0
    t_start_all = time.time()

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
            
            # 1. 确定我们要检测的目标特征(点或框)列表
            target_prompts = []
            if args.use_center:
                # 采用屏幕正中心，类型标识为 'point'
                target_prompts.append({'type': 'point', 'data': np.array([[width // 2, height // 2]])})
            else:
                # 采用 YOLO 对象检测得到的 bbox
                results = yolo_model.predict(frame, conf=0.25, classes=[0], verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        xyxy = box.xyxy.cpu().numpy()[0]
                        target_prompts.append({'type': 'box', 'data': xyxy})

            # 2. SAM 推理
            if len(target_prompts) > 0:
                predictor.set_image(frame_rgb)
            
            angle_results = []
            
            # 3. 为每个特征单独进行掩码推断和物理夹角计算
            for i, prompt in enumerate(target_prompts):
                if prompt['type'] == 'point':
                    points_np = prompt['data']
                    labels_np = np.array([1]) # 单点 Prompt，默认全都是前景点
                    masks, scores, logits = predictor.predict(
                        point_coords=points_np,
                        point_labels=labels_np,
                        multimask_output=True,
                    )
                elif prompt['type'] == 'box':
                    box_np = prompt['data']
                    masks, scores, logits = predictor.predict(
                        box=box_np,
                        multimask_output=True,
                    )
            
                # 使用最自信的一张蒙版
                best_mask = masks[np.argmax(scores)]
                
                # 用随机颜色贴遮罩
                rand_color = np.random.randint(0, 255, (3,)).tolist()
                apply_mask(frame, best_mask, rand_color, alpha=0.4)
                
                # 找外轮廓并寻找外接圆
                mask_uint8 = (best_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 100:
                        (cx, cy), radius = cv2.minEnclosingCircle(c)
                        
                        # 使用 yolo_video_infer.py 中的菊花检测逻辑
                        result = calculate_knob_angle(c, (cx, cy), radius)
                        if result:
                            draw_knob_angle(frame, result, knob_id=f"KNOB_{i}")
                            angle_results.append(f"K_{i}:{result['final_angle']:.1f}")

            # 4. 显示信息与保存
            fps_current = (frame_count + 1) / max((time.time() - t_start_all), 0.001)
            info_str = f"FPS: {fps_current:.1f} | Knobs: {len(target_prompts)}"
            cv2.putText(frame, info_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            if angle_results:
                cv2.putText(frame, " | ".join(angle_results), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if out:
                out.write(frame)
            
            cv2.imshow("Knob Angle Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nUser interrupted playback.")
                break
                
            frame_count += 1
            if pbar:
                pbar.update(1)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if pbar:
            pbar.close()
        if pipeline:
            pipeline.stop()
        if cap:
            cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
            
        print(f"\nProcessing complete! Average FPS: {frame_count / max((time.time() - t_start_all), 1):.2f}")
    print(f"Result exported to {args.output}")

