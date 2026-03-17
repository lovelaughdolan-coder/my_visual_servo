import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

import time

def show_points(coords, labels, image, marker_size=5):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    for pt in pos_points:
        cv2.circle(image, (int(pt[0]), int(pt[1])), marker_size, (0, 255, 0), -1)
        cv2.circle(image, (int(pt[0]), int(pt[1])), marker_size + 2, (255, 255, 255), 1)
    for pt in neg_points:
        cv2.circle(image, (int(pt[0]), int(pt[1])), marker_size, (0, 0, 255), -1)
        cv2.circle(image, (int(pt[0]), int(pt[1])), marker_size + 2, (255, 255, 255), 1)

def apply_mask(image, mask, color, alpha=0.5):
    """通过权重合成在原图上叠加透明 mask"""
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        colored_mask[:, :, c] = mask * color[c]
    
    # 获取掩码所在区域
    mask_indices = mask > 0
    # 在掩码区域混合原图与颜色的透明度
    image[mask_indices] = (image[mask_indices] * (1 - alpha) + colored_mask[mask_indices] * alpha).astype(np.uint8)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Switch Video Angle Detection using MobileSAM")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="video_result.mp4", help="Path to output video file")
    parser.add_argument("--sam_weights", type=str, default="no_ros/model/mobile_sam.pt", help="Path to SAM weights")
    parser.add_argument("--yolo_weights", type=str, default="no_ros/model/2-26merged.pt", help="Path to YOLO weights")
    parser.add_argument("--play", action="store_true", help="Play video while processing")
    args = parser.parse_args()

    # 1. 硬件探测与模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading YOLO model from {args.yolo_weights}...")
    yolo_model = YOLO(args.yolo_weights)
    
    print(f"Loading MobileSAM from {args.sam_weights}...")
    model_type = "vit_t"
    mobile_sam = sam_model_registry[model_type](checkpoint=args.sam_weights)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

    # 2. 视频流初始化
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # 定义编解码器并创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    X_THRESHOLD = 80 # 同一个闸刀上下旋钮的横向最大容差
    
    print(f"Starting processing. Output will be saved to {args.output}")
    pbar = tqdm(total=total_frames)
    
    frame_count = 0
    t_start_all = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 3. YOLO 识别提取并聚类为闸刀实例
        results = yolo_model.predict(frame, conf=0.25, classes=[0], verbose=False)
        
        yolo_points = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                xyxy = box.xyxy.cpu().numpy()[0]
                cx, cy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                yolo_points.append({'pt': [cx, cy], 'box': xyxy})
                
        yolo_points.sort(key=lambda item: item['pt'][0])
        
        clusters = []
        if yolo_points:
            current_cluster = [yolo_points[0]]
            for i in range(1, len(yolo_points)):
                avg_x = sum(p['pt'][0] for p in current_cluster) / len(current_cluster)
                if abs(yolo_points[i]['pt'][0] - avg_x) < X_THRESHOLD:
                    current_cluster.append(yolo_points[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [yolo_points[i]]
            clusters.append(current_cluster)
            
        switch_groups = []
        for i, cluster in enumerate(clusters):
            if len(cluster) == 2:
                # 合法的两点闸刀
                cluster.sort(key=lambda item: item['pt'][1])
                top_pt = cluster[0]['pt']
                pivot_pt = cluster[1]['pt']
                switch_groups.append({'top': top_pt, 'pivot': pivot_pt, 'id': len(switch_groups)+1})
                
        # 4. SAM 编码当前帧 (耗时大户)
        if len(switch_groups) > 0:
            predictor.set_image(frame_rgb)
        
        angle_results = []
        # 5. 对图上的每个闸刀独立预测与几何解算
        for sg in switch_groups:
            top_pt = sg['top']
            pivot_pt = sg['pivot']
            sw_id = sg['id']
            
            points_np = np.array([top_pt, pivot_pt])
            labels_np = np.array([1, 1])
            
            masks, scores, logits = predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=True,
            )
            
            best_mask = masks[np.argmax(scores)]
            
            # --- 渲染掩膜随机颜色 ---
            rand_color = np.random.randint(0, 255, (3,)).tolist()
            # 注意: OpenCV BGR 叠加需要确保类型一致
            apply_mask(frame, best_mask, rand_color, alpha=0.4)
            
            # --- 构建连线/测距与物理角度 ---
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cv2.line(frame, tuple(top_pt), tuple(pivot_pt), (255, 0, 255), 2)
            cv2.circle(frame, tuple(top_pt), 5, (0, 255, 0), -1)
            cv2.circle(frame, tuple(pivot_pt), 7, (0, 200, 255), -1)
            
            angle_text = f"SW{sw_id}: N/A"
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ref_top = np.array(top_pt)
                pivot_pt_np = np.array(pivot_pt)
                
                hull = cv2.convexHull(c)
                hull_points = hull.reshape(-1, 2)
                
                if len(hull_points) >= 4:
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
                        
                    v_up_axis = ref_top - pivot_pt_np
                    len_up_axis = np.linalg.norm(v_up_axis)
                    offset_dist = 20.0
                    if len_up_axis > 0:
                        v_up_norm = v_up_axis / len_up_axis
                        v_perp = np.array([-v_up_norm[1], v_up_norm[0]])
                        h_img, w_img = mask_uint8.shape
                        eval_pt = ref_top.copy().astype(float)
                        
                        def ray_cast(start, d):
                            pt = start.copy()
                            dist = 0
                            while 0 <= int(pt[0]) < w_img and 0 <= int(pt[1]) < h_img:
                                if mask_uint8[int(pt[1]), int(pt[0])] == 0:
                                    break
                                pt += d
                                dist += 1
                            return dist
                            
                        axis_width = ray_cast(eval_pt, v_perp) + ray_cast(eval_pt, -v_perp)
                        if axis_width > 0:
                            offset_dist = (axis_width / 2.0) * 0.85
                            
                    tip_pt = np.array([float(tip_raw[0]) + nx * offset_dist, float(tip_raw[1]) + ny * offset_dist], dtype=int)
                    
                    cv2.circle(frame, tuple(tip_raw), 3, (0, 255, 255), -1)
                    cv2.arrowedLine(frame, tuple(pivot_pt), tuple(tip_pt), (0, 165, 255), 3, tipLength=0.1)
                    cv2.circle(frame, tuple(tip_pt), 4, (0, 165, 255), -1)
                    
                    # 物理夹角算子
                    v_up = ref_top - pivot_pt_np
                    v_slant = tip_pt - pivot_pt_np
                    
                    if np.linalg.norm(v_up) > 0 and np.linalg.norm(v_slant) > 0:
                        dot = np.dot(v_up, v_slant)
                        cos_angle = np.clip(dot / (np.linalg.norm(v_up) * np.linalg.norm(v_slant)), -1.0, 1.0)
                        angle_between = np.degrees(np.arccos(cos_angle))
                    else:
                        angle_between = 0.0
                        
                    angle_text = f"SW{sw_id}: {angle_between:.1f} deg"
                    cv2.putText(frame, angle_text, (pivot_pt[0]+20, pivot_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            angle_results.append(angle_text)

        # 头部覆盖调试与状态信息
        fps_current = frame_count / max((time.time() - t_start_all), 0.001)
        info_str = f"FPS: {fps_current:.1f} | Detected SW: {len(switch_groups)}"
        cv2.putText(frame, info_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        if angle_results:
            cv2.putText(frame, " | ".join(angle_results), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)
        
        if args.play:
            cv2.imshow("Video Angle Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nUser interrupted playback.")
                break
                
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    if args.play:
        cv2.destroyAllWindows()
        
    print(f"\nProcessing complete! Average FPS: {total_frames / max((time.time() - t_start_all), 1):.2f}")
    print(f"Result exported to {args.output}")
