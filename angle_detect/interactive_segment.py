import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor
import argparse
from ultralytics import YOLO
import os

# 全局变量，用于存储用户手动点击的点以及分组后的成组点
manual_points = []
manual_labels = [] # 1 为前景(正样本)，0 为背景(负样本)
switch_groups = [] # 存储自动聚类或合法成组的闸刀点对: [{'top': [x,y], 'pivot': [x,y]}, ...]

def mouse_callback(event, x, y, flags, param):
    global manual_points, manual_labels, img_display
    
    # 左键点击：添加前景点 (绿点)
    if event == cv2.EVENT_LBUTTONDOWN:
        manual_points.append([x, y])
        manual_labels.append(1)
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image - Interactive SAM", img_display)
        print(f"Added Manual Foreground Point at ({x}, {y})")
        
    # 右键点击：添加背景点 (红点)
    elif event == cv2.EVENT_RBUTTONDOWN:
        manual_points.append([x, y])
        manual_labels.append(0)
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image - Interactive SAM", img_display)
        print(f"Added Manual Background Point at ({x}, {y})")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive MobileSAM Segmentation")
    parser.add_argument("--image", type=str, default="no_ros/images/2.jpg", help="Path to your image")
    parser.add_argument("--weights", type=str, default="no_ros/model/mobile_sam.pt", help="Path to SAM weights")
    parser.add_argument("--yolo_weights", type=str, default="no_ros/model/2-26merged.pt", help="Path to YOLO weights")
    args = parser.parse_args()

    # 1. 加载图像
    image_path = args.image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}. Please check the path.")
        exit(1)
        
    weight_path = args.weights
    
    # 转换为 RGB 格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 全局变量给 OpenCV 回调使用
    global img_display
    img_display = image.copy()

    # 2. YOLO 自动检测 (获取初始提示点)
    print(f"Loading YOLO model from {args.yolo_weights}...")
    yolo_model = YOLO(args.yolo_weights)
    yolo_results = yolo_model.predict(image, conf=0.25, classes=[0], verbose=False)
    
    yolo_points = []
    # 获取所有的检测框中心点
    if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
        for box in yolo_results[0].boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            cx, cy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
            yolo_points.append({'pt': [cx, cy], 'box': xyxy})
            # 在显示图中画出 YOLO 框 (虚线或浅色)
            cv2.rectangle(img_display, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (100, 100, 0), 1)
            cv2.circle(img_display, (cx, cy), 3, (255, 255, 0), -1)
            
    print(f"YOLO detected {len(yolo_points)} knobs.")

    # ------ 智能聚类逻辑 ------
    # 因为闸刀垂直安装，同一个闸刀的两个旋钮 x 坐标相近
    X_THRESHOLD = 80  # 放宽 x 坐标差异阈值 (考虑透视和视角倾斜)
    
    # 按 x 坐标排序方便聚类
    yolo_points.sort(key=lambda item: item['pt'][0])
    
    clusters = []
    if yolo_points:
        current_cluster = [yolo_points[0]]
        for i in range(1, len(yolo_points)):
            # 计算当前聚类的 x 坐标平均值作为中心
            avg_x = sum(p['pt'][0] for p in current_cluster) / len(current_cluster)
            # 如果当前点与中心的 x 距离在阈值内，归为一类
            if abs(yolo_points[i]['pt'][0] - avg_x) < X_THRESHOLD:
                current_cluster.append(yolo_points[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [yolo_points[i]]
        clusters.append(current_cluster)
        
    print(f"Clustered into {len(clusters)} potential switch groups based on X-coordinate.")
    
    # 解析聚类结果，寻找合法的上下旋钮对
    for i, cluster in enumerate(clusters):
        if len(cluster) == 2:
            # 找到正好一对，按 y 坐标排序，区分上下
            cluster.sort(key=lambda item: item['pt'][1])
            top_pt = cluster[0]['pt']
            pivot_pt = cluster[1]['pt']
            switch_groups.append({'top': top_pt, 'pivot': pivot_pt, 'id': len(switch_groups)+1})
            
            # 绘制配对连线和明确标识
            cv2.line(img_display, tuple(top_pt), tuple(pivot_pt), (0, 255, 0), 2)
            cv2.circle(img_display, tuple(top_pt), 6, (0, 255, 0), -1)   # Top 绿色
            cv2.circle(img_display, tuple(pivot_pt), 8, (128, 255, 0), -1) # Pivot 偏绿
            cv2.putText(img_display, f"SW{len(switch_groups)}", (int(top_pt[0])-20, int(top_pt[1])-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"  Valid Switch Group {len(switch_groups)}: Top={top_pt}, Pivot={pivot_pt}")
        else:
            print(f"  Ignored invalid cluster with {len(cluster)} points: {[p['pt'] for p in cluster]}")
    # ---------------------------

    # 3. 交互式获取点 (备用或手动补充)
    cv2.namedWindow("Image - Interactive SAM", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image - Interactive SAM", mouse_callback)
    
    print("\n--- Multi-Switch Detection Mode ---")
    print(f"Auto-detected {len(switch_groups)} switches via YOLO.")
    print("If automatic detection missed something, you can manually:")
    print("1. Click LEFT -> Top Reference Point")
    print("2. Click LEFT -> Bottom Pivot Point (These 2 consecutive clicks will form a new switch group)")
    print("3. Press ENTER or ESC to apply segmentation to ALL detected switches.")
    print("-----------------------------------")

    cv2.imshow("Image - Interactive SAM", img_display)
    
    # 等待用户按键 (Enter: 13, ESC: 27)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 13 or k == 27: 
            break
            
    cv2.destroyAllWindows()

    if len(switch_groups) == 0 and len(manual_points) < 2:
        print("No valid switch groups automatically detected AND manual points are insufficient. Exiting.")
        exit(0)

    # 整合待处理的所有靶标组
    # 将手动点击的点（如果 >= 2）作为一个补充的 switch_group 加入
    if len(manual_points) >= 2:
        m_top = manual_points[0]
        m_pivot = manual_points[1]
        switch_groups.append({'top': m_top, 'pivot': m_pivot, 'id': len(switch_groups)+1})
        print(f"Added Manual Switch Group {len(switch_groups)}: Top={m_top}, Pivot={m_pivot}")

    print(f"Initializing MobileSAM for {len(switch_groups)} switches...")

    # 4. 初始化 MobileSAM
    # 使用 CUDA 加速 (如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_type = "vit_t"

    mobile_sam = sam_model_registry[model_type](checkpoint=weight_path)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    predictor = SamPredictor(mobile_sam)

    print("Setting image to MobileSAM...")
    predictor.set_image(image_rgb)
    
    # 用于收集每个闸刀的 Mask 以便最终合并显示
    all_masks = []
    angle_results = []

    print("Running predictions for each switch group...")
    # 5. 遍历每个闸刀组进行预测和计算
    for sg in switch_groups:
        top_pt = sg['top']
        pivot_pt = sg['pivot']
        sw_id = sg['id']
        
        # 组装 Prompt 点
        points_np = np.array([top_pt, pivot_pt])
        labels_np = np.array([1, 1]) # 两个都是前景点
        
        # 执行预测
        masks, scores, logits = predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=True,
        )
        
        best_mask = masks[np.argmax(scores)]
        best_score = max(scores)
        all_masks.append(best_mask)

        # ---------------- 角度计算模块 ----------------
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angle_text = f"SW{sw_id}: N/A"
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), rect_angle = rect
            
            if w < h:
                angle_deg = rect_angle + 90
            else:
                angle_deg = rect_angle

            ref_top = np.array(top_pt)
            pivot_pt_np = np.array(pivot_pt)
            
            # 1. 绘制基准线 (Top -> Pivot)
            cv2.line(image_rgb, tuple(ref_top), tuple(pivot_pt_np), (128, 0, 128), 3) # Purple line
            
            # 2. 寻找刀尖 (基于凸包和轮廓)
            hull = cv2.convexHull(c)
            hull_points = hull.reshape(-1, 2)
            
            if len(hull_points) >= 4:
                # 寻找"最左下"的点
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
                    length = np.hypot(dx, dy)
                    edges.append({'dx': dx, 'dy': dy, 'length': length})
                
                slanted_candidates = [e for e in edges if e['dx'] < -5]
                if slanted_candidates:
                    long_slanted_edge = max(slanted_candidates, key=lambda x: x['length'])
                    l_len = long_slanted_edge['length']
                    dx_norm = long_slanted_edge['dx'] / l_len
                    dy_norm = long_slanted_edge['dy'] / l_len
                    nx, ny = -dy_norm, dx_norm
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
                        
                    w1 = ray_cast(eval_pt, v_perp)
                    w2 = ray_cast(eval_pt, -v_perp)
                    axis_width = w1 + w2
                    if axis_width > 0:
                        offset_dist = (axis_width / 2.0) * 0.85 
                
                tip_final = np.array([float(tip_raw[0]) + nx * offset_dist, float(tip_raw[1]) + ny * offset_dist], dtype=int)
                tip_pt = tip_final 
                
                cv2.circle(image_rgb, tuple(tip_raw), 4, (0, 255, 255), -1)
                cv2.arrowedLine(image_rgb, tuple(pivot_pt_np), tuple(tip_pt), (255, 165, 0), 4, tipLength=0.08)
                cv2.circle(image_rgb, tuple(tip_pt), 6, (255, 165, 0), -1)
                
                # 3. 计算物理夹角
                v_up = ref_top - pivot_pt_np
                v_slant = tip_pt - pivot_pt_np
                
                len_up = np.linalg.norm(v_up)
                len_slant = np.linalg.norm(v_slant)
                
                if len_up > 0 and len_slant > 0:
                    dot = np.dot(v_up, v_slant)
                    cos_angle = dot / (len_up * len_slant)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_between = np.degrees(np.arccos(cos_angle))
                else:
                    angle_between = 0.0
                    
                angle_text = f"SW{sw_id}: {angle_between:.1f} deg"
                # 在图像上该闸刀中心位置标示角度
                cv2.putText(image_rgb, angle_text, (pivot_pt[0]+30, pivot_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 100), 2)
            else:
                angle_text = f"SW{sw_id} Axis: {angle_deg:.1f} deg"
        
        angle_results.append(angle_text)
        print(angle_text)
        # ---------------------------------------------

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    
    # 叠加所有闸刀的 Mask
    for mask in all_masks:
        show_mask(mask, plt.gca(), random_color=True)
        
    # 显示所有的提示点
    pts_to_draw = []
    lbl_to_draw = []
    for sg in switch_groups:
        pts_to_draw.extend([sg['top'], sg['pivot']])
        lbl_to_draw.extend([1, 1])
    if pts_to_draw:
        show_points(np.array(pts_to_draw), np.array(lbl_to_draw), plt.gca())

    title_str = "Multi-Switch Angle Detection\n" + " | ".join(angle_results)
    plt.title(title_str, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    save_path = "interactive_result.jpg"
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")
    
    plt.show()
