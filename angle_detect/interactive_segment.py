import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor
import argparse

# 全局变量，用于存储用户点击的点和标签
input_points = []
input_labels = [] # 1 为前景(正样本)，0 为背景(负样本)

def mouse_callback(event, x, y, flags, param):
    global input_points, input_labels, img_display
    
    # 左键点击：添加前景点 (绿点)
    if event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(1)
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image - Click Left for Foreground, Right for Background, Enter to Segment", img_display)
        print(f"Added Foreground Point at ({x}, {y})")
        
    # 右键点击：添加背景点 (红点)
    elif event == cv2.EVENT_RBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(0)
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image - Click Left for Foreground, Right for Background, Enter to Segment", img_display)
        print(f"Added Background Point at ({x}, {y})")

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
    parser.add_argument("--image", type=str, default="images/1.png", help="Path to your image")
    parser.add_argument("--weights", type=str, default="./weights/mobile_sam.pt", help="Path to SAM weights")
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

    # 2. 交互式获取点
    cv2.namedWindow("Image - Click Left for Foreground, Right for Background, Enter to Segment", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image - Click Left for Foreground, Right for Background, Enter to Segment", mouse_callback)
    
    print("--- Instructions for Angle Calculation ---")
    print("1. Click LEFT mouse button on the TOP REFERENCE POINT (e.g., top hole).")
    print("2. Click LEFT mouse button on the PIVOT POINT (e.g., bottom knob).")
    print("   -> These two points will define the PURPLE reference line.")
    print("3. Click LEFT/RIGHT to add more Foreground/Background points if needed.")
    print("4. Press ENTER or ESC to apply segmentation.")
    print("------------------------------------------")

    cv2.imshow("Image - Click Left for Foreground, Right for Background, Enter to Segment", img_display)
    
    # 等待用户按键 (Enter: 13, ESC: 27)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 13 or k == 27: 
            break
            
    cv2.destroyAllWindows()

    if len(input_points) == 0:
        print("No points selected. Exiting.")
        exit(0)

    points_np = np.array(input_points)
    labels_np = np.array(input_labels)
    print(f"Selected {len(points_np)} points. Initializing MobileSAM...")

    # 3. 初始化 MobileSAM
    # 根据我们之前处理 gfx803 的经验，强制使用 cpu
    device = torch.device("cpu")
    model_type = "vit_t"

    mobile_sam = sam_model_registry[model_type](checkpoint=weight_path)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    predictor = SamPredictor(mobile_sam)

    print("Setting image to MobileSAM...")
    predictor.set_image(image_rgb)

    print("Running prediction...")
    # 4. 执行预测
    masks, scores, logits = predictor.predict(
        point_coords=points_np,
        point_labels=labels_np,
        multimask_output=True, # 设为 True 会返回三个不同层次的 mask 建议，我们选得分最高的一个
    )

    print("Prediction complete! Displaying result...")

    # 5. 可视化结果 (选择 score 最高的 mask 此时已经默认按 score 排序输出，0 是最高或者最优选，我们均展示)
    best_mask = masks[np.argmax(scores)]
    best_score = max(scores)

    # ---------------- 角度计算模块 ----------------
    # 假设用户的第一个前景点是“非轴心”端，或者是通过 Mask 的几何特征自动寻找长轴
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    angle_text = "Angle: N/A"
    if len(contours) > 0:
        # 找到最大的轮廓
        c = max(contours, key=cv2.contourArea)
        # 获取最小外接矩形 (中心(x,y), (宽,高), 旋转角度deg)
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), rect_angle = rect
        
        # minAreaRect 的角度范围有历史包袱，通常计算长边与 X 轴的夹角更直观
        # 判断哪条边更长，来决定物体的朝向轴
        if w < h:
            angle_deg = rect_angle + 90
        else:
            angle_deg = rect_angle

        # 需求: 提取基于两个特定点击的基准线，并计算相对夹角
        pos_points = points_np[labels_np == 1]
        
        # 只需要用户提供2个绿点: 第一个是顶部参考(基准)，第二个是底部轴心(pivot)
        if len(pos_points) >= 2:
            ref_top = pos_points[0]   # 顶部参考点
            pivot_pt = pos_points[1]  # 底部轴心点
            
            # 1. 绘制基准线 (Top -> Pivot)
            cv2.line(image_rgb, tuple(ref_top), tuple(pivot_pt), (128, 0, 128), 3) # Purple line
            cv2.circle(image_rgb, tuple(ref_top), 6, (128, 0, 128), 2)
            cv2.circle(image_rgb, tuple(pivot_pt), 8, (128, 255, 0), 2) # 标记轴心
            
            # 2. 依据用户 v5 脚本思路，使用凸包寻找“最左侧端点”及其修正路线作为刀尖
            hull = cv2.convexHull(c)
            hull_points = hull.reshape(-1, 2)
            
            if len(hull_points) >= 4:
                # 寻找"最左下"的点 (x最小且y最大)
                # 在图像坐标系中 y 向下递增，所以 score = x - y，score 最小的点即为最左下
                score = hull_points[:, 0] - hull_points[:, 1]
                leftbottom_idx = np.argmin(score)
                tip_raw = hull_points[leftbottom_idx]
                
                # 计算凸包边长，寻找“长斜边”来确定法向偏移
                n_hull = len(hull_points)
                edges = []
                for i in range(n_hull):
                    p1 = hull_points[i]
                    p2 = hull_points[(i + 1) % n_hull]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = np.hypot(dx, dy)
                    edges.append({'dx': dx, 'dy': dy, 'length': length})
                
                # 闸刀边缘特征：向左延伸，所以 dx < 0
                slanted_candidates = [e for e in edges if e['dx'] < -5]
                if slanted_candidates:
                    long_slanted_edge = max(slanted_candidates, key=lambda x: x['length'])
                    # 计算法线向量 (原直线向量为 dx, dy，则向右/下沉的法向为 -dy, dx)
                    # 注意我们希望法向是指向闸刀中心的（即如果长边在上方边缘，法相往下平移）
                    l_len = long_slanted_edge['length']
                    dx_norm = long_slanted_edge['dx'] / l_len
                    dy_norm = long_slanted_edge['dy'] / l_len
                    nx, ny = -dy_norm, dx_norm
                else:
                    # 缺省给一个向右下方的法向偏移
                    nx, ny = 0.5, 0.866
                    
                # 【动态平移量】：测量垂直那个轴的真实宽度
                v_up_axis = ref_top - pivot_pt
                len_up_axis = np.linalg.norm(v_up_axis)
                offset_dist = 20.0 # 默认后备值
                if len_up_axis > 0:
                    v_up_norm = v_up_axis / len_up_axis
                    v_perp = np.array([-v_up_norm[1], v_up_norm[0]])
                    h_img, w_img = mask_uint8.shape
                    
                    # 为了避免小角度时射线打到闸刀上，直接使用上方固定的基准点(小圆柱体)中心作为测距起始点
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
                        # 用真实厚度取代定值, 系数 0.85-0.9 比较符合 v5 的表现
                        offset_dist = (axis_width / 2.0) * 0.85 
                        print(f"Measured axis width at top ref: {axis_width} px, dynamic offset: {offset_dist:.1f} px")
                
                tip_final = np.array([tip_raw[0] + nx * offset_dist, tip_raw[1] + ny * offset_dist], dtype=int)
                tip_pt = tip_final 
                
                # 绘制最初始的“最左点”方便对比
                cv2.circle(image_rgb, tuple(tip_raw), 4, (0, 255, 255), -1)
                
                # 绘制实际经过偏置校准后的朝向线 (Pivot -> Tip)
                cv2.arrowedLine(image_rgb, tuple(pivot_pt), tuple(tip_pt), (255, 165, 0), 4, tipLength=0.08)
                cv2.circle(image_rgb, tuple(tip_pt), 6, (255, 165, 0), -1)
                
                # 3. 计算物理夹角
                # 垂直基准向量
                v_up = ref_top - pivot_pt
                # 实际活动向量
                v_slant = tip_pt - pivot_pt
                
                # 余弦定理求无向夹角 (符合物理夹角直觉 0-180)
                len_up = np.linalg.norm(v_up)
                len_slant = np.linalg.norm(v_slant)
                
                if len_up > 0 and len_slant > 0:
                    dot = np.dot(v_up, v_slant)
                    cos_angle = dot / (len_up * len_slant)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_between = np.degrees(np.arccos(cos_angle))
                else:
                    angle_between = 0.0
                    
                angle_text = f"Blade Angle: {angle_between:.1f} deg"
                print(f"Calculated Pivot: {pivot_pt}, Tip: {tip_pt}")
                print(f"Base Vector: {v_up}, Slant Vector: {v_slant}, {angle_text}")
        elif len(pos_points) == 1:
            # 兼容仅点击了单点的情况
            pivot_pt = pos_points[0]
            hull = cv2.convexHull(c)
            hull_points = hull.reshape(-1, 2)
            if len(hull_points) > 0:
                score = hull_points[:, 0] - hull_points[:, 1]
                leftbottom_idx = np.argmin(score)
                tip_pt = hull_points[leftbottom_idx]
                dy = -(tip_pt[1] - pivot_pt[1]) 
                dx = tip_pt[0] - pivot_pt[0]
                directed_angle = np.degrees(np.arctan2(dy, dx))
                angle_text = f"Absolute Angle: {directed_angle:.1f} deg"
                cv2.arrowedLine(image_rgb, tuple(pivot_pt), tuple(tip_pt), (255, 165, 0), 4, tipLength=0.1)
                cv2.circle(image_rgb, tuple(pivot_pt), 8, (255, 255, 0), 2)
        else:
            angle_text = f"Axis Angle: {angle_deg:.1f} deg (Undirected)"
    # ---------------------------------------------

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    show_mask(best_mask, plt.gca())
    show_points(points_np, labels_np, plt.gca())
    plt.title(f"MobileSAM Mask & Angle\n{angle_text}", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    
    save_path = "interactive_result.jpg"
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")
    
    plt.show()
