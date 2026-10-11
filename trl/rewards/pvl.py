import json
import numpy as np
from typing import List, Dict, Any, Union, Optional
import torch

# ================== 1. 基础工具函数 ==================

def normalize_3d_bbox(box: Union[List[float], List[List[float]]]) -> Optional[List[float]]:
    """将 [[x,y,z], [x,y,z]] 或 [x,y,z,x,y,z] 统一为扁平格式"""
    if (isinstance(box, list) and len(box) == 2 and
            isinstance(box[0], list) and len(box[0]) == 3 and
            isinstance(box[1], list) and len(box[1]) == 3):
        return box[0] + box[1]
    
    if isinstance(box, list) and len(box) == 6:
        return box
        
    return None

def get_3d_corners_from_bbox(bbox_flat):
    """
    生成 8 个顶点
    【严守逻辑】：不做任何旋转，直接生成
    """
    x1, y1, z1, x2, y2, z2 = bbox_flat
    corners = np.array([
        [x1, y1, z1], [x1, y1, z2],
        [x1, y2, z1], [x1, y2, z2],
        [x2, y1, z1], [x2, y1, z2],
        [x2, y2, z1], [x2, y2, z2]
    ], dtype=np.float32)
    return corners

def project_points_to_2d(points_3d, K, R, T):
    """
    投影函数 (NumPy 实现)
    复用验证过的逻辑：World -> Camera -> OpenCV Fix -> Image
    """
    # 1. World -> Camera (PyTorch3D style: points @ R + T)
    points_cam_p3d = np.dot(points_3d, R) + T
    
    # 2. 坐标系修正: PyTorch3D (+X Left, +Y Up) -> OpenCV (+X Right, +Y Down)
    points_cam_opencv = points_cam_p3d.copy()
    points_cam_opencv[:, 0] *= -1 
    points_cam_opencv[:, 1] *= -1 
    
    # 3. Camera -> Pixel (Intrinsic)
    points_img_homo = np.dot(K, points_cam_opencv.T).T 
    
    # 4. 透视除法
    z_safe = points_img_homo[:, 2:3] + 1e-6 
    uv = points_img_homo[:, :2] / z_safe
    
    return uv

# ================== 2. 核心 Reward 计算 (纯 Recall) ==================

def calculate_geo_consistency_reward(pred_bbox_3d, gt_bbox_2d, ext_param):
    """
    计算几何一致性 Reward (VCS - Coverage Only)
    只计算：GT 2D BBox 有百分之多少被包含在 3D 投影框内。
    """
    # 1. 数据校验与解析
    flat_bbox_3d = normalize_3d_bbox(pred_bbox_3d)
    if flat_bbox_3d is None: return 0.0
    
    if not gt_bbox_2d or len(gt_bbox_2d) != 4: return 0.0
    
    try:
        R = np.array(ext_param['R'], dtype=np.float32)
        T = np.array(ext_param['T'], dtype=np.float32)
        K = np.array(ext_param['K'], dtype=np.float32)
        img_size = float(ext_param['image_size'])
    except (KeyError, ValueError, TypeError):
        return 0.0

    # 2. 生成 8 个角点
    corners_3d = get_3d_corners_from_bbox(flat_bbox_3d)
    
    # 3. 投影到 2D
    corners_2d = project_points_to_2d(corners_3d, K, R, T)
    
    # 4. 获取投影后的 2D 包围盒 (AABB of Projection) - 红框 (Pred)
    u_min, v_min = np.min(corners_2d, axis=0)
    u_max, v_max = np.max(corners_2d, axis=0)
    
    # 限制在图像范围内 (Clip)
    pred_box = [
        max(0, u_min), max(0, v_min),
        min(img_size, u_max), min(img_size, v_max)
    ]
    
    # 5. GT 2D Box - 绿框 (GT)
    gt_box = [
        max(0, gt_bbox_2d[0]), max(0, gt_bbox_2d[1]),
        min(img_size, gt_bbox_2d[2]), min(img_size, gt_bbox_2d[3])
    ]

    # --- 计算 Recall (覆盖率) ---
    # 公式：Recall = (Pred ∩ GT) / GT_Area
    
    # 计算交集 (Intersection)
    ix1 = max(pred_box[0], gt_box[0])
    iy1 = max(pred_box[1], gt_box[1])
    ix2 = min(pred_box[2], gt_box[2])
    iy2 = min(pred_box[3], gt_box[3])
    
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    
    # 计算 GT 的面积 (分母)
    gt_w = max(1e-6, gt_box[2] - gt_box[0])
    gt_h = max(1e-6, gt_box[3] - gt_box[1])
    gt_area = gt_w * gt_h
    
    # 纯覆盖率
    coverage_ratio = inter_area / gt_area
    
    return min(1.0, max(0.0, coverage_ratio))

# ================== 3. 原有 3D 指标计算 ==================

def calculate_iou_3d(box1, box2) -> float:
    flat_box1 = normalize_3d_bbox(box1)
    flat_box2 = normalize_3d_bbox(box2)
    if flat_box1 is None or flat_box2 is None: return 0.0

    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = flat_box1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = flat_box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_z_min = max(z1_min, z2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_z_max = min(z1_max, z2_max)

    inter_vol = max(0, inter_x_max - inter_x_min) * \
                max(0, inter_y_max - inter_y_min) * \
                max(0, inter_z_max - inter_z_min)

    vol1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    vol2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)
    
    if vol1 <= 0 or vol2 <= 0: return 0.0
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0

def calculate_points_in_bbox_ratio(points, bbox) -> float:
    flat_bbox = normalize_3d_bbox(bbox)
    if flat_bbox is None or not points: return 0.0
    try:
        p_np = np.array(points, dtype=np.float32)
        if p_np.ndim != 2 or p_np.shape[1] != 3: return 0.0
    except: return 0.0

    x_min, y_min, z_min, x_max, y_max, z_max = flat_bbox
    is_inside = np.all((p_np >= [x_min, y_min, z_min]) & (p_np <= [x_max, y_max, z_max]), axis=1)
    return np.sum(is_inside) / p_np.shape[0] if p_np.shape[0] > 0 else 0.0

# ================== 4. 主函数 PVL Reward ==================

def pvl_reward(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """
    Reward = 3D IoU + Points Ratio + 2D Coverage Ratio
    """
    geo = True

    rewards = []
    rewards_iou = []
    rewards_points = []
    rewards_geo = []

    gt_conversations = kwargs.get("messages", [])
    ext_params_list = kwargs.get("ext_param", []) 

    if len(completions) != len(gt_conversations):
        raise ValueError("Mismatch between completions and GT messages.")

    for i, completion_turn in enumerate(completions):
        try:
            # 1. 解析
            pred_json_str = completion_turn[0]['content']
            pred_data = json.loads(pred_json_str)

            gt_json_str = gt_conversations[i][1]['content'][0]['text']
            gt_data = json.loads(gt_json_str)
            
            ext_param = ext_params_list[i] if i < len(ext_params_list) else None

            # 2. 提取
            pred_bbox_3d = pred_data.get("3d_bbox", [])
            pred_points_3d = pred_data.get("point_3d", [])
            
            gt_bbox_3d = gt_data.get("3d_bbox", [])
            gt_bbox_2d = gt_data.get("bbox", []) 

            # 3. 计算
            r_iou = 0.0
            r_points = 0.0
            r_geo = 0.0

            if pred_bbox_3d and gt_bbox_3d:
                r_iou = calculate_iou_3d(pred_bbox_3d, gt_bbox_3d)

            if pred_points_3d and gt_bbox_3d:
                r_points = calculate_points_in_bbox_ratio(pred_points_3d, gt_bbox_3d)

            if pred_bbox_3d and gt_bbox_2d and ext_param and geo:
                r_geo = calculate_geo_consistency_reward(pred_bbox_3d, gt_bbox_2d, ext_param)

            # 4. 加权
            if geo:
                final_reward = 2 * r_iou + r_points + 0.5 * r_geo
            else:
                final_reward = r_iou + r_points
            
            rewards.append(final_reward)
            rewards_iou.append(r_iou)
            rewards_points.append(r_points)
            if geo:
                rewards_geo.append(r_geo)   

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            rewards.append(0.0)
            rewards_iou.append(0.0)
            rewards_points.append(0.0)
            if geo:
                rewards_geo.append(0.0)

    # ================= 打印详细列表 (仅 Rank 0) =================
    if torch.cuda.current_device() == 0:
        dev = f"cuda:{torch.cuda.current_device()}"
        
        # 转 Tensor 方便打印
        t_iou = torch.tensor(rewards_iou, device=dev)
        t_pts = torch.tensor(rewards_points, device=dev)
        if geo:
            t_geo = torch.tensor(rewards_geo, device=dev)
        t_total = torch.tensor(rewards, device=dev)
        
        print("\n" + "="*40)
        print(f"[Batch Reward Details]")
        print(f"  IoU List : {t_iou.tolist()}") # 打印完整列表
        print(f"  Pts List : {t_pts.tolist()}")
        if geo:
            print(f"  Geo List : {t_geo.tolist()}")
        print(f"  Total    : {t_total.tolist()}")
 
        print("="*40 + "\n")
            
    return rewards