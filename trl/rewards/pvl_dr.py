import json
import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple

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

def normalize_3d_bbox(box: Union[List[float], List[List[float]]]) -> Optional[List[float]]:
    """将 [[x,y,z], [x,y,z]] 或 [x,y,z,x,y,z] 统一为扁平格式"""
    if (isinstance(box, list) and len(box) == 2 and
            isinstance(box[0], list) and len(box[0]) == 3 and
            isinstance(box[1], list) and len(box[1]) == 3):
        return box[0] + box[1]
    
    if isinstance(box, list) and len(box) == 6:
        return box
        
    return None

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

def calculate_points_in_bbox_ratio_3d(points, bbox) -> float:
    flat_bbox = normalize_3d_bbox(bbox)
    if flat_bbox is None or not points: return 0.0
    try:
        p_np = np.array(points, dtype=np.float32)
        if p_np.ndim != 2 or p_np.shape[1] != 3: return 0.0
    except: return 0.0

    x_min, y_min, z_min, x_max, y_max, z_max = flat_bbox
    is_inside = np.all((p_np >= [x_min, y_min, z_min]) & (p_np <= [x_max, y_max, z_max]), axis=1)
    return np.sum(is_inside) / p_np.shape[0] if p_np.shape[0] > 0 else 0.0

# --- 辅助：归一化 2D BBox ---
def normalize_2d_bbox(bbox) -> Optional[List[float]]:
    """
    将 bbox 转换为 [x_min, y_min, x_max, y_max] 格式。
    支持输入格式: [x1, y1, x2, y2]
    """
    try:
        # 展平
        b = np.array(bbox).flatten().tolist()
        if len(b) != 4:
            return None
        
        # 确保 min < max
        x1, y1, x2, y2 = b
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    except:
        return None

# --- 1. 计算 2D IoU ---
def calculate_iou_2d(box1, box2) -> float:
    flat_box1 = normalize_2d_bbox(box1)
    flat_box2 = normalize_2d_bbox(box2)
    if flat_box1 is None or flat_box2 is None: return 0.0

    x1_min, y1_min, x1_max, y1_max = flat_box1
    x2_min, y2_min, x2_max, y2_max = flat_box2

    # 计算交集坐标
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 计算交集面积
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    # 计算各自面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    if area1 <= 0 or area2 <= 0: return 0.0
    
    # 计算并集面积
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

# --- 2. 计算 2D Points Ratio ---
def calculate_points_in_bbox_ratio_2d(points, bbox) -> float:
    """
    计算有多少比例的 2D 点落在 2D bbox 内部
    """
    flat_bbox = normalize_2d_bbox(bbox)
    if flat_bbox is None or not points: return 0.0
    
    try:
        # 转换为 numpy 数组，预期 shape (N, 2)
        p_np = np.array(points, dtype=np.float32)
        # 兼容性处理：如果是 list of lists
        if p_np.ndim != 2 or p_np.shape[1] != 2: 
            return 0.0
    except: 
        return 0.0

    x_min, y_min, x_max, y_max = flat_bbox
    
    # 向量化判断点是否在矩形内
    # p_np[:, 0] 是 x, p_np[:, 1] 是 y
    is_inside = (
        (p_np[:, 0] >= x_min) & (p_np[:, 0] <= x_max) &
        (p_np[:, 1] >= y_min) & (p_np[:, 1] <= y_max)
    )
    
    return np.sum(is_inside) / p_np.shape[0] if p_np.shape[0] > 0 else 0.0

# ================== 辅助函数：查找子序列范围 ==================
def find_json_value_token_range(token_ids, tokenizer, target_key):
    """
    1. 全局解码成字符串。
    2. 在字符串中找到 Key 对应的 Value 的【起始字符位置】和【结束字符位置】。
    3. 将这两个字符位置分别映射回 Token Index。
    
    Returns:
        (start_token_idx, end_token_idx)
        都是闭区间索引（inclusive）。如果找不到返回 (-1, -1)。
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
        
    # 1. 解码整个序列
    # 必须 skip_special_tokens=True 以对齐人类阅读的文本位置
    full_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    # 2. 【字符级】定位 Value 的 Start/End
    key_start_char = full_text.find(target_key)
    if key_start_char == -1:
        return -1, -1
    
    # 从 Key 后面开始找 Value 的起始符号 [ 或 {
    search_start = key_start_char + len(target_key)
    char_start_idx = -1 # Value 的第一个字符 (如 '[') 的位置
    char_end_idx = -1   # Value 的最后一个字符 (如 ']') 的位置
    
    stack = []
    found_start = False
    
    # 限制搜索长度，避免死循环
    limit = min(len(full_text), search_start + 2000)
    
    for i in range(search_start, limit):
        char = full_text[i]
        
        if char == '[':
            if not found_start:
                char_start_idx = i # 记录起始字符位置
                found_start = True
            stack.append(char)
            
        elif char == ']' or char == '}':
            if stack:
                stack.pop()
                if len(stack) == 0:
                    char_end_idx = i # 记录结束字符位置
                    break
    
    if char_start_idx == -1 or char_end_idx == -1:
        return -1, -1

    # =========================================================
    # 3. 【核心】字符坐标 -> Token 坐标映射 (二分查找版)
    # =========================================================
    
    def get_token_index_for_char(target_char_pos):
        """找到包含 target_char_pos 这个字符的那个 token index"""
        low = 0
        high = len(token_ids) - 1
        ans = -1
        
        while low <= high:
            mid = (low + high) // 2
            # 解码前 mid+1 个 token
            sub_text = tokenizer.decode(token_ids[:mid+1], skip_special_tokens=True)
            
            if len(sub_text) > target_char_pos:
                # 说明第 mid 个 token 已经覆盖到了 target_char_pos
                ans = mid
                high = mid - 1 # 尝试往前缩，看是不是更早的 token 就已经覆盖了
            else:
                low = mid + 1
        return ans

    # 分别映射 Start 和 End
    start_token = get_token_index_for_char(char_start_idx)
    end_token = get_token_index_for_char(char_end_idx)
    
    return start_token, end_token


def pvl_dr_reward(completions, **kwargs) -> torch.Tensor:
    """
    计算细粒度 Dense Reward。
    
    Returns:
        dense_adv (torch.Tensor): 形状与 completion_ids 一致 (Batch, Seq_Len)。
        包含了归一化后的 Advantage，特定 JSON 字段区域填充特定 Adv，
        背景区域填充该样本的平均 Adv。
    """

    # 1. 提取参数
    completion_ids = kwargs.get('completion_ids')
    tokenizer = kwargs.get('tokenizer')
    gt_conversations = kwargs.get("messages", [])

    ext_params_list = kwargs.get("ext_param", []) 

    if len(completions) != len(gt_conversations):
        raise ValueError("Mismatch between completions and GT messages.")

    device = completion_ids.device if completion_ids is not None else torch.device('cpu')
    batch_size = len(completions)
    
    # 假设 num_generations 等于 batch_size (根据你的 log 是一组 8 个)
    # 如果实际 batch 包含多组，这里需要 reshape 处理，目前按单组处理
    num_generations = batch_size 

    targets = ["3d_bbox", "bbox", "point_3d", "point_2d", "geo"]
    
    # 临时存储 raw reward 和 indices
    raw_rewards = {} 
    indices_map = {}
    
    for target in targets:
        raw_rewards[target] = torch.zeros(batch_size, dtype=torch.float32, device=device)
        indices_map[target] = [(-1, -1)] * batch_size

    # ---------------------------------------------------------------------
    # 2. 遍历 Batch 计算 Raw Rewards (这部分逻辑保持不变)
    # ---------------------------------------------------------------------
    for i, completion_turn in enumerate(completions):
        current_token_ids = completion_ids[i] if completion_ids is not None else None
        ext_param = ext_params_list[i] if i < len(ext_params_list) else None
        
        # 解析 JSON
        try:
            pred_json_str = completion_turn[0]['content']
            if "```json" in pred_json_str:
                pred_json_str = pred_json_str.split("```json")[1].split("```")[0]
            elif "```" in pred_json_str:
                pred_json_str = pred_json_str.split("```")[1].split("```")[0]
            pred_data = json.loads(pred_json_str)
        except:
            pred_data = {}

        try:
            gt_json_str = gt_conversations[i][1]['content'][0]['text']
            gt_data = json.loads(gt_json_str)
        except:
            gt_data = {}
        
        # 提取字段
        gt_bbox_3d = gt_data.get("3d_bbox", [])
        gt_bbox_2d = gt_data.get("bbox", [])
        pred_bbox_3d = pred_data.get("3d_bbox", [])
        pred_bbox_2d = pred_data.get("bbox", [])
        pred_point_3d = pred_data.get("point_3d", [])
        pred_point_2d = pred_data.get("point_2d", [])

        # 计算指标 & 记录坐标
        # A. 3D BBox
        if pred_bbox_3d and gt_bbox_3d:
            raw_rewards["3d_bbox"][i] = calculate_iou_3d(pred_bbox_3d, gt_bbox_3d)
            indices_map["3d_bbox"][i] = find_json_value_token_range(current_token_ids, tokenizer, '"3d_bbox"')
        
        # B. 2D BBox
        if pred_bbox_2d and gt_bbox_2d:
            raw_rewards["bbox"][i] = calculate_iou_2d(pred_bbox_2d, gt_bbox_2d)
            indices_map["bbox"][i] = find_json_value_token_range(current_token_ids, tokenizer, '"bbox"')

        # C. 3D Points
        if pred_point_3d and gt_bbox_3d:
            raw_rewards["point_3d"][i] = calculate_points_in_bbox_ratio_3d(pred_point_3d, gt_bbox_3d)
            indices_map["point_3d"][i] = find_json_value_token_range(current_token_ids, tokenizer, '"point_3d"')

        # D. 2D Points
        if pred_point_2d and gt_bbox_2d:
            raw_rewards["point_2d"][i] = calculate_points_in_bbox_ratio_2d(pred_point_2d, gt_bbox_2d)
            indices_map["point_2d"][i] = find_json_value_token_range(current_token_ids, tokenizer, '"point_2d"')

        if pred_bbox_3d and gt_bbox_2d:
            geo_reward = calculate_geo_consistency_reward(pred_bbox_3d, gt_bbox_2d, ext_param)
            raw_rewards["geo"][i] = torch.tensor(geo_reward, device=raw_rewards["geo"].device, dtype=raw_rewards["geo"].dtype)          

        

    # ---------------------------------------------------------------------
    # 3. 归一化处理 (Normalize Rewards -> Advantages)
    # ---------------------------------------------------------------------
    norm_advs = {}
    
    for target in targets:
        rewards = raw_rewards[target] # shape: (B,)
        
        mean = rewards.mean()
        std = rewards.std()
        # 防止除以 0，加一个小 epsilon
        std = std if std > 1e-6 else torch.tensor(1.0, device=device)
        norm_advs[target] = (rewards - mean) / (std + 1e-4)
       


    # ---------------------------------------------------------------------
    # 4. 构建 Dense Reward Tensor (Painting Logic)
    # ---------------------------------------------------------------------
    # 初始化 Dense Tensor，形状与 completion_ids 一致
    dense_advantages = torch.zeros_like(completion_ids, dtype=torch.float32, device=device)
    
    # 遍历每一个样本进行填充
    for i in range(batch_size):
        # 4.1 计算该样本的“背景 Adv” (四个字段归一化后的平均值)
        # 取出该样本对应的 4 个 Normalized Adv 值
        sample_target_advs = {target: norm_advs[target][i] for target in targets}
        background_adv = 0.875 * ((sample_target_advs["3d_bbox"] + sample_target_advs["bbox"] + sample_target_advs["point_3d"] + sample_target_advs["point_2d"])) / 4.0 + 0.125 * sample_target_advs.get("geo", 0.0)
        # 4.2 【打底】将整行填充为 Background Adv
        # 注意：这里会覆盖 Padding，如果后续 Loss 计算有 Mask 则没问题，
        # 如果需要严格 Mask，可以结合 completion_ids != pad_id 使用。
        # 这里按照你的需求“填满每一个 i 样本”。
        dense_advantages[i, :] = background_adv
        
        # 4.3 【上色】将特定 Token 区域覆盖为特定的 Normalized Adv
        for target in targets:
            s_idx, e_idx = indices_map[target][i]
            target_adv = norm_advs[target][i]
            
            if s_idx != -1 and e_idx != -1:
                # 赋值区间 [start, end]。注意 Python 切片是左闭右开，所以用 e_idx + 1
                # 且需确保不越界
                end_limit = min(e_idx + 1, completion_ids.shape[1])
                dense_advantages[i, s_idx : end_limit] = target_adv

    completion_mask = kwargs.get('completion_mask')
    
    if completion_mask is not None:
        # 确保 Mask 和 Adv 在同一个 Device
        if completion_mask.device != dense_advantages.device:
            completion_mask = completion_mask.to(dense_advantages.device)
            
        # 强制将 Mask 为 0 的位置的 Advantage 置为 0
        # 注意：这里使用乘法，假设 mask 是 0/1。如果 mask 是 bool，需要转 float。
        dense_advantages = dense_advantages * completion_mask.float()

    print(f"[dev{torch.cuda.current_device()}] dense reward: {torch.mean(dense_advantages, dim=1)}")
    print(f"[dev{torch.cuda.current_device()}] raw rewards: {raw_rewards}")

    # 5. 返回最终的 Dense Tensor
    return dense_advantages, raw_rewards