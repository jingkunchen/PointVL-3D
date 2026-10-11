import json
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
import torch

# ================== 1. 几何计算辅助工具函数 (保持不变) ==================

def normalize_3d_bbox(box: Union[List[float], List[List[float]]]) -> Optional[List[float]]:
    if (isinstance(box, list) and len(box) == 2 and
            isinstance(box[0], list) and len(box[0]) == 3 and
            isinstance(box[1], list) and len(box[1]) == 3):
        return box[0] + box[1]
    if isinstance(box, list) and len(box) == 6:
        return box
    return None

def calculate_iou_3d(
    box1: Union[List[float], List[List[float]]],
    box2: Union[List[float], List[List[float]]]
) -> float:
    flat_box1 = normalize_3d_bbox(box1)
    flat_box2 = normalize_3d_bbox(box2)

    if flat_box1 is None or flat_box2 is None:
        return 0.0

    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = flat_box1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = flat_box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_z_min = max(z1_min, z2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_z_max = min(z1_max, z2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_depth = max(0, inter_z_max - inter_z_min)
    intersection_volume = inter_width * inter_height * inter_depth

    vol1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    vol2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)
    
    if vol1 < 0 or vol2 < 0: return 0.0
    
    union_volume = vol1 + vol2 - intersection_volume
    return intersection_volume / union_volume if union_volume > 0 else 0.0

def calculate_points_in_bbox_ratio(
    points: List[List[float]], 
    bbox: Union[List[float], List[List[float]]]
) -> float:
    flat_bbox = normalize_3d_bbox(bbox)
    if flat_bbox is None or not isinstance(points, list) or not points:
        return 0.0

    try:
        points_np = np.array(points, dtype=np.float32)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            return 0.0
    except ValueError:
        return 0.0

    x_min, y_min, z_min, x_max, y_max, z_max = flat_bbox
    is_inside = np.all((points_np >= [x_min, y_min, z_min]) & (points_np <= [x_max, y_max, z_max]), axis=1)
    return np.sum(is_inside) / points_np.shape[0] if points_np.shape[0] > 0 else 0.0


# ================== 2. 核心奖励函数 (修改版) ==================

def pvl_multi_reward(
    prompts: List[str],
    completions: List[Dict[str, str]],
    completion_ids: List[List[int]], # 虽然这里不用，但 TRL 接口会传
    **kwargs: Dict[str, Any],
) -> List[float]:
    """
    PVL 多阶段奖励函数。
    
    Args:
        prompts: Prompt 文本列表
        completions: 结构化输出列表，格式为 [{'role': 'assistant', 'content': '...'}, ...]
        completion_ids: Token IDs (本函数未使用)
        kwargs: 必须包含 'stage' 和 'messages' (Ground Truth)
    
    Returns:
        List[float]: 每个样本的奖励值列表
    """
    
    # 1. 获取上下文信息
    stage = kwargs.get("stage")
    gt_conversations = kwargs.get("messages", [])
    
    if stage is None:
        raise ValueError("Reward function requires 'stage' in kwargs (1 or 2).")
    if not gt_conversations:
        raise ValueError("Ground truth 'messages' not found in kwargs.")

    rewards = []
    
    # 2. 遍历 Batch
    # completions 与 gt_conversations 是一一对应的 (Batch 顺序一致)
    for i, completion in enumerate(completions):
        # 提取模型生成的文本内容
        # 这里的 completion 是 {'role': 'assistant', 'content': '...'}
        pred_text = completion.get('content', "")
        reward = 0.0

        # ---------------- STAGE 1: 内容准确性 (IoU + Points) ----------------
        if stage == 1:
            try:
                # A. 解析 Ground Truth
                # 假设 GT 格式是 [User, Assistant, ...]，取索引 1
                if len(gt_conversations[i]) < 2:
                     raise IndexError(f"GT conversations length < 2, got {len(gt_conversations[i])}")

                gt_assistant_msg = gt_conversations[i][1]
                gt_content = gt_assistant_msg['content']
                
                # 处理可能的 List[Dict] 格式 (多模态数据常见格式)
                if isinstance(gt_content, list):
                    gt_json_str = gt_content[0]['text']
                else:
                    gt_json_str = gt_content
                
                gt_data = json.loads(gt_json_str)
                
                # B. 解析 Prediction
                pred_data = json.loads(pred_text)
                
                # C. 提取 3D 信息
                pred_bbox = pred_data.get("3d_bbox", [])
                pred_points = pred_data.get("point_3d", [])
                gt_bbox = gt_data.get("3d_bbox", [])
                
                # D. 计算各项指标
                reward_iou = 0.0
                reward_points = 0.0
                
                if pred_bbox and gt_bbox:
                    reward_iou = calculate_iou_3d(pred_bbox, gt_bbox)
                
                if pred_points and gt_bbox:
                    reward_points = calculate_points_in_bbox_ratio(pred_points, gt_bbox)
                
                # E. 聚合 Stage 1 奖励
                reward = reward_iou + reward_points
                
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                # 【修改点】解析失败或数据缺失，打印详细警告
                # 截取前100个字符以避免日志过长，同时去除换行符方便查看
                clean_text = str(pred_text).replace('\n', '\\n')[:100]
                print(f"[Warn] Stage 1 Reward Failed (Sample {i}). Error: {type(e).__name__}: {e} | Content: {clean_text}...")
                reward = 0.0

        # ---------------- STAGE 2: 格式合规性 (JSON Parse Check) ----------------
        elif stage == 2:
            try:
                # 尝试解析 JSON
                json.loads(pred_text)
                # 如果成功解析，给予满分奖励
                reward = 0.3
            except (json.JSONDecodeError, TypeError) as e:
                # 【修改点】解析失败，打印详细警告
                clean_text = str(pred_text).replace('\n', '\\n')[:100]
                print(f"[Warn] Stage 2 JSON Parse Failed (Sample {i}). Error: {e} | Content: {clean_text}...")
                reward = 0.0
        
        rewards.append(reward)

    return rewards