import os
import re
import json
import math
from typing import Any, Dict, List, Tuple, Optional, Union
import io
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoProcessor
from tqdm import tqdm
import numpy as np

from model_arch.modeling_qwen2_5_pvl_batch import Qwen2_5_PVLForConditionalGeneration
from safetensors.torch import load_file

from PIL import Image, ImageDraw, ImageFont

# --- 引入 Matplotlib 用于 3D 可视化 ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 引入多进程库 ---
import torch.multiprocessing as mp


# --- 全局配置 ---
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
NUM_GPUS = 4
WORKERS_PER_GPU = 3
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

MODEL_NAME_OR_PATH = "/home/data/nas_hdd/cbx/pvllm/output_grpo_dr_geo_125/checkpoint-1100"   
# MODEL_NAME_OR_PATH = "/home/data/nas_hdd/cbx/pvllm/out_dir_v3/checkpoint-2000"
DATA_PATH = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test"
TRUST_REMOTE_CODE = True
MAX_NEW_TOKENS = 2048
OUTPUT_DIR = "/home/data/nas_hdd/cbx/pvllm/eval/output_dir"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================== 1. 新增：基础数学工具函数 ==================

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
    """生成 8 个顶点"""
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
    World -> Camera -> OpenCV Fix -> Image
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

def calculate_geo_consistency_reward(pred_bbox_3d, gt_bbox_2d, ext_param):
    """
    计算几何一致性 Reward (VCS - Coverage Only)
    计算：3D投影框覆盖了多少 GT 2D Box (Recall)
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
    
    # 4. 获取投影后的 2D 包围盒 (AABB of Projection)
    u_min, v_min = np.min(corners_2d, axis=0)
    u_max, v_max = np.max(corners_2d, axis=0)
    
    # 限制在图像范围内 (Clip)
    pred_box = [
        max(0, u_min), max(0, v_min),
        min(img_size, u_max), min(img_size, v_max)
    ]
    
    # 5. GT 2D Box
    gt_box = [
        max(0, gt_bbox_2d[0]), max(0, gt_bbox_2d[1]),
        min(img_size, gt_bbox_2d[2]), min(img_size, gt_bbox_2d[3])
    ]

    # --- 计算 Recall (覆盖率) ---
    # Intersection
    ix1 = max(pred_box[0], gt_box[0])
    iy1 = max(pred_box[1], gt_box[1])
    ix2 = min(pred_box[2], gt_box[2])
    iy2 = min(pred_box[3], gt_box[3])
    
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    
    # GT Area
    gt_w = max(1e-6, gt_box[2] - gt_box[0])
    gt_h = max(1e-6, gt_box[3] - gt_box[1])
    gt_area = gt_w * gt_h
    
    # Coverage Ratio
    coverage_ratio = inter_area / gt_area
    
    return min(1.0, max(0.0, coverage_ratio))

# ================== 原有评估函数 ==================

def calculate_2d_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个2D边界框的IoU。"""
    if not (isinstance(box1, list) and len(box1) == 4 and
            isinstance(box2, list) and len(box2) == 4):
        return 0.0
    inter_xmin = max(box1[0], box2[0]); inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2]); inter_ymax = min(box1[3], box2[3])
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if box1_area <= 0 or box2_area <= 0: return 0.0
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_3d_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个3D边界框的IoU。"""
    # 统一格式化
    b1 = normalize_3d_bbox(box1)
    b2 = normalize_3d_bbox(box2)
    
    if not (b1 and b2): return 0.0
    
    box1, box2 = b1, b2 # 使用扁平化后的
    
    inter_xmin = max(box1[0], box2[0]); inter_ymin = max(box1[1], box2[1]); inter_zmin = max(box1[2], box2[2])
    inter_xmax = min(box1[3], box2[3]); inter_ymax = min(box1[4], box2[4]); inter_zmax = min(box1[5], box2[5])
    inter_w = max(0, inter_xmax - inter_xmin); inter_h = max(0, inter_ymax - inter_ymin); inter_d = max(0, inter_zmax - inter_zmin)
    intersection_volume = inter_w * inter_h * inter_d
    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    if box1_volume <= 0 or box2_volume <= 0: return 0.0
    union_volume = box1_volume + box2_volume - intersection_volume
    return intersection_volume / union_volume if union_volume > 0 else 0.0

def calculate_2d_points_in_bbox_ratio(points_2d: List[List[float]], bbox_2d: List[float]) -> float:
    """计算 2D 关键点在 2D 边界框内的比例。"""
    if not isinstance(points_2d, list) or not points_2d:
        return 0.0
    if not (isinstance(bbox_2d, list) and len(bbox_2d) == 4):
        return 0.0
    in_box_count = 0
    xmin, ymin, xmax, ymax = bbox_2d
    valid_points_count = 0
    for point in points_2d:
        if not (isinstance(point, list) and len(point) == 2):
            continue
        valid_points_count += 1
        px, py = point
        if xmin <= px <= xmax and ymin <= py <= ymax:
            in_box_count += 1
    if valid_points_count == 0: return 0.0
    return in_box_count / valid_points_count

def calculate_points_in_bbox_ratio(points_3d: List[List[float]], bbox_3d: List[float]) -> float:
    """计算 3D 点在 3D 边界框内的比例。"""
    if not isinstance(points_3d, list) or not points_3d:
        return 0.0
    
    flat_bbox = normalize_3d_bbox(bbox_3d)
    if not flat_bbox: return 0.0
    
    in_box_count = 0
    xmin, ymin, zmin, xmax, ymax, zmax = flat_bbox
    valid_points_count = 0
    for point in points_3d:
        if not (isinstance(point, list) and len(point) == 3):
            continue
        valid_points_count += 1
        px, py, pz = point
        if xmin <= px <= xmax and ymin <= py <= ymax and zmin <= pz <= zmax:
            in_box_count += 1
    if valid_points_count == 0: return 0.0
    return in_box_count / valid_points_count

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """从文本中抽取 JSON 对象。"""
    if not isinstance(text, str): return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def ensure_pil(img) -> Image.Image:
    if isinstance(img, Image.Image): return img
    if isinstance(img, (bytes, bytearray)): return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, str): return Image.open(img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")

# def draw_annotations_on_image(
#     image: Image.Image,
#     gt_json: Optional[Dict],
#     model_json: Optional[Dict]
# ) -> Image.Image:
#     """在图片上绘制真实和预测的2D框及关键点，并添加图例。"""
#     draw = ImageDraw.Draw(image)
#     try:
#         font = ImageFont.truetype("DejaVuSans.ttf", size=20)
#     except IOError:
#         font = ImageFont.load_default()
#     draw.text((10, 10), "Ground Truth (Green)", fill="green", font=font)
#     draw.text((10, 40), "Prediction (Red)", fill="red", font=font)

#     if gt_json:
#         gt_bbox = gt_json.get("bbox")
#         if isinstance(gt_bbox, list) and len(gt_bbox) == 4:
#             draw.rectangle(gt_bbox, outline="green", width=3)
#         gt_points = gt_json.get("point_2d")
#         if isinstance(gt_points, list):
#             for pt in gt_points:
#                 if isinstance(pt, list) and len(pt) == 2:
#                     draw.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill="green", outline="green")
#     if model_json:
#         model_bbox = model_json.get("bbox")
#         if isinstance(model_bbox, list) and len(model_bbox) == 4:
#             draw.rectangle(model_bbox, outline="red", width=2)
#         model_points = model_json.get("point_2d")
#         if isinstance(model_points, list):
#             for pt in model_points:
#                 if isinstance(pt, list) and len(pt) == 2:
#                     draw.ellipse([pt[0]-3, pt[1]-3, pt[0]+3, pt[1]+3], fill="red", outline="red")
#     return image

def load_model_for_worker(device: str) -> Qwen2_5_PVLForConditionalGeneration:
    """为工作进程加载模型到指定设备。"""
    model = Qwen2_5_PVLForConditionalGeneration.from_pretrained(
        MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=torch.bfloat16, device_map=None
    )
    # 你的自定义加载逻辑
    custom_path = f"{MODEL_NAME_OR_PATH}/model-00002-of-00002.safetensors"
    if os.path.exists(custom_path):
        custom_state_dict = load_file(custom_path, device=device)
        cross_attn_state_dict = {k[len("model.cross_attn."):]: v for k, v in custom_state_dict.items() if k.startswith("model.cross_attn.")}
        pc_extractor_state_dict = {k[len("model.pc_extractor."):]: v for k, v in custom_state_dict.items() if k.startswith("model.pc_extractor.")}
        if cross_attn_state_dict:
            try: model.model.cross_attn.load_state_dict(cross_attn_state_dict)
            except Exception: pass
        if pc_extractor_state_dict:
            try: model.model.pc_extractor.load_state_dict(pc_extractor_state_dict)
            except Exception: pass
    else:
        raise FileNotFoundError(f"Custom model file not found: {custom_path}")
    model.eval()
    return model.to(device)

def evaluate_sample(
    model: Qwen2_5_PVLForConditionalGeneration,
    processor: AutoProcessor,
    item: Dict,
    device: str,
    item_idx: int
) -> Dict[str, Any]:
    """处理单个样本，计算所有指标（包括新的几何一致性），并保存输出。"""
    images, messages = item['images'], item['messages']
    
    # 提取 ext_param 用于新指标
    ext_param = item.get('ext_param')
    
    point_cloud_np = np.array(item['point_cloud'], dtype=np.float32).reshape(-1, 3)
    ground_truth_str = messages[1]['content'][0]['text']
    infer_messages = [messages[0]]

    # 准备图片
    base_image = ensure_pil(images[0]).copy()
    
    # 确保 ext_param 中有 image_size (如果数据集中没有)
    if ext_param is not None and isinstance(ext_param, dict):
        if 'image_size' not in ext_param:
             # 假设是正方形或者取最大边，这取决于相机的定义。
             # 这里的逻辑对应了原始reward代码中 img_size = float(ext_param['image_size'])
             # 通常是图像的长宽中的较大值或固定值
             w, h = base_image.size
             ext_param['image_size'] = max(w, h)
    
    text = processor.apply_chat_template(infer_messages, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=[base_image], padding=True, return_tensors="pt").to(device)
    enc["point_cloud"] = torch.from_numpy(point_cloud_np).to(device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        generated_ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS)
    response_text = processor.decode(generated_ids[0][len(enc["input_ids"][0]):], skip_special_tokens=True)

    model_json = extract_json_from_text(response_text)
    try:
        ground_truth_json = json.loads(ground_truth_str)
    except Exception:
        ground_truth_json = None
    
    # annotated_image = draw_annotations_on_image(base_image, ground_truth_json, model_json)
    # annotated_image.save(os.path.join(OUTPUT_DIR, f"{item_idx}_visualization.png"))
    
    details_content = (
        f"--- Input Messages ---\n{json.dumps(infer_messages, indent=2)}\n\n"
        f"--- Model Response ---\n{response_text}\n\n"
        f"--- Ground Truth ---\n{ground_truth_str}\n"
    )
    with open(os.path.join(OUTPUT_DIR, f"{item_idx}_details.txt"), "w", encoding='utf-8') as f:
        f.write(details_content)

    results = {
        'iou_2d': 0.0, 'iou_2d_valid': False,
        'iou_3d': 0.0, 'iou_3d_valid': False, 
        'points_2d_ratio': 0.0, 'points_2d_ratio_valid': False,
        'points_3d_ratio': 0.0, 'points_3d_ratio_valid': False,
        'geo_consistency': 0.0, 'geo_consistency_valid': False  # 新增
    }

    if model_json and ground_truth_json:
        model_2d_bbox = model_json.get("bbox"); gt_2d_bbox = ground_truth_json.get("bbox")
        model_3d_bbox = model_json.get("3d_bbox"); gt_3d_bbox = ground_truth_json.get("3d_bbox")
        model_2d_pts = model_json.get("point_2d"); model_3d_pts = model_json.get("point_3d")

        # 1. 2D IoU
        if isinstance(model_2d_bbox, list) and len(model_2d_bbox) == 4 and isinstance(gt_2d_bbox, list) and len(gt_2d_bbox) == 4:
            results['iou_2d'] = calculate_2d_iou(model_2d_bbox, gt_2d_bbox)
            results['iou_2d_valid'] = True
            if isinstance(model_2d_pts, list):
                results['points_2d_ratio'] = calculate_2d_points_in_bbox_ratio(model_2d_pts, gt_2d_bbox)
                results['points_2d_ratio_valid'] = True

        # 2. 3D IoU 和 3D Points
        if isinstance(model_3d_bbox, list) and len(model_3d_bbox) == 2: model_3d_bbox = model_3d_bbox[0] + model_3d_bbox[1]
        if isinstance(gt_3d_bbox, list) and len(gt_3d_bbox) == 2: gt_3d_bbox = gt_3d_bbox[0] + gt_3d_bbox[1]
        
        if isinstance(model_3d_bbox, list) and len(model_3d_bbox) == 6 and isinstance(gt_3d_bbox, list) and len(gt_3d_bbox) == 6:
            results['iou_3d'] = calculate_3d_iou(model_3d_bbox, gt_3d_bbox)
            results['iou_3d_valid'] = True
            if isinstance(model_3d_pts, list):
                results['points_3d_ratio'] = calculate_points_in_bbox_ratio(model_3d_pts, gt_3d_bbox)
                results['points_3d_ratio_valid'] = True
        
        # 3. Geo Consistency Reward (VCS Score) - 新增指标
        # 条件：预测了3D框，且有GT 2D框，且存在相机参数
        if model_3d_bbox and gt_2d_bbox and ext_param:
            geo_score = calculate_geo_consistency_reward(model_3d_bbox, gt_2d_bbox, ext_param)
            results['geo_consistency'] = geo_score
            results['geo_consistency_valid'] = True

    return results

def worker_process(process_id: int, device: str, task_queue: mp.Queue, result_queue: mp.Queue):
    """每个工作进程执行的函数。"""
    print(f"[Worker {process_id}] Starting on device {device}...")

    # ================== 【关键修复】 ==================
    # 显式设置当前进程的默认 CUDA 设备。
    # 否则，模型内部生成的临时 Tensor 会默认去 cuda:0，导致与 cuda:N 上的权重冲突。
    if "cuda" in device:
        try:
            # 解析设备ID，例如 "cuda:1" -> 1
            device_id = int(device.split(":")[-1])
            torch.cuda.set_device(device_id)
        except Exception as e:
            print(f"[Worker {process_id}] Warning: Failed to set default device: {e}")
    # ==================================================

    try:
        model = load_model_for_worker(device)
        processor = AutoProcessor.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    except Exception as e:
        print(f"[Worker {process_id}] CRITICAL: Failed to load model/processor. Error: {e}")
        return
    while True:
        task = task_queue.get()
        if task is None:
            print(f"[Worker {process_id}] Exiting."); break
        item_idx, item_data = task
        try:
            results = evaluate_sample(model, processor, item_data, device, item_idx)
            result_queue.put(results)
        except Exception as e:
            print(f"[Worker {process_id}] Error processing item {item_idx}: {e}")
            # 即使出错也返回空结果，避免计数器卡死
            result_queue.put({k: False for k in [
                'iou_2d_valid', 'iou_3d_valid', 
                'points_2d_ratio_valid', 'points_3d_ratio_valid',
                'geo_consistency_valid'
            ]})

def main():
    mp.set_start_method('spawn', force=True)
    print("Loading dataset...")
    dataset = load_dataset("parquet", data_files={'test': f'{DATA_PATH}/*.parquet'})['test']
    num_samples = len(dataset)
    print(f"Found {num_samples} items. Starting evaluation with {TOTAL_WORKERS} workers...")
    print(f"Results will be saved to: {OUTPUT_DIR}")

    task_queue, result_queue = mp.Queue(), mp.Queue()
    for i in range(num_samples): task_queue.put((i, dataset[i]))
    for _ in range(TOTAL_WORKERS): task_queue.put(None)

    processes = []
    for i in range(TOTAL_WORKERS):
        device = f'cuda:{i % NUM_GPUS}'
        p = mp.Process(target=worker_process, args=(i, device, task_queue, result_queue))
        p.start(); processes.append(p)
        time.sleep(10)

    totals = {
        'iou_2d': 0.0, 'iou_3d': 0.0, 
        'points_2d_ratio': 0.0, 'points_3d_ratio': 0.0,
        'geo_consistency': 0.0 # 新增
    }
    counts = {key: 0 for key in totals}

    with tqdm(total=num_samples, desc="Evaluating") as pbar:
        for _ in range(num_samples):
            result = result_queue.get()
            for key in totals:
                if result.get(f'{key}_valid', False):
                    totals[key] += result[key]
                    counts[key] += 1
            
            desc_parts = []
            if counts['iou_2d'] > 0: desc_parts.append(f"IoU-2D:{totals['iou_2d']/counts['iou_2d']:.3f}")
            if counts['iou_3d'] > 0: desc_parts.append(f"IoU-3D:{totals['iou_3d']/counts['iou_3d']:.3f}")
            if counts['points_2d_ratio'] > 0: desc_parts.append(f"Pt-R-2D:{totals['points_2d_ratio']/counts['points_2d_ratio']:.2f}")
            if counts['points_3d_ratio'] > 0: desc_parts.append(f"Pt-R-3D:{totals['points_3d_ratio']/counts['points_3d_ratio']:.2f}")
            if counts['geo_consistency'] > 0: desc_parts.append(f"Geo-Consist:{totals['geo_consistency']/counts['geo_consistency']:.3f}") # 新增
            
            pbar.set_description(" | ".join(desc_parts))
            pbar.update(1)

    for p in processes: p.join()

    print("\n\n===== Overall Evaluation Summary =====")
    metric_names = {
        'iou_2d': "Avg 2D Bbox IoU",
        'iou_3d': "Avg 3D Bbox IoU",
        'points_2d_ratio': "Avg 2D Points in GT Bbox Ratio",
        'points_3d_ratio': "Avg 3D Points in GT Bbox Ratio",
        'geo_consistency': "Avg Geometric Consistency (VCS) Score" # 新增
    }
    for key, name in metric_names.items():
        if counts[key] > 0:
            print(f"{name}: {totals[key]/counts[key]:.4f} (from {counts[key]} valid samples)")
        else:
            print(f"No valid samples were evaluated for {name.split(':')[0]}.")

if __name__ == "__main__":
    main()