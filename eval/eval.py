import os
import re
import json
import math
from typing import Any, Dict, List, Tuple, Optional
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

from PIL import Image, ImageDraw

# --- 引入 Matplotlib 用于 3D 可视化 ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 引入多进程库 ---
import torch.multiprocessing as mp


# --- 全局配置 ---
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 修改：指定使用单张显卡 (例如，第一张卡 "0")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 修改：将 GPU 数量设置为 1
NUM_GPUS = 1

# 修改：在单张卡上开 4 个模型进程
WORKERS_PER_GPU = 3

# 总工作进程数会自动计算为 1 * 4 = 4
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

MODEL_NAME_OR_PATH = "/home/data/nas_hdd/cbx/pvllm/out_dir_grpo/checkpoint-1000"
DATA_PATH = "/data/ruoshi/pcdata/sft_data_v2/test"
TRUST_REMOTE_CODE = True
MAX_NEW_TOKENS = 2048
SAVE_DIR = "/home/data/nas_hdd/cbx/pvllm/eval/test_images_2x2_with_pc_batch"
os.makedirs(SAVE_DIR, exist_ok=True)
# -----------------


def calculate_3d_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个3D边界框的IoU。"""
    if not (isinstance(box1, list) and len(box1) == 6 and
            isinstance(box2, list) and len(box2) == 6):
        return 0.0
    inter_xmin = max(box1[0], box2[0]); inter_ymin = max(box1[1], box2[1]); inter_zmin = max(box1[2], box2[2])
    inter_xmax = min(box1[3], box2[3]); inter_ymax = min(box1[4], box2[4]); inter_zmax = min(box1[5], box2[5])
    inter_w = max(0, inter_xmax - inter_xmin); inter_h = max(0, inter_ymax - inter_ymin); inter_d = max(0, inter_zmax - inter_zmin)
    intersection_volume = inter_w * inter_h * inter_d
    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    if box1_volume <= 0 or box2_volume <= 0: return 0.0
    union_volume = box1_volume + box2_volume - intersection_volume
    return intersection_volume / union_volume if union_volume > 0 else 0.0


def calculate_points_in_bbox_ratio(points_3d: List[List[float]], bbox_3d: List[float]) -> float:
    """
    计算 3D 点在 3D 边界框内的比例。
    """
    if not isinstance(points_3d, list) or not points_3d:
        return 0.0
    if not (isinstance(bbox_3d, list) and len(bbox_3d) == 6):
        return 0.0
    in_box_count = 0
    xmin, ymin, zmin, xmax, ymax, zmax = bbox_3d
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

# --- 其他辅助函数保持不变 ---
def ensure_pil(img) -> Image.Image:
    if isinstance(img, Image.Image): return img
    if isinstance(img, (bytes, bytearray)): return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, str): return Image.open(img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")

# ... (此处省略 draw_bbox, plot_3d_bbox_and_pc_to_pil 等可视化函数，它们与原代码相同)


def load_model_for_worker(device: str) -> Qwen2_5_PVLForConditionalGeneration:
    """为工作进程加载模型到指定设备。"""
    model = Qwen2_5_PVLForConditionalGeneration.from_pretrained(
        MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=torch.bfloat16, device_map=None
    )
    custom_path = "/home/data/nas_hdd/cbx/pvllm/out_dir/model-00002-of-00002.safetensors"
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
    model.eval()
    return model.to(device)


def evaluate_sample(
    model: Qwen2_5_PVLForConditionalGeneration,
    processor: AutoProcessor,
    item: Dict,
    device: str,
    item_idx: int
) -> Dict[str, Any]:
    """
    处理单个数据样本的评估逻辑。
    返回一个包含计算指标的字典。
    """
    # 1) 提取并规范化数据
    images, messages = item['images'], item['messages']
    point_cloud_list = item['point_cloud']
    point_cloud_np = np.array(point_cloud_list, dtype=np.float32).reshape(-1, 3)
    ground_truth_str = messages[1]['content'][0]['text']
    infer_messages = [messages[0]]

    # 2) 准备模型输入
    text = processor.apply_chat_template(infer_messages, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=[ensure_pil(im) for im in images], padding=True, return_tensors="pt").to(device)
    enc["point_cloud"] = torch.from_numpy(point_cloud_np).to(device, dtype=torch.float32).unsqueeze(0)

    # 3) 模型生成和解码
    with torch.no_grad():
        generated_ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS)
    response_text = processor.decode(generated_ids[0][len(enc["input_ids"][0]):], skip_special_tokens=True)

    # 4) 解析JSON并计算指标
    model_json = extract_json_from_text(response_text)
    try:
        ground_truth_json = json.loads(ground_truth_str)
    except Exception:
        ground_truth_json = None

    results = {'iou': 0.0, 'iou_valid': False, 'points_ratio': 0.0, 'points_ratio_valid': False}

    if model_json and ground_truth_json:
        # 3D IoU 计算
        model_3d = model_json.get("3d_bbox")
        gt_3d = ground_truth_json.get("3d_bbox")
        if isinstance(model_3d, list) and len(model_3d) == 2 and all(isinstance(s, list) for s in model_3d): model_3d = model_3d[0] + model_3d[1]
        if isinstance(gt_3d, list) and len(gt_3d) == 2 and all(isinstance(s, list) for s in gt_3d): gt_3d = gt_3d[0] + gt_3d[1]
        
        if isinstance(model_3d, list) and len(model_3d) == 6 and isinstance(gt_3d, list) and len(gt_3d) == 6:
            results['iou'] = calculate_3d_iou(model_3d, gt_3d)
            results['iou_valid'] = True

        # 计算 3D 点在 GT Bbox 内的比例
        model_points_3d = model_json.get("point_3d")
        if isinstance(model_points_3d, list) and isinstance(gt_3d, list) and len(gt_3d) == 6:
            results['points_ratio'] = calculate_points_in_bbox_ratio(model_points_3d, gt_3d)
            results['points_ratio_valid'] = True

    return results


def worker_process(
    process_id: int,
    device: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue
):
    """
    每个工作进程执行的函数。
    """
    print(f"[Worker {process_id}] Starting on device {device}...")
    try:
        model = load_model_for_worker(device)
        processor = AutoProcessor.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    except Exception as e:
        print(f"[Worker {process_id}] CRITICAL: Failed to load model/processor. Error: {e}")
        return

    while True:
        task = task_queue.get()
        if task is None:
            print(f"[Worker {process_id}] Exiting.")
            break
        
        item_idx, item_data = task
        try:
            results = evaluate_sample(model, processor, item_data, device, item_idx)
            result_queue.put(results)
        except Exception as e:
            print(f"[Worker {process_id}] Error processing item {item_idx}: {e}")
            result_queue.put({'iou_valid': False, 'points_ratio_valid': False})
            

def main():
    mp.set_start_method('spawn', force=True)

    print("Loading dataset...")
    dataset = load_dataset("parquet", data_files={'test': f'{DATA_PATH}/*.parquet'})['test']
    num_samples = len(dataset)
    print(f"Found {num_samples} items. Starting evaluation with {TOTAL_WORKERS} parallel workers on a single GPU...")

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for i in range(num_samples):
        task_queue.put((i, dataset[i]))

    for _ in range(TOTAL_WORKERS):
        task_queue.put(None)

    processes = []
    for i in range(TOTAL_WORKERS):
        # 所有进程都将使用 cuda:0，因为 NUM_GPUS 为 1
        device = f'cuda:{i % NUM_GPUS}'
        p = mp.Process(target=worker_process, args=(i, device, task_queue, result_queue))
        p.start()
        processes.append(p)
        # 在同一张卡上加载模型，更需要错开时间以避免瞬间的显存峰值
        time.sleep(10)

    total_iou, iou_count = 0.0, 0
    total_points_ratio, points_ratio_count = 0.0, 0

    with tqdm(total=num_samples, desc="Evaluating Samples") as pbar:
        for _ in range(num_samples):
            result = result_queue.get()

            if result.get('iou_valid', False):
                total_iou += result['iou']
                iou_count += 1
            
            if result.get('points_ratio_valid', False):
                total_points_ratio += result['points_ratio']
                points_ratio_count += 1

            desc_str = "Evaluating | "
            if iou_count > 0:
                desc_str += f"Avg IoU: {total_iou / iou_count:.4f} ({iou_count}) | "
            if points_ratio_count > 0:
                desc_str += f"Avg Pt Ratio: {total_points_ratio / points_ratio_count:.4f} ({points_ratio_count})"
            
            pbar.set_description(desc_str)
            pbar.update(1)

    for p in processes:
        p.join()

    print("\n\n===== Overall Evaluation Summary =====")
    if iou_count > 0:
        print(f"Final Avg 3D Bbox IoU: {total_iou/iou_count:.4f} (from {iou_count} valid samples)")
    else:
        print("No valid 3D Bbox pairs were evaluated for IoU.")
        
    if points_ratio_count > 0:
        print(f"Final Avg Points in GT Bbox Ratio: {total_points_ratio/points_ratio_count:.4f} (from {points_ratio_count} valid samples)")
    else:
        print("No valid 3D points/bbox pairs were evaluated for ratio.")


if __name__ == "__main__":
    main()