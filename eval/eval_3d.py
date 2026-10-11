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

from model_arch.modeling_qwen2_5_pvl_batch_3d import Qwen2_5_PVLForConditionalGeneration
from safetensors.torch import load_file

from PIL import Image

# --- 引入 Matplotlib 用于 3D 可视化 (如果需要) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 引入多进程库 ---
import torch.multiprocessing as mp


# --- 全局配置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
NUM_GPUS = 4
WORKERS_PER_GPU = 3
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

MODEL_NAME_OR_PATH = "/home/data/nas_hdd/cbx/pvllm/output_dir_3d"   
DATA_PATH = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test"
TRUST_REMOTE_CODE = True
MAX_NEW_TOKENS = 2048
OUTPUT_DIR = "/home/data/nas_hdd/cbx/pvllm/eval/output_dir"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================== 1. 基础数学工具函数 (3D 相关) ==================

def normalize_3d_bbox(box: Union[List[float], List[List[float]]]) -> Optional[List[float]]:
    """将 [[x,y,z], [x,y,z]] 或 [x,y,z,x,y,z] 统一为扁平格式 [x1,y1,z1,x2,y2,z2]"""
    if (isinstance(box, list) and len(box) == 2 and
            isinstance(box[0], list) and len(box[0]) == 3 and
            isinstance(box[1], list) and len(box[1]) == 3):
        return box[0] + box[1]
    if isinstance(box, list) and len(box) == 6:
        return box
    return None

def get_3d_corners_from_bbox(bbox_flat):
    x1, y1, z1, x2, y2, z2 = bbox_flat
    return np.array([
        [x1, y1, z1], [x1, y1, z2], [x1, y2, z1], [x1, y2, z2],
        [x2, y1, z1], [x2, y1, z2], [x2, y2, z1], [x2, y2, z2]
    ], dtype=np.float32)

def project_points_to_2d(points_3d, K, R, T):
    """投影函数 (用于计算几何一致性)"""
    points_cam_p3d = np.dot(points_3d, R) + T
    points_cam_opencv = points_cam_p3d.copy()
    points_cam_opencv[:, 0] *= -1 
    points_cam_opencv[:, 1] *= -1 
    points_img_homo = np.dot(K, points_cam_opencv.T).T 
    z_safe = points_img_homo[:, 2:3] + 1e-6 
    uv = points_img_homo[:, :2] / z_safe
    return uv

def calculate_geo_consistency_reward(pred_bbox_3d, gt_bbox_2d, ext_param):
    """几何一致性：预测的 3D 框投影到 2D 后对 GT 2D 框的覆盖率"""
    flat_bbox_3d = normalize_3d_bbox(pred_bbox_3d)
    if flat_bbox_3d is None or not gt_bbox_2d or len(gt_bbox_2d) != 4: return 0.0
    try:
        R, T, K = np.array(ext_param['R']), np.array(ext_param['T']), np.array(ext_param['K'])
        img_size = float(ext_param['image_size'])
    except: return 0.0

    corners_2d = project_points_to_2d(get_3d_corners_from_bbox(flat_bbox_3d), K, R, T)
    u_min, v_min = np.min(corners_2d, axis=0)
    u_max, v_max = np.max(corners_2d, axis=0)
    
    pred_box = [max(0, u_min), max(0, v_min), min(img_size, u_max), min(img_size, v_max)]
    gt_box = [max(0, gt_bbox_2d[0]), max(0, gt_bbox_2d[1]), min(img_size, gt_bbox_2d[2]), min(img_size, gt_bbox_2d[3])]

    inter_area = max(0, min(pred_box[2], gt_box[2]) - max(pred_box[0], gt_box[0])) * \
                 max(0, min(pred_box[3], gt_box[3]) - max(pred_box[1], gt_box[1]))
    gt_area = max(1e-6, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))
    return min(1.0, max(0.0, inter_area / gt_area))

# ================== 2. 核心 3D 评估函数 ==================

def calculate_3d_iou(box1: List[float], box2: List[float]) -> float:
    b1, b2 = normalize_3d_bbox(box1), normalize_3d_bbox(box2)
    if not (b1 and b2): return 0.0
    inter_min = [max(b1[i], b2[i]) for i in range(3)]
    inter_max = [min(b1[i+3], b2[i+3]) for i in range(3)]
    inter_vol = max(0, inter_max[0]-inter_min[0]) * max(0, inter_max[1]-inter_min[1]) * max(0, inter_max[2]-inter_min[2])
    vol1 = (b1[3]-b1[0]) * (b1[4]-b1[1]) * (b1[5]-b1[2])
    vol2 = (b2[3]-b2[0]) * (b2[4]-b2[1]) * (b2[5]-b2[2])
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0

def calculate_points_in_bbox_ratio(points_3d: List[List[float]], bbox_3d: List[float]) -> float:
    """计算预测的 3D 关键点落在 GT 3D 框内的比例"""
    if not points_3d or not isinstance(points_3d, list): return 0.0
    flat_bbox = normalize_3d_bbox(bbox_3d)
    if not flat_bbox: return 0.0
    
    xmin, ymin, zmin, xmax, ymax, zmax = flat_bbox
    in_box_count, valid_count = 0, 0
    for pt in points_3d:
        if isinstance(pt, list) and len(pt) == 3:
            valid_count += 1
            if xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax and zmin <= pt[2] <= zmax:
                in_box_count += 1
    return in_box_count / valid_count if valid_count > 0 else 0.0

# ================== 3. 模型推理与评估逻辑 ==================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None

def ensure_pil(img) -> Image.Image:
    if isinstance(img, Image.Image): return img
    return Image.open(io.BytesIO(img)).convert("RGB") if isinstance(img, (bytes, bytearray)) else Image.open(img).convert("RGB")

def load_model_for_worker(device: str):
    model = Qwen2_5_PVLForConditionalGeneration.from_pretrained(
        MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=torch.bfloat16, device_map=None
    ).to(device)
    # 加载自定义权重 (具体路径根据实际情况)
    for i in [1, 2]:
        path = f"{MODEL_NAME_OR_PATH}/model-0000{i}-of-00002.safetensors"
        sd = load_file(path, device=device)
        if i == 1: model.model.dummy_img_embeds.data.copy_(sd["model.dummy_img_embeds"])
        else:
            model.model.cross_attn.load_state_dict({k[len("model.cross_attn."):]: v for k, v in sd.items() if k.startswith("model.cross_attn.")})
            model.model.pc_extractor.load_state_dict({k[len("model.pc_extractor."):]: v for k, v in sd.items() if k.startswith("model.pc_extractor.")})
    model.eval()
    return model

def evaluate_sample(model, processor, item, device, item_idx) -> Dict[str, Any]:
    images, messages, ext_param = item['images'], item['messages'], item.get('ext_param')
    pc_np = np.array(item['point_cloud'], dtype=np.float32).reshape(-1, 3)
    gt_str = messages[1]['content'][0]['text']
    
    img = ensure_pil(images[0])
    if ext_param and 'image_size' not in ext_param: ext_param['image_size'] = max(img.size)
    
    prompt = processor.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[img], padding=True, return_tensors="pt").to(device)
    inputs["point_cloud"] = torch.from_numpy(pc_np).to(device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    resp = processor.decode(out_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    pred_json = extract_json_from_text(resp)
    try: gt_json = json.loads(gt_str)
    except: gt_json = None
    
    # 结果初始化
    res = {
        'iou_3d': 0.0, 'iou_3d_valid': False,
        'points_3d_ratio': 0.0, 'points_3d_ratio_valid': False,
        'geo_consistency': 0.0, 'geo_consistency_valid': False
    }

    if pred_json and gt_json:
        p_box_3d = pred_json.get("3d_bbox")
        g_box_3d = gt_json.get("3d_bbox")
        p_pts_3d = pred_json.get("point_3d")
        g_box_2d = gt_json.get("bbox") # 仅用于算几何一致性

        # 1. 3D IoU
        if p_box_3d and g_box_3d:
            res['iou_3d'] = calculate_3d_iou(p_box_3d, g_box_3d)
            res['iou_3d_valid'] = True
        
        # 2. 3D Points Ratio (预测点落在 GT 3D 框里的比例)
        if p_pts_3d and g_box_3d:
            res['points_3d_ratio'] = calculate_points_in_bbox_ratio(p_pts_3d, g_box_3d)
            res['points_3d_ratio_valid'] = True
            
        # 3. Geo Consistency (3D 预测与 2D GT 的对齐度)
        if p_box_3d and g_box_2d and ext_param:
            res['geo_consistency'] = calculate_geo_consistency_reward(p_box_3d, g_box_2d, ext_param)
            res['geo_consistency_valid'] = True

    return res

def worker_process(pid, device, t_q, r_q):
    torch.cuda.set_device(int(device.split(":")[-1]))
    model = load_model_for_worker(device)
    proc = AutoProcessor.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    while True:
        task = t_q.get()
        if task is None: break
        idx, data = task
        try: r_q.put(evaluate_sample(model, proc, data, device, idx))
        except: r_q.put({})

def main():
    mp.set_start_method('spawn', force=True)
    ds = load_dataset("parquet", data_files={'test': f'{DATA_PATH}/*.parquet'})['test']
    t_q, r_q = mp.Queue(), mp.Queue()
    for i in range(len(ds)): t_q.put((i, ds[i]))
    for _ in range(TOTAL_WORKERS): t_q.put(None)

    procs = [mp.Process(target=worker_process, args=(i, f'cuda:{i%NUM_GPUS}', t_q, r_q)) for i in range(TOTAL_WORKERS)]
    for p in procs: p.start(); time.sleep(2)

    totals = {'iou_3d': 0.0, 'points_3d_ratio': 0.0, 'geo_consistency': 0.0}
    counts = {k: 0 for k in totals}

    with tqdm(total=len(ds), desc="3D Eval") as pbar:
        for _ in range(len(ds)):
            res = r_q.get()
            for k in totals:
                if res.get(f'{k}_valid'):
                    totals[k] += res[k]
                    counts[k] += 1
            desc = [f"{k}:{totals[k]/max(1,counts[k]):.3f}" for k in totals if counts[k]>0]
            pbar.set_description(" | ".join(desc))
            pbar.update(1)

    for p in procs: p.join()
    print("\n\n===== Final 3D Results =====")
    for k in totals:
        val = totals[k]/counts[k] if counts[k]>0 else 0
        print(f"{k}: {val:.4f} (samples: {counts[k]})")

if __name__ == "__main__":
    main()