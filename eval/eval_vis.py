import os
import re
import json
import io
import time
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoProcessor
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 关键：设置 Agg 后端防止多进程服务器环境下 plt 死锁 ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PyTorch3D 核心组件
from pytorch3d.renderer import PerspectiveCameras

# 模型架构
from model_arch.modeling_qwen2_5_pvl_batch import Qwen2_5_PVLForConditionalGeneration
from safetensors.torch import load_file
import torch.multiprocessing as mp

# --- 全局配置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
NUM_GPUS = 4
WORKERS_PER_GPU = 1 # 如果显存足够(如A100)，可以尝试设为2；若报错则设为1
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

MODEL_NAME_OR_PATH = "/home/data/nas_hdd/cbx/pvllm/output_grpo_dr_geo_15/checkpoint-1100"   
DATA_PATH = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test"
TRUST_REMOTE_CODE = True
MAX_NEW_TOKENS = 2048
VIS_OUTPUT_DIR = "/home/data/nas_hdd/cbx/pvllm/vis_output"
os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

# ================== 辅助函数 (保持不变) ==================

def get_3d_bbox_edges(bbox_3d_limits):
    if len(bbox_3d_limits) == 2:
        min_p, max_p = bbox_3d_limits
    elif len(bbox_3d_limits) == 6:
        min_p = bbox_3d_limits[:3]
        max_p = bbox_3d_limits[3:]
    else: return None, None
    v = np.array([
        [min_p[0], min_p[1], min_p[2]], [max_p[0], min_p[1], min_p[2]],
        [max_p[0], max_p[1], min_p[2]], [min_p[0], max_p[1], min_p[2]],
        [min_p[0], min_p[1], max_p[2]], [max_p[0], min_p[1], max_p[2]],
        [max_p[0], max_p[1], max_p[2]], [min_p[0], max_p[1], max_p[2]]
    ])
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    return v, edges

def project_pts(pts_3d, ext_param, device):
    img_size = ext_param.get('image_size', 1024)
    R = torch.from_numpy(np.array(ext_param['R'])).float().to(device).unsqueeze(0)
    T = torch.from_numpy(np.array(ext_param['T'])).float().to(device).unsqueeze(0)
    K = torch.from_numpy(np.array(ext_param['K'])).float().to(device)
    cameras = PerspectiveCameras(
        device=device, R=R, T=T,
        focal_length=torch.tensor([[K[0, 0], K[1, 1]]], device=device),
        principal_point=torch.tensor([[K[0, 2], K[1, 2]]], device=device),
        in_ndc=False, image_size=((img_size, img_size),)
    )
    pts_torch = torch.from_numpy(pts_3d).float().to(device).unsqueeze(0)
    screen_pts = cameras.transform_points(pts_torch)
    u = img_size - screen_pts[0, :, 0].cpu().numpy()
    v = img_size - screen_pts[0, :, 1].cpu().numpy()
    return u, v

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return None
    return json.loads(m.group(0))

def ensure_pil(img) -> Image.Image:
    if isinstance(img, Image.Image): return img
    if isinstance(img, (bytes, bytearray)): return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, str): return Image.open(img).convert("RGB")
    return Image.fromarray(img)

# ================== 核心可视化函数 ==================

def visualize_sample(model, processor, item, device, item_idx):
    sample_dir = os.path.join(VIS_OUTPUT_DIR, f"sample_{item_idx:06d}")
    os.makedirs(sample_dir, exist_ok=True)

    images, messages = item['images'], item['messages']
    ext_param = item.get('ext_param')
    pc_raw = np.array(item['point_cloud'], dtype=np.float32).reshape(-1, 3)
    base_image = ensure_pil(images[0]).copy()
    img_size = ext_param.get('image_size', max(base_image.size))

    # --- 1. 模型推理 & 保存 Input ---
    infer_messages = [messages[0]]
    input_prompt = processor.apply_chat_template(infer_messages, tokenize=False, add_generation_prompt=True)
    
    # 保存 Input Prompt
    with open(os.path.join(sample_dir, "model_input.txt"), "w", encoding='utf-8') as f:
        f.write(input_prompt)

    enc = processor(text=[input_prompt], images=[base_image], padding=True, return_tensors="pt").to(device)
    enc["point_cloud"] = torch.from_numpy(pc_raw).to(device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        generated_ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS)
    
    response_text = processor.decode(generated_ids[0][len(enc["input_ids"][0]):], skip_special_tokens=True)
    model_json = extract_json_from_text(response_text)

    # --- 2. 基础信息保存 ---
    base_image.save(os.path.join(sample_dir, "image_original.png"))
    with open(os.path.join(sample_dir, "model_output.txt"), "w", encoding='utf-8') as f:
        f.write(response_text)
    with open(os.path.join(sample_dir, "ground_truth.json"), "w", encoding='utf-8') as f:
        json.dump(messages[1], f, indent=2, ensure_ascii=False)

    # --- 3. 点云基础投影图 ---
    u_pc, v_pc = project_pts(pc_raw, ext_param, device)
    z_values = pc_raw[:, 2]
    
    fig_grad, ax_grad = plt.subplots(figsize=(8, 8), facecolor='white')
    ax_grad.scatter(u_pc, v_pc, c=z_values, cmap='viridis', s=12, alpha=0.6, edgecolors='none')
    ax_grad.set_xlim(0, img_size); ax_grad.set_ylim(img_size, 0); ax_grad.axis('off')
    plt.savefig(os.path.join(sample_dir, "pc_original_gradient.png"), bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(fig_grad)

    if model_json:
        # --- 4. 2D Bbox 可视化 (独立文件) ---
        bbox_2d = model_json.get("bbox")
        if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
            img_bbox = base_image.copy()
            draw_bbox = ImageDraw.Draw(img_bbox)
            draw_bbox.rectangle(bbox_2d, outline="#FF3B30", width=6)
            img_bbox.save(os.path.join(sample_dir, "image_with_model_2d_bbox.png"))
        
        # --- 5. 2D Points 可视化 (独立文件) ---
        points_2d = model_json.get("point_2d")
        if isinstance(points_2d, list) and len(points_2d) > 0:
            img_pts = base_image.copy()
            draw_pts = ImageDraw.Draw(img_pts)
            for pt in points_2d:
                if len(pt) == 2:
                    draw_pts.ellipse([pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5], fill="#FF3B30", outline="white")
            img_pts.save(os.path.join(sample_dir, "image_with_model_2d_points.png"))

        # --- 6. 3D BBox 可视化 ---
        fig_3dbox, ax_3dbox = plt.subplots(figsize=(8, 8), facecolor='white')
        ax_3dbox.scatter(u_pc, v_pc, c='black', s=8, alpha=0.4, edgecolors='none') 
        bbox_3d = model_json.get("3d_bbox")
        if bbox_3d:
            verts, edges = get_3d_bbox_edges(bbox_3d)
            if verts is not None:
                u_v, v_v = project_pts(verts, ext_param, device)
                for e in edges:
                    ax_3dbox.plot([u_v[e[0]], u_v[e[1]]], [v_v[e[0]], v_v[e[1]]], color='#007AFF', linewidth=3)
        ax_3dbox.set_xlim(0, img_size); ax_3dbox.set_ylim(img_size, 0); ax_3dbox.axis('off')
        plt.savefig(os.path.join(sample_dir, "pc_with_model_3d_bbox.png"), bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig_3dbox)

        # --- 7. 3D Points 可视化 ---
        points_3d = model_json.get("point_3d")
        if points_3d and len(points_3d) > 0:
            fig_3dpts, ax_3dpts = plt.subplots(figsize=(8, 8), facecolor='white')
            ax_3dpts.scatter(u_pc, v_pc, c='black', s=8, alpha=0.4, edgecolors='none')
            u_3d, v_3d = project_pts(np.array(points_3d), ext_param, device)
            ax_3dpts.scatter(u_3d, v_3d, c='#FF3B30', s=50, marker='o', edgecolors='white', linewidths=1.5, zorder=10)
            ax_3dpts.set_xlim(0, img_size); ax_3dpts.set_ylim(img_size, 0); ax_3dpts.axis('off')
            plt.savefig(os.path.join(sample_dir, "pc_with_model_3d_points.png"), bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close(fig_3dpts)

# ================== 多进程架构 (去掉了所有的 Try-Catch 拦截) ==================

def load_model_for_worker(device: str):
    print(f"Loading model on {device}...")
    model = Qwen2_5_PVLForConditionalGeneration.from_pretrained(
        MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=torch.bfloat16, device_map=None
    )
    custom_path = f"{MODEL_NAME_OR_PATH}/model-00002-of-00002.safetensors"
    if os.path.exists(custom_path):
        custom_state_dict = load_file(custom_path, device=device)
        ca_sd = {k[len("model.cross_attn."):]: v for k, v in custom_state_dict.items() if k.startswith("model.cross_attn.")}
        pc_sd = {k[len("model.pc_extractor."):]: v for k, v in custom_state_dict.items() if k.startswith("model.pc_extractor.")}
        if ca_sd: model.model.cross_attn.load_state_dict(ca_sd, strict=False)
        if pc_sd: model.model.pc_extractor.load_state_dict(pc_sd, strict=False)
    model.eval().to(device)
    return model

def worker_process(process_id: int, device: str, task_queue: mp.Queue):
    if "cuda" in device:
        torch.cuda.set_device(int(device.split(":")[-1]))
    
    # 初始化失败将直接抛出 traceback
    model = load_model_for_worker(device)
    processor = AutoProcessor.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    
    print(f"Worker {process_id} Ready on {device}.")

    while True:
        task = task_queue.get()
        if task is None: break
        idx, item_data = task
        # 处理样本时出错也会直接抛出错误
        visualize_sample(model, processor, item_data, device, idx)

def main():
    mp.set_start_method('spawn', force=True)
    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_dataset("parquet", data_files={'test': f'{DATA_PATH}/*.parquet'})['test']
    num_samples = len(dataset)
    
    task_queue = mp.Queue()
    for i in range(num_samples):
        task_queue.put((i, dataset[i]))
    for _ in range(TOTAL_WORKERS):
        task_queue.put(None)

    processes = []
    for i in range(TOTAL_WORKERS):
        device = f'cuda:{i % NUM_GPUS}'
        p = mp.Process(target=worker_process, args=(i, device, task_queue))
        p.start()
        processes.append(p)
        time.sleep(10) # 错开加载，防止瞬时 IO 过高

    with tqdm(total=num_samples, desc="Processing") as pbar:
        last_count = 0
        while any(p.is_alive() for p in processes):
            # 通过统计 sample 文件夹数量来更新进度条
            current_files = len([d for d in os.listdir(VIS_OUTPUT_DIR) if d.startswith("sample_")])
            pbar.update(max(0, current_files - last_count))
            last_count = current_files
            time.sleep(5)
            
            # 如果进程减少但任务没完，说明出 Traceback 崩溃了
            if sum(1 for p in processes if p.is_alive()) < TOTAL_WORKERS and last_count < num_samples:
                pass 

    for p in processes: p.join()

if __name__ == "__main__":
    main()