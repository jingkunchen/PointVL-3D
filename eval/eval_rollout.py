import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional
import io

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoProcessor
from tqdm import tqdm
import numpy as np

# 假设这个模块是存在的
from model_arch.modeling_qwen2_5_pvl_batch import Qwen2_5_PVLForConditionalGeneration
from safetensors.torch import load_file

from PIL import Image

# --- 全局配置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda:0'
MODEL_NAME_OR_PATH = "/home/data/nas_hdd/cbx/pvllm/out_dir_batch/checkpoint-1000"
DATA_PATH = "/data/ruoshi/pcdata/sft_data_v2/test"
TRUST_REMOTE_CODE = True
MAX_NEW_TOKENS = 2048
SAVE_DIR = "/home/data/nas_hdd/cbx/pvllm/eval/rollout_validation" 

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 修改：核心逻辑变化
NUM_ROLLOUTS = 8  # 每个样本重复8次
DATALOADER_BATCH_SIZE = 1 # DataLoader 每次只加载一个样本
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

os.makedirs(SAVE_DIR, exist_ok=True)
# -----------------


# --- 校验函数 (无变化) ---

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
    """计算 3D 点在 3D 边界框内的比例。"""
    if not isinstance(points_3d, list) or not points_3d: return 0.0
    if not (isinstance(bbox_3d, list) and len(bbox_3d) == 6): return 0.0

    in_box_count, valid_points_count = 0, 0
    xmin, ymin, zmin, xmax, ymax, zmax = bbox_3d
    
    for point in points_3d:
        if not (isinstance(point, list) and len(point) == 3): continue
        valid_points_count += 1
        px, py, pz = point
        if xmin <= px <= xmax and ymin <= py <= ymax and zmin <= pz <= zmax:
            in_box_count += 1
            
    if valid_points_count == 0: return 0.0
    return in_box_count / valid_points_count

# --- 核心辅助函数 (无变化) ---

def compare_qa_answers(model_output_json: Dict[str, Any], ground_truth_json: Dict[str, Any]) -> Tuple[int, int]:
    try:
        gt_details = ground_truth_json.get('answer', {}).get('details', [])
        model_details = model_output_json.get('answer', {}).get('details', [])
        if not gt_details or not model_details: return 0, 0
        model_answers_map = {item['question']: item['answer'] for item in model_details if isinstance(item, dict) and 'question' in item and 'answer' in item}
        correct_count = 0
        for gt_item in gt_details:
            if isinstance(gt_item, dict) and gt_item.get('question') in model_answers_map and model_answers_map[gt_item['question']] == gt_item.get('answer'):
                correct_count += 1
        return correct_count, len(gt_details)
    except Exception as e:
        print(f"[Error] Failed to compare JSONs: {e}"); return 0, 0

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str): return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception as e:
        print(f"[Warn] json.loads failed: {e}\nCandidate head: {m.group(0)[:200]}"); return None

def ensure_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image): return img.convert("RGB")
    if isinstance(img, (bytes, bytearray)): return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, str): return Image.open(img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")

# --- 模型加载和数据整理 (无变化) ---

def load_model() -> Qwen2_5_PVLForConditionalGeneration:
    model = Qwen2_5_PVLForConditionalGeneration.from_pretrained(
        MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=torch.bfloat16, device_map=None
    )
    custom_path = os.path.join(MODEL_NAME_OR_PATH, "model-00002-of-00002.safetensors")
    if os.path.exists(custom_path):
        custom_state_dict = load_file(custom_path, device=DEVICE)
        cross_attn_state_dict = {k[len("model.cross_attn."):]: v for k, v in custom_state_dict.items() if k.startswith("model.cross_attn.")}
        pc_extractor_state_dict = {k[len("model.pc_extractor."):]: v for k, v in custom_state_dict.items() if k.startswith("model.pc_extractor.")}
        print(f"Found {len(cross_attn_state_dict)} keys for cross_attn and {len(pc_extractor_state_dict)} for pc_extractor.")
        if cross_attn_state_dict:
            try: model.model.cross_attn.load_state_dict(cross_attn_state_dict); print("Loaded cross_attn weights.")
            except Exception as e: print(f"[Warn] load cross_attn failed: {e}")
        if pc_extractor_state_dict:
            try: model.model.pc_extractor.load_state_dict(pc_extractor_state_dict); print("Loaded pc_extractor weights.")
            except Exception as e: print(f"[Warn] load pc_extractor failed: {e}")
    else: print(f"[Info] custom weights not found at {custom_path}.")
    model.eval(); return model

def collate_fn(batch: List[Dict[str, Any]]):
    batch_messages_for_inference, batch_images, batch_point_clouds, batch_ground_truth_jsons = [], [], [], []
    for item in batch:
        messages = item['messages']
        batch_messages_for_inference.append([messages[0]]) 
        batch_ground_truth_jsons.append(messages[1]['content'][0]['text'])
        batch_images.append(ensure_pil(item['images'][0]))
        pc_np = np.array(item['point_cloud'], dtype=np.float32).reshape(-1, 3)
        batch_point_clouds.append(pc_np)
    return {
        "messages_for_inference": batch_messages_for_inference,
        "images": batch_images,
        "point_clouds": torch.from_numpy(np.stack(batch_point_clouds, axis=0)).to(torch.float32),
        "ground_truths": batch_ground_truth_jsons,
    }

# --- 主函数 ---

def main():
    model = load_model().to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE, padding_side="left")

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    dataset = load_dataset("parquet", data_files={'test': f'{DATA_PATH}/*.parquet'})['test']
    dataloader = DataLoader(dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    print(f"Found {len(dataset)} items. Starting evaluation with {NUM_ROLLOUTS} rollouts per sample...")
    
    # 初始化全局指标的累加器
    global_total_iou, global_iou_count = 0.0, 0
    global_total_ratio, global_ratio_count = 0.0, 0
    global_total_correct, global_total_q = 0, 0

    for sample_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating Samples")):
        # 1) 获取单个样本的数据
        single_message = batch["messages_for_inference"][0]
        single_image = batch["images"][0]
        single_pc = batch["point_clouds"][0]  # Shape: [N, 3]
        ground_truth_str = batch["ground_truths"][0]

        # 2) 复制数据，创建 rollout 批次
        rollout_messages = [single_message] * NUM_ROLLOUTS
        rollout_images = [single_image] * NUM_ROLLOUTS
        rollout_pcs = single_pc.unsqueeze(0).repeat(NUM_ROLLOUTS, 1, 1).to(DEVICE)

        # 3) 准备模型输入
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in rollout_messages]
        inputs = processor(text=texts, images=rollout_images, padding=True, return_tensors="pt").to(DEVICE)
        inputs["point_cloud"] = rollout_pcs
        
        # 4) 模型批量生成 (带采样)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
            )
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # 5) 处理当前样本的所有 rollouts
        print(f"\n\n--- Processing Sample {sample_idx} ---")
        try: 
            ground_truth_json = json.loads(ground_truth_str)
        except Exception as e: 
            print(f"[Error] Bad GT JSON for sample {sample_idx}, skipping: {e}"); continue
            
        sample_rollouts_detailed_metrics = []

        for rollout_idx, response_text in enumerate(output_texts):
            model_json = extract_json_from_text(response_text)
            
            metrics = {
                'rollout_id': rollout_idx, 'qa_correct': None, 'qa_total': None,
                'iou': None, 'pc_ratio': None, 'parse_ok': False
            }

            if model_json and ground_truth_json:
                metrics['parse_ok'] = True
                
                # QA 准确率
                correct, total = compare_qa_answers(model_json, ground_truth_json)
                if total > 0:
                    metrics['qa_correct'] = correct; metrics['qa_total'] = total
                    global_total_correct += correct; global_total_q += total
                
                # 3D IoU
                model_3d = model_json.get("3d_bbox")
                gt_3d = ground_truth_json.get("3d_bbox")

                # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                # 修复笔误
                # 确保在修改变量之前，先用原始变量进行所有检查
                if isinstance(model_3d, list) and len(model_3d) == 2 and all(isinstance(s, list) for s in model_3d): 
                    model_3d = model_3d[0] + model_3d[1]
                
                # 这里是出错的地方，把所有的 gt_d 都改回 gt_3d
                if isinstance(gt_3d, list) and len(gt_3d) == 2 and all(isinstance(s, list) for s in gt_3d): 
                    gt_3d = gt_3d[0] + gt_3d[1]
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

                if model_3d and gt_3d:
                    iou = calculate_3d_iou(model_3d, gt_3d)
                    metrics['iou'] = iou
                    global_total_iou += iou; global_iou_count += 1

                # 3D 关键点在GT框内比例
                model_points_3d = model_json.get("point_3d")
                if model_points_3d and gt_3d:
                    ratio = calculate_points_in_bbox_ratio(model_points_3d, gt_3d)
                    metrics['pc_ratio'] = ratio
                    global_total_ratio += ratio; global_ratio_count += 1
            
            sample_rollouts_detailed_metrics.append(metrics)

        # 6) 详细打印每个rollout的24个指标
        print(f"--- Detailed Metrics for Sample {sample_idx} ---")
        valid_ious, valid_ratios, valid_qa_accs = [], [], []
        
        for m in sample_rollouts_detailed_metrics:
            status = "OK" if m['parse_ok'] else "PARSE FAILED"
            qa_str = "N/A"
            if m['qa_correct'] is not None and m['qa_total'] is not None:
                qa_acc = 100 * m['qa_correct'] / m['qa_total']
                qa_str = f"{m['qa_correct']}/{m['qa_total']} ({qa_acc:.1f}%)"
                valid_qa_accs.append(qa_acc)
            
            iou_val, pc_ratio_val = m['iou'], m['pc_ratio']
            if iou_val is not None: valid_ious.append(iou_val)
            if pc_ratio_val is not None: valid_ratios.append(pc_ratio_val)

            iou_str = f"{iou_val:.4f}" if iou_val is not None else "N/A"
            pc_ratio_str = f"{pc_ratio_val:.4f}" if pc_ratio_val is not None else "N/A"
            
            print(f"  Rollout {m['rollout_id']}:")
            print(f"    - QA Accuracy   : {qa_str}")
            print(f"    - 3D IoU        : {iou_str}")
            print(f"    - PC Ratio      : {pc_ratio_str}")
            print(f"    - Parse Status  : {status}")

        print(f"--- Statistics for Sample {sample_idx} over {NUM_ROLLOUTS} rollouts ---")
        if valid_ious: print(f"  Avg IoU    : {np.mean(valid_ious):.4f}  |  Std Dev: {np.std(valid_ious):.4f}")
        if valid_ratios: print(f"  Avg PC Ratio : {np.mean(valid_ratios):.4f}  |  Std Dev: {np.std(valid_ratios):.4f}")
        if valid_qa_accs: print(f"  Avg QA Acc   : {np.mean(valid_qa_accs):.2f}% |  Std Dev: {np.std(valid_qa_accs):.2f}%")

        # 7) 将所有 rollout 的原始输出 dump 到一个JSON文件
        dump_data = {
            "sample_index": sample_idx,
            "input_prompt": texts[0],
            "ground_truth_json": ground_truth_json,
            "rollout_outputs": [{"rollout_id": i, "raw_text": txt} for i, txt in enumerate(output_texts)],
            "detailed_metrics": sample_rollouts_detailed_metrics
        }
        json_save_path = os.path.join(SAVE_DIR, f"sample_{sample_idx}_rollouts.json")
        try:
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(dump_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[Error] Failed to save rollout JSON for sample {sample_idx}: {e}")

    # --- 最终汇总 ---
    print("\n\n===== Overall Evaluation Summary (Averaged over all rollouts) =====")
    if global_total_q > 0: print(f"Final QA Accuracy: {global_total_correct}/{global_total_q} ({100*global_total_correct/global_total_q:.2f}%)")
    if global_iou_count > 0: print(f"Final Avg 3D Bbox IoU: {global_total_iou/global_iou_count:.4f} (from {global_iou_count} valid rollouts)")
    if global_ratio_count > 0: print(f"Final Avg Points in GT Bbox Ratio: {global_total_ratio/global_ratio_count:.4f} (from {global_ratio_count} valid rollouts)")

if __name__ == "__main__":
    main()