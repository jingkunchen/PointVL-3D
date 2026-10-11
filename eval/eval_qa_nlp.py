import os
import re
import json
import time
import io
import math
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoProcessor
from tqdm import tqdm
import numpy as np
from PIL import Image

# --- NLP Metrics ---
import evaluate
from bert_score import BERTScorer

# --- Model Arch ---
from model_arch.modeling_qwen2_5_pvl_batch import Qwen2_5_PVLForConditionalGeneration
from safetensors.torch import load_file

# --- 全局配置 ---
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" 
NUM_GPUS = 2
WORKERS_PER_GPU = 3  # 显存允许可增加
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

MODEL_NAME_OR_PATH = "/home/data/nas_hdd/cbx/pvllm/ckpt_cache/output_grpo_dr/checkpoint-1100"   
DATA_PATH = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test"
TRUST_REMOTE_CODE = True
MAX_NEW_TOKENS = 4096
OUTPUT_DIR = "/home/data/nas_hdd/cbx/pvllm/eval/output_dir_qa_nlp"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================== 工具函数 ==================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """从文本中抽取 JSON 对象"""
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

def normalize_text(text: str) -> str:
    """文本标准化：去空格，转小写"""
    if text is None: return ""
    return str(text).strip().lower()

# ================== 模型加载与推理 (Worker) ==================

def load_model_for_worker(device: str) -> Qwen2_5_PVLForConditionalGeneration:
    """Worker 加载 VLM 模型"""
    if "cuda" in device:
        try:
            torch.cuda.set_device(int(device.split(":")[-1]))
        except: pass

    model = Qwen2_5_PVLForConditionalGeneration.from_pretrained(
        MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=torch.bfloat16, device_map=None
    )
    
    custom_path = f"{MODEL_NAME_OR_PATH}/model-00002-of-00002.safetensors"
    if os.path.exists(custom_path):
        custom_state_dict = load_file(custom_path, device=device)
        cross_attn = {k[len("model.cross_attn."):]: v for k, v in custom_state_dict.items() if k.startswith("model.cross_attn.")}
        pc_ext = {k[len("model.pc_extractor."):]: v for k, v in custom_state_dict.items() if k.startswith("model.pc_extractor.")}
        if cross_attn: 
            try: model.model.cross_attn.load_state_dict(cross_attn)
            except: pass
        if pc_ext: 
            try: model.model.pc_extractor.load_state_dict(pc_ext)
            except: pass
            
    model.eval()
    return model.to(device)

def process_single_sample(model, processor, item, device, item_idx):
    """返回: (qa_correct, qa_total, pred_desc, gt_desc, success)"""
    images, messages = item['images'], item['messages']
    point_cloud_np = np.array(item['point_cloud'], dtype=np.float32).reshape(-1, 3)
    
    ground_truth_str = messages[1]['content'][0]['text']
    infer_messages = [messages[0]]

    base_image = ensure_pil(images[0]).copy()
    text = processor.apply_chat_template(infer_messages, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=[base_image], padding=True, return_tensors="pt").to(device)
    enc["point_cloud"] = torch.from_numpy(point_cloud_np).to(device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        generated_ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS)
    response_text = processor.decode(generated_ids[0][len(enc["input_ids"][0]):], skip_special_tokens=True)

    with open(os.path.join(OUTPUT_DIR, f"{item_idx}_log.txt"), "w", encoding='utf-8') as f:
        f.write(f"GT:\n{ground_truth_str}\n\nPRED:\n{response_text}")

    model_json = extract_json_from_text(response_text)
    try:
        gt_json = json.loads(ground_truth_str)
    except:
        gt_json = None

    if not model_json or not gt_json:
        return 0, 0, "", "", False

    # --- QA Logic ---
    qa_correct = 0
    qa_total = 0
    
    model_ans = model_json.get('answer', {})
    gt_ans = gt_json.get('answer', {})

    gt_label = gt_ans.get('label', {}).get('answer')
    model_label = model_ans.get('label', {}).get('answer')
    if gt_label:
        qa_total += 1
        if normalize_text(model_label) == normalize_text(gt_label):
            qa_correct += 1
            
    gt_details = gt_ans.get('details', [])
    model_details = model_ans.get('details', [])
    
    def make_qa_map(details):
        m = {}
        if isinstance(details, list):
            for d in details:
                if isinstance(d, dict) and 'question' in d:
                    m[normalize_text(d['question'])] = normalize_text(str(d.get('answer', '')))
        return m

    gt_map = make_qa_map(gt_details)
    model_map = make_qa_map(model_details)

    for q, a_gt in gt_map.items():
        qa_total += 1
        if q in model_map and model_map[q] == a_gt:
            qa_correct += 1

    # --- Desc Logic ---
    pred_desc = str(model_json.get('desc', "")).strip()
    gt_desc = str(gt_json.get('desc', "")).strip()

    return qa_correct, qa_total, pred_desc, gt_desc, True

def worker_process(pid, device, task_queue, result_queue):
    print(f"[Worker {pid}] Init on {device}")
    try:
        model = load_model_for_worker(device)
        processor = AutoProcessor.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    except Exception as e:
        print(f"[Worker {pid}] Load Error: {e}")
        return

    while True:
        task = task_queue.get()
        if task is None: break
        idx, item = task
        try:
            res = process_single_sample(model, processor, item, device, idx)
            result_queue.put(res)
        except Exception as e:
            print(f"Error item {idx}: {e}")
            result_queue.put((0, 0, "", "", False))

# ================== Main Eval Loop ==================

def main():
    mp.set_start_method('spawn', force=True)
    
    print("Loading Dataset...")
    dataset = load_dataset("parquet", data_files={'test': f'{DATA_PATH}/*.parquet'})['test']
    num_samples = len(dataset)
    
    print("Loading Evaluation Metrics...")
    try:
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
        # BERTScore 使用第一张卡 (cuda:0) 进行计算
        bert_scorer = BERTScorer(lang="en", rescale_with_baseline=False, device="cuda:0") 
    except Exception as e:
        print(f"Metrics load failed: {e}")
        return

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    for i in range(num_samples): task_queue.put((i, dataset[i]))
    for _ in range(TOTAL_WORKERS): task_queue.put(None)

    procs = []
    for i in range(TOTAL_WORKERS):
        d = f'cuda:{i % NUM_GPUS}'
        p = mp.Process(target=worker_process, args=(i, d, task_queue, result_queue))
        p.start(); procs.append(p)
        time.sleep(2)

    # --- 统计累加器 ---
    acc_qa_correct = 0
    acc_qa_total = 0
    
    # 存储各项指标的总和
    # BLEU 1, 2, 3, 4
    sum_bleu = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    # ROUGE 1, 2, L
    sum_rouge = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    sum_bert_f1 = 0.0
    
    valid_qa_samples = 0
    valid_desc_samples = 0
    
    pbar = tqdm(total=num_samples, desc="Eval")
    
    for _ in range(num_samples):
        qa_c, qa_t, p_desc, g_desc, success = result_queue.get()
        
        if success:
            acc_qa_correct += qa_c
            acc_qa_total += qa_t
            valid_qa_samples += 1
            
            # --- 文本指标计算 (如果 GT 不为空) ---
            if g_desc:
                # 1. ROUGE (一次返回所有)
                try:
                    r_res = rouge_metric.compute(predictions=[p_desc], references=[g_desc])
                    # 累加 rouge1, rouge2, rougeL
                    for k in sum_rouge.keys():
                        if k in r_res:
                            sum_rouge[k] += r_res[k]
                except: pass
                
                # 2. BLEU (循环计算 1-4)
                for n in range(1, 5):
                    try:
                        b_res = bleu_metric.compute(predictions=[p_desc], references=[g_desc], max_order=n)
                        sum_bleu[n] += b_res['bleu']
                    except: pass
                
                # 3. BERTScore
                try:
                    _, _, b_f1 = bert_scorer.score([p_desc], [g_desc])
                    sum_bert_f1 += b_f1.item()
                except: pass

                valid_desc_samples += 1
        
        # --- 更新进度条 (为了简洁，只显示最关键的 QA, B4, RL, BS) ---
        avg_qa = (acc_qa_correct / acc_qa_total) if acc_qa_total > 0 else 0.0
        
        if valid_desc_samples > 0:
            avg_b4 = sum_bleu[4] / valid_desc_samples
            avg_rl = sum_rouge['rougeL'] / valid_desc_samples
            avg_bert = sum_bert_f1 / valid_desc_samples
            desc_str = f"| B4:{avg_b4:.3f} | RL:{avg_rl:.3f} | BS:{avg_bert:.3f}"
        else:
            desc_str = ""
            
        pbar.set_description(f"QA:{avg_qa:.3f} {desc_str}")
        pbar.update(1)

    pbar.close()
    for p in procs: p.join()

    # --- 最终打印详细报表 ---
    print("\n\n" + "="*40)
    print("     FINAL EVALUATION RESULTS")
    print("="*40)
    
    print(f"Total Processed Samples: {valid_qa_samples} / {num_samples}")
    
    qa_acc = (acc_qa_correct / acc_qa_total) if acc_qa_total else 0.0
    print(f"\n[QA Metrics]")
    print(f"  Accuracy (Label+Details): {qa_acc:.4f} ({acc_qa_correct}/{acc_qa_total})")

    if valid_desc_samples > 0:
        print(f"\n[Description Generation Metrics] (Avg over {valid_desc_samples} samples)")
        
        print("  BLEU Scores:")
        print(f"    BLEU-1   : {sum_bleu[1] / valid_desc_samples:.4f}")
        print(f"    BLEU-2   : {sum_bleu[2] / valid_desc_samples:.4f}")
        print(f"    BLEU-3   : {sum_bleu[3] / valid_desc_samples:.4f}")
        print(f"    BLEU-4   : {sum_bleu[4] / valid_desc_samples:.4f}")
        
        print("-" * 20)
        print("  ROUGE Scores:")
        print(f"    ROUGE-1  : {sum_rouge['rouge1'] / valid_desc_samples:.4f}")
        print(f"    ROUGE-2  : {sum_rouge['rouge2'] / valid_desc_samples:.4f}")
        print(f"    ROUGE-L  : {sum_rouge['rougeL'] / valid_desc_samples:.4f}")
        
        print("-" * 20)
        print(f"  BERTScore  : {sum_bert_f1 / valid_desc_samples:.4f}")
    else:
        print("\n[Description Generation Metrics]")
        print("  No valid descriptions found.")
        
    print("="*40)

if __name__ == "__main__":
    main()