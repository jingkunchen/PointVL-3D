import os
import re
import json
import io
import base64
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, List, Optional
import torch

# --- NLP Metrics ---
import evaluate
from bert_score import BERTScorer
from openai import OpenAI

# ================== 全局配置 ==================
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "qwen_sft"

DATA_PATH = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4_2d/test"
BERT_SCORE_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON = "vllm_detailed_eval_single.json"

# ================== 工具函数 ==================

def extract_json_from_text(text: str) -> Optional[Dict]:
    if not isinstance(text, str): return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except:
        return None

def normalize_text(text: Any) -> str:
    if text is None: return ""
    return str(text).strip().lower()

def encode_image_to_base64(image_input) -> str:
    if isinstance(image_input, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        return ""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_2d_iou(box1: List, box2: List) -> float:
    if not (isinstance(box1, list) and len(box1) == 4 and isinstance(box2, list) and len(box2) == 4):
        return 0.0
    inter_xmin, inter_ymin = max(box1[0], box2[0]), max(box1[1], box2[1])
    inter_xmax, inter_ymax = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_w, inter_h = max(0, inter_xmax - inter_xmin), max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

def calculate_pt_acc(points: List[List[float]], gt_bbox: List[float]) -> float:
    if not points or not isinstance(points, list) or len(gt_bbox) != 4: return 0.0
    count = 0
    x1, y1, x2, y2 = gt_bbox
    valid_pts = 0
    for p in points:
        if isinstance(p, list) and len(p) == 2:
            valid_pts += 1
            if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                count += 1
    return count / valid_pts if valid_pts > 0 else 0.0

# ================== 评估逻辑 ==================

class Evaluator:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        print("Loading NLP Metrics (BLEU, ROUGE)...")
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        print(f"Loading BERTScore on {BERT_SCORE_DEVICE}...")
        self.bert_scorer = BERTScorer(lang="en", device=BERT_SCORE_DEVICE)

    def compute_nlp_scores(self, pred: str, ref: str) -> Dict[str, float]:
        if not ref.strip():
            return {f"bleu{i}": 0.0 for i in range(1, 5)} | {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bert_score": 0.0}
        
        # 修正 pred 为空的情况
        if not pred.strip():
            return {f"bleu{i}": 0.0 for i in range(1, 5)} | {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bert_score": 0.0}

        scores = {}
        # BLEU 1-4
        for i in range(1, 5):
            try:
                # 注意：references 需要是嵌套列表 [[ref]]
                res = self.bleu_metric.compute(predictions=[pred], references=[[ref]], max_order=i)
                scores[f'bleu{i}'] = res['bleu']
            except: scores[f'bleu{i}'] = 0.0
            
        # ROUGE 1, 2, L
        try:
            res = self.rouge_metric.compute(predictions=[pred], references=[ref])
            scores['rouge1'] = res['rouge1']
            scores['rouge2'] = res['rouge2']
            scores['rougeL'] = res['rougeL']
        except:
            scores['rouge1'] = scores['rouge2'] = scores['rougeL'] = 0.0
            
        # BERTScore
        try:
            P, R, F1 = self.bert_scorer.score([pred], [ref])
            scores['bert_score'] = F1.item()
        except: scores['bert_score'] = 0.0
        
        return scores

def main():
    from datasets import load_dataset
    print("Loading Dataset...")
    dataset = load_dataset("parquet", data_files={'test': f'{DATA_PATH}/*.parquet'})['test']
    
    evaluator = Evaluator()
    
    metrics_sum = {
        'iou_2d': 0.0, 'pt_acc': 0.0, 'qa_acc': 0.0,
        'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0,
        'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'bert_score': 0.0
    }
    
    total_samples = len(dataset)
    valid_count = 0

    print(f"\nStarting Single-Threaded Evaluation ({total_samples} samples)...\n")

    for i in range(total_samples):
        item = dataset[i]
        
        # 1. 准备请求
        user_msgs = item['messages'][0]['content']
        image_data = item['images'][0]
        prompt_text = "".join([c['text'] + "\n" for c in user_msgs if c['type'] == 'text'])
        base64_img = encode_image_to_base64(image_data)
        
        # 2. 调用 API
        try:
            response = evaluator.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }],
                temperature=0.0
            )
            model_output = response.choices[0].message.content
        except Exception as e:
            print(f"\n[Sample {i}] API Error: {e}")
            continue

        # 3. 解析
        gt_raw = item['messages'][1]['content'][0]['text']
        gt_json = extract_json_from_text(gt_raw)
        pred_json = extract_json_from_text(model_output)

        print(f"\n{'='*20} Sample {i} {'='*20}")
        if not pred_json:
            print(f"FAILED TO PARSE JSON. RAW OUTPUT:\n{model_output}")
            continue

        # 4. 计算本条指标
        res = {}
        # Geometry
        res['iou_2d'] = calculate_2d_iou(pred_json.get('bbox'), gt_json.get('bbox'))
        res['pt_acc'] = calculate_pt_acc(pred_json.get('point_2d'), gt_json.get('bbox'))
        
        # QA
        correct, total = 0, 0
        gt_ans = gt_json.get('answer', {})
        pred_ans = pred_json.get('answer', {})
        
        # Label
        g_lab = normalize_text(gt_ans.get('label', {}).get('answer'))
        p_lab = normalize_text(pred_ans.get('label', {}).get('answer'))
        if g_lab:
            total += 1
            if g_lab == p_lab: correct += 1
        
        # Details
        g_det = {normalize_text(d.get('question')): normalize_text(d.get('answer')) for d in gt_ans.get('details', [])}
        p_det = {normalize_text(d.get('question')): normalize_text(d.get('answer')) for d in pred_ans.get('details', [])}
        for q, a in g_det.items():
            total += 1
            if p_det.get(q) == a: correct += 1
        res['qa_acc'] = correct / total if total > 0 else 0.0

        # NLP
        nlp_res = evaluator.compute_nlp_scores(pred_json.get('desc', ""), gt_json.get('desc', ""))
        res.update(nlp_res)

        # 5. 打印对比
        print(f"PRED DESC: {pred_json.get('desc')}")
        print(f"G.T. DESC: {gt_json.get('desc')}")
        print(f"--- Metrics ---")
        metric_line = " | ".join([f"{k}: {res[k]:.4f}" for k in ['iou_2d', 'qa_acc', 'bert_score']])
        print(metric_line)
        nlp_line = f"BLEU: {res['bleu1']:.3f}/{res['bleu2']:.3f}/{res['bleu3']:.3f}/{res['bleu4']:.3f} | ROUGE: {res['rouge1']:.3f}/{res['rouge2']:.3f}/{res['rougeL']:.3f}"
        print(nlp_line)

        # 累加
        valid_count += 1
        for k in metrics_sum:
            metrics_sum[k] += res.get(k, 0.0)

    # 最终汇总
    print("\n" + "#"*60)
    print(f" FINAL SUMMARY ({valid_count} samples) ")
    print("#"*60)
    if valid_count > 0:
        final_results = {k: v/valid_count for k, v in metrics_sum.items()}
        for k, v in final_results.items():
            print(f"{k:12} : {v:.4f}")
        
        with open(OUTPUT_JSON, "w") as f:
            json.dump(final_results, f, indent=4)
    else:
        print("No valid samples were processed.")

if __name__ == "__main__":
    main()