import os
import re
import json
import io
import base64
import threading
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

# --- NLP Metrics ---
import evaluate
from bert_score import BERTScorer
from openai import OpenAI

# ================== 1. 全局配置 ==================
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "qwen_sft"

DATA_PATH = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4_2d/test"
BERT_SCORE_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON = "vllm_full_eval_64threads.json"
MAX_WORKERS = 64  # API 吞吐并发数

# ================== 2. 工具函数 ==================

def extract_json_from_text(text: str) -> Optional[Dict]:
    if not isinstance(text, str): return None
    # 去除 Markdown 代码块标签
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    # 匹配最外层的花括号内容
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
    try:
        if isinstance(image_input, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            return ""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception:
        return ""

def calculate_2d_iou(box1: List, box2: List) -> float:
    if not (isinstance(box1, list) and len(box1) == 4 and isinstance(box2, list) and len(box2) == 4):
        return 0.0
    # box format: [x1, y1, x2, y2]
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

# ================== 3. 核心评估类 ==================

class Evaluator:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        print(f"Loading NLP Metrics (BLEU, ROUGE) and BERTScore on {BERT_SCORE_DEVICE}...")
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        self.bert_scorer = BERTScorer(lang="en", device=BERT_SCORE_DEVICE)
        
        # 核心锁：解决 evaluate 库非线程安全导致 bleu 为 0 的问题
        self.metric_lock = threading.Lock()
        # 数据统计锁：保证全局累加安全
        self.stats_lock = threading.Lock()

    def compute_nlp_scores(self, pred: str, ref: str) -> Dict[str, float]:
        """计算 NLP 指标，内部使用 Lock 确保线程安全"""
        p, r = pred.strip(), ref.strip()
        if not p or not r:
            return {f"bleu{i}": 0.0 for i in range(1, 5)} | {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bert_score": 0.0}

        # 所有的计算操作必须进入临界区
        with self.metric_lock:
            scores = {}
            # BLEU 1-4 (references 必须是嵌套列表 [[ref]])
            for i in range(1, 5):
                try:
                    res = self.bleu_metric.compute(predictions=[p], references=[[r]], max_order=i)
                    scores[f'bleu{i}'] = res['bleu']
                except Exception: scores[f'bleu{i}'] = 0.0
            
            # ROUGE
            try:
                res = self.rouge_metric.compute(predictions=[p], references=[r])
                scores['rouge1'] = res['rouge1']
                scores['rouge2'] = res['rouge2']
                scores['rougeL'] = res['rougeL']
            except Exception:
                scores.update({'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0})
            
            # BERTScore
            try:
                with torch.no_grad():
                    _, _, F1 = self.bert_scorer.score([p], [r])
                scores['bert_score'] = F1.item()
            except Exception:
                scores['bert_score'] = 0.0
            
            return scores

    def process_sample(self, item: Dict):
        """线程工作函数"""
        try:
            # 1. 组装输入
            user_msg_content = item['messages'][0]['content']
            image_data = item['images'][0]
            prompt = "".join([c['text'] + "\n" for c in user_msg_content if c['type'] == 'text'])
            img_b64 = encode_image_to_base64(image_data)
            
            # 2. 调用 API (多线程全速跑 IO)
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]}],
                temperature=0.0,
                timeout=90
            )
            model_output = resp.choices[0].message.content

            # 3. 解析
            gt_raw = item['messages'][1]['content'][0]['text']
            gt_json = extract_json_from_text(gt_raw)
            pred_json = extract_json_from_text(model_output)
            if not gt_json or not pred_json: return None

            # 4. 计算指标
            res = {}
            # 2D 几何
            res['iou_2d'] = calculate_2d_iou(pred_json.get('bbox'), gt_json.get('bbox'))
            res['pt_acc'] = calculate_pt_acc(pred_json.get('point_2d'), gt_json.get('bbox'))
            
            # QA 准确率
            correct, total = 0, 0
            g_ans, p_ans = gt_json.get('answer', {}), pred_json.get('answer', {})
            # Label
            gl = normalize_text(g_ans.get('label', {}).get('answer'))
            pl = normalize_text(p_ans.get('label', {}).get('answer'))
            if gl:
                total += 1
                if gl == pl: correct += 1
            # Details
            g_det = {normalize_text(d.get('question')): normalize_text(d.get('answer')) for d in g_ans.get('details', [])}
            p_det = {normalize_text(d.get('question')): normalize_text(d.get('answer')) for d in p_ans.get('details', [])}
            for q, a in g_det.items():
                total += 1
                if p_det.get(q) == a: correct += 1
            res['qa_acc'] = correct / total if total > 0 else 0.0

            # NLP 指标 (进入 Lock)
            nlp_metrics = self.compute_nlp_scores(pred_json.get('desc', ""), gt_json.get('desc', ""))
            res.update(nlp_metrics)
            
            return res
        except Exception as e:
            return None

# ================== 4. 主循环 ==================

def main():
    from datasets import load_dataset
    print("Loading Dataset from Parquet...")
    dataset = load_dataset("parquet", data_files={'test': f'{DATA_PATH}/*.parquet'})['test']
    
    evaluator = Evaluator()
    
    metrics_sum = {
        'iou_2d': 0.0, 'pt_acc': 0.0, 'qa_acc': 0.0,
        'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0,
        'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'bert_score': 0.0
    }
    valid_count = 0
    total_samples = len(dataset)

    print(f"\nStarting 64-Threaded Evaluation ({total_samples} samples)...")
    pbar = tqdm(total=total_samples, ncols=190)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        futures = {executor.submit(evaluator.process_sample, dataset[i]): i for i in range(total_samples)}
        
        for future in as_completed(futures):
            res = future.result()
            
            with evaluator.stats_lock:
                if res:
                    valid_count += 1
                    for k in metrics_sum:
                        metrics_sum[k] += res[k]
                
                # 动态刷新进度条 (显示所有指标)
                if valid_count > 0:
                    v = valid_count
                    desc = (
                        f"IoU:{metrics_sum['iou_2d']/v:.4f} | Pt:{metrics_sum['pt_acc']/v:.4f} | QA:{metrics_sum['qa_acc']/v:.4f} | "
                        f"B1-4:{metrics_sum['bleu1']/v:.4f}/{metrics_sum['bleu2']/v:.4f}/{metrics_sum['bleu3']/v:.4f}/{metrics_sum['bleu4']/v:.4f} | "
                        f"R1,2,L:{metrics_sum['rouge1']/v:.4f}/{metrics_sum['rouge2']/v:.4f}/{metrics_sum['rougeL']/v:.4f} | "
                        f"BS:{metrics_sum['bert_score']/v:.4f}"
                    )
                    pbar.set_description(desc)
            pbar.update(1)

    pbar.close()

    # 打印最终结果
    print("\n" + "="*80)
    print(f" FINAL EVALUATION REPORT ({valid_count} valid samples) ")
    print("="*80)
    if valid_count > 0:
        final_summary = {k: v/valid_count for k, v in metrics_sum.items()}
        # 格式化打印
        for k, v in final_summary.items():
            print(f"{k:<15} : {v:.4f}")
        
        # 保存结果
        with open(OUTPUT_JSON, "w") as f:
            json.dump(final_summary, f, indent=4)
        print(f"\nResults saved to {OUTPUT_JSON}")
    else:
        print("Error: No valid results were collected.")
    print("="*80)

if __name__ == "__main__":
    # 为了防止某些库在多线程中 fork 报错
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()