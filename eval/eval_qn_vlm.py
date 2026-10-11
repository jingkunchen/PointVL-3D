import os
import re
import json
import io
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# --- NLP Metrics ---
import evaluate
from bert_score import BERTScorer
from openai import OpenAI

# --- 全局配置 ---
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
API_BASE_URL = "http://localhost:8001/v1"
API_KEY = "EMPTY"
MODEL_NAME = "MiMo-VL-7B-RL"

DATA_PATH = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test" 
MAX_WORKERS = 1 
BERT_SCORE_DEVICE = "cuda:7" 
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# ================== 工具函数 ==================

def extract_json_from_text(text: str):
    if not isinstance(text, str): return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def normalize_text(text: str) -> str:
    if text is None: return ""
    return str(text).strip().lower()

def encode_image_to_base64(image_input):
    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image type")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def format_console_output(idx, qa_response, desc_response):
    """
    打印两次请求的原始输出
    """
    lines = []
    lines.append(f"\n[{idx}] {'-'*30}")
    lines.append(">>> [Round 1: QA Output]")
    lines.append(qa_response.strip())
    lines.append("-" * 10)
    lines.append(">>> [Round 2: Desc Output]")
    lines.append(desc_response.strip())
    lines.append(f"{'='*40}")
    return "\n".join(lines)

# ================== 核心处理逻辑 ==================

class Evaluator:
    def __init__(self):
        print("Loading NLP Metrics (BLEU, ROUGE, BERTScore)...")
        try:
            self.bleu_metric = evaluate.load("bleu")
            self.rouge_metric = evaluate.load("rouge")
            # BERTScore 初始化
            print(f"Loading BERTScore on {BERT_SCORE_DEVICE}...")
            self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=False, device=BERT_SCORE_DEVICE)
        except Exception as e:
            print(f"Metrics load failed: {e}")
            raise e
        
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    def call_vllm(self, messages, temperature=0.0):
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[API ERROR] {str(e)}"

    def compute_full_single_metrics(self, pred, ref):
        """
        计算单条样本的所有 NLP 指标，用于滚动更新。
        包含: BLEU 1-4, ROUGE 1/2/L, BERTScore F1
        """
        metrics = {}
        try:
            # --- BLEU 1, 2, 3, 4 ---
            # evaluate 的 bleu compute 默认返回的是 max_order 的分数，所以需要调4次
            for n in range(1, 5):
                res = self.bleu_metric.compute(predictions=[pred], references=[ref], max_order=n)
                metrics[f'bleu_{n}'] = res['bleu']

            # --- ROUGE 1, 2, L ---
            r_res = self.rouge_metric.compute(predictions=[pred], references=[ref])
            metrics['rouge1'] = r_res['rouge1']
            metrics['rouge2'] = r_res['rouge2']
            metrics['rougeL'] = r_res['rougeL']

            # --- BERTScore ---
            # 这是一个比较重的计算，逐条计算会比批量慢，但为了满足滚动显示要求必须这么做
            P, R, F1 = self.bert_scorer.score([pred], [ref])
            metrics['bert_score'] = F1.item() # 取 F1
            
            return metrics
        except Exception as e:
            print(f"Metric calc error: {e}")
            # 出错返回 0，避免程序崩溃
            return {k: 0.0 for k in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'rouge1', 'rouge2', 'rougeL', 'bert_score']}

    def process_sample(self, item, idx):
        images = item['images']
        raw_messages = item['messages']
        
        # 1. 准备 Ground Truth
        try:
            gt_text = raw_messages[1]['content'][0]['text']
            gt_json = json.loads(gt_text)
            gt_desc = str(gt_json.get('desc', "")).strip()
            gt_ans_obj = gt_json.get('answer', {})
        except:
            return 0, 0, "", "", False, f"Sample {idx}: GT JSON Parse Error"

        base64_img = encode_image_to_base64(images[0])
        img_url = f"data:image/jpeg;base64,{base64_img}"

        # ================= Round 1: QA Request =================
        try:
            raw_user_json_str = raw_messages[0]['content'][2]['text']
            user_template_json = json.loads(raw_user_json_str)
            
            qa_template = {
                "question": user_template_json.get("question", ""),
                "answer": user_template_json.get("answer", {})
            }
            qa_prompt_json_str = json.dumps(qa_template, indent=2)
            
        except:
            return 0, 0, "", "", False, f"Sample {idx}: Template Error"

        qa_instruction = "Please answer the QA questions in the following JSON based on the image. Select an option from the 'options' list and fill it into the 'answer' field. Please keep the output structure identical to the original JSON input without any changes."
        
        qa_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": qa_instruction},
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": qa_prompt_json_str}
            ]
        }]

        # Call QA
        qa_response_text = self.call_vllm(qa_messages, temperature=0.1)
        
        # Parse QA
        model_json = extract_json_from_text(qa_response_text)
        qa_correct = 0
        qa_total = 0
        
        if model_json:
            model_ans_obj = model_json.get('answer', {})
            
            # Label Check
            gt_label = gt_ans_obj.get('label', {}).get('answer')
            model_label = model_ans_obj.get('label', {}).get('answer')
            if gt_label:
                qa_total += 1
                if normalize_text(model_label) == normalize_text(gt_label):
                    qa_correct += 1
            
            # Details Check
            def make_qa_map(details_list):
                m = {}
                if isinstance(details_list, list):
                    for d in details_list:
                        if isinstance(d, dict) and 'question' in d:
                            q_norm = normalize_text(d['question'])
                            a_norm = normalize_text(str(d.get('answer', '')))
                            m[q_norm] = a_norm
                return m

            gt_map = make_qa_map(gt_ans_obj.get('details', []))
            model_map = make_qa_map(model_ans_obj.get('details', []))

            for q, a_gt in gt_map.items():
                qa_total += 1
                if model_map.get(q, "N/A") == a_gt:
                    qa_correct += 1

        # ================= Round 2: Description Request =================
        desc_instruction = "Please provide a detailed description of the object in the image in one sentence."
        
        desc_messages = [{
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": desc_instruction}
            ]
        }]

        # Call Desc
        desc_response_text = self.call_vllm(desc_messages, temperature=0.7)

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [MODIFIED] 去除 <think> 部分 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 使用正则去除 <think>...</think> 标签及其中的内容
        # re.S (DOTALL) 标志确保 . 可以匹配换行符
        cleaned_desc_text = re.sub(r"<think>.*?</think>", "", desc_response_text, flags=re.S)
        
        # 对清洗后的文本进行格式化，作为预测结果
        pred_desc = cleaned_desc_text.replace("\n", " ").strip()
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # ================= Logging =================
        # 注意：这里 log_str 依然传入原始的 desc_response_text，以便在控制台看到模型的思考过程
        # 如果你希望 Log 里也看不见 think，可以将 desc_response_text 换成 cleaned_desc_text
        log_str = format_console_output(idx, qa_response_text, desc_response_text)

        return qa_correct, qa_total, pred_desc, gt_desc, True, log_str

# ================== Main Loop ==================

def main():
    print("Loading Dataset...")
    try:
        dataset = load_dataset("parquet", data_files={'test': f'{DATA_PATH}/*.parquet'})['test']
    except:
        dataset = load_dataset(DATA_PATH, split="test")
        
    num_samples = len(dataset)
    print(f"Total samples: {num_samples}")

    evaluator = Evaluator()

    # 累加器 (Accumulators)
    acc_qa_correct = 0
    acc_qa_total = 0
    
    # NLP Metrics 累加器
    sums = {
        'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
        'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
        'bert_score': 0.0
    }
    valid_desc_count = 0

    # 保存所有结果用于最后批量算 (作为校验)
    predictions_desc = []
    references_desc = []
    valid_samples = 0
    
    print(f"Starting Evaluation with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(evaluator.process_sample, dataset[i], i): i 
            for i in range(num_samples)
        }

        # 设置更宽的进度条以显示所有指标
        pbar = tqdm(total=num_samples, desc="Init", ncols=180)
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                qa_c, qa_t, p_desc, g_desc, success, log_str = future.result()
                
                if success:
                    print(log_str) # 打印原始输出

                    acc_qa_correct += qa_c
                    acc_qa_total += qa_t
                    valid_samples += 1
                    
                    if g_desc: 
                        predictions_desc.append(p_desc)
                        references_desc.append(g_desc)
                        
                        # --- 实时计算单个样本的所有指标 ---
                        # 这里每一步都会跑一次 BERTScore，速度会受影响，但能满足实时显示
                        metrics = evaluator.compute_full_single_metrics(p_desc, g_desc)
                        
                        # 累加
                        for k, v in metrics.items():
                            if k in sums:
                                sums[k] += v
                        valid_desc_count += 1
                else:
                    print(f"!!! Error Result for Sample {idx}: {log_str}")

            except Exception as e:
                print(f"CRITICAL ERROR processing item {idx}: {e}")
            
            # --- 更新进度条 ---
            # 计算当前平均值
            avg_qa = (acc_qa_correct / acc_qa_total) if acc_qa_total > 0 else 0.0
            
            avgs = {}
            div = valid_desc_count if valid_desc_count > 0 else 1.0
            for k in sums:
                avgs[k] = sums[k] / div
            
            # 格式化显示字符串 (紧凑模式)
            # 格式: QA | B1 B2 B3 B4 | R1 R2 RL | BS
            desc_str = (
                f"QA:{avg_qa:.3f} | "
                f"B1:{avgs['bleu_1']:.2f} B2:{avgs['bleu_2']:.2f} B3:{avgs['bleu_3']:.2f} B4:{avgs['bleu_4']:.2f} | "
                f"R1:{avgs['rouge1']:.2f} R2:{avgs['rouge2']:.2f} RL:{avgs['rougeL']:.2f} | "
                f"BS:{avgs['bert_score']:.3f}"
            )
            
            pbar.set_description(desc_str)
            pbar.update(1)
            
        pbar.close()

    # ================== Final Metrics Calculation ==================
    print("\nCalculating Final Batch Metrics (Double Check)...")
    
    results = {}
    results['qa_accuracy'] = (acc_qa_correct / acc_qa_total) if acc_qa_total > 0 else 0.0
    
    # 填充滚动平均的最终结果到 results (这些就是最终结果)
    div = valid_desc_count if valid_desc_count > 0 else 1.0
    for k, v in sums.items():
        results[f'avg_{k}'] = v / div

    print("\n\n" + "="*40)
    print("     FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"Total Processed Samples: {valid_samples} / {num_samples}")
    print(f"\n[QA Metrics]")
    print(f"  Accuracy: {results.get('qa_accuracy', 0):.4f} ({acc_qa_correct}/{acc_qa_total})")
    
    if valid_desc_count > 0:
        print(f"\n[Description Metrics] (N={valid_desc_count})")
        print(f"  BLEU-1   : {results['avg_bleu_1']:.4f}")
        print(f"  BLEU-2   : {results['avg_bleu_2']:.4f}")
        print(f"  BLEU-3   : {results['avg_bleu_3']:.4f}")
        print(f"  BLEU-4   : {results['avg_bleu_4']:.4f}")
        print("-" * 20)
        print(f"  ROUGE-1  : {results['avg_rouge1']:.4f}")
        print(f"  ROUGE-2  : {results['avg_rouge2']:.4f}")
        print(f"  ROUGE-L  : {results['avg_rougeL']:.4f}")
        print("-" * 20)
        print(f"  BERTScore: {results['avg_bert_score']:.4f}")
    
    print("="*40)
    
    with open(os.path.join(os.getcwd(), "eval_metrics_api_full_rolling.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()