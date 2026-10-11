import json
import base64
import io
import re
import os
import requests
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw
from datasets import load_dataset
from tqdm import tqdm

# ================= 配置区 =================
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "MiMo-VL-7B-RL"
DATA_DIR = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test"
OUTPUT_DIR = "/home/data/nas_hdd/cbx/pvllm/eval/output_2d_xy"
MAX_WORKERS = 64  # 根据显存和CPU负载调整

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 工具函数 =================

def pil_to_base64(image):
    image.load() 
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_iou(box1, box2):
    """计算 IoU, 格式 [x1, y1, x2, y2]"""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4: return 0.0
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    inter_xmin, inter_ymin = max(x_min1, x_min2), max(y_min1, y_min2)
    inter_xmax, inter_ymax = min(x_max1, x_max2), min(y_max1, y_max2)
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def is_point_in_box(point, box):
    if not point or not box or len(point) < 2 or len(box) < 4: return False
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def extract_json_from_text(text):
    """尝试从各种格式中提取 JSON"""
    try:
        clean_text = re.sub(r'```(?:json)?\n?|```', '', text).strip()
        # 1. 尝试匹配 point_2d
        match_point = re.search(r'(\{\s*"point_2d"\s*:.*?\})', clean_text, re.DOTALL)
        if match_point:
            try:
                return json.loads(match_point.group(1))
            except: pass
        # 2. 尝试匹配 bbox
        match_bbox = re.search(r'(\{\s*"bbox"\s*:.*?\})', clean_text, re.DOTALL)
        if match_bbox:
            try:
                return json.loads(match_bbox.group(1))
            except: pass
        # 3. 尝试匹配通用 JSON 对象
        match_generic = re.search(r'(\{.*\})', clean_text, re.DOTALL)
        if match_generic:
            return json.loads(match_generic.group(1))
        return None
    except Exception:
        return None

def denormalize_bbox(bbox, width, height):
    """如果坐标是 0-1 之间，则认为是归一化坐标，转为绝对坐标"""
    if not bbox or len(bbox) != 4: return None
    # 稍微放宽判断 (<=1.05) 避免浮点误差
    if all(0.0 <= x <= 1.05 for x in bbox):
        return [bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height]
    return bbox

def denormalize_points(points, width, height):
    """如果点坐标是 0-1 之间，转为绝对坐标"""
    if not points: return None
    new_points = []
    for p in points:
        if len(p) >= 2 and all(0.0 <= x <= 1.05 for x in p[:2]):
            new_points.append([p[0]*width, p[1]*height])
        else:
            new_points.append(p)
    return new_points

def query_vllm(prompt, base64_image):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}],
        "temperature": 0.0,
        "max_tokens": 4096
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        if response.status_code != 200:
            return f"Error HTTP {response.status_code}"
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def draw_full_info(image, gt_bbox, pred_bbox, gt_points, pred_points, idx, label):
    try:
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        # GT
        if gt_bbox: draw.rectangle(gt_bbox, outline="green", width=6)
        if gt_points:
            for p in gt_points:
                if len(p)>=2: draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill="blue")
        # Pred
        if pred_bbox: draw.rectangle(pred_bbox, outline="red", width=4)
        if pred_points:
            for p in pred_points:
                if len(p)>=2: draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill="yellow")
            
        draw.rectangle([10, 10, 320, 110], fill=(255, 255, 255, 180))
        draw.text((15, 15), f"Index: {idx}\nObj: {label}\nGreen/Blue: GT, Red/Yellow: Pred", fill="black")
        draw_img.save(os.path.join(OUTPUT_DIR, f"test_{idx}.png"))
    except Exception:
        pass

# ================= 单样本处理逻辑 =================

def process_single_sample(data_tuple):
    """
    Worker 线程函数：负责网络请求、解析、计算和绘图
    返回结果字典，不操作全局变量
    """
    idx, item = data_tuple
    result = {
        "idx": idx,
        "label": "Unknown",
        "bbox_raw": "",
        "point_raw": "",
        "iou": 0.0,
        "iou_valid": False,
        "p_acc": 0.0,
        "p_valid": False,
        "error": None
    }

    try:
        messages = item['messages']
        images = item.get('images', [])
        if not images:
            result["error"] = "No image"
            return result
        
        # 解析 GT
        gt_content = json.loads(messages[1]['content'][0]['text'])
        obj_label = gt_content['answer']['label']['answer']
        gt_bbox = gt_content['bbox'] 
        gt_points = gt_content['point_2d']
        result["label"] = obj_label
        
        pil_img = images[0]
        img_w, img_h = pil_img.size
        base64_img = pil_to_base64(pil_img)

        # --- BBox ---

        bbox_prompt = (f'Locate the {obj_label} in the image, report the bbox coordinates in JSON format. '
                       f'The format MUST be [x1, y1, x2, y2]. '
                       f'Output format: {{"bbox": [x1, y1, x2, y2]}}. Only output the JSON.')
        bbox_res = query_vllm(bbox_prompt, base64_img)
        result["bbox_raw"] = bbox_res
        
        parsed_bbox = extract_json_from_text(bbox_res)
        pred_bbox = None
        if parsed_bbox:
            # 兼容 list/dict 和 key
            if isinstance(parsed_bbox, list) and parsed_bbox:
                parsed_bbox = parsed_bbox[0]
            if isinstance(parsed_bbox, dict):
                pred_bbox = parsed_bbox.get("bbox") or parsed_bbox.get("bbox_2d")
        
        # 反归一化处理
        pred_bbox = denormalize_bbox(pred_bbox, img_w, img_h)

        if pred_bbox and len(pred_bbox) == 4:
            result["iou"] = calculate_iou(pred_bbox, gt_bbox)
            result["iou_valid"] = True

        # --- Points ---
        point_prompt = (f'Output 16 keypoints of the {obj_label} in the image. '
                        f'The format MUST be [[x1, y1], [x2, y2], ...]. '
                        f'Output format: {{"point_2d": [[x1, y1], [x2, y2], ...]}}. Only output the JSON.')
        point_res = query_vllm(point_prompt, base64_img)
        result["point_raw"] = point_res
        
        parsed_point = extract_json_from_text(point_res)
        pred_points = None
        if parsed_point and isinstance(parsed_point, dict):
            pred_points = parsed_point.get("point_2d") or parsed_point.get("points_2d") or parsed_point.get("points")
        
        # 反归一化处理
        pred_points = denormalize_points(pred_points, img_w, img_h)

        if pred_points and len(pred_points) > 0:
            hits = sum([1 for p in pred_points if is_point_in_box(p, gt_bbox)])
            result["p_acc"] = hits / len(pred_points)
            result["p_valid"] = True

        # 绘图
        draw_full_info(pil_img, gt_bbox, pred_bbox, gt_points, pred_points, idx, obj_label)

    except Exception as e:
        result["error"] = str(e)
        # traceback.print_exc() 

    return result

# ================= 主流程 =================

def main():
    print(f"正在加载数据集: {DATA_DIR}")
    dataset = load_dataset("parquet", data_dir=DATA_DIR, split="test")
    tasks = list(enumerate(dataset))
    
    # 统计变量
    sum_iou = 0.0
    count_iou = 0
    sum_p_acc = 0.0
    count_p_acc = 0

    print(f"开始评估，并发线程数: {MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_sample, task) for task in tasks]
        
        # 使用 as_completed 在主线程实时处理结果
        pbar = tqdm(as_completed(futures), total=len(tasks), desc="Evaluating")
        
        for future in pbar:
            res = future.result()
            
            # 1. 打印 Raw 日志 (使用 tqdm.write 避免打断进度条)
            log_lines = []
            log_lines.append(f"\n{'-'*15} Idx: {res['idx']} | Obj: {res['label']} {'-'*15}")
            
            # BBox 日志
            if res['iou_valid']:
                log_lines.append(f"[BBox] IoU: {res['iou']:.4f}")
                # 更新统计
                sum_iou += res['iou']
                count_iou += 1
            else:
                log_lines.append(f"[BBox] \033[91mInvalid\033[0m")
            log_lines.append(f" >>> Raw: {res['bbox_raw'].strip()[:200]}") # 截断防止太长

            # Point 日志
            if res['p_valid']:
                log_lines.append(f"[Point] Acc: {res['p_acc']:.4f}")
                # 更新统计
                sum_p_acc += res['p_acc']
                count_p_acc += 1
            else:
                log_lines.append(f"[Point] \033[91mInvalid\033[0m")
            log_lines.append(f" >>> Raw: {res['point_raw'].strip()[:200]}")

            if res['error']:
                log_lines.append(f"\033[91mError: {res['error']}\033[0m")
            
            # 打印
            tqdm.write("\n".join(log_lines))

            # 2. 更新进度条指标 (计算平均值)
            avg_iou = sum_iou / count_iou if count_iou > 0 else 0.0
            avg_p_acc = sum_p_acc / count_p_acc if count_p_acc > 0 else 0.0
            
            pbar.set_postfix({
                "mIoU": f"{avg_iou:.3f}",
                "nBox": count_iou,      # Valid BBox Samples
                "mAcc": f"{avg_p_acc:.3f}",
                "nPts": count_p_acc     # Valid Point Samples
            })
    
    print(f"\n========================================")
    print(f"评估完成。")
    print(f"BBox  -> Valid Samples: {count_iou} | mIoU: {sum_iou/count_iou if count_iou else 0:.4f}")
    print(f"Point -> Valid Samples: {count_p_acc} | mAcc: {sum_p_acc/count_p_acc if count_p_acc else 0:.4f}")
    print(f"========================================")

if __name__ == "__main__":
    main()