import json
import base64
import io
import re
import os
import requests
from PIL import Image, ImageDraw
from datasets import load_dataset
from tqdm import tqdm

# ================= 配置区 =================
API_URL = "http://localhost:8001/v1/chat/completions"
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
DATA_DIR = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test"
OUTPUT_DIR = "/home/data/nas_hdd/cbx/pvllm/eval/output_2d"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 工具函数 =================

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_iou(box1, box2):
    """计算 IoU, 格式 [x1, y1, x2, y2]"""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    inter_xmin, inter_ymin = max(x_min1, x_min2), max(y_min1, y_min2)
    inter_xmax, inter_ymax = min(x_max1, x_max2), min(y_max1, y_max2)
    inter_w, inter_h = max(0, inter_xmax - inter_xmin), max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def is_point_in_box(point, box):
    """检查点 [x, y] 是否在 [x1, y1, x2, y2] 内"""
    if not point or not box or len(point) < 2 or len(box) < 4:
        return False
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def extract_json_from_text(text):
    """
    解析模型返回的各种 JSON 格式：
    1. 剥离 ```json 代码块
    2. 处理中文引号
    3. 支持解析 List 或 Dict
    """
    try:
        clean_text = re.sub(r'```(?:json)?\n?|```', '', text).strip()
        match = re.search(r'(\[.*\]|\{.*\})', clean_text, re.DOTALL)
        if match:
            json_str = match.group().replace('“', '"').replace('”', '"')
            return json.loads(json_str)
        return None
    except Exception:
        return None

def query_vllm(prompt, base64_image):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]}],
        "temperature": 0.0
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def draw_full_info(image, gt_bbox, pred_bbox, gt_points, pred_points, idx, label):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    # GT: 绿框 / 蓝点
    if gt_bbox:
        draw.rectangle(gt_bbox, outline="green", width=6)
    if gt_points:
        for x, y in gt_points:
            draw.ellipse([x-5, y-5, x+5, y+5], fill="blue")
    # Pred: 红框 / 黄点
    if pred_bbox:
        draw.rectangle(pred_bbox, outline="red", width=4)
    if pred_points:
        for x, y in pred_points:
            draw.ellipse([x-5, y-5, x+5, y+5], fill="yellow")
    # 图例
    draw.rectangle([10, 10, 300, 120], fill=(255, 255, 255, 180))
    draw.text((15, 15), f"Index: {idx}\nObj: {label}\nGreen/Blue: GT\nRed/Yellow: Pred", fill="black")
    draw_img.save(os.path.join(OUTPUT_DIR, f"test_{idx}.png"))

# ================= 主评估逻辑 =================

def main():
    print(f"正在加载数据集...")
    dataset = load_dataset("parquet", data_dir=DATA_DIR, split="test")

    # bbox 指标
    sum_iou = 0.0
    valid_bbox_n = 0

    # point 指标
    sum_point_acc = 0.0
    valid_point_n = 0

    pbar = tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating")

    for idx, item in pbar:
        try:
            messages = item['messages']
            images = item.get('images', [])
            if not images:
                continue

            # 解析 GT 数据
            gt_content = json.loads(messages[1]['content'][0]['text'])
            obj_label = gt_content['answer']['label']['answer']
            gt_bbox = gt_content['bbox']        # [x1, y1, x2, y2]
            gt_points = gt_content['point_2d']  # [[x, y], ...]

            pil_img = images[0]
            base64_img = pil_to_base64(pil_img)

            print(f"\n{'-'*30} Index: {idx} | Label: {obj_label} {'-'*30}")

            # --- 1. 处理 BBox ---
            bbox_prompt = (
                f'Locate the {obj_label} in the image, report the bbox coordinates in JSON format. '
                f'The format of output should be like {{"bbox": [x1, y1, x2, y2]}}'
            )
            bbox_res_raw = query_vllm(bbox_prompt, base64_img)
            parsed_bbox_json = extract_json_from_text(bbox_res_raw)

            pred_bbox = None
            if parsed_bbox_json:
                if isinstance(parsed_bbox_json, list) and len(parsed_bbox_json) > 0:
                    data = parsed_bbox_json[0]
                    pred_bbox = data.get("bbox_2d") or data.get("bbox")
                elif isinstance(parsed_bbox_json, dict):
                    pred_bbox = parsed_bbox_json.get("bbox_2d") or parsed_bbox_json.get("bbox")

            if pred_bbox and isinstance(pred_bbox, (list, tuple)) and len(pred_bbox) == 4:
                cur_iou = calculate_iou(pred_bbox, gt_bbox)
                sum_iou += cur_iou
                valid_bbox_n += 1
                print(f"[BBox Result]: {pred_bbox} | IoU: {cur_iou:.4f}")
            else:
                print(f"\033[91m[WARNING] BBox Parse Failed! Raw: {bbox_res_raw.strip()}\033[0m")

            # --- 2. 处理 Points ---
            point_prompt = (
                f'Output 16 keypoints of the {obj_label} in the image and return their locations in the form of coordinates. '
                f'The format of output should be like {{"point_2d": [[x1, y1], [x2, y2], ...(16 in total)]}}. '
                f'Only output the JSON result.'
            )
            point_res_raw = query_vllm(point_prompt, base64_img)
            parsed_point_json = extract_json_from_text(point_res_raw)

            pred_points = None
            if parsed_point_json and isinstance(parsed_point_json, dict):
                pred_points = parsed_point_json.get("point_2d")

            if pred_points and isinstance(pred_points, list) and len(pred_points) > 0:
                hits = sum(1 for p in pred_points if is_point_in_box(p, gt_bbox))
                cur_point_acc = hits / len(pred_points)
                sum_point_acc += cur_point_acc
                valid_point_n += 1
                print(f"[Point Result]: {len(pred_points)} points | Acc: {cur_point_acc:.4f}")
            else:
                print(f"\033[91m[WARNING] Point Parse Failed! Raw: {point_res_raw.strip()}\033[0m")

            # 可视化绘图
            draw_full_info(pil_img, gt_bbox, pred_bbox, gt_points, pred_points, idx, obj_label)

            # 更新进度条：分开显示 valid bbox / valid point
            avg_iou = (sum_iou / valid_bbox_n) if valid_bbox_n > 0 else 0.0
            avg_p_acc = (sum_point_acc / valid_point_n) if valid_point_n > 0 else 0.0
            pbar.set_postfix({
                "valid_bbox": f"{valid_bbox_n}",
                "valid_point": f"{valid_point_n}",
                "avg_iou": f"{avg_iou:.4f}",
                "avg_p_acc": f"{avg_p_acc:.4f}",
            })

        except Exception as e:
            print(f"Index {idx} System Error: {e}")
            continue

    print("\n" + "="*60)
    print("最终统计结果：")
    print(f"有效 BBox 样本: {valid_bbox_n} | 平均 IoU: {(sum_iou/valid_bbox_n) if valid_bbox_n > 0 else 0:.4f}")
    print(f"有效 Point 样本: {valid_point_n} | 平均 Point Acc: {(sum_point_acc/valid_point_n) if valid_point_n > 0 else 0:.4f}")
    print(f"可视化文件已存至: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
