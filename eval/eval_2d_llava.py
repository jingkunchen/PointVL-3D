import json
import base64
import io
import re
import os
import requests
import threading
from PIL import Image, ImageDraw
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "llama3-llava-next-8b-hf"
DATA_DIR = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test"
OUTPUT_DIR = "/home/data/nas_hdd/cbx/pvllm/eval/output_2d"
MAX_WORKERS = 32  # 调试建议1，生产可调大

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 工具函数 =================

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_iou(box1, box2):
    """计算两个框的 IoU，box格式均为 [x1, y1, x2, y2]"""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4: return 0.0
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
    """判断点是否在框内"""
    if not point or not box or len(point) < 2 or len(box) < 4: return False
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def extract_json_from_text(text):
    try:
        # 优先匹配代码块
        code_block_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        content_to_parse = code_block_match.group(1) if code_block_match else text
        # 寻找最外层的 {} 或 []
        match = re.search(r'(\[.*\]|\{.*\})', content_to_parse, re.DOTALL)
        if match:
            json_str = match.group().replace('“', '"').replace('”', '"')
            # 清理可能的末尾逗号
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            return json.loads(json_str)
        return None
    except Exception:
        return None

def denormalize_bbox(bbox, width, height):
    """将归一化坐标 [0~1] 转换为绝对像素坐标"""
    if not bbox or len(bbox) != 4:
        return None
    # 如果所有值都 <= 1.0 (且不全是0)，则认为是归一化坐标
    # 稍微放宽一点 1.1 避免边界误差，但通常是 <=1
    if all(0.0 <= x <= 1.01 for x in bbox):
        return [
            bbox[0] * width,
            bbox[1] * height,
            bbox[2] * width,
            bbox[3] * height
        ]
    return bbox

def denormalize_points(points, width, height):
    """将归一化点列表转换为绝对像素坐标"""
    if not points: return None
    new_points = []
    for p in points:
        if len(p) >= 2 and all(0.0 <= x <= 1.01 for x in p[:2]):
            new_points.append([p[0] * width, p[1] * height])
        else:
            new_points.append(p)
    return new_points

def query_vllm(prompt, base64_image):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]}],
        "temperature": 0.0,
        "max_tokens": 512
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def draw_full_info(image, gt_bbox, pred_bbox, gt_points, pred_points, idx, label):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    
    # 绘制 GT (绿色框, 蓝色点)
    if gt_bbox: 
        draw.rectangle(gt_bbox, outline="green", width=6)
    if gt_points:
        for p in gt_points:
            if len(p) >= 2: draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill="blue")

    # 绘制 Pred (红色框, 黄色点)
    if pred_bbox: 
        draw.rectangle(pred_bbox, outline="red", width=4)
    if pred_points:
        for p in pred_points:
            if len(p) >= 2: draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill="yellow")

    # 绘制图例
    draw.rectangle([10, 10, 260, 110], fill=(255, 255, 255, 180))
    draw.text((15, 15), f"Idx: {idx}\nObj: {label}\nGreen/Blue: GT\nRed/Yellow: Pred", fill="black")
    draw_img.save(os.path.join(OUTPUT_DIR, f"test_{idx}.png"))

# ================= 多线程处理 =================

def process_single_item(data_tuple):
    idx, item = data_tuple
    result = {
        "idx": idx,
        "obj_label": "Unknown",
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
            result["error"] = "No images found"
            return result
        
        # 1. 获取图片信息
        pil_img = images[0]
        img_w, img_h = pil_img.size
        base64_img = pil_to_base64(pil_img)
        
        # 2. 解析 GT
        gt_content = json.loads(messages[1]['content'][0]['text'])
        obj_label = gt_content['answer']['label']['answer']
        gt_bbox = gt_content['bbox']
        gt_points = gt_content['point_2d']
        
        result["obj_label"] = obj_label

        # ==========================
        # 3. BBox 请求与处理
        # ==========================
        bbox_prompt = f'Locate the {obj_label} in the image, report the bbox coordinates in JSON format. The format of output should be like {{"bbox": [x1, y1, x2, y2]}}. Only output the JSON data.'
        bbox_res_raw = query_vllm(bbox_prompt, base64_img)
        result["bbox_raw"] = bbox_res_raw
        
        parsed_bbox_json = extract_json_from_text(bbox_res_raw)
        local_pred_bbox = None
        
        if parsed_bbox_json:
            if isinstance(parsed_bbox_json, list) and len(parsed_bbox_json) > 0:
                data = parsed_bbox_json[0]
                local_pred_bbox = data.get("bbox_2d") or data.get("bbox")
            elif isinstance(parsed_bbox_json, dict):
                local_pred_bbox = parsed_bbox_json.get("bbox_2d") or parsed_bbox_json.get("bbox")
            
            # 反归一化
            if local_pred_bbox:
                local_pred_bbox = denormalize_bbox(local_pred_bbox, img_w, img_h)

            if local_pred_bbox and len(local_pred_bbox) == 4:
                result["iou"] = calculate_iou(local_pred_bbox, gt_bbox)
                result["iou_valid"] = True

        # ==========================
        # 4. Points 请求与处理
        # ==========================
        point_prompt = f'Output 16 keypoints of the {obj_label} in the image and return their locations in the form of coordinates. The format of output should be like {{"point_2d": [[x1, y1], [x2, y2], ...(16 in total)]}}. Only output the JSON data.'
        point_res_raw = query_vllm(point_prompt, base64_img)
        result["point_raw"] = point_res_raw
        
        parsed_point_json = extract_json_from_text(point_res_raw)
        local_pred_points = None

        if parsed_point_json and isinstance(parsed_point_json, dict):
            # 兼容 key
            local_pred_points = (
                parsed_point_json.get("point_2d") or 
                parsed_point_json.get("points_2d") or 
                parsed_point_json.get("points")
            )
            
            # 反归一化
            if local_pred_points:
                local_pred_points = denormalize_points(local_pred_points, img_w, img_h)

            if local_pred_points and len(local_pred_points) > 0:
                hits = sum([1 for p in local_pred_points if is_point_in_box(p, gt_bbox)])
                result["p_acc"] = hits / len(local_pred_points)
                result["p_valid"] = True

        # 5. 可视化并保存
        draw_full_info(pil_img, gt_bbox, local_pred_bbox, gt_points, local_pred_points, idx, obj_label)

    except Exception as e:
        result["error"] = str(e)

    return result

def main():
    print(f"正在加载数据集...")
    dataset = load_dataset("parquet", data_dir=DATA_DIR, split="test")

    print(f"启动多线程评估 (线程数: {MAX_WORKERS})...")
    
    total_iou = 0.0
    count_iou = 0
    total_p_acc = 0.0
    count_p_acc = 0
    
    tasks = list(enumerate(dataset))
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_item, task) for task in tasks]
        
        pbar = tqdm(as_completed(futures), total=len(tasks), desc="Eval")
        
        for future in pbar:
            res = future.result()
            
            # 构建日志
            log_msgs = []
            log_msgs.append(f"\n{'='*20} Index: {res['idx']} | Obj: {res['obj_label']} {'='*20}")
            
            # BBox
            if res['iou_valid']:
                total_iou += res['iou']
                count_iou += 1
                log_msgs.append(f"[BBox] IoU: {res['iou']:.4f}")
            else:
                log_msgs.append(f"[BBox] \033[91mFailed/Invalid\033[0m")
            log_msgs.append(f" >>> BBox Raw: {res['bbox_raw'].strip()}")

            # Point
            if res['p_valid']:
                total_p_acc += res['p_acc']
                count_p_acc += 1
                log_msgs.append(f"[Point] Acc: {res['p_acc']:.4f}")
            else:
                log_msgs.append(f"[Point] \033[91mFailed/Invalid\033[0m")
            log_msgs.append(f" >>> Point Raw: {res['point_raw'].strip()}")

            if res['error']:
                log_msgs.append(f"\033[91mError: {res['error']}\033[0m")

            tqdm.write("\n".join(log_msgs))
            
            # === 更新进度条指标 ===
            avg_iou = total_iou / count_iou if count_iou > 0 else 0.0
            avg_acc = total_p_acc / count_p_acc if count_p_acc > 0 else 0.0
            
            # 在这里添加了 nBox 和 nPts 的显示
            pbar.set_postfix({
                "mIoU": f"{avg_iou:.3f}",
                "mAcc": f"{avg_acc:.3f}",
                "nBox": count_iou,  # 有效bbox样本数
                "nPts": count_p_acc # 有效points样本数
            })

    print("\n" + "="*60)
    print(f"最终统计结果 ({MODEL_NAME}):")
    if count_iou > 0:
        print(f"有效 BBox 样本: {count_iou} | 平均 IoU: {total_iou/count_iou:.4f}")
    if count_p_acc > 0:
        print(f"有效 Point 样本: {count_p_acc} | 平均 Point Acc: {total_p_acc/count_p_acc:.4f}")
    print(f"可视化文件已存至: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()