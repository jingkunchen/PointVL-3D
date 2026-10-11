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
from concurrent.futures import ThreadPoolExecutor

# ================= 配置区 =================
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "InternVL3_5-4B-Instruct"
DATA_DIR = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test"
OUTPUT_DIR = "/home/data/nas_hdd/cbx/pvllm/eval/output_2d"
MAX_WORKERS = 64  # 注意：HTTP并发可以开大；你现在是 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局统计与锁
stats_lock = threading.Lock()
sum_iou = 0.0
count_iou = 0
sum_p_acc = 0.0
count_p_acc = 0

# tqdm 进度条对象（全局引用，在线程里更新 postfix）
pbar = None


# ================= 工具函数 =================

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def calculate_iou(box1, box2):
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    inter_xmin, inter_ymin = max(x_min1, x_min2), max(y_min1, y_min2)
    inter_xmax, inter_ymax = min(x_max1, x_max2), min(y_max1, y_max2)
    inter_w, inter_h = max(0, inter_xmax - inter_xmin), max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    area1 = max(0, (x_max1 - x_min1)) * max(0, (y_max1 - y_min1))
    area2 = max(0, (x_max2 - x_min2)) * max(0, (y_max2 - y_min2))
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def is_point_in_box(point, box):
    if not point or not box or len(point) < 2 or len(box) < 4:
        return False
    px, py = point[:2]
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def safe_json_from_text(text: str):
    """
    尝试从文本中提取 JSON（dict/list），兼容 ```json ...```。
    返回 python 对象或 None。
    """
    try:
        code_block_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        content = code_block_match.group(1) if code_block_match else text

        match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if not match:
            return None
        json_str = match.group().replace('“', '"').replace('”', '"')
        return json.loads(json_str)
    except Exception:
        return None


def parse_bbox_any(text: str):
    """
    解析 bbox，支持：
    1) {"bbox":[...]} / {"bbox_2d":[...]} / [{"bbox":[...]}]
    2) label[[x1,y1,x2,y2]]  或  [[x1,y1,x2,y2]]
    3) [x1,y1,x2,y2]
    返回 bbox(list[4]) 或 None
    """
    if not text:
        return None

    # 1) JSON
    obj = safe_json_from_text(text)
    if obj is not None:
        if isinstance(obj, list) and len(obj) > 0:
            first = obj[0]
            if isinstance(first, dict):
                bbox = first.get("bbox_2d") or first.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    return bbox
            if len(obj) == 4 and all(isinstance(x, (int, float)) for x in obj):
                return obj

        if isinstance(obj, dict):
            bbox = obj.get("bbox_2d") or obj.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                return bbox

    # 2) label[[...]] or [[...]]
    pat = r'\[\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\]'
    m = re.search(pat, text)
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]

    # 3) [x1,y1,x2,y2]
    pat2 = r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]'
    m2 = re.search(pat2, text)
    if m2:
        return [float(m2.group(1)), float(m2.group(2)), float(m2.group(3)), float(m2.group(4))]

    return None


def parse_points_any(text: str):
    """
    解析 points，支持：
    1) {"point_2d":[[x,y],...]}
    2) 直接输出 [[x,y], [x,y], ...]
    返回 list[list[2]] 或 None
    """
    if not text:
        return None

    # 1) JSON
    obj = safe_json_from_text(text)
    if isinstance(obj, dict):
        pts = obj.get("point_2d") or obj.get("points_2d") or obj.get("points")
        if isinstance(pts, list) and len(pts) > 0:
            clean = []
            for p in pts:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    if isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                        clean.append([float(p[0]), float(p[1])])
            return clean if clean else None

    # 2) 非 JSON：提取所有 [x,y]
    pat = r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]'
    pairs = re.findall(pat, text)
    if not pairs:
        return None
    pts = [[float(x), float(y)] for x, y in pairs]
    return pts if pts else None


def sanitize_bbox(bbox, img_w=None, img_h=None):
    """
    float -> int(round), 保证 x1<=x2, y1<=y2, 可选 clamp
    """
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if img_w is not None and img_h is not None:
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))
    return [x1, y1, x2, y2]


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
    if gt_bbox:
        draw.rectangle(gt_bbox, outline="green", width=6)
    if gt_points:
        for x, y in gt_points:
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="blue")
    if pred_bbox:
        draw.rectangle(pred_bbox, outline="red", width=4)
    if pred_points:
        for x, y in pred_points:
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="yellow")
    draw.rectangle([10, 10, 260, 110], fill=(255, 255, 255, 180))
    draw.text((15, 15), f"Idx: {idx}\nObj: {label}\nGreen/Blue: GT\nRed/Yellow: Pred", fill="black")
    draw_img.save(os.path.join(OUTPUT_DIR, f"test_{idx}.png"))


def _short(s: str, n=220):
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")


# ================= 多线程核心逻辑 =================

def process_single_item(data_tuple):
    """单线程处理函数：每个样本都打印 IoU/Acc，并更新 tqdm postfix"""
    global sum_iou, count_iou, sum_p_acc, count_p_acc, pbar
    idx, item = data_tuple

    obj_label = "Unknown"
    local_pred_bbox = None
    bbox_res_raw = ""
    local_iou = None
    bbox_ok = False

    local_pred_points = None
    point_res_raw = ""
    local_p_acc = None
    point_ok = False

    try:
        messages = item['messages']
        images = item.get('images', [])
        if not images:
            raise RuntimeError("No images field")

        # 解析 GT
        gt_content = json.loads(messages[1]['content'][0]['text'])
        obj_label = gt_content['answer']['label']['answer']
        gt_bbox = gt_content['bbox']
        gt_points = gt_content['point_2d']

        pil_img = images[0]
        img_w, img_h = pil_img.size
        base64_img = pil_to_base64(pil_img)

        # 1) BBox 请求（InternVL 常输出 label[[...]]，parser 已兼容）
        bbox_prompt = f'Please provide the bounding box coordinates of the region this sentence describes: <ref>{obj_label}</ref>'
        bbox_res_raw = query_vllm(bbox_prompt, base64_img)

        bbox_candidate = parse_bbox_any(bbox_res_raw)
        bbox_candidate = sanitize_bbox(bbox_candidate, img_w, img_h)

        if bbox_candidate is not None:
            local_pred_bbox = bbox_candidate
            local_iou = calculate_iou(local_pred_bbox, gt_bbox)
            bbox_ok = True
        else:
            bbox_ok = False
            local_iou = None

        # 2) Points 请求
        point_prompt = (
            f'Output 16 keypoints of the {obj_label} in the image and return their locations in the form of coordinates. '
            f'The format of output should be like {{"point_2d": [[x1, y1], [x2, y2], ...(16 in total)]}}. '
            f'Only output the JSON data.'
        )
        point_res_raw = query_vllm(point_prompt, base64_img)

        pts_candidate = parse_points_any(point_res_raw)

        if pts_candidate is not None and len(pts_candidate) > 0:
            # 如果你要严格 16 个点：打开下一行
            # pts_candidate = pts_candidate[:16]
            local_pred_points = [[int(round(x)), int(round(y))] for x, y in pts_candidate]
            hits = sum(1 for p in local_pred_points if is_point_in_box(p, gt_bbox))
            local_p_acc = hits / len(local_pred_points)
            point_ok = True
        else:
            point_ok = False
            local_p_acc = None

        # 可视化保存（失败也保存，便于 debug）
        draw_full_info(pil_img, gt_bbox, local_pred_bbox, gt_points, local_pred_points, idx, obj_label)

    except Exception as e:
        with stats_lock:
            tqdm.write(f"\033[91m[Exception] idx={idx} obj={obj_label} err={str(e)}\033[0m")

    # --- 无论对错，都打印，并更新 tqdm ---
    with stats_lock:
        if bbox_ok and local_iou is not None:
            sum_iou += float(local_iou)
            count_iou += 1
        if point_ok and local_p_acc is not None:
            sum_p_acc += float(local_p_acc)
            count_p_acc += 1

        avg_iou = sum_iou / count_iou if count_iou > 0 else 0.0
        avg_p = sum_p_acc / count_p_acc if count_p_acc > 0 else 0.0

        if pbar is not None:
            pbar.set_postfix({
                "mIoU": f"{avg_iou:.3f}",
                "nBox": count_iou,
                "mAcc": f"{avg_p:.3f}",
                "nPts": count_p_acc
            })

        lines = []
        lines.append(f"\n{'-'*20} Index: {idx} | Obj: {obj_label} {'-'*20}")

        if bbox_ok:
            lines.append(f"[BBox] valid=1 IoU={local_iou:.4f}")
        else:
            lines.append(f"[BBox] valid=0 IoU=NA")
        lines.append(f"  pred_bbox={local_pred_bbox}")
        lines.append(f"  raw={_short(bbox_res_raw)}")

        if point_ok:
            lines.append(f"[Point] valid=1 Acc={local_p_acc:.4f}")
        else:
            lines.append(f"[Point] valid=0 Acc=NA")

        if local_pred_points is None:
            lines.append("  pred_points=None")
        else:
            lines.append(f"  pred_points_len={len(local_pred_points)} first3={local_pred_points[:3]}")
        lines.append(f"  raw={_short(point_res_raw)}")

        tqdm.write("\n".join(lines))


def main():
    global pbar
    print("正在加载数据集...")
    dataset = load_dataset("parquet", data_dir=DATA_DIR, split="test")
    tasks = list(enumerate(dataset))

    print(f"启动评估 (线程数: {MAX_WORKERS})...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pbar = tqdm(total=len(tasks), desc="Processing")

        futures = [executor.submit(process_single_item, t) for t in tasks]

        # 主线程推进进度条
        for fut in futures:
            fut.result()
            pbar.update(1)

        pbar.close()
        pbar = None

    print("\n" + "=" * 60)
    print(f"最终统计结果 ({MODEL_NAME}):")
    if count_iou > 0:
        print(f"有效 BBox 样本: {count_iou} | 平均 IoU: {sum_iou / count_iou:.4f}")
    else:
        print("有效 BBox 样本: 0 | 平均 IoU: 0.0000")

    if count_p_acc > 0:
        print(f"有效 Point 样本: {count_p_acc} | 平均 Point Acc: {sum_p_acc / count_p_acc:.4f}")
    else:
        print("有效 Point 样本: 0 | 平均 Point Acc: 0.0000")

    print(f"可视化文件已存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
