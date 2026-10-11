import torch
import json
import re
import os
from PIL import Image, ImageDraw
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

"""
Run:
PYTHONPATH=$PYTHONPATH:/home/data/nas_hdd/cbx/pvllm/models/DeepSeek-VL2 python ./eval/eval_2d_dpsk.py
"""

# ================= 配置区 =================
MODEL_PATH = "/home/data/nas_hdd/cbx/pvllm/models/deepseek-vl2-small"
DATA_DIR = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test"
OUTPUT_DIR = "/home/data/nas_hdd/cbx/pvllm/eval/output_deepseek_tiny"

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
dtype = torch.bfloat16
device = "cuda"

RAW_TRUNC = 300
FORCE_FIRST_K_POINTS = 16

# 归一化判定阈值：更严格一点，避免 1.06 这种也被当成归一化
NORM_MAX = 1.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局统计
sum_iou = 0.0
count_iou = 0
sum_p_acc = 0.0
count_p_acc = 0

# ================= 模型加载 =================
print(f"Loading model from {MODEL_PATH} ...")
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
vl_gpt = vl_gpt.to(dtype=dtype).to(device).eval()
print("Model loaded successfully.")

# 确保 pad_token_id 可用（DeepSeek 会添加 pad token）
PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


# ================= 工具函数 =================
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


def extract_json_from_text(text):
    """从模型输出中提取 JSON（dict/list）"""
    try:
        if text is None:
            return None
        # prefer ```json ... ```
        code_block_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        content_to_parse = code_block_match.group(1) if code_block_match else text

        # first {...} or [...]
        match = re.search(r'(\[.*\]|\{.*\})', content_to_parse, re.DOTALL)
        if match:
            json_str = match.group().replace('“', '"').replace('”', '"').replace("'", '"')
            return json.loads(json_str)
        return None
    except Exception:
        return None


def extract_any_bbox(text: str):
    """
    Extract bbox from DeepSeek outputs.
    Supports:
    1) <|det|>[[x1,y1,x2,y2]]<|/det|> or <|det|>[x1,y1,x2,y2]<|/det|>
    2) <box>x1,y1,x2,y2</box>
    3) raw list: [x1, y1, x2, y2]
    """
    if not text:
        return None

    # 1) <|det|> ... <|/det|>
    m = re.search(r"<\|det\|>\s*(.*?)\s*<\|/det\|>", text, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        nums = re.findall(r"-?\d+(?:\.\d+)?", inner)
        if len(nums) >= 4:
            return list(map(float, nums[:4]))

    # 2) <box> ... </box>
    m = re.search(r"<box>\s*([^<]+?)\s*</box>", text, re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        nums = re.findall(r"-?\d+(?:\.\d+)?", inner)
        if len(nums) >= 4:
            return list(map(float, nums[:4]))

    # 3) raw list [x1,y1,x2,y2]
    m = re.search(
        r"\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]",
        text
    )
    if m:
        nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(0))
        if len(nums) == 4:
            return list(map(float, nums))
    return None


def extract_points_loose(text: str, k=16):
    """
    Extract up to k points from model output.
    Supports:
    1) {"point_2d": [[...], ...]}
    2) raw list: [[x,y], [x,y], ...] or any long text containing numbers
    It will find numbers and group by 2 (loose fallback).
    """
    if not text:
        return None

    parsed = extract_json_from_text(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("point_2d"), list):
        pts = parsed["point_2d"]
        clean = []
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    clean.append([float(p[0]), float(p[1])])
                except Exception:
                    pass
            if len(clean) >= k:
                break
        return clean if clean else None

    # loose fallback: group all numbers by 2
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) < 2:
        return None
    pts = []
    for i in range(0, len(nums) - 1, 2):
        pts.append([float(nums[i]), float(nums[i + 1])])
        if len(pts) >= k:
            break
    return pts if pts else None


def _short(s: str, n=RAW_TRUNC):
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")


def denorm_bbox_if_needed(bbox, w, h):
    """
    bbox 支持像素 / 归一化：
    - 若四个值都在 [0, NORM_MAX]，认为是归一化 => 乘 w/h
    """
    if bbox is None or len(bbox) != 4:
        return None
    try:
        xs = [float(x) for x in bbox]
    except Exception:
        return None

    if all(0.0 <= v <= NORM_MAX for v in xs):
        x1, y1, x2, y2 = xs
        return [x1 * w, y1 * h, x2 * w, y2 * h]
    return xs


def sanitize_bbox_xyxy(bbox, w=None, h=None):
    """转 int + 保证顺序 + clamp"""
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    try:
        x1, y1, x2, y2 = int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
    except Exception:
        return None

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    if w is not None and h is not None:
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

    return [x1, y1, x2, y2]


def denorm_points_if_needed(points, w, h):
    """
    points: [[x,y],...]
    - 如果“多数点”都落在 [0,NORM_MAX] 认为归一化 => 乘 w/h
    - 过滤极端离谱点（>2w/2h 或 <-0.5w/-0.5h）
    - 转 int
    """
    if points is None or len(points) == 0:
        return None

    pts = []
    norm_like = 0
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            x, y = float(p[0]), float(p[1])
        except Exception:
            continue
        if 0.0 <= x <= NORM_MAX and 0.0 <= y <= NORM_MAX:
            norm_like += 1
        pts.append([x, y])

    if len(pts) == 0:
        return None

    if norm_like >= max(1, len(pts) // 2):
        pts = [[x * w, y * h] for x, y in pts]

    clean = []
    for x, y in pts:
        if (-0.5 * w) <= x <= (2.0 * w) and (-0.5 * h) <= y <= (2.0 * h):
            clean.append([int(round(x)), int(round(y))])

    return clean if len(clean) > 0 else None


def run_inference(prompt, pil_image):
    """DeepSeek VL2 本地推理函数（保留 special tokens 以便解析）"""
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{prompt}",
            "images": ["placeholder"],
        },
        {"role": "<|Assistant|>", "content": ""}
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    with torch.no_grad():
        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=PAD_ID,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            use_cache=True
        )

    # 不要 skip_special_tokens，否则 special tag 会被吞掉
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)

    if "<|Assistant|>" in answer:
        answer = answer.split("<|Assistant|>", 1)[-1].strip()

    return answer


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

    draw.rectangle([10, 10, 360, 150], fill=(255, 255, 255, 180))
    draw.text((15, 15), f"Idx: {idx}\nObj: {label}\nGreen/Blue: GT\nRed/Yellow: Pred", fill="black")

    draw_img.save(os.path.join(OUTPUT_DIR, f"test_{idx}.png"))


# ================= 主逻辑 =================
def main():
    global sum_iou, count_iou, sum_p_acc, count_p_acc

    print(f"正在加载数据集: {DATA_DIR}")
    dataset = load_dataset("parquet", data_dir=DATA_DIR, split="test")

    print(f"开始推理... (Total: {len(dataset)})")
    pbar = tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating")

    for idx, item in pbar:
        obj_label = "Unknown"
        bbox_res_raw = ""
        point_res_raw = ""

        local_pred_bbox = None
        local_iou = None
        local_iou_valid = False

        local_pred_points = None
        local_p_acc = None
        local_p_valid = False

        err_msg = None

        try:
            messages = item["messages"]
            images = item.get("images", [])
            if not images:
                raise RuntimeError("No images")

            gt_content = json.loads(messages[1]["content"][0]["text"])
            obj_label = gt_content["answer"]["label"]["answer"]
            gt_bbox = gt_content["bbox"]
            gt_points = gt_content["point_2d"]

            pil_img = images[0]
            w, h = pil_img.size

            # ================= 1) BBox prompt (grounding style with <|ref|> and <|det|>) =================
            bbox_prompt = (
                "Locate the <|ref|>{label}<|/ref|> in the image.\n"
            ).format(label=obj_label)

            bbox_res_raw = run_inference(bbox_prompt, pil_img)

            local_pred_bbox = extract_any_bbox(bbox_res_raw)
            if local_pred_bbox and len(local_pred_bbox) == 4:
                local_pred_bbox = denorm_bbox_if_needed(local_pred_bbox, w, h)
                local_pred_bbox = sanitize_bbox_xyxy(local_pred_bbox, w, h)
                if local_pred_bbox is not None:
                    local_iou = calculate_iou(local_pred_bbox, gt_bbox)
                    local_iou_valid = True

            # ================= 2) Keypoints prompt (grounding style with <|ref|>, strict JSON) =================
            point_prompt = (
                "Find {k} 2D keypoints of the <|ref|>{label}<|/ref|>.\n"
                "Return ONLY a JSON object, no extra text.\n"
                "Format:\n"
                '{{"point_2d": [[x1, y1], [x2, y2], ...]}}\n'
                "The list MUST contain exactly {k} points.\n"
                "Use normalized coordinates in [0, 1]."
            ).format(k=FORCE_FIRST_K_POINTS, label=obj_label)

            point_res_raw = run_inference(point_prompt, pil_img)

            local_pred_points = extract_points_loose(point_res_raw, k=FORCE_FIRST_K_POINTS)
            if local_pred_points and len(local_pred_points) > 0:
                local_pred_points = denorm_points_if_needed(local_pred_points, w, h)

                if local_pred_points and len(local_pred_points) > 0:
                    if FORCE_FIRST_K_POINTS is not None and len(local_pred_points) > FORCE_FIRST_K_POINTS:
                        local_pred_points = local_pred_points[:FORCE_FIRST_K_POINTS]

                    hits = sum(1 for p in local_pred_points if is_point_in_box(p, gt_bbox))
                    local_p_acc = hits / len(local_pred_points)
                    local_p_valid = True

            # ================= 可视化 =================
            draw_full_info(pil_img, gt_bbox, local_pred_bbox, gt_points, local_pred_points, idx, obj_label)

            # ================= 统计 =================
            if local_iou_valid:
                sum_iou += float(local_iou)
                count_iou += 1
            if local_p_valid:
                sum_p_acc += float(local_p_acc)
                count_p_acc += 1

        except Exception as e:
            err_msg = str(e)

        # ================= tqdm 动态指标 =================
        avg_iou = sum_iou / count_iou if count_iou > 0 else 0.0
        avg_p = sum_p_acc / count_p_acc if count_p_acc > 0 else 0.0
        pbar.set_postfix({
            "mIoU": f"{avg_iou:.3f}",
            "nBox": count_iou,
            "mAcc": f"{avg_p:.3f}",
            "nPts": count_p_acc
        })

        # ================= 无论成功失败，都打印 =================
        lines = []
        lines.append(f"\n{'-'*20} Index: {idx} | Obj: {obj_label} {'-'*20}")

        if local_iou_valid:
            lines.append(f"[BBox] valid=1 IoU={local_iou:.4f} pred_bbox={local_pred_bbox}")
        else:
            lines.append(f"[BBox] valid=0 IoU=NA pred_bbox={local_pred_bbox}")
        lines.append(f"  raw={_short(bbox_res_raw)}")

        if local_p_valid:
            lines.append(f"[Point] valid=1 Acc={local_p_acc:.4f} pred_points_len={len(local_pred_points)}")
        else:
            ln = 0 if local_pred_points is None else len(local_pred_points)
            lines.append(f"[Point] valid=0 Acc=NA pred_points_len={ln}")
        if local_pred_points is not None:
            lines.append(f"  first3={local_pred_points[:3]}")
        lines.append(f"  raw={_short(point_res_raw)}")

        if err_msg is not None:
            lines.append(f"\033[91m[Exception] {err_msg}\033[0m")

        tqdm.write("\n".join(lines))

    # ================= 最终结果 =================
    print("\n" + "=" * 60)
    print("最终统计结果 (DeepSeek-VL2-Tiny):")
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
