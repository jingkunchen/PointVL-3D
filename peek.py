import os
import json
from datasets import load_dataset

# =====================================================
# 1. 核心：强制把缓存改到大硬盘，防止 OSError
# =====================================================

# =====================================================
# 2. 配置路径 (请确认这是你刚才跑完保存的那个目录)
# =====================================================
# 如果你刚才保存到了 ..._str 结尾的目录，请修改这里
# 如果你覆盖了原目录，就保持不变
data_dir = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4/test" 
# 或者: data_dir = "/home/data/nas_hdd/cbx/pvllm/sft_data_v4_3d/train"

print(f"正在读取目录: {data_dir}")
data_files = os.path.join(data_dir, "*.parquet")


# 加载数据集
ds = load_dataset("parquet", data_files={"train": data_files}, split="train")

print(f"\n✅ 加载成功！共 {len(ds)} 条数据")

# =====================================================
# 3. 深度分析第一条数据
# =====================================================
first_item = ds[0]
print("\n第一条数据预览:")
print(first_item.keys())
messages_content = first_item['messages']
content_type = type(messages_content)

print(f"===ext_param===")
print(f"{first_item.get('ext_param', None)}")
print(f"======")

print("\n" + "="*40)
print(" >>> 第一条数据的深度分析 <<<")
print("="*40)

# 1. 打印 Python 类型
print(f"【数据类型】: {content_type}")

# 2. 根据类型做不同处理
if isinstance(messages_content, str):
    print("【状态判断】: 是 JSON 字符串 (String)。")
    print("             说明之前的 'String化' 脚本执行成功！")
    print("-" * 40)
    
    try:
        # 尝试还原
        decoded = json.loads(messages_content)
        print("【尝试 JSON 解析】: 成功还原为 List")
        print(f"【列表长度】: {len(decoded)}")
        print("【第一轮对话结构】:")
        print(json.dumps(decoded[0], indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ JSON 解析失败，字符串可能损坏: {e}")
        print("原始内容片段:", messages_content[:500])

elif isinstance(messages_content, list):
    print("【状态判断】: 是 Python 列表 (List)。")
    print("             说明 Parquet 能够原生存储 List 结构。")
    print("-" * 40)
    print("【对话结构】:")
    # 打印列表里的第一个元素
    if len(messages_content) > 0:
        print(json.dumps(messages_content, indent=2, ensure_ascii=False))
    else:
        print("[] (空列表)")

else:
    print(f"【未知类型】: {content_type}")
    print(messages_content)

print("="*40)

