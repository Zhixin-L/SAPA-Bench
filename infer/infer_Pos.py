import os
import json
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# —— 配置区 —— #
MODEL_DIR    = "/hy-tmp/Qwen2.5-VL-7B-Instruct/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"  # 也可以是本地缓存路径
INPUT_JSON   = "/hy-tmp/Qwen2.5-VL-7B-Instruct/data/qwen_Pos.json"
OUTPUT_JSON  = "/hy-tmp/Qwen2.5-VL-7B-Instruct/results/qwen2.5vl_Pos.json"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE       = "cpu"
MAX_TOKENS   = 1024
NUM_BEAMS    = 1
# —— 配置区结束 —— #

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def infer_single(processor, model, msg):
    # --- 官方推荐流程 ---
    # 1) 把单条 messages（list of one）串成“对话”输入
    text = processor.apply_chat_template(
        [msg],
        tokenize=False,
        add_generation_prompt=True
    )
    # 2) 由 qwen_vl_utils 生成 image / video tensor
    image_inputs, video_inputs = process_vision_info([msg])
    # 3) 利用 processor 把 text + vision 转成模型输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)
    # 4) 调用模型生成
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        num_beams=NUM_BEAMS,
        do_sample=True,
        temperature=1.0,        # 温度参数，值越低生成越 “保守” （趋向贪心）
        top_k=50,               # K-采样：每步从概率前 top_k 中抽样
        top_p=0.95
    )
    # 5) 去掉 prompt 部分，仅保留模型新增
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # 6) 解码
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()
    return output_text

def run_batch():
    # 0. 断点续跑准备
    if os.path.exists(OUTPUT_JSON):
        results   = load_json(OUTPUT_JSON)
        done_imgs = { r["image_path"] for r in results }
    else:
        results   = []
        done_imgs = set()

    # 1. 读取你自己生成的 qwen 消息列表
    #    每个 entry 格式必须和官方 demo 一致：
    #    {
    #      "role": "user",
    #      "content": [
    #         {"type":"image","image":"file:///…/xxx.png"},
    #         {"type":"text", "text":"你的指令…"}
    #      ]
    #    }
    messages = load_json(INPUT_JSON)

    # 2. 初始化 processor & model
    processor = AutoProcessor.from_pretrained(MODEL_DIR,min_pixels=256*28*28,max_pixels=768*28*28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        #device_map="cpu",  # 如果有多张 GPU 可用，自动分配
        device_map="auto",
        low_cpu_mem_usage=True
    ).eval()

    # 3. 逐条推理
    for msg in tqdm(messages, desc="推理中"):
        # 从 file URI 提取本地路径
        img_uri    = msg["content"][0]["image"]
        image_path = img_uri.replace("file://", "")
        if image_path in done_imgs:
            continue

        try:
            pred = infer_single(processor, model, msg)
            results.append({
                "image_path": image_path,
                "prompt":      msg["content"][1]["text"],
                "prediction":  pred
            })
            save_json(OUTPUT_JSON, results)
        except Exception as e:
            print(f"❌ 推理失败: {image_path} → {e}")

    print(f"✅ 全部完成，结果保存在：{OUTPUT_JSON}")

if __name__ == "__main__":
    run_batch()
