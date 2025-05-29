import json
import os
import re
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
# ----------------- 辅助函数定义 -----------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    else:
        raise NotImplementedError
    if is_train:  # 数据增强
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.),
                                interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio

def image_process(image_path, config):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(is_train=False, input_size=config.vision_config.image_size,
                                  pad2square=config.pad2square, normalize_type='imagenet')
    if config.dynamic_image_size:
        images, target_aspect_ratio = dynamic_preprocess(
            image,
            min_num=config.min_dynamic_patch,
            max_num=config.max_dynamic_patch,
            image_size=config.vision_config.image_size,
            use_thumbnail=config.use_thumbnail)
    else:
        images = [image]
        target_aspect_ratio = (1, 1)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values.to(torch.bfloat16).cuda(), torch.tensor([[target_aspect_ratio[0], target_aspect_ratio[1]]], dtype=torch.long)

def parse_block_pos(str_, target_aspect_ratio):
    # target_aspect_ratio 预期为 numpy 数组
    block_num_w, block_num_h = target_aspect_ratio[0][0], target_aspect_ratio[0][1]
    action, location, direction, location_or_text = None, None, None, None
    str_ = str_.strip()
    match = re.match(r'^(.*?)\((.*?)\)$', str_)
    if match:
        action, location_or_text = match.groups()
        if action == 'CLICK':
            match = re.match(r'^\[(\d{1}), (\d{3}), (\d{3})\].*?$', location_or_text)
            if match:
                block_idx, cx, cy = match.groups()
                block_idx = int(block_idx)
                cx = int(cx)
                cy = int(cy)
                cx += (block_idx % block_num_w) * 1000
                cy += (block_idx // block_num_w) * 1000
                cx /= block_num_w * 1000
                cy /= block_num_h * 1000
                location = [cx, cy]
            else:
                print("解析CLICK参数失败：", location_or_text)
        elif action.startswith('SWIPE_'):
            action, direction = action.split('_', 1)
    return {
        'action': action,
        'location': location,
        'direction': direction,
        'content': location_or_text
    }

# ----------------- 模型与模板加载 -----------------
question_template = '''## Task: {task}
## History Actions:
{history}
## Action Space
1. CLICK([block_index, cx, cy], "text")
2. TYPE("text")
3. PRESS_BACK()
4. PRESS_HOME()
5. PRESS_ENTER()
6. SWIPE_UP()
7. SWIPE_DOWN()
8. SWIPE_LEFT()
9. SWIPE_RIGHT()
10. COMPLETED()
## Requirements: Please infer the next action according to the Task and History Actions. Think step by step. Return with Image Description, Next Action Description and Action Code. The Action Code should follow the definition in the Action Space.'''

model_path = '/hy-tmp/SpiritSight-Agent-8B/SpiritSight-Agent-8B-base'
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,              # 👈 更适合 RTX 3090
    trust_remote_code=True,
    device_map="auto",                      # 👈 自动分卡
    low_cpu_mem_usage=True
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

# ----------------- 数据加载 -----------------
def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 这里加载转换后的数据，字段包括 "task" 和 "image_path"
data_file = "/hy-tmp/SpiritSight-Agent-8B/infer/Privacy_ssAgent.json"
data_samples = load_data(data_file)

# ----------------- 推理并保存结果 -----------------
def infer_sample(sample):
    # 使用 sample 中的 task，history 为空（可以修改为其他内容）
    task_text = sample["task"]
    history_text = ""
    question = question_template.format(task=task_text, history=history_text)
    
    image_path = sample["image_path"]
    pixel_values, target_aspect_ratio = image_process(image_path, model.config)
    
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        target_aspect_ratio=target_aspect_ratio,
        generation_config=generation_config
    )
    
    # 获取最后一个 token 作为动作代码字符串
    action_code_str = response.split()[-1]
    parsed_action = parse_block_pos(action_code_str, target_aspect_ratio.cpu().numpy())
    
    # 返回所有信息构成的字典
    return {
        "input_question": question,
        "model_response": response,
        "parsed_action": parsed_action,
        "image_path": image_path,
    }

def run_inference(data_samples, output_result_file):
    results = []

    # 如果已有中间文件，先读取
    if os.path.exists(output_result_file):
        with open(output_result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        done_paths = set(item['image_path'] for item in results)
    else:
        done_paths = set()

    for sample in tqdm(data_samples, desc="推理中"):
        if sample["image_path"] in done_paths:
            continue
        try:
            infer_result = infer_sample(sample)
            results.append(infer_result)
            # 实时保存
            with open(output_result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 错误处理 {sample['image_path']}：{e}")
    
    print(f"✅ 推理完成，结果保存在：{output_result_file}")

if __name__ == '__main__':
    output_file = "/hy-tmp/SpiritSight-Agent-8B/results/output_eval.json"
    run_inference(data_samples, output_file)
