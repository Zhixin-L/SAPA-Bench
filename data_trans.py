#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def convert_a_to_b(a_path: str, b_path: str, img_prefix: str = "/hy-tmp/Positive_sample/Positive sample/"):
    """
    将 a.json 的内容转换并写入 b.json。
    
    参数:
        a_path: str，输入文件 a.json 的路径
        b_path: str，输出文件 b.json 的路径
        img_prefix: str，拼接在 images 字段前的路径前缀
    """
    # 读取 a.json
    with open(a_path, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
    
    # 转换格式
    data_b = []
    for item in data_a:
        # 安全检查
        instructions = item.get("conversations1", {}).get("instructions", "")
        image_name   = item.get("images", "")
        
        # 构造新条目
        entry = {
            "task": instructions,
            "image_path": os.path.join(img_prefix, image_name)
        }
        data_b.append(entry)
    
    # 写入 b.json
    with open(b_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保持中文可读
        json.dump(data_b, f, ensure_ascii=False, indent=4)
    
    print(f"已生成文件：{b_path}，共 {len(data_b)} 条记录。")


if __name__ == "__main__":
    # 默认文件名，可根据需要修改
    a_file = "/hy-tmp/Positive_sample/final_privacy_label_generation.json"
    b_file = "/hy-tmp/SpiritSight-Agent-2B/infer/Privacy_ssAgent.json"
    convert_a_to_b(a_file, b_file)
