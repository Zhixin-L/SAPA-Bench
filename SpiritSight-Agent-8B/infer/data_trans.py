import json
import os

def convert_data(input_path, output_path):
    # 定义新的图片目录前缀
    new_image_prefix = "/hy-tmp/download_app/processed_images"
    
    # 读取原始数据（假设是一个 JSON 数组）
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = []
    for sample in data:
        # 获取图片路径：取 images 数组的第一项
        images = sample.get("images", [])
        if images:
            # 将图片文件名与新的目录前缀拼接成完整的路径
            image_filename = images[0].strip()
            image_path = os.path.join(new_image_prefix, image_filename)
        else:
            image_path = ""
        
        # 获取 conversation 中 human 部分的内容（这里选择第一条 human 消息）
        task = ""
        for turn in sample.get("conversations", []):
            if turn.get("from", "").lower() == "human":
                task = turn.get("value", "")
                break  # 只取第一条 human 消息
        
        # 构造新格式数据：包含 task 和 image_path 两个字段
        new_sample = {
            "task": task.strip(),
            "image_path": image_path
        }
        new_data.append(new_sample)
    
    # 保存转换后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    input_file = "/hy-tmp/download_app/label_results_0116_2121.json"   # 原始数据文件名
    output_file = "/hy-tmp/SpiritSight-Agent-2B/train/label_0408(1).json" # 转换后的数据文件名
    convert_data(input_file, output_file)
    print(f"转换完成，数据保存到 {output_file}")
