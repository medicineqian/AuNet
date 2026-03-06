import json
import os

# 定义输入JSON文件路径和输出目录
json_file_path = '/home/bq/data/ICAFusion/evaluation_script/KAIST_annotation.json'
output_dir = '/home/bq/data/ICAFusion/evaluation_script/KAIST_annotation'

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取JSON文件
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 提取图像信息和注释信息
images = data['images']
annotations = data['annotations']

# 创建一个字典，将图像ID映射到图像名
image_id_to_name = {image['id']: image['im_name'] for image in images}

# 创建一个字典，将图像ID映射到注释列表
image_id_to_annotations = {image['id']: [] for image in images}
for annotation in annotations:
    image_id = annotation['image_id']
    image_id_to_annotations[image_id].append(annotation)

# 将注释写入对应的TXT文件
for image_id, image_name in image_id_to_name.items():
    annotation_list = image_id_to_annotations[image_id]
    image_name = image_name.replace('/','_')
    txt_file_path = os.path.join(output_dir, f"{image_name}.txt")
    
    with open(txt_file_path, 'w') as f:
        for annotation in annotation_list:
            category_id = annotation['category_id']
            bbox = annotation['bbox']
            bbox_str = ' '.join(map(str, bbox))
            f.write(f"{category_id} {bbox_str}\n")

print("Annotations have been successfully converted to TXT files.")