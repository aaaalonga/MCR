import re
import json
import cv2
import os
from multiprocess.connection import answer_challenge
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

def load_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxs = []
    names = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        names.append(name)
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        bbox = {'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax}
        bboxs.append(bbox)
    return names, bboxs

def load_txt(txt_path):
    f=open(txt_path)
    data=[]
    raw_data=[]
    target=[]
    coarse_target=[]
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(raw_data) > 0:
                #import pdb;pdb.set_trace()
                data.append((raw_data,target, coarse_target))
                raw_data = []
                target = []
                coarse_target=[]
            continue
        splits = line.split('\t')
        if len(splits) == 1: ## Imageid
            raw_data.append(splits[0][:-1])
        else:
            raw_data.append(splits[0])
            target.append(splits[-1][:-1]) 
            coarse_target.append("O")
    if len(raw_data) >0:
        data.append((raw_data,target, coarse_target))
        raw_data = []
        target = []
        coarse_target=[]
    print("The number of samples: "+ str(len(data)))
    return data

def parse_triple(triple_str):
    # 匹配模式：
    # 1. 实体文本：在第一个括号对中，逗号前的内容
    # 2. 实体类型：两个括号对之间的内容
    # 3. 坐标或None：最后一个括号对中的内容
    pattern = r'\(?\s*([^,]+?)\s*,\s*(\w+)\s*,\s*(\([^\)]+\)|\[[^\)]+\]|None)\s*\)?'
    matches = re.match(pattern, triple_str.strip())
    if matches:
        # 获取第一个捕获组（实体文本）
        entity_text = matches.group(1).strip()
        # 获取第二个捕获组（实体类型）
        entity_type = matches.group(2).strip()
        # 获取第三个捕获组（边界框）
        bbox = matches.group(3).strip()
        
        # 处理边界框
        if bbox != 'None':
            # 将"(7, 106, 600, 401)"转换为[7, 106, 600, 401]
            try:
                bbox = [float(x.strip()) for x in bbox[1:-1].split(',')]
            except:
                bbox = None
        else:
            bbox = None
            
        return entity_text, entity_type, bbox
    
    return None, None, None

def check_pred(triple_str):
    if 'I , Daniel Blake' in triple_str and (len(triple_str.split(',')) == 4 or len(triple_str.split(',')) == 7):
        return True
    if 'Sabrina , The Teenage Witch' in triple_str and (len(triple_str.split(',')) == 4 or len(triple_str.split(',')) == 7):
        return True
    if len(triple_str.split(',')) != 3 and len(triple_str.split(',')) != 6:
        return False
    if triple_str != '':
        try:
            entity_text, entity_type, bbox = parse_triple(triple_str)
        except:
            entity_text, entity_type, bbox = None, None, None
        if entity_text == '' or entity_text == None:
            return False
        else:
            return True
    else:
        return True

def extract_triple(str):
    triple_list = []
    # ( Ken Loach, person, None )\n( I , Daniel Blake, miscellaneous, ( 172, 30, 490, 289 ) )\n( Palme d ' Or, miscellaneous, None )\n( Festival_Cannes, miscellaneous, None )\n
    for triple_str in str.split('\n')[:-1]:
        # ( I , Daniel Blake, miscellaneous, ( 172, 30, 490, 289 ) )
        triple_str = triple_str.strip()
        triple_str = triple_str[1:-1].strip()
        # "( H . Jackson Brown , Jr, person, ( 195, 731, 478, 768 ) )\n"
        # "( James Franco, person, None )\n( Mother , May I Sleep with Danger, miscellaneous, None )\n"
        # "( Charlie Chaplin : A Silent Icon, miscellaneous, ( 2, 3, 600, 857 ) )\n( Beacon Publishing , Inc, organization, None )\n"
        # "( Martin Luther King , Jr, person, ( 295, 26, 597, 303 ) )\n"
        try:
            if len(triple_str.split(',')) == 4:
                entity_text = triple_str.split(',')[0].strip() + ' , ' + triple_str.split(',')[1].strip()
                bbox = None
                entity_type = triple_str.split(',')[2].strip()
            elif len(triple_str.split(',')) == 7:
                entity_text = triple_str.split(',')[0].strip() + ' , ' + triple_str.split(',')[1].strip()
                entity_type = triple_str.split(',')[2].strip()
                bbox = []
                bbox.append(int(float(triple_str.split(',')[3][2:].strip())))
                bbox.append(int(float(triple_str.split(',')[4].strip())))
                bbox.append(int(float(triple_str.split(',')[5].strip())))
                bbox.append(int(float(triple_str.split(',')[6][:-2].strip())))
            else:
                for i, s in enumerate(triple_str.split(',')):
                    s = s.strip()
                    if i == 0:
                        entity_text = s
                    elif i == 1:
                        entity_type = s
                    elif i == 2:
                        if s.lower() == 'none':
                            bbox = None
                        else:
                            bbox = [int(float(s[1:].strip()))]
                    elif i == 3:
                        bbox = bbox + [int(float(s.strip()))]
                    elif i == 4:
                        bbox = bbox + [int(float(s.strip()))]
                    elif i == 5:
                        bbox = bbox + [int(float(s[:-1].strip()))]
            triple_list.append((entity_text, entity_type, bbox))
        except:
            continue
    return triple_list   
                
def read_ground_truth(ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        data = json.load(f)
    # data = data[:3]
    label_list = []
    for item in data:
        labels = item['labels']
        if labels == '':
            triples = []
        else:
            try:
                triples = extract_triple(labels)
            except:
                print(labels)
        label_list.append(triples)
    return label_list

def read_lora_result_txt(lora_result_path):
    with open(lora_result_path, 'r') as f:
        data = f.readlines()
    # data = data[:10]
    str = ''
    pred_list = []
    last_data = ''
    for item in data:
        if item != '\n':
            if item.startswith('='):
                if str != '':
                    pred_list.append(str)
                str = ''
            else:
                str += item
        if last_data.startswith('=') and item == '\n':
            pred_list.append("")
            str = ''
        last_data = item
    pred_list.append(str)
    return pred_list

def read_lora_result(lora_result_path):
    pred_txt = read_lora_result_txt(lora_result_path)
    # print(pred_txt[1137:1138])
    pred_list = []
    formal_error = 0
    for item in pred_txt:
        triple_list = []
        for triple_str in item.split('\n')[:-1]:
            formal = check_pred(triple_str)
            if formal:
                if 'I , Daniel Blake' in triple_str:
                    # ( I , Daniel Blake, miscellaneous, None )
                    entity_text = 'I , Daniel Blake'
                    temp_str = ''
                    for i, s in enumerate(triple_str.split(',')):
                        if i != 1:
                            temp_str += s.strip() + ', '
                    temp_str = temp_str[:-2]
                    # print(temp_str)
                    _, entity_type, bbox = parse_triple(temp_str)
                elif 'Sabrina , The Teenage Witch' in triple_str:
                    entity_text = 'Sabrina , The Teenage Witch'
                    temp_str = ''
                    for i, s in enumerate(triple_str.split(',')):
                        if i != 1:
                            temp_str += s.strip() + ', '
                    temp_str = temp_str[:-2]
                    _, entity_type, bbox = parse_triple(temp_str)
                else:
                    entity_text, entity_type, bbox = parse_triple(triple_str)
                triple_list.append((entity_text, entity_type, bbox))
            else:
                formal_error += 1
                # triple_list.append("")
        pred_list.append(triple_list)
    return pred_list, formal_error

def iou(box1, box2):
    if box1 is None and box2 is None:
        return 1
    if box1 is None or box2 is None:
        return 0
    if len(box1) != 4 or len(box2) != 4:
        return 0
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    x_left = max(x1, x1_)
    y_top = max(y1, y1_)
    x_right = min(x2, x2_)
    y_bottom = min(y2, y2_)

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        iou = 0
    else:
        iou = intersection_area / union_area

    return iou

def convert_labels_list_to_str(labels_list):
    str = ''
    for item in labels_list:
        if item[2] == None:
            str += f'( {item[0]}, {item[1]}, None )\n'
        else:
            str += f'( {item[0]}, {item[1]}, ( {item[2][0]}, {item[2][1]}, {item[2][2]}, {item[2][3]} ) )\n'
    return str

def save_image(image_path, image, save_folder, suffix):
    image_name = image_path.split('/')[-1].split('.')[0] + suffix + '.jpg'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, image_name)
    cv2.imwrite(save_path, image)
    return save_path

def mask_special_region(image_path, bbox):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    masked_image = image.copy()
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    masked_image[y1:y2, x1:x2] = 0
    return masked_image

def only_unmask_special_region(image_path, bbox):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 255 
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def masked_image_plt(masked_image):
    img_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def transfer_bbox(labels_list, resized_width, resized_height, w, h):
    transfer_labels_list = []
    for triple in labels_list:
        if triple[2] is not None:
            # trans_bbox = np.round(triple[2] / np.array([w, h, w, h]) * np.array([resized_width, resized_height, resized_width, resized_height]), 3)
            trans_bbox = np.round(
                triple[2] / np.array([w, h, w, h]) * np.array([resized_width, resized_height, resized_width, resized_height])
            ).astype(int)
            transfer_labels_list.append((triple[0], triple[1], trans_bbox.tolist()))
        else:
            transfer_labels_list.append(triple)
    return transfer_labels_list

def read_pred_result(pred_str_list):
    pred_list = []
    formal_error = 0
    for item in pred_str_list:
        triple_list = []
        for triple_str in item.split('\n')[:-1]:
            formal = check_pred(triple_str)
            if formal:
                if 'I , Daniel Blake' in triple_str:
                    # ( I , Daniel Blake, miscellaneous, None )
                    entity_text = 'I , Daniel Blake'
                    temp_str = ''
                    for i, s in enumerate(triple_str.split(',')):
                        if i != 1:
                            temp_str += s.strip() + ', '
                    temp_str = temp_str[:-2]
                    # print(temp_str)
                    _, entity_type, bbox = parse_triple(temp_str)
                elif 'Sabrina , The Teenage Witch' in triple_str:
                    entity_text = 'Sabrina , The Teenage Witch'
                    temp_str = ''
                    for i, s in enumerate(triple_str.split(',')):
                        if i != 1:
                            temp_str += s.strip() + ', '
                    temp_str = temp_str[:-2]
                    _, entity_type, bbox = parse_triple(temp_str)
                else:
                    entity_text, entity_type, bbox = parse_triple(triple_str)
                triple_list.append((entity_text, entity_type, bbox))
            else:
                formal_error += 1
                # triple_list.append("")
        pred_list.append(triple_list)
    return pred_list, formal_error

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 280 * 280, max_pixels: int = 280 * 280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    # print(h_bar, w_bar)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
        # print(beta, h_bar, w_bar)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=280 * 280, max_pixels=280 * 280):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    # print(new_height, new_width)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]


def convert_from_qwen25vl_format(bbox_new, orig_height, orig_width, factor=28, min_pixels=280 * 280, max_pixels=280 * 280):
    """
    将转换后的bbox坐标反推回原始图像坐标
    
    参数:
        bbox_new: 转换后的坐标 [x1_new, y1_new, x2_new, y2_new]
        orig_height: 原始图像高度
        orig_width: 原始图像宽度
        factor: 缩放因子（默认28）
        min_pixels: 最小像素数（默认280*280）
        max_pixels: 最大像素数（默认280*280）
    
    返回:
        原始坐标 [x1, y1, x2, y2]
    
    注意:
        由于正向转换时使用了round()操作，反推结果可能会有1像素的误差
    """
    # 计算转换后的尺寸
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    
    # 计算缩放比例
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    # 反推原始坐标
    x1_new, y1_new, x2_new, y2_new = bbox_new
    x1 = x1_new / scale_w
    y1 = y1_new / scale_h
    x2 = x2_new / scale_w
    y2 = y2_new / scale_h
    
    # 转换为整数（因为在原始坐标系统中通常也是整数）
    x1 = round(x1)
    y1 = round(y1)
    x2 = round(x2)
    y2 = round(y2)
    
    # 确保坐标在有效范围内
    x1 = max(0, min(x1, orig_width - 1))
    y1 = max(0, min(y1, orig_height - 1))
    x2 = max(0, min(x2, orig_width - 1))
    y2 = max(0, min(y2, orig_height - 1))
    
    return [x1, y1, x2, y2]


def smart_transfer_bbox(labels_list, orig_height, orig_width, factor=28, min_pixels=280 * 280, max_pixels=280 * 280):
    transfer_labels_list = []
    for triple in labels_list:
        if triple[2] is not None:
            trans_bbox = convert_to_qwen25vl_format(triple[2], orig_height, orig_width, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)
            transfer_labels_list.append((triple[0], triple[1], trans_bbox))
        else:
            transfer_labels_list.append(triple)
    return transfer_labels_list

def transfer_bbox2str_vg(bbox, orig_height, orig_width, factor=28, min_pixels=280 * 280, max_pixels=280 * 280):
    if bbox == []:
        return 'None'
    else:
        trans_bbox = convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)
        trans_bbox = np.round(trans_bbox).astype(int)
        labels = f"( " + str(trans_bbox[0]) + ", " + str(trans_bbox[1]) + ", " + str(trans_bbox[2]) + ", " + str(trans_bbox[3]) + " )\n"
        return labels
# 提取坐标并转换成列表格式
def extract_bbox_coordinates(bbox_string):
    # 去除换行符和首尾空格
    bbox_string = bbox_string.strip()
    # 按换行符分割，取第一个坐标
    first_bbox = bbox_string.split('\n')[0]
    # 去除首尾的括号
    first_bbox = first_bbox.strip()[1:-1]
    # 按逗号分割并去除空格
    coordinates = [int(float(x.strip())) for x in first_bbox.split(',')]
    return coordinates

def extract_tuple(label_str):
    tuple_list = []
    for item in label_str.split('\n')[:-1]:
        # 去除首尾空白字符
        item = item.strip()
        if not item:
            continue
        # 检查是否以 ( 开头和 ) 结尾
        if item.startswith('(') and item.endswith(')'):
            # 去除括号
            content = item[1:-1].strip()
            # 按逗号分割
            parts = content.split(',')
            if len(parts) == 2:
                entity = parts[0].strip()
                label = parts[1].strip()
                tuple_list.append((entity, label))
            elif len(parts) > 2:
                if parts[-1].strip() in ['location', 'person', 'organization', 'miscellaneous', 'visible', 'invisible', 'None']:
                    entity = parts[0]
                    for i in range(len(parts) - 2):
                        entity += ',' + parts[i+1]
                    entity = entity.strip()
                    label = parts[-1].strip()
                    tuple_list.append((entity, label))
                elif ')' in parts[-1].strip() or ']' in parts[-1].strip():
                    x1 = parts[-4].strip()[1:]
                    y1 = parts[-3].strip()
                    x2 = parts[-2].strip()
                    y2 = parts[-1].strip()[:-1]
                    entity = parts[0]
                    for i in range(len(parts) - 5):
                        entity += ',' + parts[i+1]
                    entity = entity.strip()
                    tuple_list.append((entity, [int(x1), int(y1), int(x2), int(y2)]))
    return tuple_list

def extract_special_label_info(text, tag='<answer>'):
    import re
    if tag == '<|begin_of_box|>':
        pattern = f'<|begin_of_box|>(.*?)<|begin_of_box|>'
    else:
        pattern = f'<{tag[1:-1]}>(.*?)</{tag[1:-1]}>'
        
    # 查找所有匹配的内容
    matches = re.findall(pattern, text, re.DOTALL)
    tag_list = []
    for match in matches:
        if tag == '<answer>':
            content = match
        elif tag == '<|begin_of_box|>':
            content = match
        else:
            content = match.strip()
        if content:
            tag_list.append(content)
    if tag == '<answer>' and len(tag_list) == 0:
        tag_list.append('')
    if tag == '<|begin_of_box|>' and len(tag_list) == 0:
        tag_list.append('')
    if tag == '<check>' and len(tag_list) == 0:
        patterns = ['\(check\)(.*?)\</check>', '\(check\)(.*?)\(/check\)', '<check>(.*?)\(/check\)']
        for pattern in patterns:
            matches_1 = re.findall(pattern, text, re.DOTALL)
            for match in matches_1:
                if match:
                    tag_list.append(match.strip())
                    break
            if len(tag_list) > 0:
                break
    return tag_list

def normalize_coordinates(box, image_height, image_width):
    x1, y1, x2, y2 = box
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_box

def normalize_triple_bbox_InternVL(labels_list, image_height, image_width):
    transfer_labels_list = []
    for triple in labels_list:
        if triple[2] is not None:
            trans_bbox = normalize_coordinates(triple[2], image_height, image_width)
            transfer_labels_list.append((triple[0], triple[1], trans_bbox))
        else:
            transfer_labels_list.append(triple)
    return transfer_labels_list

def normalize_triple_bbox_Ovis(labels_list, image_height, image_width):
    transfer_labels_list = []
    for triple in labels_list:
        if triple[2] is not None:
            trans_bbox = normalize_coordinates(triple[2], image_height, image_width)
            trans_bbox = [trans_bbox[0] / 1000, trans_bbox[1] / 1000, trans_bbox[2] / 1000, trans_bbox[3] / 1000]
            transfer_labels_list.append((triple[0], triple[1], trans_bbox))
        else:
            transfer_labels_list.append(triple)
    return transfer_labels_list

def token_f1(pred, ground_truth):
    pred_tokens = pred.lower().split()
    ground_truth_tokens = ground_truth.lower().split()
    if len(pred_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0
    common_tokens = set(pred_tokens) & set(ground_truth_tokens)
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    if precision + recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)

def check_think(triples, think):
    for triple in triples:
        if triple[2] is not None:
            if f"{triple[2][0]}, {triple[2][1]}, {triple[2][2]}, {triple[2][3]}" not in think:
                if f"{triple[2][0]}, {triple[2][1]}" not in think and f"{triple[2][2]}, {triple[2][3]}" not in think:
                    if f"{triple[2][0]}" not in think or f"{triple[2][1]}" not in think or triple[2][2] is None or triple[2][3] is None:
                        return False
    return True

def extract_triples_GLM(answer_str):
    if '\n' not in answer_str:
        # (Gabriele _ Corno, person, None)(St . Petersberg, location, None)(Gordeev, person, None)
        answer_str = answer_str.replace(')(', ')\n(')
        answer = answer_str + '\n'
    elif '\n' in answer_str and answer_str[-1] == '\n':
        answer = answer_str
    elif '\n' in answer_str and answer_str[-1] != '\n':
        # (WARRIORS, organization, (200, 580, 450, 650))\n(THUNDER, organization, (565, 459, 667, 572))\n(NBAonTNT, organization, None)
        answer = answer_str + '\n'
    return extract_triple(answer)


if __name__ == '__main__':
    # labels_list = read_ground_truth('/mnt/home/user26/GMNER_DPO/data/multi_task/fmner/train.json')
    # print(len(labels_list))
    o = '( San Francisco City Hall, location, ( 2, 153, 223, 279 ) )\n'
    for triple_str in o.split('\n')[:-1]:
        entity_text, entity_type, bbox = parse_triple(triple_str)
        print(f"entity_text: {entity_text}, entity_type: {entity_type}, bbox: {bbox}")
    pred_list, formal_error = read_pred_result(o)
    print(pred_list)
    # response = "<reasoning>\nThe given text is: \"Laurel Elementary students had a great time participating International Walk to School Day !.\"  \nThe image shows a group of children outdoors, some wearing backpacks, gathered around a handmade yellow vehicle labeled \"BUS\" with a sign that reads \"Welcome Aboard! We're Walking to School!\" This suggests a school-related event, likely a walk to school activity.\n\nFirst, we identify the entities mentioned in the text:  \n1. \"Laurel Elementary\" – This is a proper noun referring to a school. In the context of the text, it is clearly an educational institution.  \n2. \"International Walk to School Day\" – This is a named event, widely recognized as an annual global event promoting walking to school, which falls under the category of a festival or public event.\n\nNext, we determine whether these entities are visually grounded in the image:  \n- \"Laurel Elementary\" is not visually present in the image. There is no sign, building, or text in the image that explicitly identifies the school as \"Laurel Elementary.\" Thus, its location is None.  \n- \"International Walk to School Day\" is also not explicitly mentioned in the image. However, the sign on the bus reads \"We're Walking to School!\" which strongly implies participation in a walk-to-school event. While this aligns with the spirit of the event, the specific name \"International Walk to School Day\" is not visually present. Therefore, its location is also None.\n\nWe assign fine-grained types based on semantic meaning, not the label:  \n- \"Laurel Elementary\" is an educational institution, which is accurate.  \n- \"International Walk to School Day\" is an event, and while it could be considered a \"festival\" in some contexts (as it is a recurring public event), it is more precisely a \"public_event\" or \"community_event\". However, given the provided fine-grained categories, if \"festival\" is the closest available type, we may use it. But since the instruction says not to easily change the predicted type because of the label, and \"festival\" is a reasonable classification for a large-scale, recurring public celebration, we will retain it for consistency with the label, but note that it could also be classified as \"event\" if that category exists.\n\nIn this case, since the label provides \"festival\" for \"International Walk to School Day\", and we are not instructed to change it unless we have a better fit, and \"festival\" is acceptable, we maintain that type.\n\nTherefore, the reasoning process concludes that:  \n- \"Laurel Elementary\" is an educational_institution and not visible in the image → location: None  \n- \"International Walk to School Day\" is a festival and not visible in the image → location: None  \n\nThe predicted types match the label types exactly.\n\n</reasoning>\n(check)True</check>"
    # l = extract_special_label_info(response, tag='<check>')
    # print(len(l))
    # print(l)
    
    # labels_str = '<answer>( Pak, building_other, None )\n</answer>'
    # labels_list = extract_triple(extract_special_label_info(labels_str, '<answer>')[0])
    # print(labels_list)
    # img = cv2.imread('/mnt/home/user26/data/MNER/IJCAI2019_data/twitter2015_images/797970.jpg')
    # h, w = img.shape[:2]
    # smart_transfer_labels_list = normalize_triple_bbox_InternVL(labels_list, h, w)
    # answer_labels = convert_labels_list_to_str(smart_transfer_labels_list)
    # print(answer_labels)
    # ipath = '/mnt/home/user26/data/MNER/IJCAI2019_data/twitter2015_images/575705.jpg'
    # img = cv2.imread(ipath)
    # h, w = img.shape[:2]
    # bbox_original = convert_from_qwen25vl_format((56, 94, 139, 201), h, w)
    # trans_bbox = convert_to_qwen25vl_format(( 151, 244, 373, 523 ), h, w)
    # print(trans_bbox)
    # print(bbox_original)
    # print(iou(( 151, 244, 373, 523 ), bbox_original))
    # triple_str = '( Webber MS, person, None )'
    # entity_text, entity_type, bbox = parse_triple(triple_str)
    # print(f'entity_text: {entity_text}, entity_type: {entity_type}, bbox: {bbox}')
    # str = "( Nick Jonas, person, None )\n( Demi Lovato, person, ( 220, 32, 448, 500 ) )\n"
    # str = '\n<|begin_of_box|>(Gabriele _ Corno, person, None)(St . Petersberg, location, None)(Gordeev, person, None)<|begin_of_box|>'
    # str2 = '<|begin_of_box|>(WARRIORS, organization, (200, 580, 450, 650))\n(THUNDER, organization, (565, 459, 667, 572))\n(NBAonTNT, organization, None)<|end_of_box|>'
    # # response = '<answer></answer>'
    # answer = extract_special_label_info(str, '<|begin_of_box|>')[0]
    # answer2 = extract_special_label_info(str2, '<|begin_of_box|>')[0]
    # print(extract_triples_GLM(answer))
    # print(extract_triples_GLM(answer2))
    # response = '<process>\n</process>'
    # answer = extract_special_label_info(response, '<process>')
    # print(answer)
    # # answer = '( Grand Concourse, location )\n( Dominican Day Parade, miscellaneous )\n( Bronx, location )\n'
    # turple = extract_tuple(answer)
    # print(turple)
    
    # ipath = '/mnt/home/user26/data/MNER/IJCAI2019_data/twitter2015_images/1450627.jpg'
    # img = cv2.imread(ipath)
    # h, w = img.shape[:2]
    # labels_list = extract_triple(str)
    # print(labels_list)
    # # transfer_labels_list = smart_transfer_bbox(labels_list, h, w, factor=28, min_pixels=280 * 280, max_pixels=280 * 280)
    # transfer_labels_list = transfer_bbox(labels_list, 280, 280, w, h)
    # print(transfer_labels_list)
    # print(iou(( 79, 24, 92, 38 ),( 78, 27, 92, 41 ) ))
    
    # s = "( Seattle, location, ( 173, 12, 323, 217 ) )\n( Wisconsin, location, None )\n"
    # s = "<think>\nTo determine the ground-truth labels for the given image-text pair, let's analyze both the text and the image step by step:\n\n1. **Text Analysis**:\n   - The text mentions \"Pak\" which is an abbreviation for Pakistan.\n   - \"Pak\" is used in the context of being a home to terrorists, indicating it refers to a geographical location.\n\n2. **Entity Identification**:\n   - From the text, \"Pak\" is identified as a meaningful and specific entity.\n   - \"Pak\" is a proper noun referring to a country, thus it is classified as a `location` type entity.\n\n3. **Image Analysis**:\n   - The image shows a cartoon where a soldier labeled \"Pakistan\" is standing next to a mailbox labeled \"Osama Bin Laden.\"\n   - However, the image does not directly depict \"Pak\" as a visible entity within the scene. It uses \"Pakistan\" as part of the soldier's label but does not show \"Pak\" as a standalone entity.\n\n4. **Location Determination**:\n   - Since \"Pak\" is not visibly represented as a separate entity in the image, its location is determined to be `None`.\n\nBased on the above analysis, the ground-truth labels can be derived as follows:\n- The entity is \"Pak\".\n- The type of the entity is `location`.\n- The location of the entity in the image is `None`.\n\nThus, the ground-truth labels are: (Pak, location, None).\n</think>\n<check>True</check>"
    # s = '<answer></answer>'
    # assert extract_special_label_info(s, tag='<answer>')[0] == ""
    # print(extract_special_label_info(s, tag='<check>')[0])
    
    # s = '( Haloti Ngata, person, ( 270, 65, 600, 450 ) )\n( Jim Csldwell, person, ( 33, 102, 316, 450 ) )\n'
    # labels_list = extract_triple(s)
    # trans_bbox = normalize_coordinates([
    #         137, 436, 1066, 890
    #     ], h, w)
    # trans_bbox = np.round(trans_bbox).astype(int)
    # labels = f"( " + str(trans_bbox[0]) + ", " + str(trans_bbox[1]) + ", " + str(trans_bbox[2]) + ", " + str(trans_bbox[3]) + " )\n"
    # smart_transfer_labels_list = smart_transfer_bbox(labels_list, h, w)
    # labels = convert_labels_list_to_str(smart_transfer_labels_list)
    # print(labels)
    # trans_bbox = normalize_coordinates(( 151, 244, 373, 523 ), h, w)
    # print(smart_transfer_labels_list)
    # trans_bbox = convert_to_qwen25vl_format(( 34, 37, 503, 399 ), h, w, factor=28, min_pixels=280 * 280, max_pixels=280 * 280)
    # iou_result = iou(( 30, 21, 284, 223 ),trans_bbox)
    # print(iou_result)
    # # 使用示例
    # bbox= [
    #         50, 104, 162, 363
    #     ]
    # org = f"( {round(bbox[0])}, {round(bbox[1])}, {round(bbox[2])}, {round(bbox[3])} )"
    # trans_bbox = convert_from_qwen25vl_format(bbox, h, w, factor=28, min_pixels=280 * 280, max_pixels=280 * 280)
    # bbox_coords = extract_bbox_coordinates(string)
    # print(f"提取的坐标: {bbox_coords}")
    # trans_bbox = convert_to_qwen25vl_format(bbox_coords, h, w, factor=28, min_pixels=280 * 280, max_pixels=280 * 280)
    # trans_bbox = np.round(trans_bbox).astype(int)
    # print(trans_bbox)
    
    # s = "( Carmelo Anthony, ( 10, 1, 118, 195 ) )\n"
    
    # print(extract_tuple(s))

    