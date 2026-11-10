from swift.llm import DatasetMeta, MessagesPreprocessor, load_dataset, register_dataset
from typing import Any, Dict, Optional
import random
random.seed(42)
import numpy as np
import sys
import os
import cv2
sys.path.append('/mnt/home/user26/GMNER_DPO')
from utils.utils import *
from utils.template import template, select_template
# image path
gmner_path = '/mnt/home/user26/data/MNER/IJCAI2019_data/'
mner_mi_path = '/mnt/home/user26/data/MNER/MNER-MI/images'
coco_path = '/mnt/home/user26/data/VQA/coco2014/train2014/'
# data path 
base_path = '/mnt/home/user26/MCR/data'
datasets = ['gmner', 'mner', 'mner-mi', 'grefcoco']
# datasets = ['gmner']
train_cot_paths = {}
for dataset in datasets:
    train_cot_paths[f'{dataset}_filtered_data'] = f'{base_path}/{dataset}/filtered_data.json'
for key, value in train_cot_paths.items():
    globals()[key] = value

MAX_PIXELS = 78400
MIN_PIXELS = 78400



class MNERSFTCOTPreprocessor(MessagesPreprocessor): 
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if os.path.exists(gmner_path + 'twitter2015_images/' + row['img_id'] + '.jpg'):
            image_path = gmner_path + 'twitter2015_images/' + row['img_id'] + '.jpg'
        elif os.path.exists(gmner_path + 'twitter2017_images/' + row['img_id'] + '.jpg'):
            image_path = gmner_path + 'twitter2017_images/' + row['img_id'] + '.jpg'
        else:
            image_path = row['image_path']
        type = row['type']
        version = row['version']
        prompt = '<image> {text_input}' + template['mner']['instruction'] + template['mner'][type][version]
        labels = extract_special_label_info(row['solution'], '<answer>')[0]
        return super().preprocess({
            'messages': [
                {
                    "role": "user",
                    "content": prompt.format(text_input=row['text_input'])
                }
            ],
            'images': [image_path],
            'solution':labels,
            'task':row['dataset']
        })
class MNERMISFTCOTPreprocessor(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        images_path = [os.path.join(mner_mi_path, img_id) for img_id in row['img_ids']]
        type = row['type']
        version = row['version']
        prompt = '<image>' * len(images_path) + '{text_input}' + template['mner-mi']['instruction'] + template['mner-mi'][type][version]
        labels = extract_special_label_info(row['solution'], '<answer>')[0]
        return super().preprocess({
            'messages': [
                {
                    "role": "user",
                    "content": prompt.format(text_input=row['text_input'])
                }
            ],
            'images': images_path,
            'solution':labels,
            'task':row['dataset']
        })
class GMNERSFTCOTPreprocessor(MessagesPreprocessor):
    max_pixels = MAX_PIXELS
    min_pixels = MIN_PIXELS
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if os.path.exists(gmner_path + 'twitter2015_images/' + row['img_id'] + '.jpg'):
            image_path = gmner_path + 'twitter2015_images/' + row['img_id'] + '.jpg'
        elif os.path.exists(gmner_path + 'twitter2017_images/' + row['img_id'] + '.jpg'):
            image_path = gmner_path + 'twitter2017_images/' + row['img_id'] + '.jpg'
        else:
            image_path = row['image_path']
        type = row['type']
        version = row['version']
        prompt = '<image> {text_input}' + template['gmner']['instruction'] + template['gmner'][type][version]
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        labels_list = extract_triple(extract_special_label_info(row['solution'], '<answer>')[0])
        smart_transfer_labels_list = smart_transfer_bbox(labels_list, h, w, factor=28, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        labels = convert_labels_list_to_str(smart_transfer_labels_list)
        return super().preprocess({
            'messages': [
                {
                    "role": "user",
                    "content": prompt.format(text_input=row['text_input'])
                }
            ],
            'images': [image_path],
            'solution':labels,
            'task':row['dataset']
        })
class GREFCOCOSFTCOTPreprocessor(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image_path = os.path.join(coco_path, row['img_id'] + '.jpg')
        type = row['type']
        version = row['version']
        prompt = '<image>' + template['grefcoco']['instruction'] + template['grefcoco'][type][version]
        bbox = row['bbox']
        w = row['width']
        h = row['height']
        turple_str = extract_special_label_info(row['cot_formal']['v1'], '<entailment>')[0]
        turple = extract_tuple(turple_str+'\n')[0]
        for k, v in row['text_input'].items():
             if v.lower().strip() == turple[0].lower().strip():
                 text_input = turple[0]
                 break
        if bbox == []:
            labels = 'None'
        else:
            bbox_str = extract_special_label_info(row['solution'], '<answer>')[0]
            org_bbox = extract_bbox_coordinates(bbox_str)
            orgs1 = f"{org_bbox[0]}, {org_bbox[1]}"
            orgs2 = f"{org_bbox[2]}, {org_bbox[3]}"
            trans_bbox = convert_to_qwen25vl_format(bbox, h, w, factor=28, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
            trans_bbox = np.round(trans_bbox).astype(int)
            new = f"( {trans_bbox[0]}, {trans_bbox[1]}, {trans_bbox[2]}, {trans_bbox[3]} )"
            new1 = f"{trans_bbox[0]}, {trans_bbox[1]}"
            new2 = f"{trans_bbox[2]}, {trans_bbox[3]}"
            labels = new
            if orgs1 in row[type][version]:
                row[type][version] = row[type][version].replace(orgs1, new1)
            if orgs2 in row[type][version]:
                row[type][version] = row[type][version].replace(orgs2, new2)
        return super().preprocess({
                'messages': [
                    {
                        "role": "user",
                        "content": prompt.format(text_input=text_input)
                    }
                ],
                'images': [image_path],
                'solution':labels,
                'task':row['dataset']
            })



for dataset in datasets:
    if dataset == 'gmner':
        preprocess_func = GMNERSFTCOTPreprocessor()
    elif dataset == 'mner':
        preprocess_func = MNERSFTCOTPreprocessor()
    elif dataset == 'mner-mi':
        preprocess_func = MNERMISFTCOTPreprocessor()
    elif dataset == 'grefcoco':
        preprocess_func = GREFCOCOSFTCOTPreprocessor()
    dataset_path = train_cot_paths[f'{dataset}_filtered_data']
    register_dataset(
        DatasetMeta(
            dataset_name=f'{dataset}_filtered_data',
            dataset_path=dataset_path,
            preprocess_func=preprocess_func,
        )
    )
if __name__ == '__main__':
    tasks = ['gmner_filtered_data', 'mner_filtered_data', 'mner-mi_filtered_data', 'grefcoco_filtered_data']
    for task in tasks:
        dataset = load_dataset([task])[0]
        print(f'dataset: {dataset}')
        print(dataset[3])
    