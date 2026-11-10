from swift.llm import DatasetMeta, MessagesPreprocessor, load_dataset, register_dataset
from typing import Any, Dict, Optional
import random
random.seed(42)
import numpy as np
import sys
import os
import cv2
sys.path.append('/mnt/home/user26/MCR')
from utils.utils import *
from utils.template import template, select_template
# image path
gmner_path = '/mnt/home/user26/data/MNER/IJCAI2019_data/'
mner_mi_path = '/mnt/home/user26/data/MNER/MNER-MI/images'
coco_path = '/mnt/home/user26/data/VQA/coco2014/train2014/'
# data path 
base_path = '/mnt/home/user26/MCR/data'
datasets = ['gmner', 'mner', 'mner-mi', 'grefcoco']
train_cot_paths = {}
for dataset in datasets:
    if dataset == 'grefcoco':
        train_cot_paths[f'{dataset}A_cot_test_path'] = f'{base_path}/{dataset}/testA.json'
        train_cot_paths[f'{dataset}B_cot_test_path'] = f'{base_path}/{dataset}/testB.json'
    else:
        train_cot_paths[f'{dataset}_cot_test_path'] = f'{base_path}/{dataset}/test.json'
for key, value in train_cot_paths.items():
    globals()[key] = value

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
        return super().preprocess({
            'messages': [
                {
                    "role": "user",
                    "content": prompt.format(text_input=row['text_input'])
                }
            ],
            'images': [image_path],
        })
class MNERMISFTCOTPreprocessor(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        images_path = [os.path.join(mner_mi_path, img_id) for img_id in row['img_ids']]
        type = row['type']
        version = row['version']
        prompt = '<image>' * len(images_path) + '{text_input}' + template['mner-mi']['instruction'] + template['mner-mi'][type][version]
        return super().preprocess({
            'messages': [
                {
                    "role": "user",
                    "content": prompt.format(text_input=row['text_input'])
                }
            ],
            'images': images_path,
        })
class GMNERSFTCOTPreprocessor(MessagesPreprocessor):
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
        return super().preprocess({
            'messages': [
                {
                    "role": "user",
                    "content": prompt.format(text_input=row['text_input'])
                }
            ],
            'images': [image_path],
        })
class GREFCOCOSFTCOTPreprocessor(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image_path = os.path.join(coco_path, row['img_id'] + '.jpg')
        type = row['type']
        version = row['version']
        prompt = '<image>' + template['grefcoco']['instruction'] + template['grefcoco'][type][version]
        text_input = row['text_input']
        return super().preprocess({
                'messages': [
                    {
                        "role": "user",
                        "content": prompt.format(text_input=text_input)
                    }
                ],
                'images': [image_path],
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
    if dataset == 'grefcoco':
        dataset_path_A = train_cot_paths[f'{dataset}A_cot_test_path']
        dataset_path_B = train_cot_paths[f'{dataset}B_cot_test_path']
        register_dataset(
            DatasetMeta(
                dataset_name=f'{dataset}A_cot_test_multitype',
                dataset_path=dataset_path_A,
                preprocess_func=preprocess_func,
            )
        )
        register_dataset(
            DatasetMeta(
                dataset_name=f'{dataset}B_cot_test_multitype',
                dataset_path=dataset_path_B,
                preprocess_func=preprocess_func,
            )
        )
    else:
        dataset_path = train_cot_paths[f'{dataset}_cot_test_path']
        register_dataset(
            DatasetMeta(
                dataset_name=f'{dataset}_cot_test_multitype',
                dataset_path=dataset_path,
                preprocess_func=preprocess_func,
            )
        )
if __name__ == '__main__':
    tasks = ['gmner_cot_test_multitype']
    for task in tasks:
        dataset = load_dataset([task])[0]
        print(f'dataset: {dataset}')
        print(dataset[3])
    