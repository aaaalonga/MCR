import sys
import os
import cv2
import re
import json
import numpy as np
import pandas as pd
import argparse

from pandas.core.arrays import boolean
sys.path.append('/mnt/home/user26/MCR')
from utils.utils import *
parser = argparse.ArgumentParser()
parser.add_argument('--result_floder', type=str, help='the folder of the result')
parser.add_argument('--test_path', type=str, help='the path of the test data')
parser.add_argument('--v', type=str, default='grpo', help='the version of the result')
parser.add_argument('--min_pixels', type=int, default=78400, help='the minimum pixels of the image')
parser.add_argument('--max_pixels', type=int, default=78400, help='the maximum pixels of the image')
parser.add_argument('--cot', type=bool, default=False, help='whether to use the cot')
parser.add_argument('--normalized', type=bool, default=False, help='whether to use the normalized bbox 0-1000')
parser.add_argument('--normalized_1', type=bool, default=False, help='whether to use the normalized bbox 0-1')
parser.add_argument("--task", type=str, default="gmner")
args = parser.parse_args()

if __name__ == '__main__':
    test_path = args.test_path
    result_floder = args.result_floder + '/' + args.v
    result_files = [f for f in os.listdir(result_floder) if f.endswith('.jsonl')]
    def get_checkpoint_num(filename):
        num = filename.split('_')[-1].split('.')[0]
        return int(num)
    result_files.sort(key=get_checkpoint_num)
    if args.task == 'gmner' or args.task == 'fmner':
        with open(test_path, 'r') as f:
            ground_truth_data = json.load(f)
        label_list = read_ground_truth(test_path)
        result_list = []
        for file in result_files:
            result_path = os.path.join(result_floder, file)
            with open(result_path, 'r') as f:
                results = [json.loads(line) for line in f]
            pred_str_list = [result["response"] for result in results]
            if args.cot:
                pred_str_list_temp = [extract_special_label_info(pred_str, tag='<answer>')[0] for pred_str in pred_str_list]
            else:
                pred_str_list_temp = pred_str_list
            pred_list, formal_error = read_pred_result(pred_str_list_temp)
            # print(formal_error)
            if len(pred_list) == len(label_list):
                res = {
                    'entity_error': 0,  
                    'entity_boundary_error': 0,
                    'entity_type_error': 0,
                    'right': 0,
                    'only_entity_type_error': 0,
                    'both_type_bbox_error': 0,
                    'only_bbox_error': 0,
                    'bbox_error_pred_bbox_label_none': 0,
                    'bbox_error_pred_none_label_bbox': 0,
                    'bbox_error_pred_bbox_label_bbox': 0,
                    'entity_boundary_type_error': 0,
                    'only_entity_boundary_type_error': 0,
                    'both_boundary_type_bbox_error': 0,
                    'only_entity_boundary_error': 0,
                    'entity_boundary_bbox_error': 0,
                    'entity_boundary_bbox_error_pred_bbox_label_none': 0,
                    'entity_boundary_bbox_error_pred_none_label_bbox': 0,
                    'entity_boundary_bbox_error_pred_bbox_label_bbox': 0,
                    'label_contain_pred': 0,
                    'pred_contain_label': 0,
                    'boundary_start_end_error': 0,
                    'recall_error': 0,
                    'pred_num': 0,
                    'label_num': 0,
                    'right_none': 0,
                    'mner_right': 0,
                    'eeg_right': 0,
                }
                for i in range(len(pred_list)):
                    # print(pred_list[i])
                    if pred_list[i] != []:
                        res["pred_num"] += len(pred_list[i])
                    else:
                        res["pred_num"] += 1
                    if label_list[i] != []:
                        res["label_num"] += len(label_list[i])
                    else:
                        res["label_num"] += 1
                    if pred_list[i] == [] and label_list[i] == []:
                        res["right_none"] += 1
                            
                for i in range(len(pred_list)):
                    img = cv2.imread(ground_truth_data[i]["image_path"])
                    h, w = img.shape[:2]
                    val_entities = []
                    for j in range(len(pred_list[i])):
                        check_entity_text = False
                        for k in range(len(label_list[i])):
                            # entity text right
                            if label_list[i][k][2] is not None:
                                resized_w = results[i].get("resized_width")
                                resized_h = results[i].get("resized_height")
                                min_px = results[i].get("min_pixels")
                                max_px = results[i].get("max_pixels")
                                if args.normalized:
                                    trans_bbox = normalize_coordinates(label_list[i][k][2], h, w)
                                elif args.normalized_1:
                                    trans_bbox = normalize_coordinates(label_list[i][k][2], h, w)
                                    trans_bbox = [trans_bbox[0] / 1000, trans_bbox[1] / 1000, trans_bbox[2] / 1000, trans_bbox[3] / 1000]
                                else:
                                    if resized_w is not None and resized_h is not None:
                                        trans_bbox = np.round(label_list[i][k][2] / np.array([w, h, w, h]) * np.array([resized_w, resized_h, resized_w, resized_h])).astype(int)
                                    elif min_px is not None and max_px is not None:
                                        trans_bbox = convert_to_qwen25vl_format(label_list[i][k][2], h, w, 28, min_pixels=min_px, max_pixels=max_px)
                                    else:
                                        trans_bbox = convert_to_qwen25vl_format(label_list[i][k][2], h, w, 28, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
                            else:
                                trans_bbox = None
                            if pred_list[i][j][0].lower() == label_list[i][k][0].lower():
                                if label_list[i][k][0].lower() not in val_entities:
                                    val_entities.append(label_list[i][k][0].lower())
                                    check_entity_text = True
                                    # type wrong
                                    if pred_list[i][j][1] != label_list[i][k][1]:
                                        res["entity_type_error"] += 1
                                        
                                        if iou(pred_list[i][j][2], trans_bbox) >= 0.5:
                                            res["only_entity_type_error"] += 1
                                            res['eeg_right'] += 1
                                        else:
                                            res["both_type_bbox_error"] += 1
                                    # type right
                                    else:
                                        res['mner_right'] += 1
                                        # bbox right
                                        if iou(pred_list[i][j][2], trans_bbox) >= 0.5:
                                            res["right"] += 1
                                            res['eeg_right'] += 1
                                        # bbox wrong
                                        else:
                                            res["only_bbox_error"] += 1
                                            if pred_list[i][j][2] != None and label_list[i][k][2] == None:
                                                res["bbox_error_pred_bbox_label_none"] += 1
                                            elif pred_list[i][j][2] == None and label_list[i][k][2] != None:
                                                res["bbox_error_pred_none_label_bbox"] += 1
                                            elif pred_list[i][j][2] != None and label_list[i][k][2] != None:
                                                res["bbox_error_pred_bbox_label_bbox"] += 1  
                                    break                     
                        if check_entity_text == False:
                            check_entailment = False
                            res["entity_error"] += 1
                            num = 3
                            for k in range(len(label_list[i])):
                                if label_list[i][k][2] is not None:
                                    resized_w = results[i].get("resized_width")
                                    resized_h = results[i].get("resized_height")
                                    min_px = results[i].get("min_pixels")
                                    max_px = results[i].get("max_pixels")
                                    if args.normalized:
                                        trans_bbox = normalize_coordinates(label_list[i][k][2], h, w)
                                    elif args.normalized_1:
                                        trans_bbox = normalize_coordinates(label_list[i][k][2], h, w)
                                        trans_bbox = [trans_bbox[0] / 1000, trans_bbox[1] / 1000, trans_bbox[2] / 1000, trans_bbox[3] / 1000]
                                    else:
                                        if resized_w is not None and resized_h is not None:
                                            trans_bbox = np.round(label_list[i][k][2] / np.array([w, h, w, h]) * np.array([resized_w, resized_h, resized_w, resized_h])).astype(int)
                                        elif min_px is not None and max_px is not None:
                                            trans_bbox = convert_to_qwen25vl_format(label_list[i][k][2], h, w, 28, min_pixels=min_px, max_pixels=max_px)
                                        else:
                                            trans_bbox = convert_to_qwen25vl_format(label_list[i][k][2], h, w, 28, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
                                else:
                                    trans_bbox = None
                                if pred_list[i][j][0] in label_list[i][k][0] or label_list[i][k][0] in pred_list[i][j][0]:
                                    check_entailment = True
                                    res["entity_boundary_error"] += 1
                                    if pred_list[i][j][0] in label_list[i][k][0] :
                                        res["label_contain_pred"] += 1
                                    elif label_list[i][k][0] in pred_list[i][j][0]:
                                        res["pred_contain_label"] += 1
                                        # print(pred_list[i][j][0], label_list[i][k][0])
                                    if pred_list[i][j][1] != label_list[i][k][1]:
                                        res["entity_boundary_type_error"] += 1
                                        if iou(pred_list[i][j][2], trans_bbox) >= 0.5:
                                            res["only_entity_boundary_type_error"] += 1
                                        else:
                                            res["both_boundary_type_bbox_error"] += 1
                                    # type right
                                    else:
                                        # bbox right
                                        if iou(pred_list[i][j][2], trans_bbox) >= 0.5:
                                            res["only_entity_boundary_error"] += 1
                                        # bbox wrong
                                        else:
                                            res["entity_boundary_bbox_error"] += 1
                                            if pred_list[i][j][2] != None and label_list[i][k][2] == None:
                                                res["entity_boundary_bbox_error_pred_bbox_label_none"] += 1
                                            elif pred_list[i][j][2] == None and label_list[i][k][2] != None:
                                                res["entity_boundary_bbox_error_pred_none_label_bbox"] += 1
                                            elif pred_list[i][j][2] != None and label_list[i][k][2] != None:
                                                res["entity_boundary_bbox_error_pred_bbox_label_bbox"] += 1
                                    break 
                                # pred或label的entity_text的尾部和label或pred的entity_text的头部重合的情况
                                elif pred_list[i][j][0][-num:] in label_list[i][k][0] or label_list[i][k][0][-num:] in pred_list[i][j][0]:
                                    check_entailment = True
                                    res["boundary_start_end_error"] += 1
                            if check_entailment == False:
                                res["recall_error"] += 1
                # if '600' in file or '1050' in file or '2180' in file:
                # Print_detail(res)
                res["right"] += res["right_none"]
                res["mner_right"] += res["right_none"]
                res['eeg_right'] += res["right_none"]
                # 计算指标
                precision = res["right"] / res["pred_num"] if res["pred_num"] > 0 else 0
                recall = res["right"] / res["label_num"] if res["label_num"] > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                mner_precision = res["mner_right"] / res["pred_num"] if res["pred_num"] > 0 else 0
                mner_recall = res["mner_right"] / res["label_num"] if res["label_num"] > 0 else 0
                mner_f1 = 2 * mner_precision * mner_recall / (mner_precision + mner_recall) if (mner_precision + mner_recall) > 0 else 0
                eeg_precision = res["eeg_right"] / res["pred_num"] if res["pred_num"] > 0 else 0
                eeg_recall = res["eeg_right"] / res["label_num"] if res["label_num"] > 0 else 0
                eeg_f1 = 2 * eeg_precision * eeg_recall / (eeg_precision + eeg_recall) if (eeg_precision + eeg_recall) > 0 else 0
                result = {
                    'checkpoint': file,
                    'Precision': round(precision * 100, 2),
                    'Recall': round(recall * 100, 2), 
                    'F1': round(f1 * 100, 2),
                    'pred_bbox_label_none': res["bbox_error_pred_bbox_label_none"],
                    'pred_none_label_bbox': res["bbox_error_pred_none_label_bbox"],
                    'pred_bbox_label_bbox': res["bbox_error_pred_bbox_label_bbox"],
                    'pred_num': res["pred_num"],
                    'label_num': res["label_num"],
                    'right': res["right"],
                    'mner_right': res["mner_right"],
                    'mner_precision': round(mner_precision * 100, 2),
                    'mner_recall': round(mner_recall * 100, 2),
                    'mner_f1': round(mner_f1 * 100, 2),
                    'eeg_right': res["eeg_right"],
                    'eeg_precision': round(eeg_precision * 100, 2),
                    'eeg_recall': round(eeg_recall * 100, 2),
                    'eeg_f1': round(eeg_f1 * 100, 2)
                }
                result_list.append(result)
            else:
                print(f'{file} 的预测结果与ground truth长度不一致')
    elif args.task == 'mner' or args.task == 'mner_mi':
        with open(test_path, 'r') as f:
            ground_truth_data = json.load(f)
        result_list = []
        for file in result_files:
            with open(os.path.join(result_floder, file), 'r') as f:
                results = [json.loads(line) for line in f]
            pred_str_list = [result["response"] for result in results]
            if args.cot:
                pred_str_list_temp = [extract_special_label_info(pred_str, tag='<answer>')[0] for pred_str in pred_str_list]
            else:
                pred_str_list_temp = pred_str_list
            if len(ground_truth_data) == len(pred_str_list_temp):
                res = {
                    'label_num': 0,  
                    'pred_num': 0,
                    'mner_right': 0,
                    'mner_precision': 0,
                    'mner_recall':0,
                    'mner_f1':0,
                }
                pred_list = [extract_tuple(item) for item in pred_str_list_temp] 
                label_list = [extract_tuple(item['labels']) for item in ground_truth_data]
                for i in range(len(pred_list)):
                    if pred_list[i] != []:
                        res["pred_num"] += len(pred_list[i])
                    else:
                        res["pred_num"] += 1
                    if label_list[i] != []:
                        res["label_num"] += len(label_list[i])
                    else:
                        res["label_num"] += 1
                    if pred_list[i] == [] and label_list[i] == []:
                        res["mner_right"] += 1
                for i in range(len(pred_list)):
                    for pred_turple in pred_list[i]:
                        for label_turple in label_list[i]:
                            if pred_turple[0].lower() == label_turple[0].lower() and pred_turple[1].lower() == label_turple[1].lower():
                                res["mner_right"] += 1
                                break    
                            # else:
                            #     print(pred_turple[0], label_turple[0])
                mner_precision = res["mner_right"] / res["pred_num"] if res["pred_num"] > 0 else 0
                mner_recall = res["mner_right"] / res["label_num"] if res["label_num"] > 0 else 0
                mner_f1 = 2 * mner_precision * mner_recall / (mner_precision + mner_recall) if (mner_precision + mner_recall) > 0 else 0
                result = {
                    'checkpoint': file,
                    'mner_precision': round(mner_precision * 100, 2),
                    'mner_recall': round(mner_recall * 100, 2),
                    'mner_f1': round(mner_f1 * 100, 2)
                }
                # print(result)
                # print('========'*10)
                result_list.append(result)
            else:
                print(f'{file} 的预测结果与ground truth长度不一致')
            
    elif args.task == 'grefcoco':
        image_floder = '/mnt/home/user26/data/VQA/coco2014/train2014'
        with open(test_path, 'r') as f:
            ground_truth_data = json.load(f)
        result_list = []
        for file in result_files:
            with open(os.path.join(result_floder, file), 'r') as f:
                results = [json.loads(line) for line in f]
            pred_str_list = [result["response"] for result in results]
            if args.cot:
                pred_str_list_temp = [extract_special_label_info(pred_str, tag='<answer>')[0] for pred_str in pred_str_list]
            else:
                pred_str_list_temp = pred_str_list
            if len(ground_truth_data) == len(pred_str_list_temp):
                res = {
                    'bbox_error': 0,  
                    'none_error': 0,
                    'bbox_right': 0,
                    'none_right': 0,
                    'N-acc':0,
                    'bbox-acc':0,
                    'acc':0,
                }
                for item, pred_str in zip(ground_truth_data, pred_str_list_temp):
                    if item['labels'] != "":
                        h = item['height']
                        w = item['width']
                        # print(item['labels'])
                        bbox_label = extract_bbox_coordinates(item['labels'])
                        if pred_str != "None":
                            try:
                                bbox_pred = extract_bbox_coordinates(pred_str)
                            except:
                                res['bbox_error'] += 1
                                continue
                            if args.normalized:
                                trans_bbox = normalize_coordinates(bbox_label, h, w)
                            elif args.normalized_1:
                                trans_bbox = normalize_coordinates(bbox_label, h, w)
                                trans_bbox = [trans_bbox[0] / 1000, trans_bbox[1] / 1000, trans_bbox[2] / 1000, trans_bbox[3] / 1000]
                            else:
                                trans_bbox = convert_to_qwen25vl_format(bbox_label, h, w, 28, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
                            if iou(trans_bbox, bbox_pred) >= 0.5:
                                res['bbox_right'] += 1
                            else:
                                res['bbox_error'] += 1
                        else:
                            res['bbox_error'] += 1
                    else:
                        bbox_label = 'None'
                        if bbox_label == pred_str:
                            res['none_right'] += 1
                        else:
                            res['none_error'] += 1
                res['N-acc'] = res['none_right'] / (res['none_right'] + res['none_error'])
                res['bbox-acc'] = res['bbox_right'] / (res['bbox_right'] + res['bbox_error'])
                res['acc'] = (res['bbox_right'] + res['none_right']) / (res['bbox_right'] + res['bbox_error'] + res['none_right'] + res['none_error'])
                result = {
                    'checkpoint': file,
                    'N-acc': round(res['N-acc'] * 100, 2),
                    'bbox-acc': round(res['bbox-acc'] * 100, 2),
                    'acc': round(res['acc'] * 100, 2)
                }
                result_list.append(result)
            else:
                print(f'{file} 的预测结果与ground truth长度不一致')
        

    df = pd.DataFrame(result_list)
    df.to_excel(result_floder + '/evaluation_results.xlsx', index=False)
    