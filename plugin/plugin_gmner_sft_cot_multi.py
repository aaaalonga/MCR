import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

import json 
import sys
sys.path.append('/mnt/home/user26/MCR')
from utils.utils import *
from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()

class GMNEREntityCountORM(ORM):
    def __call__(self, completions, solution, task, **kwargs) -> List[float]: 
        rewards = []
        for content, label, t in zip(completions, solution, task):
            if t == 'gmner':
                reward = 1.0
                answer = extract_special_label_info(content, '<answer>')[0]
                pred_triple_list = extract_triple(content)
                true_triple_list = extract_triple(label)
                entity_cou = len(true_triple_list)
                pred_cou = len(pred_triple_list)
                if entity_cou ==0 and pred_cou == 0:
                    reward = 1.0
                diff_count = abs(entity_cou - pred_cou)
                if entity_cou == 0 or pred_cou == 0:
                    if diff_count > 0:
                        reward = 0.0
                # 改成百分比，比如label中n有个实体，pred中有m个实体
                if pred_cou > entity_cou:
                    if entity_cou > 1 and entity_cou <= 2:
                        reward = reward - diff_count * 0.4
                    elif entity_cou >= 3 and entity_cou <= 4:
                        reward = reward - diff_count * 0.2
                    elif entity_cou >= 5:
                        reward = reward - diff_count * 0.1
                elif pred_cou < entity_cou:
                    if pred_cou >= 0 and pred_cou <= 2:
                        reward = reward - diff_count * 0.5
                    elif pred_cou >= 3 and pred_cou <= 4:
                        reward = reward - diff_count * 0.3
                    elif pred_cou >= 5:
                        reward = reward - diff_count * 0.2
                if reward < 0.0:
                    reward = 0.0
                # record_dict = {}
                # record_dict['entity_count_reward'] = reward
                # record_dict['content'] = content
                # record_dict['label'] = label
                # with open(record_entity_num_path, 'a') as f:
                #     f.write(json.dumps(record_dict) + '\n')
                rewards.append(reward)
            else:
                rewards.append(None)
        return rewards
orms['gmner_entity_count_reward'] = GMNEREntityCountORM

class GMNERORM(ORM):  
    def __call__(self, completions, solution, task, **kwargs) -> List[float]:
        rewards = []
        for content, label, t in zip(completions, solution, task):
            if t == 'gmner':
                # record_dict = {}
                answer = extract_special_label_info(content, '<answer>')[0]
                pred_triple_list = extract_triple(answer)
                true_triple_list = extract_triple(label)
                pred_count = len(pred_triple_list)
                true_count = len(true_triple_list)
                if pred_count == 0 and true_count == 0:
                    reward = 1.0
                    # record_entity_score = 0.5
                    # record_type_score = 0.1
                    # record_iou_score = 0.12
                    # record_entail_score = 0.28
                elif pred_count == 0 and true_count != 0:
                    reward = 0.0
                    # record_entity_score = 0
                    # record_type_score = 0
                    # record_iou_score = 0
                    # record_entail_score = 0
                elif pred_count != 0 and true_count == 0:
                    reward = 0.0
                    # record_entity_score = 0
                    # record_type_score = 0
                    # record_iou_score = 0
                    # record_entail_score = 0
                else:
                    each_entity_reward = 1.0 / true_count
                    entity_score = each_entity_reward * 0.5
                    ground_score = each_entity_reward * 0.4
                    iou_score = ground_score * 0.3
                    entail_score = ground_score * 0.7
                    type_score = each_entity_reward * 0.1
                    recall_error_penalty = 0.4
                    pre_error_penalty = 0.4
                    true_entities = []
                    pred_entities = []
                    strict_true_entities = []
                    strict_pred_entities = []
                    reward = 0.0
                    # record_entity_score = 0
                    # record_type_score = 0
                    # record_iou_score = 0
                    # record_entail_score = 0
                    for true_triple in true_triple_list:
                        true_entity = true_triple[0].lower()
                        candidate_scores = []
                        candidate_entities_index = []
                        for i, pred_triple in enumerate(pred_triple_list):
                            pred_entity = pred_triple[0].lower()
                            if pred_entity not in pred_entities:
                                if pred_entity == true_entity:
                                    pred_entities.append(pred_entity)
                                    true_entities.append(true_entity)
                                    reward += entity_score
                                    # record_entity_score += entity_score
                                    iou_num = iou(pred_triple[2], true_triple[2])
                                    if pred_triple[2] is not None and true_triple[2] is not None:
                                        if iou_num >= 0.3:
                                            reward += iou_score * iou_num + entail_score
                                            # record_iou_score += iou_score * iou_num
                                            # record_entail_score += entail_score
                                        else:
                                            reward += entail_score
                                            # record_entail_score += entail_score
                                    elif pred_triple[2] is None and true_triple[2] is None:
                                        reward += ground_score
                                        # record_iou_score += iou_score
                                        # record_entail_score += entail_score
                                    if pred_triple[1] == true_triple[1]:
                                        reward += type_score
                                        # record_type_score += type_score
                                    if iou_num >= 0.5 and pred_triple[1] == true_triple[1]:
                                        strict_pred_entities.append(pred_entity)
                                        strict_true_entities.append(true_entity)
                                    candidate_scores = []
                                    candidate_entities_index = []
                                    break
                                else:
                                    t_f1 = token_f1(pred_entity, true_entity)
                                    if t_f1 > 0:
                                        candidate_entities_index.append(i)
                                        candidate_scores.append(t_f1)
                        if len(candidate_scores) > 0:
                            max_score = max(candidate_scores)
                            max_index = candidate_scores.index(max_score)
                            index = candidate_entities_index[max_index]
                            pred_triple = pred_triple_list[index]
                            pred_entity = pred_triple[0].lower()
                            pred_entities.append(pred_entity)
                            true_entities.append(true_entity)
                            t_f1 = token_f1(pred_entity, true_entity)
                            reward += entity_score
                            # record_entity_score += entity_score * t_f1
                            iou_num = iou(pred_triple[2], true_triple[2])
                            if pred_triple[2] is not None and true_triple[2] is not None:
                                if iou_num >= 0.3:
                                    reward += iou_score * iou_num + entail_score
                                    # record_iou_score += iou_score * iou_num * t_f1 
                                    # record_entail_score += entail_score * t_f1 
                                else:
                                    reward += entail_score
                                    # record_entail_score += entail_score * t_f1
                            elif pred_triple[2] is None and true_triple[2] is None:
                                reward += ground_score
                                # record_iou_score += iou_score * t_f1
                                # record_entail_score += entail_score * t_f1
                            if pred_triple[1] == true_triple[1]:
                                reward += type_score  
                                # record_type_score += type_score * t_f1
                            reward = reward * t_f1 
                    # 多召回的惩罚
                    recall_error_len = pred_count - len(strict_pred_entities)
                    if recall_error_len > 0:
                        recall_penalty = recall_error_len / pred_count * recall_error_penalty
                        # record_dict['recall_penalty'] = recall_penalty
                        reward = reward - recall_penalty
                    pre_error_len = true_count - len(strict_true_entities)
                    if pre_error_len > 0:
                        pre_penalty = pre_error_len / true_count * pre_error_penalty
                        # record_dict['pre_penalty'] = pre_penalty
                        reward = reward - pre_error_len / true_count * pre_error_penalty
                    if reward < 0.0:
                        reward = 0.0
                # record_dict['entity_score'] = record_entity_score
                # record_dict['type_score'] = record_type_score
                # record_dict['iou_score'] = record_iou_score
                # record_dict['entail_score'] = record_entail_score
                # record_dict['triple_reward'] = reward
                # record_dict['content'] = content
                # record_dict['label'] = label
                # with open(record_triple_path, 'a') as f:
                #     f.write(json.dumps(record_dict) + '\n')
                rewards.append(reward)
            else:
                rewards.append(None)
        return rewards
orms['gmner'] = GMNERORM

class MNEREntityCountORM(ORM):
    def __call__(self, completions, solution, task, **kwargs) -> List[float]:
        rewards = []
        for content, label, t in zip(completions, solution, task):
            if t == 'mner' or t == 'mner-mi':
                reward = 1.0
                answer = extract_special_label_info(content, '<answer>')[0]
                try:
                    pred_turple_list = extract_tuple(answer)
                except:
                    reward = 0.0
                    rewards.append(reward)
                    continue
                true_turple_list = extract_tuple(label)
                pred_cou = len(pred_turple_list)
                true_cou = len(true_turple_list)
                if pred_cou == 0 and true_cou == 0:
                    reward = 1.0
                elif pred_cou == 0 and true_cou != 0:
                    reward = 0.0
                elif pred_cou != 0 and true_cou == 0:
                    reward = 0.0
                else:
                    diff_count = abs(pred_cou - true_cou)
                    if pred_cou > true_cou:
                        if true_cou > 1 and true_cou <= 2:
                            reward = reward - diff_count * 0.4
                        elif true_cou >= 3 and true_cou <= 4:
                            reward = reward - diff_count * 0.2
                        elif true_cou >= 5:
                            reward = reward - diff_count * 0.1
                    elif pred_cou < true_cou:
                        if pred_cou >= 0 and pred_cou <= 2:
                            reward = reward - diff_count * 0.5
                        elif pred_cou >= 3 and pred_cou <= 4:
                            reward = reward - diff_count * 0.3
                        elif pred_cou >= 5:
                            reward = reward - diff_count * 0.2
                    if reward < 0.0:
                        reward = 0.0
                rewards.append(reward)
            else:
                rewards.append(None)
        return rewards
orms['mner_entity_count_reward'] = MNEREntityCountORM

class MNERORM(ORM):
    def __call__(self, completions, solution, task, **kwargs) -> List[float]:
        rewards = []
        for content, label, t in zip(completions, solution, task):
            if t == 'mner' or t == 'mner-mi':
                answer = extract_special_label_info(content, '<answer>')[0]
                try:
                    pred_turple_list = extract_tuple(answer)
                except:
                    reward = 0.0
                    rewards.append(reward)
                    continue
                true_turple_list = extract_tuple(label)
                pred_count = len(pred_turple_list)
                true_count = len(true_turple_list)                 
                if pred_count == 0 and true_count == 0:
                    reward = 1.0
                elif pred_count == 0 and true_count != 0:
                    reward = 0.0
                elif pred_count != 0 and true_count == 0:
                    reward = 0.0
                else:
                    each_entity_reward = 1.0 / true_count
                    entity_score = each_entity_reward * 0.7
                    type_score = each_entity_reward * 0.3
                    recall_error_penalty = 0.3
                    pre_error_penalty = 0.4
                    true_entities = []
                    pred_entities = []
                    strict_true_entities = []
                    strict_pred_entities = []
                    reward = 0.0
                    for true_turple in true_turple_list:
                        true_entity = true_turple[0].lower()
                        candidate_scores = []
                        candidate_entities_index = []
                        for i, pred_turple in enumerate(pred_turple_list):
                            pred_entity = pred_turple[0].lower()
                            if pred_entity not in pred_entities:
                                if pred_entity == true_entity:
                                    pred_entities.append(pred_entity)
                                    true_entities.append(true_entity)
                                    reward += entity_score
                                    if pred_turple[1] == true_turple[1]:
                                        reward += type_score
                                        strict_pred_entities.append(pred_entity)
                                        strict_true_entities.append(true_entity)
                                        break
                                else:
                                    t_f1 = token_f1(pred_entity, true_entity)
                                    if t_f1 > 0:
                                        candidate_entities_index.append(i)
                                        candidate_scores.append(t_f1)
                        if len(candidate_scores) > 0:
                            max_score = max(candidate_scores)
                            max_index = candidate_scores.index(max_score)
                            index = candidate_entities_index[max_index]
                            pred_turple = pred_turple_list[index]
                            pred_entity = pred_turple[0].lower()
                            pred_entities.append(pred_entity)
                            true_entities.append(true_entity)
                            t_f1 = token_f1(pred_entity, true_entity)
                            reward += entity_score
                            if pred_turple[1] == true_turple[1]:
                                reward += type_score  
                            reward = reward * t_f1 
                    # 多召回的惩罚
                    recall_error_len = pred_count - len(strict_pred_entities)
                    if recall_error_len > 0:
                        reward = reward - recall_error_len / pred_count * recall_error_penalty
                    pre_error_len = true_count - len(strict_true_entities)
                    if pre_error_len > 0:
                        reward = reward - pre_error_len / true_count * pre_error_penalty
                    if reward < 0.0:
                        reward = 0.0  
                rewards.append(reward)    
            else:
                rewards.append(None)
        return rewards
orms['mner'] = MNERORM


class GREFCOCOORM(ORM):
    def __call__(self, completions, solution, task, **kwargs) -> List[float]:
        rewards = []
        for content, label, t in zip(completions, solution, task):
            if t == 'grefcoco':
                answer = extract_special_label_info(content, '<answer>')[0]
                if answer == '':
                    answer = 'None'
                if answer == 'None' and label == 'None':
                    reward = 1.0
                elif answer != 'None' and label != 'None':
                    entail_score = 0.6
                    bbox_score = 0.4
                    reward = 0.0
                    try:
                        answer_bbox = extract_bbox_coordinates(answer)
                    except:
                        reward = 0.0
                        rewards.append(reward)
                        continue
                    label_bbox = extract_bbox_coordinates(label)
                    iou_score = iou(answer_bbox, label_bbox)
                    if iou_score >= 0.3:
                        reward += entail_score + iou_score * bbox_score
                    else:
                        reward += entail_score
                else:
                    reward = 0.0
                rewards.append(reward)
            else:
                rewards.append(None)
        return rewards
orms['grefcoco'] = GREFCOCOORM