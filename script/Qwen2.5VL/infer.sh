#!/bin/bash

#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH --job-name=qwen2.5_test
module load cuda/12.1
export NCCL_P2P_LEVEL=NVL

for checkpoint in xxxxx
do
FOLDER_PATH=/mnt/home/user26/MCR/output/Qwen2.5VL/grpo_multi/v4-20251101-042635

CUDA_VISIBLE_DEVICES=0,1 \
MASTER_PORT=28300 \
NPROC_PER_NODE=2 \
MAX_PIXELS=78400 \
MIN_PIXELS=78400 \
swift infer \
        --model ${FOLDER_PATH}/checkpoint-${checkpoint}-merged \
        --model_kwargs '{"max_pixels":78400, "min_pixels":78400}' \
        --result_path /mnt/home/user26/MCR/results/Qwen2.5VL/grpo/results_${checkpoint}.jsonl \
        --infer_backend vllm \
        --vllm_gpu_memory_utilization 0.9 \
        --vllm_max_num_seqs 16 \
        --vllm_max_model_len 8192 \
        --val_dataset gmner_cot_test_multitype \
        --custom_register_path /mnt/home/user26/MCR/custom/dataset_cot_multi_type_test.py \
        --max_new_tokens 4096

rm -rf ${FOLDER_PATH}/checkpoint-${checkpoint}-merged

done