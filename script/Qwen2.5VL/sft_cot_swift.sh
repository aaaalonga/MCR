#!/bin/bash

#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH --job-name=train_swift
module load cuda/12.1
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=4
# /mnt/home/user26/model/MiMo-VL-7B-RL-2508     mimo_vl
# /mnt/home/user26/model/Qwen2.5-VL-7B-Instruct    qwen2_5_vl
nproc_per_node=4
MIN_PIXELS=78400
MAX_PIXELS=78400


MAX_PIXELS=${MAX_PIXELS} \
MIN_PIXELS=${MIN_PIXELS} \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_PORT=29900 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /mnt/home/user26/model/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl \
    --train_type lora \
    --freeze_vit False \
    --freeze_aligner False \
    --custom_register_path /mnt/home/user26/MCR/custom/dataset_sft_cot_multi_type.py \
    --dataset 'gmner_sft_cot' \
              'mner-mi_sft_cot' \
              'grefcoco_sft_cot' \
              'fmner_sft_cot' \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 8 / $nproc_per_node) \
    --save_strategy epcho \
    --eval_steps 200 \
    --save_steps 500 \
    --save_total_limit 45 \
    --logging_steps 200 \
    --max_length 4096 \
    --output_dir /mnt/home/user26/MCR/output/Qwen2.5VL/multi_task_multi_style \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4