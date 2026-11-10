#!/bin/bash

#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH --job-name=sample

module load cuda/12.1
export NCCL_P2P_LEVEL=NVL
MIN_PIXELS=78400
MAX_PIXELS=78400

MAX_PIXELS=${MAX_PIXELS} \
MIN_PIXELS=${MIN_PIXELS} \
MASTER_PORT=28300 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sample \
    --model /mnt/home/user26/MCR/output/Qwen2.5VL/multi_task_multi_style/v4-20251101-042635/chechpoint-xxxxx-merged \
    --model_type qwen2_5_vl \
    --sampler_engine vllm \
    --num_return_sequences 8 \
    --num_sampling_per_gpu_batch_size 8 \
    --custom_register_path /mnt/home/user26/MCR/custom/dataset_grpo_multi_type_sample.py \
    --dataset gmner_grpo_data_sample \
    --dataset_shuffle false \
    --output_dir /mnt/home/user26/MCR/data/sample/Qwen2.5VL/multi_task_multi_style/grpo \
    --output_file grpo.jsonl \
    --override_exist_file True \
    --top_k 200 \
    --top_p 0.95 \
    --temperature 1.5 \
    --dataset_num_proc 4 \
    --engine_kwargs '{"tensor_parallel_size":4,"max_model_len":4096,"gpu_memory_utilization":0.8}' \
    --max_length 2048 