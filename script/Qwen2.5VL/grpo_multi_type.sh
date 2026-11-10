#!/bin/bash

#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:8
#SBATCH --job-name=grpo
module load cuda/12.1
export OMP_NUM_THREADS=8
export NCCL_P2P_LEVEL=NVL
nproc_per_node=8
MIN_PIXELS=78400
MAX_PIXELS=78400

MAX_PIXELS=${MAX_PIXELS} \
MIN_PIXELS=${MIN_PIXELS} \
MASTER_PORT=29900 \
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/home/user26/MCR/output/Qwen2.5VL/multi_task_multi_style/v4-20251101-042635/chechpoint-xxxxx-merged \
    --model_type qwen2_5_vl \
    --external_plugins /mnt/home/user26/MCR/plugin/plugin_gmner_sft_cot_multi.py \
    --reward_funcs gmner_entity_count_reward gmner mner_entity_count_reward mner grefcoco\
    --reward_weights 0.5 1.0 0.5 1.0 1.0 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 4096 \
    --custom_register_path /mnt/home/user26/MCR/custom/dataset_grpo_multi_type.py \
    --dataset 'gmner_filtered_data' \ 'mner-mi_filtered_data' \ 'mner_filtered_data' \ 'grefcoco_filtered_data' \
    --output_dir /mnt/home/user26/MCR/output/Qwen2.5VL/grpo_multi \
    --split_dataset_ratio 0 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit True \
    --freeze_aligner True \
    --torch_dtype bfloat16 \
    --max_completion_length 2048 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2\
    --learning_rate 5e-6 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 25 \
    --save_steps 100 \
    --save_total_limit 35 \
    --logging_steps 10 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --epsilon 0.15 \
    --epsilon_high 0.25 \
    --temperature 1.5 \
    --top_k 200 \
    --top_p 0.95 \
    --log_completions true \
    --num_iterations 1 \
    --beta 0.005

# /mnt/home/user26/GMNER_DPO/merged_model/gmner_mner_mner_mi_gref14000_1e4_78400_78400_bs8_swift_rank8_alpha32_simple_instruction_cot_formal_warm0.1/checkpoint-4360
# /mnt/home/user26/GMNER_DPO/merged_model/gmner400/checkpoint-25

    # --use_vllm true \
    # --vllm_mode colocate \
    # --vllm_gpu_memory_utilization 0.4 \
    # --vllm_tensor_parallel_size 4 \
    # --vllm_max_model_len 4096 \
        # --vllm_enforce_eager \
        # --importance_sampling_level sequence \

    # --reward_funcs gmner_entity_count_reward gmner mner_entity_count_reward mner grefcoco \
    # --reward_weights 0.5 1.0 0.5 1.0 1.0 \
    #     --dynamic_sample True \
    # --max_resample_times 1 \