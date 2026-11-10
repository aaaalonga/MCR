#!/bin/bash

#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --job-name=merge_lora

module load cuda/12.1
swift export \
    --adapters xxxxxxx \
    --merge_lora true