#!/bin/bash

#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:0
#SBATCH --job-name=eval

# MODEL=MimoVL
MODEL=Qwen2.5VL

GMNER_PATH=/mnt/home/user26/MCR/data/gmner/test.json
GREFCOCO_A_PATH=/mnt/home/user26/MCR/data/grefcoco/testA.json
GREFCOCO_B_PATH=/mnt/home/user26/MCR/data/grefcoco/testB.json
MNER_PATH=/mnt/home/user26/MCR/data/mner/test.json
MNER_MI_PATH=/mnt/home/user26/MCR/data/mner-mi/test.json

python /mnt/home/user26/MCR/eval/eval.py \
    --result_floder /mnt/home/user26/MCR/results/Qwen2.5VL \
    --test_path ${GMNER_PATH} \
    --v grpo \
    --task gmner \
    --cot True \
    --min_pixels 78400 \
    --max_pixels 78400