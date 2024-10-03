#!/bin/bash
#SBATCH --job-name=train-motif
#SBATCH --output=./logs/train_output_report-%j.out
#SBATCH --error=./logs/train_output_report-%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:A100:1

echo "TRAINING MotIF"
eval "$(conda shell.bash hook)"
conda activate motif
cd <absolute path to your workspace>/LLaVA

bash ./scripts/v1_5/finetune_task_lora_cotrain.sh
# bash ./scripts/v1_5/finetune_task_lora_human_motion.sh
# bash ./scripts/v1_5/finetune_task_lora_stretch_motion.sh