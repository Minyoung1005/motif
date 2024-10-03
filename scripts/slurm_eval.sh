#!/bin/bash
#SBATCH --job-name=eval-motif
#SBATCH --output=./logs/eval_output_report-%j.out
#SBATCH --error=./logs/eval_output_report-%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:A5000:1

# Your job commands go here
echo "EVALUATING MotIF"
eval "$(conda shell.bash hook)"
conda activate llava
cd <absolute path to your workspace>/LLaVA

bash ./scripts/v1_5/eval/stretch_motion.sh
# bash ./scripts/v1_5/eval/human_motion.sh
# bash ./scripts/v1_5/eval/human_stretch_cotrain.sh
