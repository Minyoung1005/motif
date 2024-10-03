#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa \
    --model-path ../path/to/your/checkpoint \
    --model-base liuhaotian/llava-v1.5-7b \
    --question-file ../data/eval/stretch_motion/stretch-motion-val_129_questions.jsonl \
    --image-folder ../ \
    --answers-file ../data/eval/answers/stretch-motion-val_129_questions.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1