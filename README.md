# MotIF: Motion Instruction Fine-tuning
<p align="center">
  <img width="95.0%" src="docs/images/MotIF_all_motions_short.gif">
</p>

This repository contains the official release of data collection code, model training, and datasets for the paper "MotIF: Motion Instruction Fine-tuning". 

The released dataset contains 1K trajectories with a variety of tasks (13 categories) and various feasible motions per task.

Website: https://motif-1k.github.io

Paper: https://arxiv.org/abs/2409.10683

# Table of Contents
- [MotIF: Motion Instruction Fine-tuning](#motif-motion-instruction-fine-tuning)
- [Table of Contents](#table-of-contents)
- [Dependencies](#dependencies)
- [Data Collection](#data-collection)
  - [Setup Camera Devices](#setup-camera-devices)
  - [Collect Human Demonstration](#collect-human-demonstration)
  - [Collect Robot Demonstration](#collect-robot-demonstration)
  - [Visual Motion Representations](#visual-motion-representations)
  - [Create Data Configuration Files](#create-data-configuration-files)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [LoRA Fine-tuning](#lora-fine-tuning)
  - [Evaluation (optionally w/ logits)](#evaluation-optionally-w-logits)
  - [SLURM](#slurm)
- [Run your own Gradio Server with Web UI](#run-your-own-gradio-server-with-web-ui)
- [Troubleshooting](#troubleshooting)


# Dependencies
```
# create conda env
conda create -n motif python=3.10

# install cudatoolkit and torch packages
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# install llava and additional packages
cd LLaVA && pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

# Data Collection
## Setup Camera Devices
We use two realsense D435 cameras to collect multiview (topdown and sideview) RGBD image observations. Set `'DEVICE_ID_topdown'` and `'DEVICE_ID_sideview'` as the strings of device ids. For instance, if the device id of the sideview camera is `123457689012`, set as `config_2.enable_device('123457689012')` in `data_collection_scripts/collect_human_demo.py` or `data_collection_scripts/collect_stretch_demo.py`

## Collect Human Demonstration
Press `s` to restart the episode or `e` to end it. If you don't press `s`, the system will automatically record the trajectory after the previous episode. To quit, just press `q` at any time.
```
# 2 cameras: topdown + side
python data_collection_scripts/collect_human_demo.py --save

# topdown only
python data_collection_scripts/collect_human_demo.py --save --view topdown

# side only
python data_collection_scripts/collect_human_demo.py --save --view side
```

## Collect Robot Demonstration
We teleoperate Hello Robot Stretch 2 (software upgraded to Stretch 3 version) to collect robot demonstrations. It's easy to set your own robot for data collection - just chance few lines for collecting joint states! If you want to collect joint states synchronously to RGBD image observations, the code should run *on* the robot's onboard computer. Otherwise, we recommend running the data collection code in a separate computer to avoid lag on the cameras!
```
# 2 cameras: topdown + side
python data_collection_scripts/collect_stretch_demo.py --save

# topdown only
python data_collection_scripts/collect_stretch_demo.py --save --view topdown

# side only
python data_collection_scripts/collect_stretch_demo.py --save --view side
```

## Visual Motion Representations
We provide three options: (1) single keypoint (e.g., robot end-effector or human hand) tracking, (2) optical flow, and (3) NxN storyboard to construct visual motion representations.
```

```

## Create Data Configuration Files
```

```


# Model Training and Evaluation
## LoRA Fine-tuning
For training in a local machine, use the following scripts.
```
cd LLaVA

# training only on human data
bash ./scripts/v1_5/finetune_task_lora_human_motion.sh

# training only on robot data
bash ./scripts/v1_5/finetune_task_lora_robot_motion.sh

# training on both human and robot data
bash ./scripts/v1_5/finetune_task_lora_cotraining.sh
```

## Evaluation (optionally w/ logits)
For model evaluation, run the following script. You can replace `llava.eval.model_vqa` with `llava.eval.model_vqa_probs` to output logits.
```
cd LLaVA
bash ./scripts/v1_5/eval.sh
```

In the script, you should set paths to the model checkpoint, questions, and answers. Outputs will be saved at `--answers-file` location. To run inference on a specific GPU when you have multiple GPUs, set `CUDA_VISIBLE_DEVICES` as the device id.
```
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa \
    --model-path ../path/to/your/checkpoint \
    --model-base liuhaotian/llava-v1.5-7b \
    --question-file ../data/eval/stretch_motion/stretch-motion-val_129_questions.jsonl \
    --image-folder ../ \
    --answers-file ../data/eval/answers/stretch-motion-val_129_questions.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
```

## SLURM
We also provide SLURM scripts for training and evaluating our model. Make sure to replace `<absolute path to your workspace>` with a proper path. For training, we recommend using at least 2x A6000 GPUs or a single A100 GPU. For inference, most NVIDIA GPUs would be compatible - we've tested with a single RTX 3090, A5000, A6000, and A100.

```
cd LLaVA

# train
sbatch ./scripts/slurm_train.sh

# eval
sbatch ./scripts/slurm_eval.sh
```

# Run your own Gradio Server with Web UI
After training, you can run a gradia server on your local machine to test your model with web UI. If you plan to launch multiple model workers to compare between different checkpoints, you only need to launch the controller and the web server ONCE. Please follow [LLaVA gradio demo instructions](https://github.com/haotian-liu/LLaVA#demo) for detailed instructions.

```
# Launch a controller
python -m llava.serve.controller --host 0.0.0.0 --port 10000

# Launch a gradio web server
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

# Troubleshooting
- import library error
  If you face errors while importing libraries, they might often be due to path configurations. One direct solution is to add `<absolute path to the workspace>/LLaVA` to `sys.path`, before loading any packages under `llava`.
  ```
  import <normal packages>
  ...
  import sys
  sys.path.append("<absolute path to the workspace>/LLaVA")
  ...
  import llava.<llava dependent packages>
  from llava.<path> import <packages>
  ```