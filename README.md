# MotIF: Motion Instruction Fine-tuning
<p align="center">
  <img width="95.0%" src="docs/images/MotIF_all_motions_short.gif">
</p>

This repository contains the official release of data collection code, model training, and datasets for the paper "MotIF: Motion Instruction Fine-tuning". 

The released dataset contains 1K trajectories with a variety of tasks (13 categories) and various feasible motions per task.

Website: https://motif-1k.github.io

Paper: https://arxiv.org/abs/2409.10683

Feel free to reach out to [myhwang@mit.edu](myhwang@mit.edu) for further questions/collaborations!

# Table of Contents
- [MotIF: Motion Instruction Fine-tuning](#motif-motion-instruction-fine-tuning)
- [Table of Contents](#table-of-contents)
- [Dependencies](#dependencies)
- [MotIF-1K Dataset](#motif-1k-dataset)
  - [Visual Motion Representations](#visual-motion-representations)
  - [Data Structure](#data-structure)
  - [Metadata (EEF trajectories, language annotations, image and video paths, ...)](#metadata-eef-trajectories-language-annotations-image-and-video-paths-)
- [Custom Data Collection](#custom-data-collection)
  - [Setup Camera Devices](#setup-camera-devices)
  - [Collect Human Demonstration](#collect-human-demonstration)
  - [Collect Robot Demonstration](#collect-robot-demonstration)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [LoRA Fine-tuning](#lora-fine-tuning)
  - [Evaluation (optionally w/ logits)](#evaluation-optionally-w-logits)
  - [Model Checkpoints](#model-checkpoints)
    - [Co-trained Models](#co-trained-models)
    - [Models trained only on stretch data](#models-trained-only-on-stretch-data)
    - [Model trained only on human data](#model-trained-only-on-human-data)
    - [Models with Different Visual Motion Representations](#models-with-different-visual-motion-representations)
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

# MotIF-1K Dataset
## Visual Motion Representations
We provide three options: (1) single keypoint (e.g., robot end-effector or human hand) tracking, (2) optical flow, and (3) N-keyframe storyboard to construct visual motion representations. Preprocessed visual motion representations are available in the MotIF-1K dataset.

## Data Structure
Download `MotIF.zip` (1.4GB) and unzip the folder under `./data`.
```
mkdir data && cd data
wget https://storage.googleapis.com/motif-1k/MotIF.zip && unzip MotIF.zip .
```

`./data/MotIF` should contain three subfolders: `annotations`, `human_motion`, and `stretch_motion`. See the following data structure for details.
```
MotIF
├── annotations
│   ├── human_motion_data_info.json
│   ├── stretch_motion_data_info.json
│   ├── cotrain
│   ├── human_motion
│   └── stretch_motion
├── human_motion
│   ├── last_frame_raw
│   ├── last_frame_trajviz
│   ├── opticalflow
│   ├── storyboard_key16
│   ├── storyboard_key16_trajviz
│   ├── storyboard_key2
│   ├── storyboard_key2_trajviz
│   ├── storyboard_key4
│   ├── storyboard_key4_trajviz
│   ├── storyboard_key9
│   ├── storyboard_key9_trajviz
│   ├── videos_raw
│   └── videos_trajviz
└── stretch_motion
    ├── last_frame_raw
    ├── last_frame_trajviz
    ├── opticalflow
    ├── storyboard_key16
    ├── storyboard_key16_trajviz
    ├── storyboard_key2
    ├── storyboard_key2_trajviz
    ├── storyboard_key4
    ├── storyboard_key4_trajviz
    ├── storyboard_key9
    ├── storyboard_key9_trajviz
    ├── videos_raw
    └── videos_trajviz
```

## Metadata (EEF trajectories, language annotations, image and video paths, ...)
For human and robot demonstrations in MotIF-1K, we provide `data_info.json` that includes trajectory index, number of timesteps, end-effector (EEF) trajectory in the image, image and video paths, task instruction, and motion description for each episode. For instance, `data/MotIF/stretch_motion_data_info.json` contains a list of dictionaries as follows:
```
[
    {
        "traj_idx": 0,
        "num_steps": 38,
        "trajectory": [
            [210, 123], [211, 122], [211, 122], [211, 122], [211, 123], [209, 126], [209, 127], [207, 125], [231, 115], [268, 110], [313, 100], [321, 84], [317, 85], [275, 86], [224, 84], [197, 87], [196, 102], [194, 112], [227, 104], [273, 101], [293, 99], [293, 96], [277, 93], [229, 89], [206, 86], [205, 92], [235, 99], [299, 80], [358, 76], [357, 78], [353, 82], [312, 92], [266, 80], [243, 76], [238, 72], [235, 73], [235, 74], [235, 88]
        ],
        "last_frame_path": "../data/MotIF/stretch_motion/last_frame_trajviz/traj0.jpg",
        "video_path": "../data/MotIF/stretch_motion/videos_trajviz/traj0.mp4",
        "task_instruction": "shake the boba",
        "motion_description": "move to the right and to the left, repeating this sequence 3 times"
    },
]
```

*Note*: A subset of our robot demonstrations contain agent joint states and end effector trajectories in real 3D space. Please reach out to the first author to get access to this data.

# Custom Data Collection
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

The default setup is to use MotIF motion representation (single point tracking). To train models other available motion representations (raw image, optical flow, and storyboard), change `--data_path` accordingly. For instance, you can replace `../data/MotIF/cotrain/human_full_653_robot_train_100.json` with `../data/MotIF/cotrain/human_full_653_robot_train_100_storyboard_key9.json` to use storyboard representation with 9 keyframes per storyboard.

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

## Model Checkpoints
We provide model checkpoints trained on various combinations of human and robot demonstrations. We also provide checkpoints trained with different visual motion representations. To download all checkpoints (4.8GB), use the following command.
```
wget https://storage.googleapis.com/motif-1k/MotIF-checkpoints.zip
```

To download specific checkpoints, see the following tables. Each checkpoint zip file is around 515MB. You can use wget to download any of these checkpoints!
### Co-trained Models
| (# Human Demos, # Stretch Demos) | (653, 100) | (653, 50) | (653, 20) |
|----------------|-----|----|----|
| Checkpoint     | [cotrain_human_full_robot_train_100](https://storage.googleapis.com/motif-1k/MotIF-checkpoints/cotrain-human_full_robot_train_100.zip)   | [cotrain_human_full_robot_train_50](https://storage.googleapis.com/motif-1k/MotIF-checkpoints/cotrain-human_full_robot_train_50.zip)  | [cotrain_human_full_robot_train_20](https://storage.googleapis.com/motif-1k/MotIF-checkpoints/cotrain-human_full_robot_train_20.zip)  |

### Models trained only on stretch data
| # Stretch Demos | 100 | 50 | 20 |
|----------------|-----|----|----|
| Checkpoint     | [robot_train_100](https://storage.googleapis.com/motif-1k/MotIF-checkpoints/stretch_train_100.zip)   | [robot_train_50](https://storage.googleapis.com/motif-1k/MotIF-checkpoints/stretch_train_50.zip)  | [robot_train_20](https://storage.googleapis.com/motif-1k/MotIF-checkpoints/stretch_train_20.zip)  |

### Model trained only on human data
| # Human Demos | 653 |
|----------------|-----|
| Checkpoint     | [human_full](https://storage.googleapis.com/motif-1k/MotIF-checkpoints/human_full.zip)  |

### Models with Different Visual Motion Representations
Models are co-trained on a dataset with full human data (653 trajectories) and 100 robot demonstrations.
| Visual Motion Representation | Raw | Optical Flow | 2x2 Storyboard |
|----------------|-----|----|----|
| Checkpoint     | [cotrain_human_full_robot_train_100_raw](https://storage.googleapis.com/motif-1k/MotIF-checkpoints/cotrain-human_full_robot_train_100_raw.zip)   | [cotrain_human_full_robot_train_100_opticalflow]()  | [cotrain_human_full_robot_train_100_storyboard_key4](https://storage.googleapis.com/motif-1k/MotIF-checkpoints/cotrain-human_full_robot_train_100_storyboard_key4.zip)  |

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
- deepspeed port issue
  If you run multiple scripts in a single machine (either local computer or cluster), be sure to set `--master_port` differently! The port number is set to `60000` in default in our code.