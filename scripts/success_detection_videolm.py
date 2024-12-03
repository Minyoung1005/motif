import os
import sys
sys.path.append("../")
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import base64
import requests
import json
import torch
from pathlib import Path
from videolm.videollama2 import model_init, mm_infer
from utils.parse_utils import parse_result, calculate_metrics
# import google.generativeai as genai
# GOOGLE_API_KEY=""
# genai.configure(api_key=GOOGLE_API_KEY)

from huggingface_hub import login

token = Path("../.hf_token").read_text().strip()
login(token=token)


from utils.video_utils import sample_frames
import argparse

def none_or_str(value):
    if value == 'None':
        return None
    return value

def get_args():
  parser = argparse.ArgumentParser(description='VLM-Success-Detector')  
  parser.add_argument(
      '--max_trial',
      default=5,
      type=int,
      help='maximum number of trials per query')
  
  parser.add_argument(
      '--max_episodes',
      default=10000,
      type=int,
      help='maximum number of trajectories')
  
  parser.add_argument(
      '--video',
      default=False,
      action='store_true',
      help='whether to use video or not')
  
  parser.add_argument(
      '--video-sampling-frames',
      default=8,
      type=int,
      help='number of frames to sample from the video')
  
  parser.add_argument(
      '--api',
      default='motif',
      help='which VLM to use')
  
  parser.add_argument(
      '--question',
      required=True,
      help='path to the questions jsonl file')
  
  parser.add_argument(
      '--model_path',
      required=False,
      default=None,
      help='path to the model')
  
  parser.add_argument(
      '--seed',
      required=False,
      default=None,
      type=int,
  )

  parser.add_argument(
    '--temperature',
    default=0.0,
    type=float,
    help='temperature for sampling')

  args = parser.parse_args()

  return args

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

args = get_args()

if args.seed is not None:
  set_seed(args.seed)

questions = []
with open(args.question, "r") as f:
  for line in f:
    questions.append(json.loads(line))

if args.model_path is None:
  save_data_path = "../data/eval/answers/{}_{}_{}.jsonl".format(args.question.split("/")[-1].replace(".jsonl", ""), args.api, "pretrained")
else:
  save_data_path = "../data/eval/answers/{}_{}_{}.jsonl".format(args.question.split("/")[-1].replace(".jsonl", ""), args.api, args.model_path.split("/")[-1])
if args.seed is not None:
  save_data_path = save_data_path.replace(".jsonl", "_seed{}.jsonl".format(args.seed))
if not os.path.exists(os.path.dirname(save_data_path)):
    os.makedirs(os.path.dirname(save_data_path))

traj_idx = 0
eval_data = []

if os.path.exists(save_data_path):
  with open(save_data_path, "r") as f:
      for line in f:
          eval_data.append(json.loads(line))
  start_traj_idx = len(eval_data)
else:
  start_traj_idx = 0
print("Starting from traj_idx: {}".format(start_traj_idx))
max_episodes = min(args.max_episodes, len(questions))

if start_traj_idx == max_episodes:
  print("Done")
  exit()

if args.api == 'gpt-4v' or args.api == 'gpt-4o':
  # OpenAI API Key
  openai_org = None
  api_key = None
  openai_org = None

  if api_key is None:
    raise ValueError("OpenAI API Key not found!")
  
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
  }

  if openai_org is not None:
    headers["OpenAI-Organization"] = openai_org

elif 'gemini' in args.api:
  model = genai.GenerativeModel('gemini-1.5-pro-latest')

elif args.api == 'motif':
  model, processor, tokenizer = model_init(args.model_path, token=token)
else:
  raise ValueError("Model not found!")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def discriminate(image_path_list, language_instruction):
  misc = {}
  text_prompt = language_instruction

  if args.api == 'gpt-4v':
    if args.video:
      base64_image_list = sample_frames(image_path_list[0], args.video_sampling_frames, encode=True)
    else:
      base64_image_list = [encode_image(image_path) for image_path in image_path_list]
    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": text_prompt
            },
            *[
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
            for base64_image in base64_image_list
            ]
          ]
        }
      ],
      "max_tokens": 300
    }
    sleep_time = 30

  elif args.api == 'gpt-4o':
    if args.video:
      base64_image_list = sample_frames(image_path_list[0], args.video_sampling_frames, encode=True)
    else:
      base64_image_list = [encode_image(image_path) for image_path in image_path_list]
    payload = {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": text_prompt
            },
            *[
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
            for base64_image in base64_image_list
            ]
          ]
        }
      ],
      "max_tokens": 300
    }
    sleep_time = 30

  elif 'gemini' in args.api:
    if args.video:
      video_path = image_path_list[0]
      images = sample_frames(video_path, args.video_sampling_frames)
    else:
      image = Image.open(image_path_list[0])
    sleep_time = 30
  elif 'motif' in args.api:
    modal = 'video'
    modal_path = image_path_list[0]
    instruct = text_prompt
    sleep_time = 0

  retry_flag = 0

  while True:
    if retry_flag == args.max_trial:
        print("{} Trials ... Timeout !!".format(args.max_trial))
        return None, None
    try:
      time.sleep(sleep_time)
      if args.api == "gpt-4v" or args.api == "gpt-4o":
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()['choices'][0]['message']['content'].strip()

      elif 'gemini' in args.api:
        if args.video:
          api_response = model.generate_content([text_prompt, *images], stream=True)
        else:
          api_response = model.generate_content([text_prompt, image], stream=True)
        api_response.resolve()

        response = api_response.text

      elif 'motif' in args.api:        
        if args.seed is None:
          output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
        else:
          output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=True, modal=modal, temperature=args.temperature)
        response = output

      print(response)
      break
      
    except Exception as e:
      print(e)
      print("Error! Trying again...")
      retry_flag += 1
    time.sleep(sleep_time)
  
  misc["text_prompt"] = text_prompt

  return response, misc



for traj_idx in tqdm(range(start_traj_idx, max_episodes)):
  start_time = time.time()

  print("==== EPISODE {}/{} ====".format(traj_idx, max_episodes))
  episode_question = questions[traj_idx]
  image_path_list = [os.path.join("../", episode_question["video"])]
  language_instruction = episode_question["text"]
  
  response, misc = discriminate(image_path_list, language_instruction)

  episode_data_dict = {"question_id": episode_question["question_id"], "prompt": episode_question["text"], "response": response}

  eval_data.append(episode_data_dict)

  # save data
  with open(save_data_path, "w") as f:
    for item in eval_data:
      f.write(json.dumps(item) + "\n")

  print("Time: {}".format(time.time() - start_time))
  print("Current Success Detector: {} | EVAL QUESTION DATASET: {} ".format(args.api, args.question))

full_results_dict = {"precision": [], "recall": [], "f1": [], "accuracy": [], "tp": [], "fp": [], "tn": [], "fn": [], "misc": []}
results = parse_result(save_data_path, args.question, None, print_results=False)
        
for key in full_results_dict:
    full_results_dict[key].append(results[key])

tp = np.sum(full_results_dict["tp"])
fp = np.sum(full_results_dict["fp"])
tn = np.sum(full_results_dict["tn"])
fn = np.sum(full_results_dict["fn"])

precision, recall, f1, accuracy = calculate_metrics(tp, fp, tn, fn)
print("Total")
print("TP: {} | FP: {} | TN: {} | FN: {}".format(tp, fp, tn, fn))
print("Precision: {:.2f} | Recall: {:.2f} | F1: {:.2f} | Accuracy: {:.2f}".format(precision, recall, f1, accuracy))

print("="*50)