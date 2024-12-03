import os
import sys
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

positive_keys = ["has been solved", "is solved", "were completed", "were made", "Answer: 1", "answer is 1","has completed"
                 "answer to the task is 1", "the answer is 1", "answer to the task being solved is 1", "is completed successfully",
                 "has been successfully completed", "has completed", "the answer to the task is 1", "indicating success",
                 "appears to be solved","Answer: \( 1 \)", "agent appears to be following", "agent is following",
                 "answer is **1**", "**1**", "**Answer:** 1", "\n1\n", "\n1", "1\n", "this as a 1", "Yes"]
negative_keys = ["has not been solved", "is not solved", "did not complete", "answer would be 0", "was not solved", "not appear to be solved",
                 "incomplete", "does not match", "Answer: 0", "answer is 0", "has not completed", "not fully solved", "not completed",
                 "is not fully solved", "answer, it would be 0", "not yet fully solved", "is not yet complete", "partially solved",
                 "is not completed", "0 (the agent", "agent's trajectory does not follow", "agent is not following",
                 "agent's path does not follow", "**Answer:** 0 ", "answer is **0**", "**0**",
                 "Answer: **0**", "The agent is **not** following", "**Answer:** 0", "the answer is: 0", "The agent's path is not ", "{0}"
                  "\n0\n", "\n0", "0\n", "No", "does not follow"
                 ]

def is_positive(response):
    if response is None:
        return False
    if response[0] == "1" or response[-1] == "1":
        return True
    if len([key for key in positive_keys if key in response]) > 0:
        return True
    return False

def is_negative(response):
    if response is None:
        return False
    if response[0] == "0" or response[-1] == "0":
        return True
    if len([key for key in negative_keys if key in response]) > 0:
        return True
    return False

def parse_result(agent_answer_path, gt_answer_path, custom_episode_indices=None, print_results=True):
    with open(gt_answer_path, "r") as f:
        gt_answers = [json.loads(line) for line in f]
    tp, fp, tn, fn, misc = 0, 0, 0, 0, 0
    results_by_instr_dict = {}
    wrong_indices = []
    wrong_cases = []
    results = []

    # read agent answers
    with open(agent_answer_path, "r") as f:
        agent_answers = [json.loads(line) for line in f]
        for ans_idx in tqdm(range(len(agent_answers))):
            ans = agent_answers[ans_idx]
            episode_id = int(ans["question_id"])
            gt_answer = gt_answers[episode_id]["answer"]
            if custom_episode_indices is not None:
                if int(gt_answers[episode_id]["image"].split("/")[-1].split("traj")[-1].split(".")[0]) not in custom_episode_indices:
                    continue
            if "response" in ans:
                agent_answer = ans["response"]
            elif "text" in ans:
                agent_answer = ans["text"]
            instr = ans["prompt"]
            if instr not in results_by_instr_dict:
                results_by_instr_dict[instr] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "misc": 0}
            if agent_answer == gt_answer:
                if agent_answer == "1":
                    tp += 1
                    results.append("tp")
                    results_by_instr_dict[instr]["tp"] += 1
                elif agent_answer == "0":
                    tn += 1
                    results.append("tn")
                    results_by_instr_dict[instr]["tn"] += 1
                elif is_positive(agent_answer):
                    tp += 1
                    results.append("tp")
                    results_by_instr_dict[instr]["tp"] += 1
                elif is_negative(agent_answer):
                    tn += 1
                    results.append("tn")
                    results_by_instr_dict[instr]["tn"] += 1
                else:
                    misc += 1
                    results.append("misc")
                    results_by_instr_dict[instr]["misc"] += 1
                    print(agent_answer)

            else:
                if agent_answer == "1":
                    fp += 1
                    results.append("fp")
                    results_by_instr_dict[instr]["fp"] += 1
                    wrong_cases.append("fp")
                elif agent_answer == "0":
                    fn += 1
                    results.append("fn")
                    results_by_instr_dict[instr]["fn"] += 1
                    wrong_cases.append("fn")
                elif is_positive(agent_answer):
                    if gt_answer == "1":
                        tp += 1
                        results.append("tp")
                        results_by_instr_dict[instr]["tp"] += 1
                    else:
                        fp += 1
                        results.append("fp")
                        results_by_instr_dict[instr]["fp"] += 1
                        wrong_cases.append("fp")
                elif is_negative(agent_answer):
                    if gt_answer == "0":
                        tn += 1
                        results.append("tn")
                        results_by_instr_dict[instr]["tn"] += 1
                    else:
                        fn += 1
                        results.append("fn")
                        results_by_instr_dict[instr]["fn"] += 1
                        wrong_cases.append("fn")
                else:
                    misc += 1
                    results.append("misc")
                    results_by_instr_dict[instr]["misc"] += 1
                    wrong_cases.append("misc")
                    print(agent_answer)
                wrong_indices.append(episode_id)

    if tp + fp == 0:
        precision = np.nan
    else:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = np.nan
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = np.nan
    if tp + tn + fp + fn > 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        accuracy = np.nan

    if print_results:
        # Total number of samples
        print("EVAL RESULTS")
        print("EVAL PERCENTAGE: {} / {}".format(len(agent_answers), len(gt_answers)))

        # print tp, fp, tn, fn, misc
        print("TP: {} | FP: {} | TN: {} | FN: {} | Misc: {}".format(tp, fp, tn, fn, misc))
        print("Precision: {:.2f} | Recall: {:.2f} | F1: {:.2f} | Accuracy: {:.2f}".format(precision, recall, f1, accuracy))

        # print wrong indices
        print("Wrong indices: ", wrong_indices)
        print("Wrong cases: ", wrong_cases)

        # print instructions for tp
        tp_instr = []
        for instr in results_by_instr_dict:
            if results_by_instr_dict[instr]["tp"] > 0:
                tp_instr.append(instr)

        print("TP instructions: ", tp_instr)
    
    # return results
    results_dict = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "misc": misc,
        "wrong_indices": wrong_indices,
        "results": results,
    }
    return results_dict

def calculate_metrics(tp, fp, tn, fn):
    if tp + fp == 0:
        precision = np.nan
    else:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = np.nan
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = np.nan
    if tp + tn + fp + fn > 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        accuracy = np.nan
    return precision, recall, f1, accuracy

