""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from jinja2 import Template
from typing import List
import random
random.seed(1234)

def process(text):
    prompt = f"""<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prompt


if __name__ == '__main__':
    
    file_path = "/workspace/dataset/ORZ/orz_math_57k_collected.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(len(data))
    
    # 将嵌套列表转换为适合 Dataset 的格式
    formatted_data = []
    for conversation in data:
        human_message = conversation[0]["value"]  # 获取 human 的消息
        assistant_message = conversation[1]["ground_truth"]["value"]  # 获取 assistant 的回复
        formatted_data.append({
            "human": human_message,
            "assistant": assistant_message
        })
    random.shuffle(formatted_data)

    data_source = 'orz'
    n = 0.95
    train_dataset = Dataset.from_list(formatted_data[:int(len(formatted_data) * n)])
    test_dataset = Dataset.from_list(formatted_data[int(len(formatted_data) * n):])
    
    def make_map_fn(split):

        def process_fn(example, idx):
            human = example["human"]
            question = process(text = human)
            
            # print(question)
            solution = example["assistant"]
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet('/workspace/dataset/ORZ/train_coldstart.parquet')
    test_dataset.to_parquet('/workspace/dataset/ORZ/test_coldstart.parquet')
