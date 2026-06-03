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

# def make_prefix(dp, template_type):
#     quiz = dp['quiz']
#     if template_type == 'base':
#         prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
#     elif template_type == 'qwen-instruct':
#         prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
#     return prefix

checkpoint = "/workspace/models/Qwen2.5-7B-Instruct-1M/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

def process(text):
        prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
"""
        prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
This is the problem:
{{prompt}}
"""
        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=text)
        prompt_template = Template(prompt_template_jinja)
        if tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = tokenizer.decode([tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)
        return prompt

if __name__ == '__main__':
    
    file_path = "/workspace/data/Open-Reasoner/orz_math_57k_collected.json"
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

    train_dataset.to_parquet('/workspace/data/Open-Reasoner/train.parquet')
    test_dataset.to_parquet('/workspace/data/Open-Reasoner/test.parquet')