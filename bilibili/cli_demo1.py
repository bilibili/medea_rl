import os
import torch
import platform
import signal
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import readline
import json

# checkpoint = "/workspace/bvac-open-r1/output/asir-r1-zero-v7-Qwen2.5-Math-7B"
# checkpoint = "/workspace/bvac-open-r1/models/llama3_8b_continue_pretrain_stable500B_cosinedecay400B_task_2039267_iter_0208000"
checkpoint = "/workspace/verl/checkpoints/GRPO_logic_KK/Qwen2.5-7B-Instruct-1M-step1-v12/actor/global_step_2810/"


if torch.cuda.is_available():
   print("Use GPU")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
print(tokenizer.special_tokens_map)
model = AutoModelForCausalLM.from_pretrained(checkpoint, 
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                            #  attn_implementation="flash_attention_2",
                                            # quantization_config=quantization_config,
                                             trust_remote_code=True)
# model = PeftModel.from_pretrained(model, adapter_path)
model.config.pad_token_id = tokenizer.pad_token_id
print(model.config)
model = model.eval()
print('model loaded', checkpoint)

# SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
SYSTEM_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
"""


def main():
    global stop_stream
    print("输入stop 终止程序")
    top_k = 5
    top_p = 0.8
    temperature = 0.3
    repetition_penalty = 1.1
    max_new = 128
    system_message = "你是由哔哩哔哩自主研发的大语言模型，名为“Index”。你能够根据用户传入的信息，帮助用户完成指定的任务，并生成恰当的、符合要求的回复。"
    print_all = False
    while True:
        #query = input("\nUser:")
        #while True:
        #    extra_q = input("")
        #    if extra_q.strip() == "go":
        #        break
        #    query += '\n'+extra_q
        query = input("\nUser:")
        if len(query) == 0:
            continue
        if query.startswith('json"'):
            try:
              query = json.loads(query[4:])
            except:
              print('json format error !!')
        if query.startswith('top_p"'):
            try:
              top_p = float(json.loads(query[5:]))
              print('reset top_p:', top_p)
            except:
              print('json format error !!')
            continue
        if query.startswith('system"'):
            try:
              system_message = str(query[6:])
              system_message = system_message.strip("\"")
              print('reset system:', system_message)
            except:
              print('system format error !!')
            continue
        if query.startswith('temp"'):
            try:
              temperature = float(json.loads(query[4:]))
              print('reset temperature:', temperature)
            except:
              print('json format error !!')
            continue
        if query.startswith('repp"'):
            try:
              repetition_penalty = float(json.loads(query[4:]))
              print('reset repetition_penalty:', repetition_penalty)
            except:
              print('json format error !!')
            continue
        if query.startswith('max_new"'):
            try:
              max_new = int(json.loads(query[7:]))
              print('reset max_new:', max_new)
            except:
              print('json format error !!')
            continue
        if query.startswith('!!print all'):
            print_all = True
            print('set: print all output')
            continue
        if query.startswith('!!print new'):
            print_all = False
            print('set: only print new output')
            continue
        if query.startswith('file"'):
            try:
              pass
            except:
              print('file does not exist!!!')
            continue
        if query == 'show para':
            print('top_p:', top_p, 'temperature:', temperature, 'repetition_penalty:', repetition_penalty)
            continue

#         prompt = f"<system>:\n{SYSTEM_PROMPT}\n<user>:\n{query}\n<assistant>:\n<think>"
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n<think>"
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
# #             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": query},
#         ]
#         prompt = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs,
                        max_new_tokens=max_new,
                        # num_beams=1,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        do_sample=True
                        )
        if not print_all:
            outputs = outputs[0][len(inputs[0]):]
            response = tokenizer.decode(outputs)
            response = response.rstrip(tokenizer.eos_token)
            print('\nModel:', response)
        else:
            response = tokenizer.decode(outputs[0])
            print('\nModel:', response)

if __name__ == "__main__":
    main()
