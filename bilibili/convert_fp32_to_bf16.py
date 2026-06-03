from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

origin_path = "/workspace/models/Qwen2.5-7B-Instruct-1M-step1-v12-global_step_2500"
# 加载模型
model = AutoModelForCausalLM.from_pretrained(origin_path, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(origin_path, trust_remote_code=True)

# 将模型转换为 bfloat16
model = model.to(torch.bfloat16)

# 保存模型
save_path = "/workspace/models/v12-global_step_2500-bf16"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Model converted and saved.")
