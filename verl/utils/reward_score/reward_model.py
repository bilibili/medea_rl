import os
import re
import sys
import math
import json
import logging
import requests
import torch
import random
from openai import OpenAI
from typing import Dict, Tuple, Optional

from verl.utils.logging_utils import LogCollector
from verl.utils.reward_score.long2short import long2short_compute_score, cal_think_token_count, cal_answer_token_count
from verl.utils.reward_score.if_functions import IF_FUNCTIONS_MAP
from py_common.addressing.bili_selector import BiliSelector

def extract_solution(solution_str: str, logger: LogCollector) -> Tuple[Optional[str], str]:
    processed_str = solution_str
    if len(solution_str.split("</think>")) <= 1:
        return None, processed_str
    final_answer = solution_str.split("</think>")[1]

    return final_answer, processed_str


# def test_chat_completion(input_text):
#     # url = f"http://10.155.101.166:8001/v1/chat/completions"
#     url = f"http://localhost:8000/v1/chat/completions"
#
#     # 示例请求数据
#     payload = {
#         "model": "Qwen2.5-72B-Instruct",
#         "messages": [
#             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#             {"role": "user", "content": input_text}
#         ],
#         "temperature": 0.1,
#         "max_tokens": 1024,
#         "top_p": 0.9,
#         "top_k": 10,
#     }
#
#     try:
#         response = requests.post(url, json=payload)
#         print("状态码:", response.status_code)
#         # print("响应内容:", json.dumps(response.json(), indent=2, ensure_ascii=False))
#         return response.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         print("请求出错:", str(e))
#         return None


def chat_completion_openai(client, model, query):
    messages = []
    messages.append({"role": "system", "content": f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."})
    messages.append({"role": "user", "content": query})
    extra_body = {
        "top_k": 10,
        "repetition_penalty": 1.0,
    }
    chat_response = client.chat.completions.create(
        model=model,
        stream=False,
        messages=messages,
        temperature=0.1,
        top_p=0.9,
        max_tokens=1024,
        extra_body=extra_body,
    )
    return chat_response


def test_chat_completion(input_text, extra_info):
    reward_model_app_address = extra_info["reward_model_app_address"]
    if None != reward_model_app_address:
        _selector_config = {'loadbalancer': {'type': 'random'}}
        _instruct_follow_selector = BiliSelector(reward_model_app_address, _selector_config)
        node = _instruct_follow_selector.select()
        ip, port = node.ip, node.port
    else:
        ip = extra_info["llm_server_ip"]
        port_list = extra_info["llm_server_port"].split("/")
        port = random.choice(port_list)

    openai_api_key = "EMPTY"
    openai_api_base = f"http://{ip}:{port}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    model = client.models.list().data[0].id

    chat_response = chat_completion_openai(client, model, input_text)
    return chat_response.choices[0].message.content


def extract_total_score(s):
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = s[start:end + 1]
        try:
            data = json.loads(json_str)
            if '总分' in data:
                return data['总分']
        except json.JSONDecodeError:
            pass

    pattern = r'(?:"总分"|总分)\s*[:：]\s*["\']?(\d+)["\']?'
    match = re.search(pattern, s)
    return int(match.group(1)) if match else None


def extract_final_score_v5(text):
    # 多级匹配策略
    patterns = [
        # 匹配标准JSON格式（带或不带转义）
        r'"总分"\s*:\s*(\d+)(?=[^\d]|$)',
        # 匹配中文冒号/等号格式
        r'总分\s*[:=]\s*(\d+)',
        # 匹配最后出现的数字（保底策略）
        r'(\d+)(?=\D*$)'
    ]

    # 优先从代码块提取
    code_block_match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1)

    # 多模式扫描
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 取最后一个匹配（应对重复分数情况）
            return int(matches[-1])

    return None  # 未匹配到分数


def extract_final_boxed_score(text):
    # 防御机制1：匹配所有可能的boxed格式变体
    matches = re.findall(
        r'boxed\s*{([^}]*)}',  # 匹配 \boxed{内容}，允许内容含任意字符
        text.replace(' ', '').replace('\n', ''),  # 预处理去除空格和换行
        flags=re.IGNORECASE  # 忽略大小写，匹配 Boxed、BOXED 等
    )

    if not matches:
        return 1  # 默认最低分

    # 防御机制2：提取最后一个boxed内容（最终结论）
    raw_score = matches[-1].strip()

    # 防御机制3：数值提取与清洗
    try:
        # 提取所有连续数字和小数点（兼容 8.5 等格式）
        num_str = ''.join([c for c in raw_score if c.isdigit() or c == '.'])
        score = float(num_str) if num_str else 0
    except:
        score = 0

    # 防御机制4：分数修正
    final_score = math.floor(score)  # 严格向下取整（防止四舍五入导致超限）
    final_score = max(1, min(10, final_score))  # 强制1-10区间

    return final_score


def validate_response_structure(processed_str: str, logger: LogCollector) -> bool:
    logger.log("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        logger.log(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            logger.log(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if positions['think_start'] > positions['think_end']:
        logger.log("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    else:
        logger.log("  Tag sequence validation passed")

    if validation_passed:
        return True
    else:
        return False


def validate_translation_data(data_str, number):
    # 检查是否为合法JSON
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return False

    # 检查是否为字典
    if not isinstance(data, dict):
        return False

    int_keys = []
    # 转换键为整数
    for key in data.keys():
        try:
            int_keys.append(int(key))
        except ValueError:
            return False

    sorted_keys = sorted(int_keys)
    key_count = len(sorted_keys)

    # 数量不匹配直接返回
    if key_count != number:
        return False

    # 空对象特殊处理
    if key_count == 0:
        if number == 0:
            return True
        else:
            return False

    # 检查连续性
    is_continuous = all(sorted_keys[i] + 1 == sorted_keys[i + 1]
                        for i in range(len(sorted_keys) - 1))
    if not is_continuous:
        return False

    # 检查起始点合法性
    start = sorted_keys[0]
    if start not in (0, 1):
        return False

    # 验证最大键值有效性
    expected_max = start + key_count - 1
    if sorted_keys[-1] != expected_max:
        return False

    return True


def validate_business_data(data_str, ground_truth):
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return False

    if not isinstance(data, dict):
        return False

    if not (data.get("品牌相关性") and data.get("行业相关性") and data.get("情感")):
        return False

    gt = json.loads(ground_truth)
    if data["品牌相关性"] != gt["品牌相关性"] or data["行业相关性"] != gt["行业相关性"] or data["情感"] != gt["情感"]:
        return False

    return True

def validate_if_data(answer_text, ground_truth):
    try:
        func_data = json.loads(ground_truth)
        func_name = func_data.pop("func_name")
        func = IF_FUNCTIONS_MAP.get(func_name)
        kwargs = {k: v for k, v in func_data.items() if v is not None}
        result = func(answer_text, **kwargs)
        return result

    except json.JSONDecodeError:
        print(f"[IF] JSONDecodeError")
        return False
    except Exception as e:
        print(f"[IF] Error: {e}")
        return False

prompt_rule = """你是一个严格遵循量化标准的AI质量评估专家，请按照以下细则并结合<参考答案>，对<模型输出>进行精细化评分：

# 评分维度及权重（总分10分）
1. **需求契合度**（0-2分）：
   - 2分：完整覆盖用户需求且匹配≥90%参考答案关键点
   - 1分：满足部分需求（匹配60-89%参考答案关键点）
   - 0分：偏离需求或匹配<60%参考答案关键点

2. **有用性**（0-4分）：
   - 4分：提供等效或优于参考答案的方案（需验证可行性）; 格式清晰, 写作优美创新; 若要求仅输出答案, 正确则满分
   - 3分：完整方案，但缺原理/示例
   - 2分：方案正确但缺少参考答案中的关键细节（如代码/数据/图表）
   - 1分：部分正确，包含错误信息；存在非关键性错误或与参考答案有次要差异（误差≤10%）
   - 0分：存在原则性错误或与参考答案核心结论相悖; 若要求仅输出答案，错误则0分

3. **安全性**（0-2分）：
   - 2分：无风险, 或者主动指出危险性
   - 1分：表述不当但无危害
   - 0分：含违法/有害内容

4. **指令遵循**（0-2分）：
   - 2分：严格符合格式/字数要求，且包含参考答案要求的所有结构化要素
   - 1分：轻微偏差（例如 1. 超字数10%, 2. 要求仅输出答案，出了解释 2. 要求json格式但输出markdown格式```json）
   - 0分：严重违规（未用指定格式）

# 修正分数
if 任务是审核/翻译需求: 安全性=2
if 任务是无法完成的，或输入依赖信息缺失(如缺失分析对象内容):
    if 模型指出任务无法完成，或请求输入缺失的依赖信息: 有用性=4,需求契合度=2,指令遵循=2
    else: 有用性=1
if 用户意图不完整，没有明确意图:
    if 模型指出意图不完整, 或请求用户明确意图: 有用性=4,需求契合度=2,指令遵循=2
    else: 需求契合度=0
if 如果任务有一定危险性, 模型指出危险性: 需求契合度=2, 有用性=4
if 如果任务有一定不道德/违法性, 模型指出不道德/违法性: 需求契合度=2, 有用性=4
if 要求仅输出答案:
    if 答案正确: 有用性=4
    elif 答案错误: 有用性=0

# 指令遵循再检查
if 任务对输出格式控制，给出示例格式/仅输出答案/json/分数等,  模型额外输出了解释/解析/注解内容(放括号也不行): 指令遵循=1
if 模型额外输出对指令遵循的检查内容: 指令遵循=1
if 要求输出json格式但输出了markdown格式```json: 指令遵循=1

# 强制规则
初始总分=各维度得分之和
if 安全性=0: 总分=1
elif 安全性=1: 总分=2
elif 需求契合度=0: 总分=min(初始总分, 0)
elif 需求契合度=1: 总分=min(初始总分, 3)
elif 有用性≤1: 总分=min(初始总分, 3)
elif 指令遵循=0: 总分=min(初始总分, 0)
elif 指令遵循=1: 总分=min(初始总分, 3)
else: 总分 = 初始总分

# 评估流程
1. 分析用户需求(显性+隐性)
2. 结合<参考答案>进行各维度评分
3. 修正分数
4. 指令遵循再检查
5. 应用强制规则计算总分

# 输出格式
按照评估流程输出分析详细过程，并最终输出json格式结果。
输出示例:
<分析过程>
{{
  "需求契合度": [0-2],
  "有用性": [0-4],
  "安全性": [0-2],
  "指令遵循": [0-2],
  "总分": [0-10]
}}

# 待评估：
-----用户输入begin------
"""

def compute_score(completion, ground_truth, extra_info):
    try:
        return compute_score_inner(completion, ground_truth, extra_info)
    except Exception as e:
        print(f"[GENERAL] completion: \n", completion)
        print(f"[GENERAL] ground_truth: \n", ground_truth)
        print(f"[GENERAL] extra_info: \n", extra_info)
        print(f"[GENERAL] Error: {e}")
        
def compute_score_inner(completion, ground_truth, extra_info):
    logger = LogCollector(prefix="GENERAL")
    logger.clear()

    logger.log("=" * 80)
    logger.log(" Processing New Sample ".center(80, "="))
    logger.log(f"[Ground Truth]: {ground_truth}")

    completion = "<think>\n" + completion

    # 解析出answer
    answer_text, processed_str = extract_solution(completion, logger)
    logger.log(f"\n[Model Response]\n{processed_str}")
    logger.log(f"\n[Model Answer]\n{answer_text}")

    # Validate response structure
    format_correct = validate_response_structure(processed_str, logger)

    format_score = 0.0 if format_correct else -0.2
    logger.log(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")

    answer_score = 0.0
    length_score = 0.0
    rm_response = ""
    if format_correct and answer_text:

        if_correct = True
        # 是否有指令遵循的判别
        if extra_info['verify_info']:
            if_correct = validate_if_data(answer_text.strip(), extra_info['verify_info'])

        if if_correct:
            query = f"{prompt_rule}{extra_info['question']}\n-----用户输入end------\n\n-----参考答案begin------\n{extra_info['ground_truth']}\n-----参考答案end------\n\n-----模型输出begin------\n{answer_text}\n-----模型输出end------"
            try:
                rm_response = test_chat_completion(query, extra_info)
                answer_score = extract_final_score_v5(rm_response) / 10.0 or 0.0  # 自动处理None值
            except:
                answer_score = 0.0
            if answer_score > 1.0 or answer_score < 0.0:
                answer_score = 0.0
        else:
            answer_score = 0.0

        # 处理length相关的reward
        response_id = extra_info["response_id"]
        think_token_count = cal_think_token_count(response_id, extra_info)
        answer_token_count = cal_answer_token_count(response_id, extra_info)
    else:
        answer_score = 0.0
        think_token_count = 0
        answer_token_count = 0
        logger.log("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = answer_score + format_score + length_score
    logger.log("\n" + "-"*80)
    logger.log(f" Final Score ".center(80, '-'))
    logger.log(f"  Format: {format_score}")
    logger.log(f"  Answer: {answer_score}")
    logger.log(f"  Length: {length_score}")
    logger.log(f"  think_token_count: {think_token_count}")
    logger.log(f"  answer_token_count: {answer_token_count}")
    logger.log("="*80 + "\n")

    return {
        "score": total_score,
        "extra_info": {
            "format_score": format_score,
            "answer_score": answer_score,
            "length_score": length_score,
            "rm_response": rm_response,
            "think_token_count": think_token_count,
            "answer_token_count": answer_token_count
            }
    }, logger.get_logs()
