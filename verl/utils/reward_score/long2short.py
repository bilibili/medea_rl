import json
import re
import torch

def find_first_occurrence(tensor: torch.Tensor, sequence: list) -> int:
    seq_len = len(sequence)
    if tensor.ndim != 1 or len(tensor) < seq_len:
        return -1

    # 将目标和输入张量对齐设备（CPU/GPU）和数据类型
    target = torch.tensor(sequence, dtype=tensor.dtype, device=tensor.device)

    # 生成滑动窗口（形状为 [n - seq_len + 1, seq_len]）
    windows = tensor.unfold(0, seq_len, 1)

    # 比较窗口与目标是否完全匹配
    matches = (windows == target).all(dim=1)

    # 找到第一个匹配的索引
    indices = torch.nonzero(matches, as_tuple=True)[0]
    return indices[0].item() if indices.numel() > 0 else -1


def cal_think_token_count(response_id, extra_info):
    if extra_info["token_upper"] == 0:
        think_token_count = 0
    else:
        mask_think_end = find_first_occurrence(response_id, [151668])  # </think>
        assert response_id[mask_think_end].item() == 151668, "string: </think> is not in response_id"
        # 处理一下\n
        if response_id[mask_think_end - 1].item() == 198:
            think_token_count = mask_think_end - 1
        else:
            think_token_count = mask_think_end
    return think_token_count


def cal_answer_token_count(response_id, extra_info):
    mask_answer_end = len(response_id)
    if extra_info["token_upper"] == 0:
        # 处理一下\n
        if response_id[mask_answer_end - 1].item() == 198:
            answer_token_count = mask_answer_end - 1
        else:
            answer_token_count = mask_answer_end
    else:
        mask_answer_start = find_first_occurrence(response_id, [151668])  # </think>
        assert response_id[mask_answer_start].item() == 151668, "string: </think> is not in response_id"

        # 处理一下\n
        buffer = 0
        if response_id[mask_answer_start + 1].item() == 198:
            buffer += 1
        
        if response_id[mask_answer_end - 1].item() == 198:
            buffer += 1
        answer_token_count = mask_answer_end - mask_answer_start - buffer
    return answer_token_count


def long2short_compute_score(extra_info, logger, think_buffer=100, answer_buffer=100):

    response_id = extra_info["response_id"]
    data_source = extra_info["data_source"]
    think_token_count = cal_think_token_count(response_id, extra_info)
    answer_token_count = cal_answer_token_count(response_id, extra_info)

    think_token_budget = int(extra_info["token_upper"])
    answer_token_origin = int(extra_info["answer_token_origin"])
    think_budget_ratio = (think_buffer + think_token_count) / (think_buffer + think_token_budget)
    answer_change_ratio = (answer_buffer + answer_token_count) / (answer_buffer + answer_token_origin)

    logger.log(f"  think_token_budget: {think_token_budget}")
    logger.log(f"  answer_token_origin: {answer_token_origin}")
    logger.log(f"  think_token_count: {think_token_count}")
    logger.log(f"  answer_token_count: {answer_token_count}")

    length_score = 0.0
    if think_token_budget == 0 or think_token_budget == 32768:
        think_length_success = True
    elif think_budget_ratio < 0.25 or think_budget_ratio > 1.1:
        length_score += -0.4
        think_length_success = False
    else:
        think_length_success = True

    if data_source in ["long2short"]:
        if answer_change_ratio > 1.5 or answer_change_ratio < 0.8:
            length_score += -0.4
            answer_length_success = False
        else:
            answer_length_success = True
    elif data_source in ["orz", "sandbox"]:
        if answer_change_ratio > 1.5:
            length_score += -0.4
            answer_length_success = False
        else:
            answer_length_success = True
    else:
        answer_length_success = True

    return length_score, think_length_success, answer_length_success, think_token_count, answer_token_count