from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


def compute_reward_metrics(batch: DataProto, reward_extra_infos_dict: Dict):
    reward_tensor = batch.batch['token_level_scores'].sum(-1)

    reward_metrics = {}
    reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()

    if reward_extra_infos_dict:
        data_sources = batch.non_tensor_batch["data_source"]
        answer_scores = batch.non_tensor_batch["answer_score"]

        errors = batch.non_tensor_batch["error"]
        reward_metrics[f"reward/errors"] = np.mean(errors)

        for key in ["answer_score", "format_score", "length_score", "r_collection", "r_rank", "think_token_count", "answer_token_count", "overlong_reward"]:
            values = batch.non_tensor_batch[key]
            reward_metrics[f"reward/{key}"] = np.mean(values)

            if key in ["think_token_count", "answer_token_count"]:
                answer_info = defaultdict(list)
                for answer_score, value in zip(answer_scores, values):
                    if float(answer_score) > 0.0:
                        answer_info[f"{key}_correctness"].append(value)
                
                for name, rewards in answer_info.items():
                    if len(rewards) < 1:
                        reward_metrics[f'reward/{name}'] = 0.0
                    else:
                        reward_metrics[f'reward/{name}'] = np.mean(rewards)

            reward_extra_info = defaultdict(list)
            for data_source, value in zip(data_sources, values):
                reward_extra_info[data_source].append(value)
  
            for data_source, rewards in reward_extra_info.items():
                reward_metrics[f'{data_source}/{key}'] = np.mean(rewards)

        for key in ["llm_equal", "if_llm_equal"]:
            values = batch.non_tensor_batch[key]
            reward_extra_info = defaultdict(list)
            for data_source, value in zip(data_sources, values):
                reward_extra_info[data_source].append(value)

            for data_source, rewards in reward_extra_info.items():
                reward_metrics[f'{data_source}/{key}'] = float(np.sum(rewards))

        if 'orz/if_llm_equal' in reward_metrics and 'orz/llm_equal' in reward_metrics:
            reward_metrics['orz/llm_equal_ratio'] = reward_metrics['orz/if_llm_equal'] / (reward_metrics['orz/llm_equal'] + 1e-7)

    return reward_metrics