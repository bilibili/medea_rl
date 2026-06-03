# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel

from verl import DataProto
from verl.workers.reward_manager import register
from verl.utils.py_functional import NestedNamespace, str2bool


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=1800, log_print=False):
    loop = asyncio.get_running_loop()
    # print("origin completion: \n", completion)
    # print("origin reference: \n", reference)
    # print("origin task_extra_info: \n", task_extra_info)
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(evaluation_func, task, completion, reference, task_extra_info)
            ),
            timeout=timeout
        )

        # 处理结果
        score, logs = result
        if log_print and logs:
            print(f"[Evaluation Logs]\n{logs}")

        return score
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return {"score": 0.0}
    except Exception as e:
        print(f"Error processing completion: {completion} \n{reference}, Error: {e}, Exception type: {type(e)}")
        return {"score": 0.0}


# async def parallel_compute_score_async(evaluation_func,
#                                        completions,
#                                        references,
#                                        tasks,
#                                        extra_info=None,
#                                        num_processes=64,
#                                        log_print=False):
#     scores = []
#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         if extra_info is None:
#             extra_info = [None] * len(tasks)
#         # Create tasks for all rows
#         tasks_async = [
#             single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=600, log_print=log_print)
#             for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)
#         ]
#         # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
#         try:
#             results = await asyncio.gather(*tasks_async, return_exceptions=False)
#         except:
#             for pid, proc in executor._processes.items():
#                 try:
#                     proc.kill()
#                 except Exception as kill_err:
#                     print('shut down failed: ' + str(kill_err))
#             raise
#
#     # 处理结果，因为 single_compute_score 始终返回 float 类型，所以直接添加
#     for result in results:
#         # 如果结果为 None 或异常则返回默认值 0.0，否则直接添加结果
#         if result is None or isinstance(result, Exception):
#             scores.append({"score": 0.0})
#         else:
#             scores.append(result)
#     return scores


async def parallel_compute_score_async(
    evaluation_func,
    completions,
    references,
    tasks,
    extra_info=None,
    num_workers=1,   # 强制 1
):
    scores = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)

        tasks_async = [
            single_compute_score(
                evaluation_func,
                completion,
                reference,
                task,
                task_extra_info,
                executor,
                timeout=1800
            )
            for completion, reference, task, task_extra_info
            in zip(completions, references, tasks, extra_info)
        ]

        results = await asyncio.gather(*tasks_async)

    for r in results:
        scores.append(r if r is not None else {"score": 0.0})

    return scores


class EmbeddingBatcher:
    def __init__(self, tokenizer, model, device, batch_size=256):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts):
        """
        texts: List[str]
        return: Tensor [len(texts), hidden_dim]
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(**batch)
                last_hidden = outputs.last_hidden_state
                attention_mask = batch["attention_mask"]

                # === 你原来的 last_token_pool ===
                seq_len = attention_mask.sum(dim=1) - 1
                emb = last_hidden[
                    torch.arange(last_hidden.size(0), device=self.device),
                    seq_len
                ]

                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)


@register("index_pr")
class IndexPrRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key: str="data_source", \
                 reward_num_processes: int=32, max_resp_len: int=1024, grm: dict=None, overlong_buffer: dict=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

        self.reward_num_processes = 1

        # self.embedding_tokenizer = AutoTokenizer.from_pretrained(
        #     'Qwen/Qwen3-Embedding-0.6B', padding_side='left'
        # )
        # self.embedding_model = AutoModel.from_pretrained(
        #     'Qwen/Qwen3-Embedding-0.6B'
        # )

        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_device = device

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen3-Embedding-0.6B',
            padding_side='left'
        )
        self.embedding_model = AutoModel.from_pretrained(
            'Qwen/Qwen3-Embedding-0.6B'
        ).to(device)

        self.embedding_model.eval()
        for p in self.embedding_model.parameters():
            p.requires_grad = False

        print(f"[PrimeRewardManager] Embedding model loaded on {device}")

        self.embedding_batcher = EmbeddingBatcher(
            tokenizer=self.embedding_tokenizer,
            model=self.embedding_model,
            device=self.embedding_device,
            batch_size=256,  # A800 推荐起步
        )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # data.save_to_disk(f"/workspace/verl/prime_data_proto_{time.time()}.pkl")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get('extra_info', [None] * len(data_sources))

        assert len(response_str) == len(ground_truth) == len(data_sources) == len(extra_info)
        print("response_str length: \n", len(response_str))

        extra_info_process = []
        for info, response in zip(extra_info, response_str):
            if isinstance(info, dict):
                # 由于代码中的repeat逻辑，这这里不能直接修改info，需要copy一份
                copied_info = copy.deepcopy(info)
                copied_info["response_id"] = torch.tensor(self.tokenizer.encode(response, add_special_tokens=False))
                # copied_info["use_llm_verify"] = self.use_llm_verify
                # copied_info["llm_server_ip"] = self.llm_server_ip
                # copied_info["llm_server_port"] = self.llm_server_port
                # copied_info["genRM_app_address"] = self.genRM_app_address
                # copied_info["instruct_follow_app_address"] = self.instruct_follow_app_address
                # copied_info["reward_model_app_address"] = self.reward_model_app_address

                copied_info["embedding_tokenizer"] = self.embedding_tokenizer
                copied_info["embedding_model"] = self.embedding_model

                copied_info["embedding_batcher"] = self.embedding_batcher

                if "token_upper" not in copied_info:
                    copied_info["token_upper"] = 32768
                extra_info_process.append(copied_info)
            else:
                tmp = {
                        "token_upper": 32768,
                        # "use_llm_verify": self.use_llm_verify,
                        # "llm_server_ip": self.llm_server_ip,
                        # "llm_server_port": self.llm_server_port,
                        # "genRM_app_address": self.genRM_app_address,
                        # "instruct_follow_app_address": self.instruct_follow_app_address,
                        # "reward_model_app_address": self.reward_model_app_address,
                        "response_id": torch.tensor(self.tokenizer.encode(response, add_special_tokens=False)),
                        "embedding_tokenizer": self.embedding_tokenizer,
                        "embedding_model": self.embedding_model,
                        "embedding_batcher": self.embedding_batcher,
                    }
                extra_info_process.append(tmp)

        assert len(extra_info_process) == len(response_str)

        try:
            scores = asyncio.run(
                parallel_compute_score_async(self.compute_score,
                                             response_str,
                                             ground_truth,
                                             data_sources,
                                             extra_info_process,
                                             # num_processes=self.reward_num_processes,
                                             # log_print=str2bool(os.environ.get('PRINT_REWARD_LOG', 'false'))))
                                             num_workers=1
                )
            )
        except asyncio.TimeoutError as e:
            print('Global timeout in reward computing! Setting all as 0.')
            scores = [{"score": 0.0} for _ in range(len(response_str))]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [{"score": 0.0} for _ in range(len(response_str))]

        assert len(scores) == len(data)

        final_reward_list = []
        reward_extra_info = defaultdict(list)
        for i in range(len(scores)):
            data_source = data_sources[i]
            gt = ground_truth[i]
            final_reward = scores[i]['score']
            if 'extra_info' in scores[i]:
                error_ = 0
                answer_score = scores[i]['extra_info']['answer_score']
                format_score = scores[i]['extra_info']['format_score']
                length_score = scores[i]['extra_info']['length_score']
                rm_response = scores[i]['extra_info']['rm_response']
                think_token_count = scores[i]['extra_info']['think_token_count']
                answer_token_count = scores[i]['extra_info']['answer_token_count']
                llm_equal = scores[i]['extra_info'].get("llm_equal", 0)
                if_llm_equal = scores[i]['extra_info'].get("if_llm_equal", 0)
                llm_response = scores[i]['extra_info'].get("llm_response", "")
                count_score = scores[i]['extra_info']['count_score']
                sim_score = scores[i]['extra_info']['sim_score']

                overlong_reward = 0.0
                # if self.overlong_buffer_cfg.enable:
                #     overlong_buffer_len = self.overlong_buffer_cfg.len
                #     expected_len = self.max_resp_len - overlong_buffer_len
                #     exceed_len = valid_response_length[i].item() - expected_len
                #     overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                #     overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                #     final_reward += overlong_reward
                #     if self.overlong_buffer_cfg.log:
                #         reward_extra_info["overlong_reward"].append(overlong_reward)
                #         reward_extra_info["overlong"].append(overlong_reward < 0)
            else:
                error_ = 1
                answer_score = 0.0
                format_score = 0.0
                length_score = 0.0
                rm_response = ""
                think_token_count = 0
                answer_token_count = 0
                llm_equal = 0
                if_llm_equal = 0
                overlong_reward = 0.0
                llm_response = ""
                count_score = 0.0
                sim_score = 0.0

            reward_extra_info["data_source"].append(data_source)
            reward_extra_info["error"].append(error_)
            reward_extra_info["answer_score"].append(answer_score)
            reward_extra_info["format_score"].append(format_score)
            reward_extra_info["length_score"].append(length_score)
            reward_extra_info["rm_response"].append(rm_response)
            reward_extra_info["ground_truth"].append(gt)
            reward_extra_info["think_token_count"].append(think_token_count)
            reward_extra_info["answer_token_count"].append(answer_token_count)
            reward_extra_info["llm_equal"].append(llm_equal)
            reward_extra_info["if_llm_equal"].append(if_llm_equal)
            reward_extra_info["overlong_reward"].append(overlong_reward)
            reward_extra_info["llm_response"].append(llm_response)
            reward_extra_info["count_score"].append(count_score)
            reward_extra_info["sim_score"].append(sim_score)

            final_reward_list.append(final_reward)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = final_reward_list[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[Response]", response_str[i])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
