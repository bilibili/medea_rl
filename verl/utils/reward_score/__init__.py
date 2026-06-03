# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl.utils.import_utils import deprecated
from verl.utils.reward_score import general, reward_model, renaissance_dvc_pr


def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    if data_source == 'renaissance_dvc_pr':
        extra_info["data_source"] = "renaissance_dvc_pr"
        return renaissance_dvc_pr.compute_score(solution_str, ground_truth, extra_info)
    elif data_source == 'general' or data_source == "general_nbs":
        extra_info["data_source"] = "general"
        return reward_model.compute_score(solution_str, ground_truth, extra_info)
    elif data_source == 'long2short':
        extra_info["data_source"] = "long2short"
        return general.compute_score(solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore)


__all__ = ["default_compute_score"]
