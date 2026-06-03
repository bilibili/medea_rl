# Copyright 2025 bilibili Ltd. and/or its affiliates
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

from recipe.bilibili.reward_score import renaissance_dvc
from verl.utils.reward_score import renaissance


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if "renaissance_dvc" == data_source:
        extra_info["data_source"] = "renaissance_dvc"
        return renaissance_dvc.compute_score(solution_str, ground_truth, extra_info)
    elif "renaissance" == data_source:
        extra_info["data_source"] = "renaissance"
        return renaissance.compute_score(solution_str, ground_truth, extra_info)
    elif 'translation' == data_source:  # hack translation dataset to train with renaissance_dvc, just for debug
        extra_info["data_source"] = "renaissance_dvc"
        ground_truth = "低质内容"
        return renaissance_dvc.compute_score(solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")


__all__ = ["compute_score"]
