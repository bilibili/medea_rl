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

from typing import Dict, Tuple, Optional
from verl.utils.logging_utils import LogCollector
from verl.utils.reward_score.long2short import cal_think_token_count, cal_answer_token_count


def validate_response_structure(processed_str: str, answer_text: str, logger: LogCollector) -> bool:
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
    # Verify box位置
    return validation_passed


def extract_solution(solution_str: str, logger: LogCollector) -> Tuple[Optional[str], str]:
    processed_str = solution_str
    if len(solution_str.split("</think>")) <= 1:
        return None, processed_str
    final_answer = solution_str.split("</think>")[1]

    return final_answer, processed_str


def validate_renaissance_label(data_str, ground_truth):
    if data_str == ground_truth:
        return 1.0
    else:
        return 0.0


def compute_score(solution_str, ground_truth, extra_info):
    try:
        return compute_score_inner(solution_str, ground_truth, extra_info)
    except Exception as e:
        print(f"[RENAISSANCE_DVC] completion: \n", solution_str)
        print(f"[RENAISSANCE_DVC] ground_truth: \n", ground_truth)
        print(f"[RENAISSANCE_DVC] extra_info: \n", extra_info)
        print(f"[RENAISSANCE_DVC] Error: {e}")


def compute_score_inner(solution_str, ground_truth, extra_info):
    logger = LogCollector(prefix="RENAISSANCE_DVC")
    logger.clear()

    logger.log("\n" + "=" * 80)
    logger.log(" Processing New Sample ".center(80, '='))

    solution_str = "<think>\n" + solution_str
    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str, logger)
    logger.log(f"\n[Model Response]\n{processed_str}")
    logger.log(f"\n[Model Answer]\n{answer_text}")
    # Validate response structure
    format_correct = validate_response_structure(processed_str, answer_text, logger)
    format_score = 0.0 if format_correct else -1.0
    logger.log(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")

    answer_score = 0.0
    length_score = 0.0
    think_token_count = 0
    answer_token_count = 0
    if format_correct and answer_text:
        logger.log(f"\n[Content Validation]")
        logger.log(f"  Expected: {extra_info['ground_truth']}")
        logger.log(f"  Predicted: {answer_text.strip()}")

        answer_score = validate_renaissance_label(answer_text.strip(), extra_info['ground_truth'].strip())
        # 处理length相关的reward
        # response_id = extra_info["response_id"]
        # think_token_count = cal_think_token_count(response_id, extra_info)
        # answer_token_count = cal_answer_token_count(response_id, extra_info)
    else:
        answer_score = 0.0
        think_token_count = 0
        answer_token_count = 0
        logger.log("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = answer_score + format_score + length_score
    logger.log("\n" + "-" * 80)
    logger.log(f" Final Score ".center(80, '-'))
    logger.log(f"  Format: {format_score}")
    logger.log(f"  Answer: {answer_score}")
    logger.log(f"  Length: {length_score}")
    logger.log(f"  think_token_count: {think_token_count}")
    logger.log(f"  answer_token_count: {answer_token_count}")
    logger.log("=" * 80 + "\n")

    return {
        "score": total_score,
        "extra_info": {
            "format_score": format_score,
            "answer_score": answer_score,
            "length_score": length_score,
            "rm_response": "",
            "think_token_count": think_token_count,
            "answer_token_count": answer_token_count
        }
    }, logger.get_logs()
