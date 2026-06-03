from typing import Dict, Tuple, Optional
import numpy as np
import re
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from verl.utils.logging_utils import LogCollector
from verl.utils.reward_score.long2short import long2short_compute_score, cal_think_token_count, cal_answer_token_count


def validate_response_structure(processed_str: str, answer_text: str, logger: LogCollector) -> bool:
    logger.log("\n[Structure Validation]")
    validation_passed = True

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

    if positions['think_start'] > positions['think_end']:
        logger.log("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    else:
        logger.log("  Tag sequence validation passed")

    return validation_passed


def extract_solution(solution_str: str, logger: LogCollector) -> Tuple[Optional[str], str]:
    processed_str = solution_str
    if len(solution_str.split("</think>")) <= 1:
        return None, processed_str
    final_answer = solution_str.split("</think>")[1]

    return final_answer, processed_str


def validate_renaissance_label(data_str, ground_truth):
    return 1.0 if data_str == ground_truth else 0.0


def extract_comments_from_think(think_text: str):
    support_pattern = r"【(.*?)】[（(]支持评论\d+[）)]"
    oppose_pattern = r"【(.*?)】[（(]反对评论\d+[）)]"

    supports = re.findall(support_pattern, think_text)
    opposes = re.findall(oppose_pattern, think_text)

    comments = [[c, "支持"] for c in supports] + [[c, "反对"] for c in opposes]
    return comments, len(supports), len(opposes)


def compute_comment_similarity_embedding(gen_comments, real_comments, extra_info):
    if not gen_comments or not real_comments:
        return -0.5

    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    embedding_tokenizer = extra_info["embedding_tokenizer"]
    embedding_model = extra_info["embedding_model"]

    all_texts = gen_comments + real_comments
    max_length = 256

    batch_dict = embedding_tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict = {k: v.to(embedding_model.device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = embedding_model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    gen_vecs = embeddings[:len(gen_comments)]
    real_vecs = embeddings[len(gen_comments):]

    sim_matrix = torch.matmul(gen_vecs, real_vecs.T)
    max_sims = sim_matrix.max(dim=1).values
    mean_sim = max_sims.mean().item()

    return float(mean_sim) if torch.isfinite(max_sims).all() else -0.5


def compute_comment_similarity_embedding_unique(gen_comments, real_comments, extra_info):
    if not gen_comments or not real_comments:
        return -0.5

    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    embedding_tokenizer = extra_info["embedding_tokenizer"]
    embedding_model = extra_info["embedding_model"]

    device = embedding_model.device
    all_texts = gen_comments + real_comments
    max_length = 256

    batch_dict = embedding_tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = embedding_model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    gen_vecs = embeddings[:len(gen_comments)]
    real_vecs = embeddings[len(gen_comments):]

    sim_matrix = torch.matmul(gen_vecs, real_vecs.T)  # [N_gen, N_real]

    matched = torch.zeros(real_vecs.size(0), dtype=torch.bool, device=device)
    matched_sims = []

    for i in range(sim_matrix.size(0)):
        sim_row = sim_matrix[i]
        sim_row = sim_row.masked_fill(matched, -1e9)
        max_sim, max_idx = sim_row.max(dim=0)
        if max_sim > -1e8:
            matched[max_idx] = True
            matched_sims.append(max_sim.item())

    if not matched_sims:
        return -0.5

    mean_sim = float(sum(matched_sims) / len(matched_sims))
    return mean_sim


def compute_comment_similarity_embedding_unique_batch(
    gen_comments, real_comments, extra_info
):
    if not gen_comments or not real_comments:
        return -0.5

    batcher = extra_info["embedding_batcher"]

    all_texts = gen_comments + real_comments

    embeddings = batcher.encode(all_texts)

    gen_vecs = embeddings[:len(gen_comments)]
    real_vecs = embeddings[len(gen_comments):]

    sim_matrix = torch.matmul(gen_vecs, real_vecs.T)

    matched = torch.zeros(real_vecs.size(0), dtype=torch.bool, device=sim_matrix.device)
    matched_sims = []

    for i in range(sim_matrix.size(0)):
        row = sim_matrix[i].masked_fill(matched, -1e9)
        max_sim, idx = row.max(dim=0)
        if max_sim > -1e8:
            matched[idx] = True
            matched_sims.append(max_sim.item())

    if not matched_sims:
        return -0.5

    return sum(matched_sims) / len(matched_sims)


def compute_score(solution_str, ground_truth, extra_info):
    try:
        return compute_score_inner(solution_str, ground_truth, extra_info)
    except Exception as e:
        print(f"[RENAISSANCE_DVC_PR] completion: \n", solution_str)
        print(f"[RENAISSANCE_DVC_PR] ground_truth: \n", ground_truth)
        print(f"[RENAISSANCE_DVC_PR] extra_info: \n", extra_info)
        print(f"[RENAISSANCE_DVC_PR] Error: {e}")


def compute_score_inner(solution_str, ground_truth, extra_info):
    logger = LogCollector(prefix="RENAISSANCE_DVC_PR")
    logger.clear()

    solution_str = "<think>\n" + solution_str
    answer_text, processed_str = extract_solution(solution_str, logger)

    format_correct = validate_response_structure(processed_str, answer_text, logger)
    format_score = 0.0 if format_correct else -1.0

    answer_score = 0.0
    length_score = 0.0
    count_score = 0.0
    sim_score = 0.0

    if format_correct and answer_text:
        gen_comments, gen_support, gen_oppose = extract_comments_from_think(processed_str)
        if len(gen_comments) < 15:
            sim_score = -0.5
        else:
            real_comments = extra_info["real_comments"]

            texts = [c[0] for c in gen_comments]
            dup_penalty = 0.0
            freq = Counter(texts)
            for c, f in freq.items():
                if f > 1:
                    dup_penalty -= 0.1 * (f - 1)

            count_score = dup_penalty

            sim_score = compute_comment_similarity_embedding_unique(texts, real_comments, extra_info)

        answer_score = validate_renaissance_label(answer_text.strip(), extra_info['ground_truth'].strip())

        response_id = extra_info["response_id"]
        think_token_count = cal_think_token_count(response_id, extra_info)
        answer_token_count = cal_answer_token_count(response_id, extra_info)
    else:
        think_token_count, answer_token_count = 0.0, 0.0

    count_weight = 0.5
    sim_weight = 1.0
    answer_weight = 2.0
    total_score = format_score + count_weight * count_score + sim_weight * sim_score + answer_weight * answer_score

    logger.log("\n" + "-" * 80)
    logger.log(f" Final Score ".center(80, '-'))
    logger.log(f"  Format: {format_score}")
    logger.log(f"  Answer: {answer_score}")
    logger.log(f"  Length: {length_score}")
    logger.log(f"  Count: {count_score}")
    logger.log(f"  Similarity: {sim_score}")
    logger.log(f"  think_token_count: {think_token_count}")
    logger.log(f"  answer_token_count: {answer_token_count}")
    logger.log("=" * 80 + "\n")

    return {
        "score": total_score,
        "extra_info": {
            "format_score": format_score,
            "answer_score": answer_score,
            "length_score": length_score,
            "rm_response": extra_info["real_comments"],
            "count_score": count_score,
            "sim_score": sim_score,
            "think_token_count": think_token_count,
            "answer_token_count": answer_token_count
        }
    }, logger.get_logs()
