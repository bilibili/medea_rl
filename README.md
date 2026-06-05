# MEDEA: Multimodal Engagement-Driven Evaluation Architecture

[![Paper](https://img.shields.io/badge/arXiv-CASTER-b31b1b.svg)](https://arxiv.org/abs/2606.01897)
[![Code](https://img.shields.io/badge/GitHub-medea__rl-blue.svg)](https://github.com/bilibili/medea_rl)
[![Model](https://img.shields.io/badge/HuggingFace-MEDEA-yellow.svg)](https://huggingface.co/IndexTeam/MEDEA)
[![Dataset](https://img.shields.io/badge/HuggingFace-CASTER--Bench-green.svg)](https://huggingface.co/datasets/IndexTeam/CASTER-Bench)

This repository contains the reinforcement learning training code for **MEDEA** (Multimodal Engagement-Driven Evaluation Architecture), as described in:

> **Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation**
>
> Tianjiao Li, Kai Zhao, Xiang Li, Yang Liu, Huyang Sun
>
> ACL 2026

## Overview

MEDEA redefines UGC quality assessment by shifting focus from aesthetic fidelity to **social-cognitive resonance**. It introduces *Social Chain-of-Thought (Social-CoT)*, where the model simulates diverse viewer personas and generates empathetic reasoning paths before making a quality judgment.

The RL stage uses **GRPO** (Group Relative Policy Optimization) with a composite **Social Alignment Reward** to align the model's reasoning with authentic human social cognition:

```
r = r_format + r_label + r_diversity + r_social
```

- **r_format**: Ensures `<think>...</think>` structure compliance
- **r_label**: Correctness of the final binary quality prediction
- **r_diversity**: Penalizes repetitive/collapsed comment generation
- **r_social**: Cosine similarity between generated reasoning paths and real user comments (via embedding model)

## Installation

```bash
pip install -e .
pip install vllm  # inference engine (or sglang)
```

## Training

```bash
bash train_medea.sh
```

Key parameters can be configured in the script. For multi-node training, set `trainer.nnodes=N`.

## Reward Function

The core reward implementation is in `verl/utils/reward_score/medea.py`.

The scoring formula:
```
total = format_score + 0.5 * count_score + 1.0 * sim_score + 2.0 * answer_score
```

| Component | Description |
|-----------|-------------|
| `format_score` | 0 if `<think>...</think>` structure is correct, -1 otherwise |
| `count_score` | Diversity penalty: -0.1 for each duplicate comment |
| `sim_score` | Social Alignment: greedy-matched cosine similarity between generated and real comments |
| `answer_score` | 1.0 if predicted label matches ground truth, 0.0 otherwise |

## Data Format

Training data should be in parquet format with:
- `prompt`: Input prompt (UGC metadata + instruction)
- `data_source`: `"medea"`
- `reward_model.ground_truth`: Expected quality label ("High-Quality" / "Low-Quality")
- `reward_model.real_comments`: List of real user comments for social alignment

## Project Structure

```
verl/
├── trainer/                    # GRPO trainer (Ray distributed)
├── workers/
│   └── reward_manager/         # Social Alignment Reward orchestration
├── utils/reward_score/
│   └── medea.py                # Composite reward function
train_medea.sh                  # Training launch script
```

## Citation

```bibtex
@inproceedings{li2026caster,
  title={Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation},
  author={Li, Tianjiao and Zhao, Kai and Li, Xiang and Liu, Yang and Sun, Huyang},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

## Acknowledgements

Built on [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLMs). Licensed under Apache-2.0.
