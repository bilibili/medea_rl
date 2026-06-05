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

## Deployment with vLLM

```bash
vllm serve your-org/MEDEA \
    --served-model-name "MEDEA" \
    --max-model-len 20480 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.8 \
    --dtype float16 \
    --port 8006
```

## Inference

### Input Format

Prepare a JSON file where each sample has `input_text` and `images` fields:

```json
{
  "input_text": "<任务定义>\n你是一个B站内容分析专家。请综合给定稿件的视觉特征和文本内容，然后根据视频内容推测评论区针对该视频创作质量可能出现的评论（至少15条）及其倾向，最后总结评论区的总倾向并直接输出结果。\n\n### 输入数据\n1. 封面: 稿件的封面图像\n2. 关键帧: 7张视频的关键帧图像\n3. 标题: {title}\n4. 标签: {tag}\n5. ASR: {asr}\n6. 一级分区: {new_tid_name}\n7. 二级分区: {new_sub_tid_name}\n8. 时长: {duration}\n9. 分辨率: {resolution}\n10. 是否竖屏: {vertical}\n\n### 评论区总倾向判断准则\n1. 模拟的评论必须至少有15条，不能重复，并且明确出现在思考过程中\n2. 假设在模拟的评论中，支持类评论共X条，反对类评论共Y条\n3. 计算 Sigma-标准化差额（Skellam z 分数）z = (X - Y) / sqrt(X + Y)\n4. 如果 z ≥ 1.5 则评论区总倾向为\"支持\"，否则为\"非明确支持\"\n5. 在输出时，z值必须用boxed{}包裹，例如\"z = boxed{-2}\"\n6. 总结时的支持类评论数和反对类评论数需要和思考过程中模拟的评论数严格匹配，不能随意虚构\n\n<当前任务>\n封面: <image>\n关键帧: <image><image><image><image><image><image><image>\n标题: \"给三角洲一点震撼\"\n标签: \"游戏,射击游戏,三角洲行动\"\nASR: \"\"\n一级分区: \"游戏\"\n二级分区: \"射击游戏\"\n时长: \"28秒\"\n分辨率: \"2160x3840\"\n是否竖屏: \"是\"\n\n请直接输出结果（支持/非明确支持），不要包含任何额外解释。",
  "images": [
    "/path/to/cover.jpg",
    "/path/to/frame_0.jpg",
    "/path/to/frame_1.jpg",
    "/path/to/frame_2.jpg",
    "/path/to/frame_3.jpg",
    "/path/to/frame_4.jpg",
    "/path/to/frame_5.jpg",
    "/path/to/frame_6.jpg"
  ]
}
```

Each `<image>` placeholder corresponds to an image path in the `images` list (1 cover + 7 key frames). The `{title}`, `{tag}`, etc. in the task definition are already filled in the `<当前任务>` section.

### Batch Inference Script

```python
import base64
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(base_url="http://localhost:8006/v1", api_key="EMPTY")

def file_to_data_url(path):
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def build_multimodal_content(text, image_paths):
    parts = text.split("<image>")
    content = []
    for i, seg in enumerate(parts):
        if seg.strip():
            content.append({"type": "text", "text": seg.strip()})
        if i < len(image_paths):
            content.append({"type": "image_url", "image_url": {"url": file_to_data_url(image_paths[i])}})
    return content

def infer(sample):
    user_msg = next(m for m in sample["messages"] if m["role"] == "user")
    content = build_multimodal_content(user_msg["content"], sample.get("images", []))
    resp = client.chat.completions.create(
        model="MEDEA",
        messages=[{"role": "user", "content": content}],
        temperature=0.6,
        max_tokens=4096,
        top_p=0.95,
    )
    return resp.choices[0].message.content.strip()

# Batch inference
samples = json.load(open("test_data.json"))
with ThreadPoolExecutor(max_workers=32) as ex:
    futures = {ex.submit(infer, s): i for i, s in enumerate(samples)}
    for fut in tqdm(as_completed(futures), total=len(samples)):
        i = futures[fut]
        samples[i]["pred"] = fut.result()

json.dump(samples, open("results.json", "w"), ensure_ascii=False, indent=2)
```
## Output Format

The model generates structured Social-CoT reasoning with simulated community reactions, followed by a final quality label:

<details>
<summary>Click to expand full example output</summary>

```
<think>
基于给定稿件信息，该视频属于「游戏」分区下的「射击游戏」类别，标题「给三角洲一点震撼，"真"（大红包）」明确指出视频主题是关于游戏内获得大量稀有物品。视频时长「28秒」，分辨率「2160x3840」，且为「竖屏」拍摄。ASR内容未提供。关键帧和封面清晰展示了视频内容，即通过「手机拍摄显示屏」的方式，呈现了《三角洲行动》游戏中一个「背包」被搜刮后，里面装满了「主战坦克模型、军用信息终端、显卡、医疗机器人、浮力设备、黄金眼玲、实验枪械、名贵怀表、盘缠等」这些稀有的「红色品质」游戏物资，与标题中的"大红包"概念完美契合。

观众在观看这类以游戏稀有物品展示为主题的视频时，通常会关注以下几个方面：首先是内容是否真实，其次是展示的物品稀有度是否足够震撼，再者是对up主运气或实力的评价，最后可能也会关注视频本身的制作质量。

当观众看到「视频标题"真"（大红包）与屏幕上充满红色稀有物品的背包」时，会因其视觉冲击力产生强烈的肯定。
【真的特别给力，放心冲[星星眼][星星眼][星星眼]】(支持评论1)

针对「视频中背包被稀有红色物品填满」的景象，观众会将其与标题「大红包」联系起来，表达对内容的认可和赞叹。
【牛逼，真红包🧧】(支持评论2)

当观众看到「如此之多的稀有物品汇集一包」时，会由衷地赞叹up主的实力或运气。
【我去[星星眼]这么有实力】(支持评论3)

「视频内容直接展现了一背包红色稀有物品的震撼效果」，这足以让多数观众对其稀有度和价值表示肯定。
【有点强】(支持评论4)

「视频直接展示了稀有物品的收集结果」，这种直观的震撼力，能轻易引发观众的赞叹。
【牛逼】(支持评论5)

「视频展示的『全红』背包内容极其夸张且罕见」，这会让一些观众感到难以置信或惊叹，并用"难绷"来表达这种复杂情绪，但整体倾向是正面的惊讶。
【难绷】(支持评论6)

面对「视频中前所未见的『全红』背包」，许多观众会感到极其惊讶，认为这是难以想象的。
【我做梦都不敢这么梦。】(支持评论7)

「视频直观地展示了整个背包都是稀有红色物品」，这种极为罕见的场景会让观众印象深刻，并强调其稀有性。
【666第一次见全红】(支持评论8)

「视频中满满当当的稀有高价值物品」，能够直接勾起游戏玩家的占有欲和羡慕。
【想要。馋】(支持评论9)

「视频中清晰可见的、铺满屏幕的红色物品」，其视觉效果本身就能让观众感到赏心悦目，并肯定其内容的稀有性。
【全是红的包就是好看[doge]】(支持评论10)

「视频核心内容是展示一背包红色稀有物品」，这种稀有性与up主的"牛逼"表现结合，让观众发出赞叹。
【全红啊nb】(支持评论11)

对于「这类展示罕见"大红包"的视频，且呈现出前所未有的『全红』背包」，观众会认为这极大地体现了up主的运气或实力。
【这个大红包全红视频没见过，是新的[doge]真有实力啊】(支持评论12)

「视频中的『全红』背包意味着极高的游戏资产价值」，观众会以夸张的语气来表达其带来的财富感。
【全红！！！！！，你直接发家致富了】(支持评论13)

「视频展示的『全红』背包是极其罕见的现象」，许多观众在游戏中可能从未见过类似情况，从而表达出极度的惊讶和难以置信。
【这辈子没见过3红，视频里直接一背包红，人看傻了[笑哭]】(支持评论14)

「视频展示的稀有物品和获得方式，体现了up主在游戏中的某种"实力"」，观众会直接对其表现进行肯定。
【实力这一块】(支持评论15)

然而，并非所有观众都会对「视频中展示的『大红包』」感到惊讶或支持，有些玩家可能会认为这只是普通操作，不值得大惊小怪。
【也就那样吧，我搜一个背包的速度比他快】(反对评论1)

当视频展示了「高价值物品的获取」，可能让一些观众误以为易于复制，从而引发盲目尝试，因此会有观众出于警示目的发出不同意见。
【不要学主播猛攻破产了[大哭]】(反对评论2)

面对「视频中过于完美和稀有的『全红』背包」，部分观众可能会出于怀疑，质疑其内容的真实性，认为可能存在作弊行为。
【物资透吧】(反对评论3)

尽管视频内容稀有，但其「竖屏且为手机拍摄显示屏」的制作方式，可能会被部分注重视频质量的观众批评。
【依旧手机拍显示屏】(反对评论4)

当观众对「视频展示的『全红』背包」不感到稀奇，或者认为其价值被夸大时，会以一种轻描淡写的方式表达反对或不屑。
【没心，一般[doge]】(反对评论5)

在模拟的评论中，支持类评论共15条，反对类评论共5条。
计算 Sigma-标准化差额（Skellam z 分数）z = (15 - 5) / sqrt(15 + 5) = 10 / sqrt(20) ≈ boxed{2.236}。
因 z ≥ 1.5，故整体倾向为支持。
</think>

支持
```

</details>

The output structure:
1. **`<think>...</think>`**: Social-CoT reasoning — content analysis, perspective-taking from diverse viewer personas, simulated supportive/opposing comments
2. **Skellam scoring**: Statistical consensus mechanism aggregating support vs. opposition counts
3. **Final label**: "支持" (High-Quality) or "反对" (Low-Quality)

## Performance

Results on CASTER-Bench:

| | High-Quality F1 | Low-Quality F1 | Macro F1 |
|---|---|---|---|
| **MEDEA** | **0.650** | **0.847** | **0.749** |
| Claude-4.5-opus (Long-CoT) | 0.528 | 0.517 | 0.522 |
| Qwen3-VL-Plus (Social-CoT) | 0.508 | 0.647 | 0.578 |
| GPT-5.2 | 0.506 | 0.489 | 0.498 |

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
