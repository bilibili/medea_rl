# CASTER website sources

Reference information checked on October 10, 2026.

## Publication and resources

- [ACL Anthology — official publication and citation](https://aclanthology.org/2026.acl-long.985/)
- [Official paper PDF](https://aclanthology.org/2026.acl-long.985.pdf)
- [arXiv:2606.01897](https://arxiv.org/abs/2606.01897)
- [MEDEA training code and public inference example](https://github.com/bilibili/medea_rl)
- [MEDEA model](https://huggingface.co/IndexTeam/MEDEA)
- [CASTER-Bench dataset](https://huggingface.co/datasets/IndexTeam/CASTER-Bench)

The model and dataset URLs are published in the official code repository. The ACL record supplies the title, author order, conference, page numbers, DOI, and downloadable [BibTeX](assets/citation.bib).

## Data and metric definitions

- **CASTER-Bench:** 1,485 videos across 30 categories, averaging 442 seconds, with 182.5 hours of video in total. These are the paper's reported statistics (Section 3 and Table 1).
- **Annotation:** 10 professional content operation experts assess production quality, perceived value, information utility, and narrative excellence. The original labels are Excellent, Good, Average, and Poor; the reported model evaluation uses High-Quality / Low-Quality classification.
- **Results:** High-Quality F1 and Macro F1 come from Table 2. F1 is not accuracy. MEDEA obtains 0.650 High-Quality F1 and 0.749 Macro F1. If shown on a 0–100 scale, these become 65.0 and 74.9.
- **Improvement:** Compared with GPT-5.2 (reasoning), 0.650 versus 0.555 is a 17.1% relative increase in High-Quality F1, or 9.5 points on a 0–100 scale. Macro F1 increases from 0.595 to 0.749, or 15.4 points.
- **Comparisons:** The paper reports averages over five independent runs. Traditional VQA methods use threshold sweeps; LMM baselines perform zero-shot prediction. Claims are limited to CASTER-Bench.
- **Ablations:** Component and modality comparisons come from Tables 3 and 6. Reasoning faithfulness and diversity scores come from Table 7, on a five-point scale. The 0.536 High-Quality F1 variant is “RL-pseudo+human.”

## Method and examples

MEDEA uses a Qwen3-VL-8B-Think backbone. Its training pipeline combines 54K pseudo-labeled Social-CoT samples and 3K human-annotated samples for supervised fine-tuning, followed by GRPO on human-curated samples. Real comments provide training supervision and the Social Alignment Reward. At inference, MEDEA uses multimodal content and metadata without requiring real comments (Sections 4–5).

Social-CoT comments displayed as model outputs are **simulated reactions**, not testimonials or observed audience feedback. The consensus score is `z = (supportive − opposing) / sqrt(supportive + opposing)`; the model uses `z ≥ 1.5` for the supportive / High-Quality decision.

- **Iceland travel vlog:** Appendix E, Figure 6. MEDEA's published example generates 15 supportive and 0 opposing comments, giving `z ≈ 3.873`. The comparison without Social Alignment is in Figure 7.
- **Rare gaming loot:** The official repository's public inference example generates 15 supportive and 5 opposing comments, giving `z ≈ 2.236`.
- **Giveaway-driven engagement:** Appendix H, Figure 4. This is an expert-annotation example illustrating the difference between popularity and intrinsic content quality; it is not a reported successful individual MEDEA prediction.

## Design reference

[LAIT](https://lait.cs.sfu.ca/) is the visual inspiration for the site's academic project-page presentation. CASTER's research claims and resources are attributed to the sources above.
