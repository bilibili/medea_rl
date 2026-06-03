#!/bin/bash
# MEDEA — GRPO Training with Social Alignment Reward
#
# Usage: bash train_medea.sh
#
# Prerequisites:
#   pip install -e .
#   pip install vllm  # or sglang

set -x

MODEL_PATH="Qwen/Qwen3-8B"  # or local path
TRAIN_DATA="data/train.parquet"
VAL_DATA="data/val.parquet"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=30 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.project_name=medea \
    trainer.experiment_name=grpo_run \
    trainer.logger=[console,tensorboard] \
    trainer.val_before_train=True \
    reward_model.reward_manager=index_pr
