set -x
# MODEL_PATH=/workspace/models/Qwen2.5-7B-Instruct-1M-step1-bf16
MODEL_PATH=/workspace/models/Qwen2.5-7B-Instruct-1M
# MODEL_PATH=/workspace/models/Qwen2.5-7B
# DATA_PATH_PREFIX=/workspace/data/kk/instruct/5ppl
DATA_PATH_PREFIX=/workspace/data/kk/instruct
# DATA_PATH_PREFIX=/workspace/data/orz


# export WANDB_MODE="offline"
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY='7d935869a9bcdfe5abfcd8dc7de64914f5d5a6d1'
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files="${DATA_PATH_PREFIX}"/ppl3_7_train_shuf.parquet \
    data.val_files="${DATA_PATH_PREFIX}"/ppl3_7_test_shuf.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=4e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='GRPO_logic_KK' \
    trainer.experiment_name='Qwen2.5-7B-Instruct-1M-step1-v14' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@