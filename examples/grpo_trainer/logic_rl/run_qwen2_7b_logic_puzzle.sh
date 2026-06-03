set -x
MODEL_PATH="${MODEL_PATH:-s3://llm_snapshot/Qwen2.5-7B-Instruct-1M}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-s3://llm-data/benjamin/datasets/kk/instruct/3ppl/train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-s3://llm-data/benjamin/datasets/kk/instruct/3ppl/test.parquet}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS

echo "MODEL_PATH"
echo "TRAIN_DATA_PATH"
echo "VAL_DATA_PATH"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=400 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='GRPO_logic_KK' \
    trainer.experiment_name='Qwen2.5-7B-Instruct-1M-step1' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@ 2>&1 | tee grpo.log