set -x
MODEL_PATH=/workspace/models/Qwen2.5-7B
DATA_PATH_PREFIX=/workspace/data/R1-dataset

# export WANDB_MODE="offline"
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY='7d935869a9bcdfe5abfcd8dc7de64914f5d5a6d1'
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_PATH_PREFIX}"/train_5kcodeforce_5korz.parquet \
    data.val_files="${DATA_PATH_PREFIX}"/test_5kcodeforce_5korz.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=prime \
    reward_model.prime_num_processes=32 \
    reward_model.prime_batch_size=32 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='GRPO_MISTURE' \
    trainer.experiment_name='Qwen2.5-7B-mistrue' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=2 \
    trainer.val_generations_to_log_to_txt=100 $@ 2>&1 | tee grpo.log
    

# ps -ef | grep verl.trainer.main_ppo | grep -v grep | awk -F' ' '{print $2}' | xargs kill -9
# ps -ef | grep run.sh | grep -v grep | awk -F' ' '{print $2}' | xargs kill -9
# ps -ef | grep run_qwen2_7b_zero_mixture.sh | grep -v grep | awk -F' ' '{print $2}' | xargs kill -9