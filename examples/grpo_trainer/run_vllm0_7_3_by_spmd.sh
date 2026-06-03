set -x
MODEL_PATH=/workspace/models/Qwen2.5-3B
DATA_PATH_PREFIX=/workspace/data/Open-Reasoner

# export WANDB_MODE="offline"
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY='7d935869a9bcdfe5abfcd8dc7de64914f5d5a6d1'
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_PATH_PREFIX}"/train.parquet \
    data.val_files="${DATA_PATH_PREFIX}"/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=4e-7 \
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
    actor_rollout_ref.rollout.n=256 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='VLLM_TEST' \
    trainer.experiment_name='review_vllm_0_7_3_n256_leng16384_sp1_tp1_mbs16' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=-1 \
    trainer.test_freq=10000 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.val_generations_to_log_to_txt=100 $@ 2>&1 | tee grpo.log
    

# ps -ef | grep verl.trainer.main_ppo | grep -v grep | awk -F' ' '{print $2}' | xargs kill -9
# ps -ef | grep run.sh | grep -v grep | awk -F' ' '{print $2}' | xargs kill -9
# ps -ef | grep run_vllm0_7_3_by_spmd.sh | grep -v grep | awk -F' ' '{print $2}' | xargs kill -9