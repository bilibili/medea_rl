#!/bin/bash

# set -x
set -e

trap "echo '检测到 Ctrl+C, 退出脚本'; exit 1" SIGINT
ulimit -n 32768
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"
# 判断设备类型，并设置相应的启动参数
if [ -e "/dev/davinci_manager" ]; then
    _DEVICE_TYPE=npu
    _DEVICE_PER_NODE=16
    _SP_SIZE=8
    _USE_TORCH_COMPILE=False
    #TASK_QUEUE_ENABLE，下发优化，图模式设置为1，非图模式设置为2
    export TASK_QUEUE_ENABLE=1
    export HCCL_ASYNC_ERROR_HANDLING=0
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600
else
    _DEVICE_TYPE=cuda
    _DEVICE_PER_NODE=8
    _SP_SIZE=1
    _USE_TORCH_COMPILE=True
fi

# Prepare for debugging in vscode-remote
if [ ! -n "$PET_NNODES" ]; then
    source /mnt/group/daiyi/tmp/npu_beats_verl_env_bak.sh
fi

cd $SCRIPT_ROOT/../../..
bash recipe/bilibili/ray_start_with_task.sh

_PROJ_NAME=GRPO_WYFX
_EXP_NAME=$(echo $RANDOM | md5sum | cut -c1-4)

_USE_NNODES="${PET_NNODES:-1}"
_MAX_PROMPT_L=4096
_MAX_RES_L=4096
_MAX_MODEL_L=$((_MAX_PROMPT_L + _MAX_RES_L))

_actor_ppo_max_token_len=$((_MAX_MODEL_L / _SP_SIZE))
_infer_ppo_max_token_len=$((_MAX_MODEL_L / _SP_SIZE))

MODEL_ARGS=()
MODEL_ARGS+=(
    actor_rollout_ref.model.path=/mnt/group/opensource_models/GLM-4.5-Air-Base
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=True
    # reward model & rule
    custom_reward_function.path=recipe/bilibili/wyfx/wyfx_custom_func.py
    custom_reward_function.name=compute_score
    reward_model.reward_manager=index
    reward_model.reward_kwargs.reward_num_processes=32
    reward_model.reward_kwargs.max_resp_len=${_MAX_RES_L}
    reward_model.reward_kwargs.grm.use_llm_verify=False
    reward_model.reward_kwargs.grm.llm_server_ip=null
    reward_model.reward_kwargs.grm.llm_server_port=null
    reward_model.reward_kwargs.grm.genRM_app_address=http://infra.beats-app.prod-beats-infer-2
    reward_model.reward_kwargs.grm.instruct_follow_app_address=http://infra.beats-app.prod-beats-infer-2
    reward_model.reward_kwargs.grm.reward_model_app_address=http://infra.beats-app.prod-beats-infer-2
    reward_model.reward_kwargs.overlong_buffer.enable=False
    reward_model.reward_kwargs.overlong_buffer.len=0
    reward_model.reward_kwargs.overlong_buffer.penalty_factor=0.0
    reward_model.reward_kwargs.overlong_buffer.log=False
)

DATA_ARGS=()
DATA_ARGS+=(
    data.train_files=/mnt/group/datasets/individual-dataset/mangoz/deepseek_r1/data_grpo/train_wyfx_norm11k_cvrlow11k_sft_v6_think.parquet
    data.val_files=/mnt/group/datasets/individual-dataset/mangoz/deepseek_r1/data_grpo/test_wyfx_norm11k_cvrlow11k_sft_v6_think.parquet
    data.max_prompt_length=${_MAX_PROMPT_L}
    data.max_response_length=${_MAX_RES_L}
    data.train_batch_size=32
    data.val_batch_size=8
    data.shuffle=False
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=8
)

MEGATRON_OVERRIDE_TRANSFORMER_CONFIG=()
if [ -e "/dev/davinci_manager" ]; then
    MEGATRON_OVERRIDE_TRANSFORMER_CONFIG+=(
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
        ++actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=5
        ++actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=5
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
        ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True
    )
else
    MEGATRON_OVERRIDE_TRANSFORMER_CONFIG+=(
        actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity="selective"
        actor_rollout_ref.actor.megatron.override_transformer_config.recompute_modules=["core_attn","moe_act","layernorm","mlp","moe"]
        +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="flex"
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True
        # GLM-4.5-Air has 46 layers in total, fit to account embed/output as single layer => total 48 layers
        +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=True 
        +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=True
    )
fi

ACTOR_ARGS=()
ACTOR_ARGS+=(
    # actor
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${_actor_ppo_max_token_len}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.entropy_coeff=0.001
    # optim, verl/trainer/config/optim/megatron.yaml
    actor_rollout_ref.actor.optim.lr=1e-6

    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True 

    # checkpoint
    actor_rollout_ref.actor.checkpoint._target_=verl.trainer.config.BiliCheckpointConfig
    actor_rollout_ref.actor.checkpoint.save_contents=['model']
    actor_rollout_ref.actor.checkpoint.export=True
    actor_rollout_ref.actor.checkpoint.default_local_dir=checkpoints/${_PROJ_NAME}/${_EXP_NAME}
    actor_rollout_ref.actor.checkpoint.delete_after_upload=False
    # policy_loss
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=1
    # megatron_config
    actor_rollout_ref.actor.megatron.use_mbridge=True  # or else use_dist_checkpointing=True
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null
    actor_rollout_ref.actor.megatron.context_parallel_size=4
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
)

ROLLOUT_ARGS=()
ROLLOUT_ARGS+=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.tensor_model_parallel_size=8
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.top_k=20
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${_MAX_MODEL_L}
    actor_rollout_ref.rollout.max_model_len=${_MAX_MODEL_L}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${_infer_ppo_max_token_len}
    # valuate
    actor_rollout_ref.rollout.val_kwargs.top_k=20
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
)

REF_ARGS=()
REF_ARGS+=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${_infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=null
    actor_rollout_ref.ref.megatron.context_parallel_size=4
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
)

TRAINER_ARGS=()
TRAINER_ARGS+=(
    trainer.device=${_DEVICE_TYPE}
    trainer.n_gpus_per_node=${_DEVICE_PER_NODE}
    trainer.project_name=${_PROJ_NAME}
    trainer.experiment_name=${_EXP_NAME}
    trainer.default_local_dir=checkpoints/${_PROJ_NAME}/${_EXP_NAME}
    trainer.nnodes=${_USE_NNODES}
    trainer.critic_warmup=0
    trainer.logger=['console','tensorboard']
    trainer.save_freq=50
    trainer.test_freq=10
    trainer.total_epochs=4
    trainer.default_hdfs_dir=null
    trainer.val_before_train=False
    trainer.val_generations_to_log_to_txt=200
    trainer.summary_dir=/summary_dir
)

if [ ! -n "$TASK_TYPE" ]; then
    echo "Missing TASK_TYPE!!!"
    exit 0
fi

set -o pipefail   # make sure catch failure signal

if [ "${TASK_TYPE}" == "master" ]; then
    python3 -m recipe.bilibili.wyfx.main_ijp \
            --config-path=../../../verl/trainer/config \
            --config-name=index_ppo_megatron_trainer.yaml \
            algorithm.adv_estimator=grpo \
            algorithm.use_kl_in_reward=False \
            algorithm.kl_ctrl.kl_coef=0.001 \
            ${DATA_ARGS[@]} \
            ${ACTOR_ARGS[@]} \
            ${ROLLOUT_ARGS[@]} \
            ${TRAINER_ARGS[@]} \
            ${REF_ARGS[@]} \
            ${MODEL_ARGS[@]} \
            ${PLACEHOLD_ARGS[@]} \
            ${MEGATRON_OVERRIDE_TRANSFORMER_CONFIG[@]} \
            $@ 2>&1 | tee verl_demo.log
fi
