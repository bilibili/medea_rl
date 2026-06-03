#!/bin/bash

# set -x
set -e

trap "echo '检测到 Ctrl+C, 退出脚本'; exit 1" SIGINT
ulimit -n 32768

# Prepare for debugging in vscode-remote
if [ ! -n "$PET_NNODES" ]; then
    source /mnt/group/daiyi/tmp/npu_beats_verl_env_bak.sh
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)"
# 判断设备类型，并设置相应的启动参数
if [ -e "/dev/davinci_manager" ]; then
    _DEVICE_TYPE=npu
    _DEVICE_PER_NODE=16
    _SP_SIZE=1
    _PP="${_PP:-16}"
    _GEN_TP=64
    _ROLLOUT_EP=True
    _USE_TORCH_COMPILE=False
    #TASK_QUEUE_ENABLE，下发优化，图模式设置为1，非图模式设置为2
    export TASK_QUEUE_ENABLE=2
    export HCCL_ASYNC_ERROR_HANDLING=0
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600

    _RAW_MASTER_IP=$(getent hosts $PET_MASTER_ADDR | awk '{print $1}')
    if [[ "$(hostname -I | awk '{print $1}')" == "${_RAW_MASTER_IP}" ]]; then
        export HCCL_SOCKET_IFNAME=$(ip route show default | awk '/default/ {print $5}')
    else
        _ROUTE=$(ip route get ${_RAW_MASTER_IP})
        export HCCL_SOCKET_IFNAME=$(echo ${_ROUTE} | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
    fi

    export GLOO_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}

    echo "HCCL_SOCKET_IFNAME: $HCCL_SOCKET_IFNAME"
    echo "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
else
    export NCCL_NVLS_ENABLE=0
    _DEVICE_TYPE=cuda
    _DEVICE_PER_NODE=8
    _SP_SIZE=1
    _PP="${_PP:-8}"
    _GEN_TP=16
    _ROLLOUT_EP=False
    _USE_TORCH_COMPILE=True
fi

cd $SCRIPT_ROOT/../../..
bash recipe/bilibili/ray_start_with_task.sh

_PROJ_NAME=GRPO_WYFX
_EXP_NAME=$(echo $RANDOM | md5sum | cut -c1-4)

_REMOTE_ONLY="${_REMOTE_ONLY:-1}"
# Don't export ckpts when using remote-dir only.
if [ "${REMOTE_ONLY}" == "0" ]; then
    DEFAULT_LOCAL_DIR=checkpoints/${_PROJ_NAME}/${_EXP_NAME}
    _EXPORT=True
else
    _JOB_ID="${JOB_ID:-$_PROJ_NAME}"
    # "/mnt/group/models" is remote-dir(cubefs).
    DEFAULT_LOCAL_DIR=/mnt/group/models/${_JOB_ID}
    _EXPORT=False
fi

_USE_NNODES="${PET_NNODES:-1}"

_TP=4
_EP=$((_USE_NNODES * _DEVICE_PER_NODE / _PP))
_ETP=1
_CP=1

_OPTIM_OFFLOAD_FRAC=${_OPTIM_OFFLOAD_FRAC:-1.}

_MAX_PROMPT_L="${_MAX_PROMPT_L:-1024}"
_MAX_RES_L="${_MAX_RES_L:-1024}"
_MAX_MODEL_L=$((_MAX_PROMPT_L + _MAX_RES_L))

_actor_ppo_max_token_len=$((_MAX_MODEL_L / _SP_SIZE))
_infer_ppo_max_token_len=$((_MAX_MODEL_L / _SP_SIZE))

MODEL_ARGS=()
MODEL_ARGS+=(
    data.train_files=/mnt/group/datasets/individual-dataset/mangoz/deepseek_r1/data_grpo/train_wyfx_norm11k_cvrlow11k_sft_v6_think.parquet
    data.val_files=/mnt/group/datasets/individual-dataset/mangoz/deepseek_r1/data_grpo/test_wyfx_norm11k_cvrlow11k_sft_v6_think.parquet
    actor_rollout_ref.model.path=/mnt/group/opensource_models/Qwen3-235B-A22B
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
    data.max_prompt_length=${_MAX_PROMPT_L}
    data.max_response_length=${_MAX_RES_L}
    data.train_batch_size=64
    data.shuffle=False
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=8
)

MEGATRON_OVERRIDE_TRANSFORMER_CONFIG=()
MEGATRON_OVERRIDE_TRANSFORMER_CONFIG+=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

if [ -e "/dev/davinci_manager" ]; then
    MEGATRON_OVERRIDE_TRANSFORMER_CONFIG+=(
        ++actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=5
        ++actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=5
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
        ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True
    )
else
    MEGATRON_OVERRIDE_TRANSFORMER_CONFIG+=(
        +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="flex"
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=True
    )
fi

ACTOR_ARGS=()
ACTOR_ARGS+=(
    # actor
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${_actor_ppo_max_token_len}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.2
    actor_rollout_ref.actor.entropy_coeff=0.001
    actor_rollout_ref.actor.use_torch_compile=${_USE_TORCH_COMPILE}
    # optim
    actor_rollout_ref.actor.optim.lr_warmup_steps=1
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_decay_style=cosine
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.clip_grad=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${_OPTIM_OFFLOAD_FRAC} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    # megatron_config
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${_PP}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${_TP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${_ETP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${_CP}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    # checkpoint
    actor_rollout_ref.actor.checkpoint._target_=verl.trainer.config.BiliCheckpointConfig
    # ['model','optimizer','extra','hf_model']
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model']
    actor_rollout_ref.actor.checkpoint.export=${_EXPORT}
    actor_rollout_ref.actor.checkpoint.default_local_dir=${DEFAULT_LOCAL_DIR}
    actor_rollout_ref.actor.checkpoint.delete_after_upload=True
    # policy_loss
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=1
)

ROLLOUT_ARGS=()
ROLLOUT_ARGS+=(
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${_infer_ppo_max_token_len}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85
    actor_rollout_ref.rollout.tensor_model_parallel_size=${_GEN_TP}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${_MAX_MODEL_L}
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.top_k=20
    actor_rollout_ref.rollout.enable_expert_parallel=${_ROLLOUT_EP}
    # valuate
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_k=20
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
)

TRAINER_ARGS=()
TRAINER_ARGS+=(
    trainer.device=${_DEVICE_TYPE}
    trainer.n_gpus_per_node=${_DEVICE_PER_NODE}
    trainer.project_name=${_PROJ_NAME}
    trainer.experiment_name=${_EXP_NAME}
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR}
    trainer.nnodes=${_USE_NNODES}
    trainer.critic_warmup=0
    trainer.logger=['console','tensorboard']
    trainer.save_freq=-1
    trainer.test_freq=10
    trainer.total_epochs=4
    trainer.default_hdfs_dir=null
    # trainer.val_before_train=True
    trainer.val_generations_to_log_to_txt=200
    trainer.summary_dir=/summary_dir
)

REF_ARGS=()
REF_ARGS+=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${_PP}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${_TP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${_ETP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${_CP}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${_infer_ppo_max_token_len}
)

if [ ! -n "$TASK_TYPE" ]; then
    echo "Missing TASK_TYPE!!!"
    exit 0
fi

if [ "${TASK_TYPE}" == "master" ]; then
    python3 -m recipe.bilibili.wyfx.main_ijp \
            --config-path=config \
            --config-name=wyfx_megatron_trainer.yaml \
            algorithm.adv_estimator=grpo \
            algorithm.use_kl_in_reward=False \
            algorithm.kl_ctrl.kl_coef=0.001 \
            actor_rollout_ref.nccl_timeout=3600 \
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
