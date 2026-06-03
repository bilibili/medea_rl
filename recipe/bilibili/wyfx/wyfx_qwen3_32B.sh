#!/bin/bash

# set -x
set -e

trap "echo '检测到 Ctrl+C, 退出脚本'; exit 1" SIGINT
ulimit -n 32768
# Prepare for debugging in vscode-remote
if [ ! -n "$PET_NNODES" ]; then
    source /mnt/group/daiyi/tmp/npu_beats_verl_env_bak.sh
fi

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

    export HCCL_SOCKET_IFNAME=$(ip route get $(getent hosts $PET_MASTER_ADDR | awk '{print $1}') \
                                | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
    export GLOO_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}

    echo "HCCL_SOCKET_IFNAME: $HCCL_SOCKET_IFNAME"
    echo "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
else
    _DEVICE_TYPE=cuda
    _DEVICE_PER_NODE=8
    _SP_SIZE=1
    _USE_TORCH_COMPILE=True
fi

cd $SCRIPT_ROOT/../../..
bash recipe/bilibili/ray_start_with_task.sh

_PROJ_NAME=GRPO_WYFX
_EXP_NAME=$(echo $RANDOM | md5sum | cut -c1-4)

_USE_NNODES="${PET_NNODES:-1}"
_MAX_PROMPT_L=8192
_MAX_RES_L=4096
_MAX_MODEL_L=$((_MAX_PROMPT_L + _MAX_RES_L))

_actor_ppo_max_token_len=$((_MAX_MODEL_L / _SP_SIZE))
_infer_ppo_max_token_len=$((_MAX_MODEL_L / _SP_SIZE))

MODEL_ARGS=()
MODEL_ARGS+=(
    data.train_files=/mnt/group/datasets/individual-dataset/mangoz/deepseek_r1/data_grpo/train_wyfx_norm11k_cvrlow11k_sft_v6_think.parquet
    data.val_files=/mnt/group/datasets/individual-dataset/mangoz/deepseek_r1/data_grpo/test_wyfx_norm11k_cvrlow11k_sft_v6_think.parquet
    actor_rollout_ref.model.path=/mnt/group/daiyi/tmp/hf/qwen3_32b_sft_beats_157185_ep5
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
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

ACTOR_ARGS=()
ACTOR_ARGS+=(
    # actor
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${_actor_ppo_max_token_len}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.2
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.entropy_coeff=0.001
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${_SP_SIZE}
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.use_torch_compile=${_USE_TORCH_COMPILE}
    # optim
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.warmup_style=cosine
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1
    # fsdp_config
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    # checkpoint
    actor_rollout_ref.actor.checkpoint._target_=verl.trainer.config.BiliCheckpointConfig
    actor_rollout_ref.actor.checkpoint.save_contents=['model']
    actor_rollout_ref.actor.checkpoint.export=True
    actor_rollout_ref.actor.checkpoint.default_local_dir=checkpoints/${_PROJ_NAME}/${_EXP_NAME}
    actor_rollout_ref.actor.checkpoint.delete_after_upload=False
    # policy_loss
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=1
)

ROLLOUT_ARGS=()
ROLLOUT_ARGS+=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
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
    # trainer.val_before_train=True
    trainer.val_generations_to_log_to_txt=200
    trainer.summary_dir=/summary_dir
)

REF_ARGS=()
REF_ARGS+=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${_infer_ppo_max_token_len}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

if [ ! -n "$TASK_TYPE" ]; then
    echo "Missing TASK_TYPE!!!"
    exit 0
fi

if [ "${TASK_TYPE}" == "master" ]; then
    python3 -m recipe.bilibili.wyfx.main_ijp \
            --config-path=config \
            --config-name=wyfx_trainer.yaml \
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
            $@ 2>&1 | tee verl_demo.log
fi
