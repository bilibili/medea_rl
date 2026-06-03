set -o pipefail 

TASK_TYPE="${TASK_TYPE:-master}"
MODEL_PATH="${MODEL_PATH:-s3://llm_snapshot/mangoz/deepseek_r1/model/mcore_qwen2_5_32b_base_2943638_ep1}"
DATA_PATH_PREFIX="${DATA_PATH_PREFIX:-s3://llm-data/benjamin/datasets/Open-Reasoner}"

ARG_PACKS="${ARG_PACKS:-}"
USE_HYDRA=1

#### config args, 
if [ -z "$ARG_PACKS" ]; then
    echo "ARG_PACKS is not set, no extra parameter file provided."
    arg_pack_files=()
else
    # 如果包含逗号，则按照逗号分割成数组；否则就是单个文件
    IFS=',' read -r -a arg_pack_files <<< "$ARG_PACKS"
fi

_EXTRA_ARGS=()
# 遍历每个文件，从中按行读取参数，放到 _EXTRA_ARGS 中
for file in "${arg_pack_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Args pack file not found: $file"
        continue
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        # 忽略空行或以 # 开头的注释行
        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        # 如果行以 "--" 开头，则转换为 "++"（Hydra 用 ++ 来 add/override 参数）
        if [ "${USE_HYDRA}" -ne "0" ] && [[ $line == --* ]]; then
            _EXTRA_ARGS+=( "++${line:2}" )
        else
            _EXTRA_ARGS+=( "$line" )
        fi
    done < "$file"
done

# 处理命令行传入的参数
for arg in "$@"; do
    # 如果以 "--" 开头，则转换为 "++"
    if [ "${USE_HYDRA}" -ne "0" ] && [[ $arg == --* ]]; then
        _EXTRA_ARGS+=( "++${arg:2}" )
    else
        _EXTRA_ARGS+=( "$arg" )
    fi
done

echo "Parsed _EXTRA_ARGS :"
for arg in "${_EXTRA_ARGS[@]}"; do
    echo "  $arg"
done
echo "================================="

# 安装verl
pip install -e . -i https://pypi.bilibili.co/repository/pypi-public/simple

# 安装其他依赖
if [ -n "$INSTALL_JAVIS_REQS" ] && [ -f "$INSTALL_JAVIS_REQS" ]; then
    echo "Installing requirements in $INSTALL_JAVIS_REQS"
    pip install -r $INSTALL_JAVIS_REQS -i https://pypi.bilibili.co/repository/pypi-public/simple
fi


export VLLM_ATTENTION_BACKEND=XFORMERS

# 启动ray
bash setup_ray.sh

if [ "${TASK_TYPE}" == "master" ]; then

    export WANDB_MODE="offline"
    # export WANDB_API_KEY='759a057f637aefd0dd03cee635ce85109f9664df'

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="${DATA_PATH_PREFIX}"/train_coldstart.parquet \
        data.val_files="${DATA_PATH_PREFIX}"/test_coldstart.parquet \
        data.train_batch_size=32 \
        data.val_batch_size=32 \
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
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.top_p=1 \
        actor_rollout_ref.rollout.repetition_penalty=1.2 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name='GRPO_ORZ' \
        trainer.experiment_name='Qwen2.5-32B-ZERO' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=4 \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=/model_dir/GRPO_ORZ/Qwen2.5-32B-ZERO \
        trainer.save_freq=200 \
        trainer.test_freq=50 \
        trainer.total_epochs=1 "${_EXTRA_ARGS[@]}" \
        +hydra.job.config.allow_unknown_args=True 2>&1 | tee grpo.log
else
    sleep infinity
fi


# verl/workers/fsdp_workers.py
# verl/workers/actor/dp_actor.py

# 以actor的更新为例，
# 一个batch的数据大小（也就是prompt的数量）为train_batch_size，rollout.n后的数据大小为 train_batch_size * rollout.n， 也就是这些数据会用来更新actor policy

# policy更新中，先将 batch = train_batch_size * rollout.n 的数据按照 ppo_mini_batch_size 进行切分为 minibatch
# 每一份 minibatch 再按照 ppo_micro_batch_size_per_gpu 进行切分，计算 gradient_accumulation = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu
# 每一份 minibatch 累积完loss后，更新一次 policy

# ref和critic也是同理的做法