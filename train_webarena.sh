#!/bin/bash
set -eux

# === Config from env vars ===
AGENTGYM_TASK="${AGENTGYM_TASK:-webarena}"
AGENTGYM_ENV_PORT="${AGENTGYM_ENV_PORT:-36005}"
MODEL_PATH="${MODEL_PATH:-/mnt/group/opensource_models/Qwen3-8B}"
SAVE_DIR="${SAVE_DIR:-/workspace/saves/agentgym_rl}"

WORKSPACE=/workspace/agentgym-rl

# === 1. Clone AgentGym-RL ===
if [ ! -d "${WORKSPACE}/.git" ]; then
    rm -rf ${WORKSPACE}
    git clone --depth 1 https://github.com/WooooDyy/AgentGym-RL.git ${WORKSPACE}
fi
cd ${WORKSPACE}

# === 2. Install AgentGym-RL (skip vllm version constraint, use existing env) ===
# Patch pyproject.toml to remove strict vllm/torch version pins (we use the image's versions)
sed -i 's/"vllm<=0.6.3"/"vllm"/g' AgentGym-RL/pyproject.toml
sed -i "s/'torch==2.4.0'//g" AgentGym-RL/pyproject.toml

pip install -e AgentGym-RL/ --no-deps -i https://pypi.bilibili.co/repository/pypi-public/simple
# Install missing deps that aren't already in the image
pip install hydra-core codetiming dill pylatexenc tensordict -i https://pypi.bilibili.co/repository/pypi-public/simple

# Install agentenv client
cd ${WORKSPACE}
git clone --depth 1 https://github.com/WooooDyy/AgentGym.git ${WORKSPACE}/AgentGym 2>/dev/null || true
if [ -d "${WORKSPACE}/AgentGym/agentenv" ]; then
    pip install -e ${WORKSPACE}/AgentGym/agentenv/ -i https://pypi.bilibili.co/repository/pypi-public/simple 2>/dev/null || true
fi

# === 3. Download training data ===
DATA_DIR=${WORKSPACE}/AgentGym-RL/AgentItemId
mkdir -p ${DATA_DIR}
if [ ! -f "${DATA_DIR}/${AGENTGYM_TASK}_train.json" ]; then
    echo "Downloading AgentGym-RL-Data-ID from HuggingFace..."
    pip install huggingface_hub -i https://pypi.bilibili.co/repository/pypi-public/simple
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='AgentGym/AgentGym-RL-Data-ID', repo_type='dataset', local_dir='${DATA_DIR}')
"
fi

# === 4. Start AgentGym environment server (background) ===
echo "Starting AgentGym environment server for task: ${AGENTGYM_TASK} on port ${AGENTGYM_ENV_PORT}..."
if [ "${AGENTGYM_TASK}" == "webarena" ]; then
    echo "WARNING: WebArena requires external environment setup (docker containers for shopping/forum/gitlab/cms)."
    echo "If you have a pre-deployed server, set AGENTGYM_ENV_ADDR env var."
    echo "Attempting to start agentenv server..."
    cd ${WORKSPACE}/AgentGym
    python -m agentenv.server --task ${AGENTGYM_TASK} --port ${AGENTGYM_ENV_PORT} &
    ENV_PID=$!
    cd ${WORKSPACE}
    sleep 15
else
    cd ${WORKSPACE}/AgentGym
    python -m agentenv.server --task ${AGENTGYM_TASK} --port ${AGENTGYM_ENV_PORT} &
    ENV_PID=$!
    cd ${WORKSPACE}
    sleep 10
fi

ENV_SERVER_URL="${AGENTGYM_ENV_ADDR:-http://127.0.0.1:${AGENTGYM_ENV_PORT}}"

# === 5. Start RL Training ===
echo "Starting AgentGym-RL training..."
cd ${WORKSPACE}/AgentGym-RL

mkdir -p ${SAVE_DIR}

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=15 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_file=AgentItemId/${AGENTGYM_TASK}_train.json \
    data.train_batch_size=32 \
    data.max_prompt_length=750 \
    data.max_response_length=14098 \
    actor_rollout_ref.agentgym.task_name=${AGENTGYM_TASK} \
    actor_rollout_ref.agentgym.env_addr=${ENV_SERVER_URL} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_tokens=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.rollout_log_dir=${SAVE_DIR}/rollout_logs \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.project_name=AgentGym-RL \
    trainer.experiment_name=${AGENTGYM_TASK}_qwen3_8b \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.total_epochs=25 \
    trainer.logger=['console','tensorboard'] \
    trainer.summary_dir=/summary_dir

status=$?

# Cleanup env server
if [ -n "${ENV_PID:-}" ]; then
    kill ${ENV_PID} 2>/dev/null || true
fi

exit $status
