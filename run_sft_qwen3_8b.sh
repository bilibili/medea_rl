#!/bin/bash
# Qwen3-8B SFT — 4-node H800S (32 GPUs) — 128K context
#
# Parallelism: TP=4, CP=4, PP=1, DP=2 (32 / 4 / 4 = 2)
# Sequence length: 131072 (128K)
#
# Data: 11 agentic + SWE + OpenCode datasets blended
#
# Data format (JSONL, one JSON per line):
#   {"conversations": [{"from": "User", "value": "..."}, {"from": "Assistant", "value": "..."}]}
# or:
#   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

set -euo pipefail

# Activate environment if exists
if [ -f /opt/venv/bin/activate ]; then
    source /opt/venv/bin/activate
fi

# Find Megatron-Bridge installation
if [ -d /workspace/Megatron-Bridge ]; then
    MBRIDGE_DIR=/workspace/Megatron-Bridge
elif [ -d /opt/Megatron-Bridge ]; then
    MBRIDGE_DIR=/opt/Megatron-Bridge
else
    # pip installed — find via python
    MBRIDGE_DIR=$(python -c "import megatron.bridge; import os; print(os.path.dirname(os.path.dirname(megatron.bridge.__file__)))" 2>/dev/null || echo "")
    if [ -z "$MBRIDGE_DIR" ]; then
        echo "ERROR: Cannot find Megatron-Bridge installation"
        exit 1
    fi
fi

# Find Megatron-LM
if [ -d "${MBRIDGE_DIR}/3rdparty/Megatron-LM" ]; then
    export PYTHONPATH=${MBRIDGE_DIR}/3rdparty/Megatron-LM/:${MBRIDGE_DIR}/src:${PYTHONPATH:-}
else
    export PYTHONPATH=${MBRIDGE_DIR}/src:${PYTHONPATH:-}
fi

cd ${MBRIDGE_DIR}

# Data blend: 11 files with equal weights (each gets ~9.09% weight)
DATA_BLEND="["
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-Agentic-v2/data/interactive_agent.jsonl', 1.0),"
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-Agentic-v2/data/search.jsonl', 1.0),"
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-Agentic-v2/data/tool_calling.jsonl', 1.0),"
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-SWE-v2/data/agentless.jsonl', 1.0),"
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-SWE-v2/data/swe.jsonl', 1.0),"
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/agent_skills_question_tool/data.jsonl', 1.0),"
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/bash_only_tool_skills/data.jsonl', 1.0),"
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/agent_skills/data.jsonl', 1.0),"
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/bash_only_tool/data.jsonl', 1.0),"
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/general/data.jsonl', 1.0),"
DATA_BLEND+="('/mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/question_tool/data.jsonl', 1.0)"
DATA_BLEND+="]"

# Prepare data directory: symlink all JSONL files into a single directory
DATA_DIR=/tmp/sft_data
mkdir -p ${DATA_DIR}
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-Agentic-v2/data/interactive_agent.jsonl ${DATA_DIR}/01_interactive_agent.jsonl
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-Agentic-v2/data/search.jsonl ${DATA_DIR}/02_search.jsonl
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-Agentic-v2/data/tool_calling.jsonl ${DATA_DIR}/03_tool_calling.jsonl
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-SWE-v2/data/agentless.jsonl ${DATA_DIR}/04_agentless.jsonl
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-SWE-v2/data/swe.jsonl ${DATA_DIR}/05_swe.jsonl
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/agent_skills_question_tool/data.jsonl ${DATA_DIR}/06_agent_skills_question_tool.jsonl
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/bash_only_tool_skills/data.jsonl ${DATA_DIR}/07_bash_only_tool_skills.jsonl
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/agent_skills/data.jsonl ${DATA_DIR}/08_agent_skills.jsonl
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/bash_only_tool/data.jsonl ${DATA_DIR}/09_bash_only_tool.jsonl
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/general/data.jsonl ${DATA_DIR}/10_general.jsonl
ln -sf /mnt/group/mangoz/agent/data/Nemotron-SFT-OpenCode-v1/question_tool/data.jsonl ${DATA_DIR}/11_question_tool.jsonl

# Make HF model path available locally (no internet on training nodes)
# Patch: monkey-patch AutoBridge to use local path before importing recipe
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Create a wrapper that patches the model path before running
cat > /tmp/run_sft_wrapper.py << 'PYTHON_EOF'
import sys
import os

# Patch: make "Qwen/Qwen3-8B" resolve to local path
LOCAL_MODEL_PATH = "/mnt/group/opensource_models/Qwen3-8B"

# Monkey-patch AutoConfig to use local path
import transformers
_original_from_pretrained = transformers.AutoConfig.from_pretrained.__func__

@classmethod
def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "Qwen/Qwen3-8B":
        pretrained_model_name_or_path = LOCAL_MODEL_PATH
    return _original_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

transformers.AutoConfig.from_pretrained = _patched_from_pretrained

# Now run the actual training script
sys.argv = [sys.argv[0]] + sys.argv[1:]
exec(open(os.path.join(os.environ.get("MBRIDGE_DIR", "/opt/Megatron-Bridge"), "scripts/training/run_recipe.py")).read())
PYTHON_EOF

CUDA_DEVICE_MAX_CONNECTIONS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
NCCL_NVLS_ENABLE=0 \
WANDB_MODE=disabled \
MBRIDGE_DIR=${MBRIDGE_DIR} \
python -m torch.distributed.run \
--nproc_per_node=8 \
/tmp/run_sft_wrapper.py \
--recipe qwen3_8b_sft_config \
--dataset llm-finetune-preloaded \
--hf_path /mnt/group/opensource_models/Qwen3-8B \
--seq_length 131072 \
dataset.dataset_root=${DATA_DIR} \
dataset.num_workers=8 \
checkpoint.save=/mnt/group/mangoz/nemo_sft/saves/qwen3_8b_sft_128k \
checkpoint.save_interval=500 \
train.train_iters=1000000 \
train.global_batch_size=8 \
train.micro_batch_size=1 \
optimizer.lr=1.0e-6 \
optimizer.min_lr=1.0e-6 \
optimizer.use_distributed_optimizer=False \
logger.tensorboard_dir=/workspace/summary \
logger.log_interval=1 \
model.seq_length=131072 \
model.recompute_granularity=full \
model.recompute_method="uniform" \
model.recompute_num_layers=2 \
model.tensor_model_parallel_size=4 \
model.pipeline_model_parallel_size=1 \
model.context_parallel_size=4 \
model.cp_comm_type=a2a \
model.sequence_parallel=True \
model.cross_entropy_loss_fusion=False \
model.calculate_per_token_loss=True \
ddp.use_distributed_optimizer=False \
ddp.overlap_grad_reduce=True \
ddp.overlap_param_gather=True \
ddp.average_in_collective=False \
ddp.grad_reduce_in_fp32=False
