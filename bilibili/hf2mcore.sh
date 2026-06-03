#!/bin/bash

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)"

# _HF=/mnt/group/daiyi/alluxio/Qwen3-235B-A22B
_HF=/daiyi_file/ai/llm/daiyi/alluxio/Qwen3-235B-A22B
_MCORE=/mnt/group/daiyi/alluxio/Qwen3-235B-A22B-mcore-torchrun

echo "Listing bilibili-env parameters:"
echo "PET_NNODES: $PET_NNODES"
echo "PET_NPROC_PER_NODE: $PET_NPROC_PER_NODE"
echo "PET_NODE_RANK: $PET_NODE_RANK"
echo "PET_MASTER_ADDR: $PET_MASTER_ADDR"
echo "PET_MASTER_PORT: $PET_MASTER_PORT"
echo "TASK_TYPE: $TASK_TYPE"
echo

if [ -e "/dev/davinci_manager" ]; then
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600
    # 各个机子对应网卡名
    # _RAW_MASTER_IP=$(getent hosts $PET_MASTER_ADDR | awk '{print $1}')

    # if [[ "$(hostname -I | awk '{print $1}')" == "${_RAW_MASTER_IP}" ]]; then
    #     export HCCL_SOCKET_IFNAME=$(ip route show default | awk '/default/ {print $5}')
    # else
    #     _ROUTE=$(ip route get ${_RAW_MASTER_IP})
    #     export HCCL_SOCKET_IFNAME=$(echo ${_ROUTE} | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
    # fi
else
    echo
fi

start_s=$(date +%s)

if true; then
    cd $SCRIPT_ROOT/..
    torchrun --master_addr ${PET_MASTER_ADDR} \
            --nproc_per_node ${PET_NPROC_PER_NODE} \
            --nnodes ${PET_NNODES} \
            --node_rank ${PET_NODE_RANK} \
            scripts/converter_hf_to_mcore.py \
            --hf_model_path ${_HF} \
            --output_path ${_MCORE} \
            --use_cpu_initialization    # Only work for MoE models
else
    cd $SCRIPT_ROOT/../scripts
    python converter_hf_to_mcore.py --hf_model_path ${_HF} --output_path ${_MCORE} --use_cpu_initialization
fi

end_s=$(date +%s)
diff_m=$(((end_s - start_s) / 60))
printf "Elapsed: %d minutes.\n" "$diff_m"
