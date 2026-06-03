#!/bin/bash

# set -x
# set -o pipefail
set -eu
trap "echo '检测到 Ctrl+C, 退出脚本'; exit 1" SIGINT
ray stop --force
ulimit -n 32768
export RAY_DEDUP_LOGS=0

SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"
echo "Listing bilibili-env parameters:"
echo "RAY SCRIPT_ROOT: $SCRIPT_ROOT"
echo "PET_NNODES: $PET_NNODES"
echo "PET_NPROC_PER_NODE: $PET_NPROC_PER_NODE"
echo "PET_NODE_RANK: $PET_NODE_RANK"
echo "PET_MASTER_ADDR: $PET_MASTER_ADDR"
echo "PET_MASTER_PORT: $PET_MASTER_PORT"
echo "TASK_TYPE: $TASK_TYPE"
echo

start_ray() {
    local max_retries=10
    local retry_delay=10
    local retries=0
    until "$@"; do
        echo "执行命令失败: $*"
        retries=$((retries+1))
        if [ $retries -ge $max_retries ]; then
            echo "超过最大重试次数 ($max_retries)，退出脚本"
            exit 1
        fi
        echo "重试 ($retries/$max_retries) ，等待 ${retry_delay} 秒..."
        sleep $retry_delay
    done
}

RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8260}"
RAY_PORT="${RAY_PORT:-16344}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-30002}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-39999}"
# Preset port for following params: `--dashboard-agent-grpc-port`,`--runtime-env-agent-port`,`--metrics-export-port`,
# or we might encounter with issues like `Ray component worker_ports is trying to use a port number xxx that is used by other components.`
# https://docs.ray.io/en/latest/ray-core/configure.html
_RAY_PORT_ARGS=(
    "--min-worker-port=$RAY_MIN_WORKER_PORT"
    "--max-worker-port=$RAY_MAX_WORKER_PORT"
    "--dashboard-host=0.0.0.0"
    "--dashboard-port=${RAY_DASHBOARD_PORT}"
    "--dashboard-agent-grpc-port=$((RAY_DASHBOARD_PORT+2))"
    "--runtime-env-agent-port=$((RAY_DASHBOARD_PORT+3))"
    "--metrics-export-port=$((RAY_DASHBOARD_PORT+4))"
)

_ALL_DEVICE=$(seq -s, 0 $((PET_NPROC_PER_NODE - 1)))
# 如果在ray的日志中发生spill（memory和disk的存储交换现象），可以调大object-store-memory，其单位为字节
# Object store memory: memory used when your application creates objects in the object store via ray.put
# and when it returns values from remote functions. By default, when starting an instance,
# Ray reserves 30% of available memory.
# _OSM=""
_OSM="--object-store-memory=15000000000"
_RES="\"pet_node_rank_${PET_NODE_RANK}\":1"
# 判断设备类型，并设置相应的启动参数
if [ "${TASK_TYPE}" == "master" ]; then
    _RES="\"master_node\":1,$_RES"
fi
if [ -e "/dev/davinci_manager" ]; then
    export ASCEND_RT_VISIBLE_DEVICES=$_ALL_DEVICE
    _RES="\"NPU\":${PET_NPROC_PER_NODE},$_RES"
    _RES_ARG="$_OSM --resources={$_RES}"
else
    export CUDA_VISIBLE_DEVICES=$_ALL_DEVICE
    _RES_ARG="$_OSM --num-gpus ${PET_NPROC_PER_NODE} --resources={$_RES}"
fi

echo "Listing ray parameters:"
echo "_ALL_DEVICE: $_ALL_DEVICE"
echo "_RES_ARG: $_RES_ARG"
echo "_RAY_PORT_ARGS: ${_RAY_PORT_ARGS[@]}"
echo
# start ray
if [ "${TASK_TYPE}" == "master" ]; then
    echo "Bringing up ray pool as master"
    # 显式指定 dashboard agent gRPC 端口，避免与 worker_ports 冲突
    _CMD="ray start --head --port ${RAY_PORT} ${_RAY_PORT_ARGS[@]} ${_RES_ARG}"
    echo "Executing: $_CMD"

    start_ray $_CMD
    echo "master started, waiting for workers"
    while true; do
        sleep 10
        ACTIVE_NODES=$(ray status | awk '/Active:/,/Pending:/' | grep -E "node_" | wc -l)
        if [ ${ACTIVE_NODES} -ge ${PET_NNODES} ]; then
            echo "所有 worker 已就绪，开始训练"
            break
        fi
        echo "等待 worker 加入... current ready worker = ${ACTIVE_NODES}, expected = ${PET_NNODES}"
    done

    ray status
    # start task
    echo
    if [[ -v BILI_RAY_TASK ]]; then
        echo "Start task: $BILI_RAY_TASK"
        bash $BILI_RAY_TASK
    else
        echo "No task, just start ray."
    fi
else
    echo "Bringing up ray pool as worker"
    echo "Checking if master at ${PET_MASTER_ADDR}:${RAY_PORT} is ready..."
    while ! timeout 5 nc -z "${PET_MASTER_ADDR}" "${RAY_PORT}"; do
        echo "Waiting for master at ${PET_MASTER_ADDR}:${RAY_PORT}..."
        sleep 2
    done

    _CMD="ray start --address=${PET_MASTER_ADDR}:${RAY_PORT} ${_RAY_PORT_ARGS[@]} ${_RES_ARG}"
    echo "Executing: $_CMD"

    start_ray $_CMD
    sleep infinity
fi
