#!/bin/bash
set -euo pipefail

# 捕获 Ctrl+C 信号
trap "echo '检测到 Ctrl+C，退出脚本'; exit 1" SIGINT

# 解析命令行参数，检测是否包含 --no-wait 标志
NO_WAIT=false
for arg in "$@"; do
    if [ "$arg" == "--no-wait" ]; then
        NO_WAIT=true
        break
    fi
done

echo "Listing bvac nodes parameters"
echo "$PET_NNODES"
echo "$PET_NPROC_PER_NODE"
echo "$PET_NODE_RANK"
echo "$PET_MASTER_ADDR"
echo "$PET_MASTER_PORT"

RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8260}"
RAY_PORT="${RAY_PORT:-6344}"
TASK_TYPE="${TASK_TYPE:-master}"
EXPECTED_NODES="${PET_NNODES:-1}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-10002}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-19999}"

# 判断设备类型，并设置相应的启动参数
if [ -e "/dev/davinci_manager" ]; then
    DEVICE_TYPE="NPU"
    resource_arg="--resources={\"NPU\":${PET_NPROC_PER_NODE},\"pet_node_rank_${PET_NODE_RANK}\":1}"
else
    DEVICE_TYPE="GPU"
    resource_arg="--num-gpus ${PET_NPROC_PER_NODE} --resources={\"pet_node_rank_${PET_NODE_RANK}\":1}"
fi

# 设置重试参数
MAX_RETRIES=10
RETRY_DELAY=10

start_ray() {
    local retries=0
    until "$@"; do
        echo "执行命令失败: $*"
        retries=$((retries+1))
        if [ $retries -ge $MAX_RETRIES ]; then
            echo "超过最大重试次数 ($MAX_RETRIES)，退出脚本"
            exit 1
        fi
        echo "重试 ($retries/$MAX_RETRIES) ，等待 ${RETRY_DELAY} 秒..."
        sleep $RETRY_DELAY
    done
}


# Preset port for following params: `--dashboard-agent-grpc-port`,`--runtime-env-agent-port`,`--metrics-export-port`,
# or we might encounter with issues like `Ray component worker_ports is trying to use a port number xxx that is used by other components.`
# https://docs.ray.io/en/latest/ray-core/configure.html
_RAY_PORT_ARGS=(
    "--min-worker-port=$RAY_MIN_WORKER_PORT"
    "--max-worker-port=$RAY_MAX_WORKER_PORT"
    "--dashboard-host=0.0.0.0"
    "--dashboard-port=${RAY_DASHBOARD_PORT}"
    "--dashboard-grpc-port=$((RAY_DASHBOARD_PORT+1))"
    "--dashboard-agent-grpc-port=$((RAY_DASHBOARD_PORT+2))"
    "--runtime-env-agent-port=$((RAY_DASHBOARD_PORT+3))"
    "--metrics-export-port=$((RAY_DASHBOARD_PORT+4))"
)

if [ "${TASK_TYPE}" == "master" ]; then
    echo "Bringing up ray pool as master"
    # 显式指定 dashboard agent gRPC 端口，避免与 worker_ports 冲突
    echo "Executing: ray start --head --port ${RAY_PORT} ${_RAY_PORT_ARGS[@]} ${resource_arg}"
    start_ray ray start --head --port "${RAY_PORT}" "${_RAY_PORT_ARGS[@]}" ${resource_arg}

    if [ "$NO_WAIT" = true ]; then
        echo "--no-wait 参数检测到，跳过等待 worker 检查"
    else
        echo "master started, waiting for workers"
        sleep 5
        while true; do
            sleep 2
            ACTIVE_NODES=$(ray status | awk '/Active:/,/Pending:/' | grep -E "node_" | wc -l)
            if [ ${ACTIVE_NODES} -ge ${EXPECTED_NODES} ]; then
                echo "所有 worker 已就绪，开始训练"
                break
            fi
            echo "等待 worker 加入... current ready worker = ${ACTIVE_NODES}, expected = ${EXPECTED_NODES}"
        done
    fi
else
    echo "Bringing up ray pool as worker"
    if [ "$NO_WAIT" = true ]; then
        echo "--no-wait 参数检测到，跳过 master 可用性检查"
    else
        echo "Checking if master at ${PET_MASTER_ADDR}:${RAY_PORT} is ready..."
        while ! timeout 5 nc -z "${PET_MASTER_ADDR}" "${RAY_PORT}"; do
            echo "Waiting for master at ${PET_MASTER_ADDR}:${RAY_PORT}..."
            sleep 2
        done
    fi
    echo "Executing: ray start --address=${PET_MASTER_ADDR}:${RAY_PORT} ${_RAY_PORT_ARGS[@]} ${resource_arg}"
    start_ray ray start --address="${PET_MASTER_ADDR}:${RAY_PORT}" "${_RAY_PORT_ARGS[@]}" ${resource_arg}
fi

ray status
