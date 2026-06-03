set -o pipefail 

TASK_TYPE="${TASK_TYPE:-master}"
MODEL_PATH=/workspace/models/origin_dir
MODEL_DOWNLOAD_PATH="${MODEL_DOWNLOAD_PATH:-s3://llm_snapshot/mangoz/deepseek_r1/model/Qwen2.5-32B}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-32B}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILZATION="${GPU_MEMORY_UTILZATION:-0.9}"

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

# 安装其他依赖
if [ -n "$INSTALL_JAVIS_REQS" ] && [ -f "$INSTALL_JAVIS_REQS" ]; then
    echo "Installing requirements in $INSTALL_JAVIS_REQS"
    pip install -r $INSTALL_JAVIS_REQS -i https://pypi.bilibili.co/repository/pypi-public/simple
fi

export NCCL_TIMEOUT=7200

# 下载模型
python3 verl/bl_utils/boss_transfer.py \
    --config=/workspace/verl/boss_credentials.toml \
    --source="${MODEL_DOWNLOAD_PATH}" \
    --dest="${MODEL_PATH}"

# 运行vllm
python3 -m vllm.entrypoints.openai.api_server \
    --served-model-name ${MODEL_NAME} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILZATION} \
    --trust-remote-code \
    --max-model-len ${MAX_MODEL_LEN}
