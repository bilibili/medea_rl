#!/bin/bash

set -eux

ACTUAL_ENTRY="${ACTUAL_ENTRY:-run.sh}"
COMMIT="${COMMIT:-}"
NSYS="${NSYS:-}"
NSYS="${NSYS^^}"
NSYS_DELAY="${NSYS_DELAY:-180}"
PET_NODE_RANK="${PET_NODE_RANK:-0}"
VERSION="${VERSION:-}"
JAVIS_DEV_DIR="${JAVIS_DEV_DIR:-/workspace/verl}"
JAVIS_DEV_GIT="${JAVIS_DEV_GIT:-git@git.bilibili.co:zhengbin/verl.git}"
USE_HYDRA="${USE_HYDRA:-1}"
INSTALL_PROJ="${INSTALL_PROJ:-0}"   # whether `pip install -e .``
INSTALL_PIP_REQS="${INSTALL_PIP_REQS:-}"   # the path of custom requirements.txt to install
ARG_PACKS="${ARG_PACKS:-}"

python_module_check()
{
  if python -c 'from importlib import util;exit(None != util.find_spec("'$1'"))' >/dev/null 2>&1
  then
      # module is not installed.
      echo "0"
  else
      # module is installed.
      echo "1"
  fi
}

if [ ! -d "${JAVIS_DEV_DIR}/.git" ]; then
  # NOTE: 当 Javis 仓库不存在时对其进行初始化 (这使得 BVAC 任务可以直接使用标
  #       准镜像, 而不是已经内嵌 Javis 仓库和产物的发布镜像).

  rm -f -r "${JAVIS_DEV_DIR}"
  mkdir -p "${JAVIS_DEV_DIR}"

  git -c advice.detachedHead="false" \
      clone \
      --branch="main" \
      --depth="1" \
      "${JAVIS_DEV_GIT}" \
      "${JAVIS_DEV_DIR}"
fi

cd -P "${JAVIS_DEV_DIR}"
_VERSION=""
# NOTE: 可以通过环境变量 `VERSION` 或 `COMMIT` 来指定 Javis 仓库的版本, 其中
#       `VERSION` 具有更高的优先级并且是推荐的用法, 指定的版本可以是 "提交" (
#       即 SHA1 Commit), "分支" (即 Branch) 或 "标签" (即 Tag), 缺省情况下使用
#       "master".
if [ ! -z "${VERSION}" ]; then
  _VERSION="${VERSION}"
elif [ ! -z "${COMMIT}" ]; then
  _VERSION="${COMMIT}"
else
  _VERSION="master"
fi

_SHA1_HEAD="$(git rev-parse --verify "HEAD")"
[ ! -z "${_SHA1_HEAD}" ]

_SHA1_REMOTE="$(git ls-remote --refs "origin" "${_VERSION}" | cut -f "1")"
_SHA1_REMOTE="${_SHA1_REMOTE:-${_VERSION}}"

if [ "${_SHA1_REMOTE}" != "${_SHA1_HEAD}" ]; then
  # NOTE: 请求的 Javis 仓库版本与当前不符, 尝试在切换版本时保持可能存在的本地
  #       修改及忽略产物等.

  git fetch --depth="1" "origin" "${_VERSION}"

  _SHA1_FETCH_HEAD="$(git rev-parse --verify "FETCH_HEAD")"
  [ ! -z "${_SHA1_FETCH_HEAD}" ]

  if [ "${_SHA1_FETCH_HEAD}" != "${_SHA1_HEAD}" ]; then
    _WITH_MODIFICATIONS="0"

    if [ ! -z "$(git status --porcelain)" ]; then
      _WITH_MODIFICATIONS="1"
    fi

    if [ "${_WITH_MODIFICATIONS}" -ne "0" ]; then
      git stash push -u --quiet
    fi

    git checkout --force --detach "${_SHA1_FETCH_HEAD}"

    if [ "${_WITH_MODIFICATIONS}" -ne "0" ]; then
      if ! git stash pop --quiet; then
        git stash drop --quiet
        git reset --hard
        git clean -d -f
      fi
    fi
  fi
fi

result=`python_module_check torch_npu`
if [ $result == 0 ]
then
  echo "Not in NPU env."
else
  echo "In NPU env."
fi
# NOTE: 以开发模式安装 当前repo项目 (这可能是一个覆盖安装, 但是如果此前的构建过程
#       产物在仓库路径中, 并且版本变更并未引入需要重新构建 C 扩展的修改, 那么
#       安装过程本身并不会花费太多时间).

if [ "${INSTALL_PROJ}" -ne "0" ]; then
  pip install -e . -i https://pypi.bilibili.co/repository/pypi-public/simple
fi

# 安装其他依赖
if [ -n "$INSTALL_PIP_REQS" ] && [ -f "$INSTALL_PIP_REQS" ]; then
    echo "Installing requirements in $INSTALL_PIP_REQS"
    pip install -r $INSTALL_PIP_REQS -i https://pypi.bilibili.co/repository/pypi-public/simple
fi

_ARGUMENTS=()
# NOTE: 调用入口脚本, 可以选择使用 `nsys` 来开启性能调试.
if [ "${NSYS}" == "ON" ] && [ "${PET_NODE_RANK}" -eq "0" ]; then
  _ARGUMENTS+=(
      nsys profile
      --trace-fork-before-exec="true"
      --kill="sigkill"
      --duration="30"
      --trace="cuda,cudnn,cublas,osrt,nvtx"
      --gpu-metrics-device="0"
      --cuda-memory-usage="true"
      --delay="${NSYS_DELAY}"
      --output="/result_dir/nsys_report"
  )
fi

_ARGUMENTS+=(
    bash "${ACTUAL_ENTRY}"
)

#### Parse ARG_PACKS and BVAC -T args
if [ -z "$ARG_PACKS" ]; then
    echo "ARG_PACKS is not set, no extra parameter file provided."
    arg_pack_files=()
else
    # 如果包含逗号，则按照逗号分割成数组；否则就是单个文件
    IFS=',' read -r -a arg_pack_files <<< "$ARG_PACKS"
fi
unset ARG_PACKS

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

exec "${_ARGUMENTS[@]}" "${_EXTRA_ARGS[@]}"
