#!/bin/bash

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)"
cd $SCRIPT_ROOT/..

# --tie-word-embedding \
python -m verl.model_merger merge \
       --backend megatron \
       --use_cpu_initialization \
       --local_dir /mnt/group/models/5256003/global_step_10/actor \
       --target_dir tmp
