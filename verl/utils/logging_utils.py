# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import torch


def set_basic_config(level):
    """
    This function sets the global logging format and level. It will be called when import verl
    """
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=level)


def log_to_file(string):
    print(string)
    if os.path.isdir('logs'):
        with open(f'logs/log_{torch.distributed.get_rank()}', 'a+') as f:
            f.write(string + '\n')

class LogCollector:
    """简单的日志收集器"""
    def __init__(self, prefix=None):
        self.logs = []
        self.prefix = prefix
        
    def log(self, message):
        """添加日志"""
        if self.prefix:
            self.logs.append("[" + self.prefix + "]" + message)
        else:
            self.logs.append(message)
        
    def get_logs(self):
        """获取所有日志并用换行符连接"""
        return "\n".join(self.logs)
    
    def clear(self):
        """清空所有日志"""
        self.logs = []
