# Copyright 2025 bilibili Ltd. and/or its affiliates
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

import os
import shutil
import logging
import subprocess


def list2file(src_prefix: str, dst_prefix: str, lst: list[str]) -> int:
    success = 0
    for i in lst:
        src = os.path.join(src_prefix, i)
        dst = os.path.join(dst_prefix, i)
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)
            success += 1
    return success


def tag_item(key:str, tag:str):
    return {key:{"tag":tag, "len":len(tag), "val":[]}}


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Modify as needed: repo_path, dst_prefix
    repo_path = r"/mnt/group/daiyi/tmp/verl"
    dst_prefix = r"bilibili/verl"
    result = subprocess.run(['git', 'status'], capture_output=True, text=True, cwd=repo_path)

    logging.info(result.stdout)
    logging.info("*" * 96)

    tag_map = {}
    tag_map.update(tag_item("modified", "\tmodified:   "))
    tag_map.update(tag_item("new", "\t"))
    tag_map.update(tag_item("del", "\tdeleted:    "))
    tag_map.update(tag_item("rename", "\trenamed:    "))
    # Sorted by tag-len
    tag_map = dict(sorted(tag_map.items(), key=lambda x: x[1]["len"], reverse=True))

    tmp = result.stdout.split('\n')
    for i in tmp:
        for _, v in tag_map.items():
            if i.startswith(v["tag"]):
                v["val"].append(i[v["len"]:])
                break

    for k, v in tag_map.items():
        if "del" == k:
            for i in v["val"]:
                logging.info(f"{k}: {os.path.join(repo_path, i)}")
        else:
            rtv = list2file(repo_path, dst_prefix, v["val"])
            logging.info(f"{k}: {rtv}")
    print()
