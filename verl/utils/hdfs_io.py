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
import toml
import shutil
import logging
from verl.bl_utils.boss_transfer import HybridBossFileHandler

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

_HDFS_PREFIX = "hdfs://"
_BOSS_PREFIX = "s3://"

_HDFS_BIN_PATH = shutil.which("hdfs")

_global_boss_file_handler = None

def init_boss_file_handler(config_path=None):
    global _global_boss_file_handler
    try:
        _global_boss_file_handler = HybridBossFileHandler(config_path=config_path)
    except Exception as e:
        logging.error(f"BOSS handler 初始化失败: {e}")
        _global_boss_file_handler = None
    return _global_boss_file_handler


def exists(path: str, **kwargs) -> bool:
    r"""Works like os.path.exists() but supports hdfs.

    Test whether a path exists. Returns False for broken symbolic links.

    Args:
        path (str): path to test

    Returns:
        bool: True if the path exists, False otherwise
    """
    if _is_non_local(path):
        return _exists(path, **kwargs)
    return os.path.exists(path)


def _exists(file_path: str):
    """hdfs capable to check whether a file_path is exists"""
    if file_path.startswith("hdfs"):
        return _run_cmd(_hdfs_cmd(f"-test -e {file_path}")) == 0
    return os.path.exists(file_path)


def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    r"""Works like os.makedirs() but supports hdfs.

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    Args:
        name (str): directory to create
        mode (int): file mode bits
        exist_ok (bool): if True, do not raise an exception if the directory already exists
        kwargs: keyword arguments for hdfs

    """
    if _is_non_local(name):
        # TODO(haibin.lin):
        # - handle OSError for hdfs(?)
        # - support exist_ok for hdfs(?)
        _mkdir(name, **kwargs)
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


def _mkdir(file_path: str) -> bool:
    """hdfs mkdir"""
    if file_path.startswith("hdfs"):
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))
    else:
        os.makedirs(file_path, exist_ok=True)
    return True


def _safe_delete(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        logging.info(f"已删除本地源文件 {path}")
    except Exception as e:
        logging.warning(f"删除本地源文件失败 {path}: {e}")


def copy(src: str, dst: str, **kwargs) -> bool:
    r"""Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs.

    Copy data and mode bits ("cp src dst"). Return the file's destination.
    The destination may be a directory.
    If source and destination are the same file, a SameFileError will be
    raised.

    Arg:
        src (str): source file path
        dst (str): destination file path
        kwargs: keyword arguments for hdfs copy

    Returns:
        bool: success or not

    """
    try:
        rtv = True
        _src_is_dir = os.path.isdir(src)
        if _is_oss_path(src) or _is_oss_path(dst):
           rtv = _copy_with_boss(src, dst)
        elif _is_non_local(src) or _is_non_local(dst):
            # TODO(haibin.lin):
            # - handle SameFileError for hdfs files(?)
            # - return file destination for hdfs files
            rtv = _copy(src, dst)
        else:
            parent = os.path.dirname(dst)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

            if _src_is_dir:
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)

        if "delete_after_upload" in kwargs:
            if True == kwargs["delete_after_upload"]:
                if _src_is_dir:
                    shutil.rmtree(src)
                else:
                    os.remove(src)
        return rtv
    except Exception as e:
        logger.warning(f"Failed: {e}, copy {src} to {dst}")
        return False


def _copy_with_boss(from_path: str, to_path: str, timeout: int = None) -> bool:
    global _global_boss_file_handler
    if not _global_boss_file_handler:
        _global_boss_file_handler = init_boss_file_handler()  # init with os.environ['BOSS_TOML']
    source_is_s3 = _global_boss_file_handler.is_oss_path(from_path)
    dest_is_s3 = _global_boss_file_handler.is_oss_path(to_path)
    ret = -1

    if source_is_s3 and not dest_is_s3 and _global_boss_file_handler:
        # 下载：S3 -> 本地
        local_file = _global_boss_file_handler.download_from_oss(from_path)
        if local_file is None:
            logging.error("下载操作失败")
            return ret
        shutil.move(local_file, to_path)
        logging.info(f"文件已下载到 {to_path}")
        ret = 0
    elif not source_is_s3 and dest_is_s3 and _global_boss_file_handler:
        # 上传：本地 -> S3
        if not os.path.exists(from_path):
            logging.error(f"本地文件 {from_path} 不存在")
            return ret
            
        _global_boss_file_handler.upload_to_oss(to_path, from_path)
        logging.info(f"已将文件从 {from_path} 上传到 {to_path}")
        ret = 0
    else:
        try:
            shutil.copy(from_path, to_path)
            ret = 0
        except shutil.SameFileError:
            ret = 0
        except Exception as e:
            logger.warning(f"copy {from_path} {to_path} failed: {e}")
            ret = -1
    return ret == 0


def _copy(from_path: str, to_path: str, timeout: int = None) -> bool:
    # 如果是本地拷贝，先确保目标目录存在
    if not to_path.startswith("hdfs"):
        parent_dir = os.path.dirname(to_path)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"创建目录 {parent_dir} 失败: {e}")

    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(_hdfs_cmd(f"-cp -f {from_path} {to_path}"), timeout=timeout)
        else:
            returncode = _run_cmd(_hdfs_cmd(f"-put -f {from_path} {to_path}"), timeout=timeout)
    else:
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(
                _hdfs_cmd(
                    f"-get \
                {from_path} {to_path}"
                ),
                timeout=timeout,
            )
        else:
            try:
                shutil.copy(from_path, to_path)
                returncode = 0
            except shutil.SameFileError:
                returncode = 0
            except Exception as e:
                logger.warning(f"copy {from_path} {to_path} failed: {e}")
                returncode = -1
    return returncode == 0


def _run_cmd(cmd: str, timeout=None):
    return os.system(cmd)


def _hdfs_cmd(cmd: str) -> str:
    return f"{_HDFS_BIN_PATH} dfs {cmd}"


def _is_oss_path(path: str):
    return path.startswith(_BOSS_PREFIX)


def _is_non_local(path: str):
    return path.startswith(_HDFS_PREFIX) or path.startswith(_BOSS_PREFIX)
