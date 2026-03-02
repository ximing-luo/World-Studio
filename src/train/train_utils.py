import torch
import os
from datetime import datetime

def get_log_dir(base_path):
    """
    生成带时间戳的日志目录
    例如: logs/sekiro/rssm/20240302-153000
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_path, timestamp)
    return log_dir
