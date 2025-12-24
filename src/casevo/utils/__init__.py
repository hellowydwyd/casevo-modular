"""
Casevo 工具模块

包含日志、缓存、线程池等实用工具。
"""

from casevo.utils.log import MesaLog
from casevo.utils.tot_log import TotLog
from casevo.utils.tot_log_stream import TotLogStream
from casevo.utils.cache import RequestCache
from casevo.utils.thread_send import ThreadSend
from casevo.utils.random_name import (
    random_chinese_name,
    random_two_name,
    random_three_name,
    random_three_names,
    random_four_name,
)

# 便捷别名
get_random_name = random_chinese_name

__all__ = [
    "MesaLog",
    "TotLog",
    "TotLogStream",
    "RequestCache",
    "ThreadSend",
    "random_chinese_name",
    "random_two_name",
    "random_three_name",
    "random_three_names",
    "random_four_name",
    "get_random_name",
]

