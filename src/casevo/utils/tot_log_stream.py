"""
ToT 流式日志模块

支持实时写入的日志系统。
"""

import json
import os


class TotLogStream(object):
    """ToT 流式日志类"""
    model_log = []
    agent_log = []
    agent_num = 0
    offset = 0
    event_log = []
    event_flag = False
    tar_folder = None
    buffer_size = 20
    current_num = 0
        
    @classmethod
    def init_log(cls, agent_num, tar_folder, if_event=False, buffer_size=20):
        """
        初始化日志
        
        参数:
            agent_num: 代理数量
            tar_folder: 目标文件夹路径
            if_event: 是否启用事件日志
            buffer_size: 缓冲区大小
        """
        cls.model_log = []
        cls.agent_log = [[] for i in range(agent_num)]
        cls.agent_num = agent_num
        cls.offset = 0
        cls.event_log = []
        cls.event_flag = if_event
        cls.tar_folder = tar_folder
        cls.buffer_size = buffer_size

    @classmethod
    def set_offset(cls, tar_offset):
        """设置时间偏移量"""
        cls.offset = tar_offset

    @classmethod
    def add_model_log(cls, tar_ts, tar_type, tar_item):
        """添加模型日志"""
        cls.model_log.append({
            'ts': tar_ts + cls.offset,
            'type': tar_type,
            'item': tar_item
        })
        if cls.event_flag:
            cls.event_log.append({
                'ts': tar_ts + cls.offset,
                'owner': 'model',
                'type': tar_type,
                'item': tar_item
            })
        
        cls.current_num += 1
        if cls.current_num >= cls.buffer_size:
            cls.write_log()

    @classmethod
    def add_agent_log(cls, tar_ts, tar_type, tar_item, tar_agent_id):
        """添加智能体日志"""
        cls.agent_log[tar_agent_id].append({
            'ts': tar_ts + cls.offset,
            'type': tar_type,
            'item': tar_item
        })
        if cls.event_flag:
            cls.event_log.append({
                'ts': tar_ts + cls.offset,
                'owner': 'agent_{}'.format(tar_agent_id),
                'type': tar_type,
                'item': tar_item
            })
        cls.current_num += 1
        if cls.current_num >= cls.buffer_size:
            cls.write_log()

    @classmethod
    def get_agent_log(cls, tar_agent_id):
        """获取指定智能体的日志"""
        return cls.agent_log[tar_agent_id]

    @classmethod
    def get_event_log(cls):
        """获取事件日志"""
        return cls.event_log

    @classmethod
    def write_log(cls):
        """将日志写入文件"""
        if len(cls.model_log) > 0:
            res_str = ""
            for item in cls.model_log:
                res_str += json.dumps(item, ensure_ascii=False) + '\n'
            with open(os.path.join(cls.tar_folder, 'model.txt'), 'a') as f:
                f.write(res_str)
        
        for i in range(cls.agent_num):
            if len(cls.agent_log[i]) == 0:
                continue
            res_str = ""
            for item in cls.agent_log[i]:
                res_str += json.dumps(item, ensure_ascii=False) + '\n'
            with open(os.path.join(cls.tar_folder, 'agent_{}.txt'.format(i)), 'a') as f:
                f.write(res_str)
        
        if cls.event_flag and len(cls.event_log) > 0:
            res_str = ""
            for item in cls.event_log:
                res_str += json.dumps(item, ensure_ascii=False) + '\n'
            with open(os.path.join(cls.tar_folder, 'event.txt'), 'a') as f:
                f.write(res_str)
        
        cls.current_num = 0
        cls.model_log = []
        cls.agent_log = [[] for i in range(cls.agent_num)]
        cls.event_log = []

