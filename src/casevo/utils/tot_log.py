"""
ToT 日志模块

用于记录和管理日志数据，支持多智能体日志追踪。
"""

import json
import os


class TotLog(object):
    """ToT 日志类"""
    model_log = []
    agent_log = []
    agent_num = 0
    extra_log = {}
    offset = 0
    event_log = []
    event_flag = False

    @classmethod
    def init_log(cls, agent_num, if_event=False):
        """初始化日志"""
        cls.model_log = []
        cls.agent_log = [[] for i in range(agent_num)]
        cls.agent_num = agent_num
        cls.extra_log = {}
        cls.offset = 0
        cls.event_log = []
        cls.event_flag = if_event
    
    @classmethod
    def set_log(cls, tar_file, tar_offset, extra_list=[]):
        """从文件加载日志"""
        cls.offset = tar_offset
        with open(os.path.join(tar_file, 'model.json'), 'r') as f:
            cls.model_log = json.load(f)
        for i in range(cls.agent_num):
            with open(os.path.join(tar_file, 'agent_{}.json'.format(i)), 'r') as f:
                cls.agent_log[i] = json.load(f)
        if cls.event_flag:
            with open(os.path.join(tar_file, 'event.json'), 'r') as f:
                cls.event_log = json.load(f)
        for item in extra_list:
            with open(os.path.join(tar_file, '{}.json'.format(item)), 'r') as f:
                cls.extra_log[item] = json.load(f)

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

    @classmethod
    def add_extra_log(cls, tar_ts, tar_type, tar_item, tar_name):
        """添加额外日志"""
        cls.extra_log[tar_name].append({
            'ts': tar_ts + cls.offset,
            'type': tar_type,
            'item': tar_item
        })
        if cls.event_flag:
            cls.event_log.append({
                'ts': tar_ts + cls.offset,
                'owner': tar_name,
                'type': tar_type,
                'item': tar_item
            })

    @classmethod
    def get_agent_log(cls, tar_agent_id):
        """获取指定智能体的日志"""
        return cls.agent_log[tar_agent_id]

    @classmethod
    def get_event_log(cls):
        """获取事件日志"""
        return cls.event_log

    @classmethod
    def write_log(cls, tar_file):
        """将日志写入文件"""
        with open(os.path.join(tar_file, 'model.json'), 'w') as f:
            json.dump(cls.model_log, f, ensure_ascii=False)
        for i in range(cls.agent_num):
            with open(os.path.join(tar_file, 'agent_{}.json'.format(i)), 'w') as f:
                json.dump(cls.agent_log[i], f, ensure_ascii=False)
        if cls.event_flag:
            with open(os.path.join(tar_file, 'event.json'), 'w') as f:
                json.dump(cls.event_log, f, ensure_ascii=False)
        for item in cls.extra_log:
            with open(os.path.join(tar_file, '{}.json'.format(item)), 'w') as f:
                json.dump(cls.extra_log[item], f, ensure_ascii=False)

