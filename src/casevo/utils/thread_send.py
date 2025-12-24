"""
线程发送模块
"""

import threading
import queue
import time


class ThreadSend:
    """线程池任务发送器"""

    def __init__(self, thread_num=8):
        self.status = 'init' 
        self.task_list = queue.Queue()
        self.threads_num = thread_num

    def add_task(self, func, args):
        """添加任务"""
        self.task_list.put([func, args])
    
    def get_task_num(self):
        """获取待处理任务数量"""
        return self.task_list.qsize()

    def worker(self):
        """工作线程"""
        while True:
            try:
                task = self.task_list.get(timeout=5)
                task[0](*task[1])
                self.task_list.task_done()
                time.sleep(5)
                if self.task_list.empty():
                    break
            except:
                break
    
    def start_thread(self):
        """启动线程池"""
        self.thread_list = []
        for i in range(self.threads_num):
            t = threading.Thread(target=self.worker)
            t.start()
            self.thread_list.append(t)
        self.task_list.join()

