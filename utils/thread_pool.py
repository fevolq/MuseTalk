import os
import threading
from concurrent.futures import ThreadPoolExecutor

from configs import config


class ThreadPool:
    """
    线程池。池中每增加一个任务，即增加一个线程，各自执行任务。
    """

    __lock = threading.RLock()
    __pools = config.MAX_POOL

    def __new__(cls, *args, **kwargs):
        # 构造单例
        if hasattr(cls, 'instance'):
            return cls.instance

        # 线程锁
        with cls.__lock:
            if not hasattr(cls, 'instance'):
                cls.instance = super(ThreadPool, cls).__new__(cls)
            return cls.instance

    def __init__(self):
        self.executor = ThreadPoolExecutor(ThreadPool.__pools)

    def submit(self, func, *args, **kwargs) -> bool:
        flag = True
        try:
            self.executor.submit(func, *args, **kwargs)
        except:
            flag = False
        return flag


pool_instance = ThreadPool()


def submit(func, *args, **kwargs) -> bool:
    return pool_instance.submit(func, *args, **kwargs)
