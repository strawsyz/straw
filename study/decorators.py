import signal
import time
from functools import wraps


def calculate_time(func):
    """计算运行时间的装饰器"""

    def decorator():
        start_time = time.time()  # 开始计时
        func()
        end_time = time.time()  # 计时结束
        print("Total Spend time：", str((end_time - start_time) / 60)[0:6] + "分钟")

    return decorator()


def set_cwd(func):
    @wraps(func)
    def decorator():
        import argparse
        import os
        parser = argparse.ArgumentParser()
        # 默认是当前文件夹
        parser.add_argument('-p', type=str, default=os.getcwd())
        args = parser.parse_args()
        func()
        return args.p

    return decorator()


def set_delay(seconds=0):
    """设置一定时间之后运行程序"""

    def decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            time.sleep(seconds)
            return func(*args, **kwargs)

        return wrapped_function

    return decorator


def catch_exception(is_log=False):
    def decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                pass
                # raise e

        return wrapped_function

    return decorator


def set_timeout(timeout):
    def decorator(func):
        def handle(signum, frame):
            raise RuntimeError("Run function {} time out".format(func.__name__))

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, handle)
            # 设定超过timeout秒就报错
            signal.alarm(timeout)
            # 运行被装饰的函数
            result = func(*args, **kwargs)
            # 重置警报
            signal.alarm(0)
            return result

        wrapper.func_name = func.__name__
        return wrapper

    return decorator

def log_output(file):
    def wrapper(func):
        def inner(*args, **kwargs):
            with open(file, 'a', encoding='utf-8') as f:
                result = func(*args, **kwargs)
                f.write('[{}]:{}] Out:< {} > \n'.format(func.__name__, time.strftime('%Y %m-%d %X'), result))
                return result

        return inner

    return wrapper
