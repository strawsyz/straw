import logging
import os
import time


class Log:
    def __init__(self, logger_name=None, log_cate='main'):
        """
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        """
        # 创建一个logger
        self.logger = logging.getLogger(logger_name)
        # 设置日志级别
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(levelname)s]<%(asctime)s> %(filename)s->%(funcName)s line:%(lineno)d %(message)s')

        # 保存日志的文件
        file_dir = os.getcwd() + '/Logs'
        from file_util import make_directory
        make_directory(file_dir)
        self.log_path = file_dir
        self.log_name = self.log_path + "/" + log_cate + "." + time.strftime("%Y_%m_%d") + '.log'
        # fh = logging.FileHandler(self.log_name, 'a')  # 追加模式  这个是python2的
        fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')  # 这个是python3的
        fh.setLevel(logging.DEBUG)
        # 定义handler的输出格式
        fh.setFormatter(formatter)
        # 给logger添加handler
        self.logger.addHandler(fh)

        # 再创建一个handler，用于输出到控制台
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        # 关闭打开的文件
        fh.close()
        console.close()

    def get_log(self):
        return self.logger
