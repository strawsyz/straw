import logging
import os
import time
from logging import handlers

from configs import log_config as config
from utils.file_utils import make_directory


class Logger:
    logger = None

    @classmethod
    def init_logger(cls, logger_name=None):
        """
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        """
        cls.logger = logging.getLogger(logger_name)
        cls.logger.setLevel(config.level)

        cls.log_path = config.log_path
        make_directory(cls.log_path)
        cls.log_filename = os.path.join(cls.log_path, '{}.log'.format(time.strftime("%Y_%m_%d")))
        # fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')  # 这个是python3的
        # fh.setLevel(logging.DEBUG)
        # # 定义handler的输出格式
        # fh.setFormatter(formatter)
        # # 给logger添加handler
        # self.logger.addHandler(fh)
        if config.console_output:
            console = logging.StreamHandler()
            formatter = logging.Formatter(fmt=config.console_format, datefmt=config.console_datefmt)
            console.setFormatter(formatter)
            cls.logger.addHandler(console)
            console.close()

        if config.file_output:
            th = handlers.TimedRotatingFileHandler(filename=cls.log_filename,
                                                   when="midnight",
                                                   backupCount=0,
                                                   encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
            formatter = logging.Formatter(fmt=config.file_format, datefmt=config.file_datefmt)
            th.setFormatter(formatter)
            cls.logger.addHandler(th)

            th.close()

    @classmethod
    def get_logger(cls):
        if cls.logger is None:
            cls.init_logger()
        return cls.logger


if __name__ == '__main__':
    logger1 = Logger.get_logger()
    logger2 = Logger.get_logger()
    if logger1 == logger2:
        if id(logger2) == id(logger2):
            print(1)
            logger1.error(111)
            logger2.debug(222)
