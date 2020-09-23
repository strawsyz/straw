import logging
import os
import time
from logging import handlers

from configs import log_config as config
from utils.file_utils import make_directory


class Logger:
    logger = None
    tag_loggers = {}
    @classmethod
    def init_logger(cls, logger_name=None):
        """
            initialize logger class
        """
        cls.logger = logging.getLogger(logger_name)
        cls.logger.setLevel(config.level)
        if logger_name is not None:
            cls.log_path = os.path.join(cls.log_path, logger_name)
        cls.log_path = config.log_path
        make_directory(cls.log_path)
        cls.log_filename = os.path.join(cls.log_path, '{}.log'.format(time.strftime("%Y_%m_%d")))

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
                                                   encoding='utf-8')
            formatter = logging.Formatter(fmt=config.file_format, datefmt=config.file_datefmt)
            th.setFormatter(formatter)
            cls.logger.addHandler(th)

            th.close()

    @classmethod
    def get_logger(cls,tag=None):
        if tag is None:
            if cls.logger is None:
                cls.init_logger()
            return cls.logger
        else:
            tage_logger = cls.tag_loggers.get(tag,None)
            if tage_logger is None:
                tage_logger = cls.init_logger(tag=tag)
                cls.tag_loggers[tag] = tage_logger

            return tage_logger



def get_logger(tag=None):
    return Logger.get_logger(tag)
