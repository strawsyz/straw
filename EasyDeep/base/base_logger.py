from utils.log_utils import Logger


class BaseLogger:
    def __init__(self):
        super(BaseLogger, self).__init__()
        self.logger = Logger.get_logger()

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)
