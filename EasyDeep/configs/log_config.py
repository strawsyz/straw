import logging
import platform
_system = platform.system()
if _system=="Windows":

    log_path = "C:\\Users\Administrator\PycharmProjects\straw\EasyDeep\Log"
    # tmp
    log_path = "C:\\Log"
elif _system == "Linux":
    log_path = "/home/straw/PycharmProjects/straw/EasyDeep/Logs"

level = logging.INFO
format = "[%(levelname)s]<%(asctime)s> %(filename)s->%(funcName)s line:%(lineno)d { %(message)s }"
console_output = True
console_format = "[%(levelname)s]<%(asctime)s> { %(message)s }"
console_datafmt = None
file_output = True
file_format = "[%(levelname)s]<%(asctime)s> %(filename)s->%(funcName)s line:%(lineno)d { %(message)s }"
file_datafmt = "%y-%m-%d_%H:%M:%S"

