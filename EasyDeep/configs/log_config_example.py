import logging
import platform

_system = platform.system()
if _system == "Windows":
    log_path = ""
elif _system == "Linux":
    log_path = ""

level = logging.INFO
format = "[%(levelname)s]<%(asctime)s> %(filename)s->%(funcName)s line:%(lineno)d { %(message)s }"
console_output = True
console_format = "[%(levelname)s]<%(asctime)s> { %(message)s }"
console_datefmt = None
file_output = True
file_format = "[%(levelname)s]<%(asctime)s> %(filename)s->%(funcName)s line:%(lineno)d { %(message)s }"
file_datefmt = "%y-%m-%d_%H:%M:%S"
