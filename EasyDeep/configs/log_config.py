import logging
import platform

_system = platform.system()
if _system == "Windows":
    # log_path = "C:\\Log"
    log_path = r"C:\Users\syz11\PycharmProjects\straw\EasyDeep\experiments"
elif _system == "Linux":
    log_path = "/straw/EasyDeep/experiments/Logs"

level = logging.DEBUG
format = "[%(levelname)s]<%(asctime)s> %(filename)s->%(funcName)s line:%(lineno)d { %(message)s }"
console_output = True
console_format = "[%(levelname)s]<%(asctime)s> { %(message)s }"
console_datefmt = None
file_output = True
file_format = "[%(levelname)s]<%(asctime)s> %(filename)s->%(funcName)s line:%(lineno)d { %(message)s }"
file_datefmt = "%y-%m-%d_%H:%M:%S"
# The passwords of THUMOS15 and 14 are THUMOS15_challenge_REGISTERED and THUMOS14_REGISTERED, respectively.