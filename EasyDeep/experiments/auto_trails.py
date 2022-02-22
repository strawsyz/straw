#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/04/11 1:08
# @Author  : strawsyz
# @File    : trails.py
# @desc:
import os
import threading
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
from concurrent.futures import ThreadPoolExecutor

from utils.print_utils import print_red
import time
import platform


def add_record(file_path, record):
    with open(file_path, mode="a") as f:
        f.write(record)


def run_command(command):
    p = os.popen(command)
    result = p.readlines()
    assert len(result) > 0
    if result[-1] == "\x1b[0m":
        result = result[-2]
    else:
        result = result[-1]
    record = f"{command}:\t{result}"
    return record


def run_thread(command_):
    GPU_id = useable_GPU_ids[int(threading.current_thread().name.split("_")[-1])]
    command_ = f"{command_} --GPU {GPU_id}"

    print("START\t", command_)
    try:
        result = run_command(command_)
        print_red(result)
        add_record(RECORD_FILEPATH, result)
    except Exception as e:
        result = "ERROR" + str(e)
        result = f"{command_}:\t{result}\n"
        add_record(RECORD_FILEPATH, result)

    print("END\t", command_)
    print("=========================================")
    return result


def select_free_gpu_by_memory(memory_threshold=None):
    import numpy as np
    command = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Free"
    p = os.popen(command)
    result = p.readlines()
    memory_gpu = [int(x.split()[2]) for x in result]
    free_gpu_id = np.argmax(memory_gpu)
    if memory_threshold is not None and memory_gpu[free_gpu_id] < memory_threshold:
        return None
    return str(free_gpu_id)


def create_commands(main_command):
    params_commands = []
    # params_commands.append(" --chunk_size 10")
    # params_commands.append(" --chunk_size 15")
    # params_commands.append(" --chunk_size 20")
    params_commands.append(" --model_name FIE2")
    params_commands.append(" --clip_length 5")
    params_commands.append(" --clip_length 10")
    params_commands.append(" --clip_length 15")
    params_commands.append(" --clip_length 20")
    params_commands.append(" --clip_length 25")
    params_commands.append(" --clip_length 30")
    params_commands.append(" --batch_size 512")
    params_commands.append(" --batch_size 256")
    params_commands.append(" --batch_size 128")
    params_commands.append(" --batch_size 64")
    params_commands.append(" --batch_size 32")
    params_commands.append(" --embedding 32")
    params_commands.append(" --embedding 64")
    params_commands.append(" --embedding 128")
    params_commands.append(" --embedding 256")
    params_commands.append(" --LR 0.001")
    params_commands.append(" --LR 0.002")
    params_commands.append(" --LR 0.003")
    params_commands.append(" --LR 0.004")
    params_commands.append(" --LR 0.005")
    params_commands.append(" --LR 0.008")
    params_commands.append(" --LR 0.01")
    params_commands.append(" --LR 0.03")
    params_commands.append(" --LR 0.08")
    # params_commands.append(" --heads 1")
    # params_commands.append(" --heads 2")
    # params_commands.append(" --heads 4")
    # params_commands.append(" --heads 8")
    params_commands.append(" --n_layers 1")
    params_commands.append(" --n_layers 2")
    params_commands.append(" --n_layers 3")
    params_commands.append(" --n_layers 4")
    params_commands.append(" --n_layers 5")
    params_commands.append(" --n_layers 6")

    # params_commands.append(" --chunk_size 30")
    # params_commands.append(" --chunk_size 40")
    params_commands.append(" --batch_size 1024")
    params_commands.append(" --batch_size 16")
    params_commands.append(" --batch_size 8")
    # params_commands.append(" --batch_size 512")
    # params_commands.append(" --batch_size 256")

    commands = []
    # commands = append_commands("NMS_threshold", args.NMS_threshold, commands, grid=True)
    for params_command in params_commands:
        commands.append(main_command + params_command)
    if len(commands) == 0:
        commands.append(main_command)
    print("Commands : ")
    print(commands)
    return commands


if __name__ == '__main__':
    # RECORD_FILEPATH = r"record.txt"
    DIR_PATH = "records"
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    RECORD_FILEPATH = fr"{DIR_PATH}/{int(time.time())}"
    # 用于解决一个error，不知道原因
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    useable_GPU_ids = [str(i) for i in range(torch.cuda.device_count())]
    num_workers = len(useable_GPU_ids)
    parser = ArgumentParser(description='trail for network', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--OA', required=False, default="", help='other arguments')

    args = parser.parse_args()
    if platform.system().lower() == 'windows':
        GPU = "0"
    elif platform.system().lower() == 'linux':
        GPU = select_free_gpu_by_memory()

    # create main command
    main_command = fr"python ../experiments/video_feature_experiment.py --GPU {GPU} {args.OA} "
    commands = create_commands(main_command)

    # create threadpool
    # pool = ThreadPoolExecutor(max_workers=num_workers)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for command_ in commands:
            future1 = pool.submit(run_thread, command_)
            time.sleep(3)
