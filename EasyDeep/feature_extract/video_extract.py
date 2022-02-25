#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/9 14:18
# @Author  : strawsyz
# @File    : video_extractor.py
# @desc:
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.autograd import Variable
from video_loader import FrameCV
import numpy as np
from feature_extract.extractor import get_extractor
from utils.feature_resnet_utils import analyze
from utils.file_utils import make_directory


def extract_sample(video_path, feature_path, model, start=None, duration=None, overwrite=False, FPS=2,
                   transform="resize"):
    print("extract video", video_path, "from", start, duration)
    if os.path.exists(feature_path) and not overwrite:
        return None
    # 读取视频
    videoLoader = FrameCV(video_path, FPS=FPS, transform=transform, start=start,
                          duration=duration)
    frames = videoLoader.frames
    num_frames = frames.shape[0]
    # 用于resnet152的预处理
    # if model_name == "resnet152":
    #     frames = preprocess_input(videoLoader.frames)
    print(f"frames : {frames.shape}")
    frames = frames.transpose(3, 0, 1, 2)
    frames = frames[None, :]
    # b, c, t, h, w = frames.shape
    # print(f"frames : {frames.shape}")
    features = []
    # for start in range(1, t - 56, 1600):
    #     end = min(t - 1, start + 1600 + 56)
    #     start = max(1, start - 48)
    frames = Variable(torch.from_numpy(frames).cuda(), volatile=True).float()
    features.append(model.extract_features(frames).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
    features = np.concatenate(features, axis=0)
    features = features.squeeze()
    np.save(feature_path, features)

    if duration is None:
        duration = videoLoader.time_second
    print("frames", frames.shape, "fps=", num_frames / duration)

    # predict the featrues from the frames (adjust batch size for smalled GPU)
    # features = model.predict(frames, batch_size=64, verbose=1)

    print("features", features.shape, "fps=", features.shape[0] / duration)

    num_frames = features.shape[0]

    # save the featrue in .npy format
    # os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    # np.save(feature_path, features)

    print(f"Save  features at {feature_path}")
    return num_frames


def set_gpu(gpu: str):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def set_path():
    import socket

    hostname = socket.gethostname()
    if hostname == "26d814d5923d":
        # video_path = r"/workspace/datasets/UCF101/val"
        # feature_path = f"/workspace/datasets/features/UCF101/val/{FPS}FPS-{model_name}"
        video_path = "/workspace/datasets/kinetics400/val_256"
        feature_path = f"/workspace/datasets/features/kinetics400/val_256/{FPS}FPS-{model_name}"
    else:
        video_path = r"C:\(lab\datasets\UCF101\val"
        feature_path = rf"C:\(lab\datasets\UCF101\features\{FPS}FPS-{model_name}"
    return video_path, feature_path


def extract_feature_dataset():
    for label in os.listdir(video_path):
        for filename in os.listdir(os.path.join(video_path, label)):
            video_sample_path = os.path.join(video_path, label, filename)
            feature_folder_path = os.path.join(feature_path, label)
            feature_sample_path = os.path.join(feature_folder_path, filename.split(".")[0])
            make_directory(video_sample_path)
            make_directory(feature_folder_path)

            if os.path.exists(feature_sample_path + ".npy"):
                print(f"already have {feature_sample_path}")
                continue

            print(f"video path : {video_sample_path}")
            print(f"feature path : {feature_sample_path}")

            num_frames = extract_sample(video_path=video_sample_path, feature_path=feature_sample_path,
                                        model=model, FPS=FPS)

            all_num_frames.append(num_frames)
            print("======================================")


class Args():
    def __init__(self):
        GPU = "0"

        parser = ArgumentParser(description='transformer for soccernet', formatter_class=ArgumentDefaultsHelpFormatter)

        parser.add_argument('--GPU', required=False, type=str, default=GPU, help='ID of the GPU to use')
        parser.add_argument('--video_path', required=False, type=str, default=None, help='Video(Source) Dir Path')
        parser.add_argument('--feature_path', required=False, type=str, default=None, help='Feature(Traget) Dir Path')

        # parser.add_argument("--split_data", required=False, type=int, default=1,
        #                     help='split_data')
        # parser.add_argument("--model_names_in_type1", nargs='+', required=False, type=str,
        #                     default=["MyOptimizeTransformer6", "MyOptimizeTransformer7", "MyOptimizeTransformer8",
        #                              "MyOptimizeTransformer61"],
        #                     help='spot_model_path')
        # parser.add_argument("--test_4_highlights", required=False, type=bool, default=False,
        #                     help='test_4_highlights')
        # parser.add_argument("--loss_weight", required=False, type=float, default=False,
        #                     help='loss_weight')

        self.args = parser.parse_args()


if __name__ == '__main__':
    model_name = "i3d"
    FPS = 16
    args = Args().args
    video_path, feature_path = set_path()
    if args.video_path is not None:
        video_path = args.video_path
    if args.feature_path is not None:
        feature_path = args.feature_path

    set_gpu(args.GPU)

    # if use_gpu:
    #     set_gpu("0")

    # model = get_feature_extractor("ResNet152")
    model = get_extractor(model_name)

    all_num_frames = []
    video_sample_path = r"C:\(lab\OtherProjects\pytorch-i3d-master\models\9vOJ_EAXmhY.mkv"
    feature_sample_path = r"C:\(lab\datasets\tmp1.npy"
    num_frames = extract_sample(video_path=video_sample_path, feature_path=feature_sample_path,
                                model=model, FPS=FPS)
    # extract_feature_dataset()
    # analyze(all_num_frames)

    # extract_sample()
    # video_extractor()
