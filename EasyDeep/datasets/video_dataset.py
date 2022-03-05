#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/6 13:18
# @Author  : strawsyz
# @File    : videodataset.py
# @desc:
import os

from torch.utils.data import DataLoader
from torchvision.datasets import HMDB51, UCF101
import numpy as np

from tqdm import tqdm

from base.base_dataset import BaseDataSet
from configs.dataset_config import VideoFeatureDatasetConfig
from feature_extract.video_loader import FrameCV
from utils import file_utils
from utils.common_utils import copy_need_attr
import torchvision

from torch import nn

print("Torchvision Version: ", torchvision.__version__)

import torch
from torch.utils.data import DataLoader
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)


class UCF101DataSet(VideoFeatureDatasetConfig):
    def __init__(self):
        super(UCF101DataSet, self).__init__()

    def create_dataset(self):
        self.train_dataset = VideoFeatureDataset(split="train")
        self.num_train = len(self.train_dataset)
        self.test_dataset = VideoFeatureDataset(split="test")
        self.num_valid = self.num_test = len(self.test_dataset)
        print("num train : {}, num_test : {}, num valid : {}".format(self.num_train, self.num_test, self.num_valid))

    def load_data(self):
        pass

    def get_dataloader(self, target):
        self.create_dataset()
        if self.test_model:
            self.train_dataset.test()
            self.test_dataset.test()
        else:
            self.train_dataset.train()
            self.test_dataset.train()
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_4_test, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_4_test, shuffle=False)
        copy_need_attr(self, target, ["valid_loader", "train_loader", "test_loader"])

    def train(self):
        self.test_model = False

    def test(self):
        self.test_model = True


class UCF1012SDataSet(VideoFeatureDatasetConfig):
    def __init__(self):
        super(UCF1012SDataSet, self).__init__()

    def create_dataset(self):
        self.train_dataset = Video2SDataset(split="train")
        self.num_train = len(self.train_dataset)
        self.test_dataset = Video2SDataset(split="test")
        self.num_valid = self.num_test = len(self.test_dataset)
        print("num train : {}, num_test : {}, num valid : {}".format(self.num_train, self.num_test, self.num_valid))

    def load_data(self):
        pass

    def get_dataloader(self, target):
        self.create_dataset()
        if self.test_model:
            self.train_dataset.test()
            self.test_dataset.test()
        else:
            self.train_dataset.train()
            self.test_dataset.train()
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_4_test, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_4_test, shuffle=False)
        copy_need_attr(self, target, ["valid_loader", "train_loader", "test_loader"])

    def train(self):
        self.test_model = False

    def test(self):
        self.test_model = True


def load_annotations(annotation_filepath):
    labels_dict = {}
    for line in open(annotation_filepath):
        index, name = line.strip().split(' ')
        labels_dict[name] = int(index) - 1
    return labels_dict


class VideoFeatureDataset(BaseDataSet, VideoFeatureDatasetConfig):
    def __init__(self, split="train"):
        super(VideoFeatureDataset, self).__init__()
        self.split = split
        self.X = []
        # self.Y = []

        if self.split == "train":
            self.annotation_filepath = self.train_annotation_filepath
            self.dataset_root_path = self.train_dataset_root_path
        elif self.split == "test":
            self.annotation_filepath = self.test_annotation_filepath
            self.dataset_root_path = self.test_dataset_root_path
        else:
            raise NotImplementedError('No Such split')


        labels = load_annotations(self.label_filepath)
        num_classes = len(labels)
        from utils.file_utils import get_line_number
        num_samples = get_line_number(self.annotation_filepath)
        num_samples = int(num_samples * self.use_rate)

        self.Y = np.zeros((num_samples, num_classes))
        self.feature_root_path = r"C:\(lab\datasets\UCF101\features\RGB"
        for idx, line in enumerate(tqdm(open(self.annotation_filepath, 'r').readlines()[:num_samples], ncols=50)):
            class_name, filename = line.strip().split(r"/")
            video_filepath = os.path.join(self.dataset_root_path, class_name, filename.split(".")[0] + ".avi")
            # self.X.append(feat2clip(np.load(video_filepath), self.clip_length))
            videoLoader = FrameCV(video_filepath, FPS=self.FPS, transform=self.transform, start=self.start,
                                  duration=self.duration)
            frames = videoLoader.frames
            if frames is None:
                print(f"loading video Failed{video_filepath}")
                continue
            # num_frames = frames.shape[0]
            frames = frames.transpose(3, 0, 1, 2)
            file_utils.make_directory(os.path.join(self.feature_root_path, class_name))
            print(type(frames))
            # np.save(os.path.join(self.feature_root_path, class_name, filename.split(".")[0]), frames)
            self.Y[idx][labels[class_name]] = 1
            self.X.append(frames)

        self.X = self.X[:num_samples]
        self.Y = self.Y[:num_samples]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class Video2SDataset(BaseDataSet, VideoFeatureDatasetConfig):
    def __init__(self, split="train"):
        super(Video2SDataset, self).__init__()
        self.video_paths = []
        self.side_size = 256
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.num_frames = 32
        self.sampling_rate = 2
        self.frames_per_second = 32
        self.alpha = 4
        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(self.mean, self.std),
                    ShortSideScale(
                        size=self.side_size
                    ),
                    CenterCropVideo(self.crop_size),
                    PackPathway(self.alpha)
                ]
            ),
        )

        self.split = split
        self.fast_flow = []
        self.slot_flow = []
        # self.Y = []

        if self.split == "train":
            self.annotation_filepath = self.train_annotation_filepath
            self.dataset_root_path = self.train_dataset_root_path
        elif self.split == "test":
            self.annotation_filepath = self.test_annotation_filepath
            self.dataset_root_path = self.test_dataset_root_path
        else:
            raise NotImplementedError('No Such split')

        labels = load_annotations(self.label_filepath)
        num_classes = len(labels)
        from utils.file_utils import get_line_number
        num_samples = get_line_number(self.annotation_filepath)
        num_samples = int(num_samples * self.use_rate)

        self.Y = np.zeros((num_samples, num_classes))
        self.feature_root_path = r"C:\(lab\datasets\UCF101\features\RGB"
        for idx, line in enumerate(tqdm(open(self.annotation_filepath, 'r').readlines()[:num_samples], ncols=50)):
            class_name, filename = line.strip().split(r"/")
            video_filepath = os.path.join(self.dataset_root_path, class_name, filename.split(".")[0] + ".avi")
            # self.X.append(feat2clip(np.load(video_filepath), self.clip_length))
            video_data = self.load_two_stream(video_filepath)
            slow_flow, fast_flow = video_data
            # np.save(os.path.join(self.feature_root_path, class_name, filename.split(".")[0]), frames)
            self.Y[idx][labels[class_name]] = 1
            self.fast_flow.append(fast_flow)
            self.slot_flow.append(slow_flow)

        self.fast_flow = self.fast_flow[:num_samples]
        self.slot_flow = self.slot_flow[:num_samples]
        self.Y = self.Y[:num_samples]

    def load_two_stream(self, video_path):
        # The duration of the input clip is also specific to the model.
        clip_duration = (self.num_frames * self.sampling_rate) / self.frames_per_second

        # Load the example video
        # video_path = r"C:\(lab\datasets\UCF101\all\ApplyLipstick\v_ApplyLipstick_g01_c02.avi"

        # Select the duration of the clip to load by specifying the start and end duration
        # The start_sec should correspond to where the action occurs in the video
        start_sec = 0
        end_sec = start_sec + clip_duration

        # Initialize an EncodedVideo helper class
        video = EncodedVideo.from_path(video_path)
        # Load the desired clip
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        # Apply a transform to normalize the video input
        video_data = self.transform(video_data)
        # Move the inputs to the desired device
        inputs = video_data["video"]

        # inputs = [i.to(device)[None, ...] for i in inputs]
        # print(inputs.shape)
        return inputs

    def __getitem__(self, index):
        return self.slot_flow[index], self.fast_flow[index], self.Y[index]

    def __len__(self):
        return len(self.slot_flow)


# 问题 ：使用的slow fast的帧，但是没有光流抽取的功能
# 使用提案手法没有办法对没有使用帧特征的。
# 将提案手法作为另外一个branch，用另外一个branch来抽取帧之间的特征
# 在最后的部分联合起来




def test():
    # dataset = HMDB51(frames_per_clip=5, step_between_clips=5)
    dataset = UCF101(root=r"C:\(lab\datasets\UCF101\val",
                     annotation_path=r"C:\(lab\datasets\UCF101\UCF101TrainTestSplits-RecognitionTask\ucfTrainTestlist\testlist01.txt",
                     frames_per_clip=5, step_between_clips=5)
    print(dataset.shape)

    #  创建数据集

    #  抽取特征

    #  直接使用显存的特征量


if __name__ == '__main__':
    import cv2

    video_path = r"C:\(lab\datasets\UCF101\val\ApplyEyeMakeup\v_ApplyEyeMakeup_g01_c03.avi"
    vidcap = cv2.VideoCapture(video_path)
    print(os.path.exists(video_path))
    # get number of frames
    # self.numframe = int(self.time_second * self.fps_video)

    # frame drop ratio
    # drop_extra_frames = self.fps_video / self.FPS

    # init list of frames
    # self.frames = []

    # TQDM progress bar
    # pbar = tqdm(range(self.numframe), desc='Grabbing Video Frames', unit='frame')
    # i_frame = 0
    ret, frame = vidcap.read()
    print(ret)

    # test()
