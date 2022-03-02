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
from my_cv.utils import file_util
from utils.common_utils import copy_need_attr


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
            file_util.make_directory(os.path.join(self.feature_root_path, class_name))
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