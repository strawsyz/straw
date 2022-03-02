import logging
import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from base.base_dataset import BaseDataSet
from configs.dataset_config import VideoFeatureDatasetConfig
import torch

from utils.common_utils import copy_need_attr


def load_annotations(annotation_filepath):
    labels_dict = {}
    for line in open(annotation_filepath):
        index, name = line.strip().split(' ')
        labels_dict[name] = int(index) - 1
    return labels_dict


def feat2clip(feat, clip_length):
    num_frame = feat.shape[0]

    if num_frame < clip_length:
        n = (clip_length - num_frame) // 2
        idx = [0 for _ in range(n)] + [i for i in range(num_frame)] + [num_frame - 1 for _ in
                                                                       range(clip_length - n - num_frame)]
        return feat[idx, ...]
    elif num_frame > clip_length:
        start = max((num_frame - clip_length) // 2, 0)
        return feat[start:start + clip_length]
    else:
        return feat


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


        if self.annotation_filepath is None:
            labels = os.listdir(self.dataset_root_path)
            label_dict = {}
            for idx, label in enumerate(labels):
                label_dict[label] = idx
            if self.split == "train":
                num_samples = 240618
            elif self.split == "test":
                num_samples = 19404
            num_classes = len(labels)
            self.Y = np.zeros((num_samples, num_classes))
            idx = 0
            for dir_name in os.listdir(self.dataset_root_path):
                feature_dirpath = os.path.join(self.dataset_root_path, dir_name)
                for filename in os.listdir(feature_dirpath):
                    filepath = os.path.join(feature_dirpath, filename)
                    self.X.append(feat2clip(np.load(filepath), self.clip_length))
                    self.Y[idx][label_dict[dir_name]] = 1
                    idx += 1
        else:
            labels = load_annotations(self.label_filepath)
            num_classes = len(labels)
            from utils.file_utils import get_line_number
            num_samples = get_line_number(self.annotation_filepath)
            num_samples = int(num_samples * self.use_rate)

            self.Y = np.zeros((num_samples, num_classes))

            for idx, line in enumerate(tqdm(open(self.annotation_filepath, 'r').readlines()[:num_samples], ncols=50)):
                class_name, filename = line.strip().split(r"/")
                filepath = os.path.join(self.dataset_root_path, class_name, filename.split(".")[0] + ".npy")
                features = np.load(filepath)
                features = features.transpose(1, 0, 2, 3)
                features = feat2clip(features, self.clip_length)
                features = features.transpose(1, 0, 2, 3)
                self.X.append(features)
                self.Y[idx][labels[class_name]] = 1
                # break
        self.X = self.X[:num_samples]
        self.Y = self.Y[:num_samples]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


class K400DataSet(VideoFeatureDatasetConfig):
    def __init__(self):
        super(K400DataSet, self).__init__()

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    dataset_Train = VideoFeatureDataset(split="test")
    print(len(dataset_Train))
    all_labels = []

    for i in tqdm(range(len(dataset_Train))):
        feats, label1 = dataset_Train[i]
        # print(feats.shape)
        all_labels.append(label1)

    all_labels = np.stack(all_labels)
    print(all_labels.shape)
    print(np.sum(all_labels, axis=0))
    print(np.sum(all_labels, axis=1))
