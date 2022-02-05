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
        # self.train_path = r"/home/shi/Downloads/dataset/polyp/TMP/09/train/"
        # self.test_path = r"/home/shi/Downloads/dataset/polyp/TMP/09/test/"
        self.train_dataset = VideoFeatureDataset(split="train")
        self.num_train = len(self.train_dataset)
        self.test_dataset = VideoFeatureDataset(split="test")
        self.num_valid = self.num_test = len(self.test_dataset)
        print("num train : {}, num_test : {}, num valid : {}".format(self.num_train, self.num_test, self.num_valid))

    def load_data(self):
        pass

    def get_dataloader(self, target):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_4_test, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_4_test, shuffle=False)
        copy_need_attr(self, target, ["valid_loader", "train_loader", "test_loader"])

    def train(self):
        self.train_dataset.train()
        self.test_dataset.train()

    def test(self):
        self.train_dataset.test()
        self.test_dataset.test()


class VideoFeatureDataset(BaseDataSet, VideoFeatureDatasetConfig):
    def __init__(self, split="train"):
        super(VideoFeatureDataset, self).__init__()
        self.split = split
        self.X = []
        self.Y = []
        self.clip_length = 10
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

        for idx, line in enumerate(tqdm(open(self.annotation_filepath, 'r').readlines()[:num_samples])):
            class_name, filename = line.strip().split(r"/")
            filepath = os.path.join(self.dataset_root_path, class_name, filename.split(".")[0] + ".npy")
            self.X.append(feat2clip(np.load(filepath), self.clip_length))
            self.Y[idx][labels[class_name]] = 1

        self.X = self.X[:num_samples]
        self.Y = self.Y[:num_samples]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


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
