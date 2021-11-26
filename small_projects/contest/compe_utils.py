#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/03/18 15:50
# @Author  : strawsyz
# @File    : compe_utils.py
# @desc:
import csv
import os
import random
import time

import albumentations
import numpy as np
import torch
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip, \
    Rotate, RandomContrast, IAAAdditiveGaussianNoise
from albumentations.pytorch import ToTensorV2
from torch import nn
from torchvision import models
from torchvision.models import resnet18
from tqdm import tqdm


def create_relabel_dataset(csv_paths, save_csv_filepath=None):
    """compare different prediction result and create relabel dataset"""
    dataset = {}
    different_filepaths = set()
    for csv_path in csv_paths:
        with open(csv_path) as f:
            reader = csv.reader(f)
            for line in reader:
                if line[1] == dataset.get(line[0], line[1]):
                    dataset[line[0]] = line[1]
                else:
                    different_filepaths.add(line[0])
        print(len(different_filepaths), csv_path)
    print(different_filepaths)
    print(dataset)
    if save_csv_filepath is not None:
        with open(save_csv_filepath, "w", newline='') as f:
            writer = csv.writer(f)
            for key in dataset:
                if key not in different_filepaths:
                    writer.writerow([key, dataset[key]])


def get_transforms(*, data):
    if data == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=180, p=0.5),
            RandomContrast(p=0.5),
            albumentations.RandomBrightness(limit=0.3, p=0.5),
            # albumentatiDons.RandomBrightnessContrast
            IAAAdditiveGaussianNoise(p=0.25),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data in ['valid', 'test']:
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


def set_seed(random_state: int = 0):
    if random_state is not None:
        torch.manual_seed(random_state)  # cpu
        torch.cuda.manual_seed(random_state)  # gpu
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)  # numpy
        random.seed(random_state)  # random and transforms
        torch.backends.cudnn.deterministic = True  # cudnn
        torch.backends.cudnn.benchmark = True
        os.environ['PYTHONHASHSEED'] = str(random_state)


def image_shape_analysis(path):
    """analysis images' shape in one directory"""
    shape = None
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        import cv2
        image = cv2.imread(filepath)
        if shape is None and image.shape is not None:
            shape = image.shape
        else:
            if shape != image.shape:
                print(f"find different size {image.shape}")


def ensemble_prediction_by_models(models, data, plus):
    features = []
    from torch.nn import LogSoftmax
    softmax = LogSoftmax()

    if plus:
        predict_result = None
        for model in models:
            predict = model(data)
            predict = softmax(predict)
            _, label = torch.max(predict, 1)
            if predict_result is None:
                predict_result = predict
            else:
                predict_result += predict
        _, y_preds_labels = torch.max(predict_result, 1)
        if label != y_preds_labels:
            print(f"{label}:{y_preds_labels}")
        return y_preds_labels.to('cpu').detach().numpy()
    else:
        for model in models:
            predict_result = model(data)
            _, y_preds_labels = torch.max(predict_result, 1)
            features.extend(y_preds_labels.to('cpu').detach().numpy())
    return features


def create_features_by_models(models, test_loader, feature_path, device):
    """create features using multiple models"""
    start_time = time.time()
    tk_test = tqdm(enumerate(test_loader), total=len(test_loader))

    with open(feature_path, "w", newline='') as f:
        writer = csv.writer(f)
        for i, (image_filenames, images) in tk_test:
            images = images.to(device)
            features = ensemble_prediction_by_models(models, images, plus=True)
            for image_filename, feature in zip(image_filenames, features):
                row_data = [image_filename, feature]
                writer.writerow(row_data)

    print("save results at {}".format(feature_path))
    print("use {:.1f} seconds".format(time.time() - start_time))



def save_2_csv(row_record: dict, csv_save_path: str, init=False):
    with open(csv_save_path, 'a+', newline='') as csvfile:
        fieldnames = row_record.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if init:
            writer.writeheader()
        writer.writerow(row_record)


def get_resnet18_model(trained, class_num):
    model = models.resnet18(pretrained=trained)
    model.fc = nn.Linear(512, class_num)
    return model


def get_resnet34_model(trained, class_num):
    model = models.resnet34(pretrained=trained)
    model.fc = nn.Linear(512, class_num)
    return model


def get_resnet50_model(trained, class_num):
    model = models.resnet50(pretrained=trained)
    model.fc = nn.Linear(2048, class_num)
    return model


def select_model(model_name, trained, class_num):
    if model_name == "resnet18":
        return get_resnet18_model(trained, class_num)
    elif model_name == "resnet34":
        return get_resnet34_model(trained, class_num)
    elif model_name == "resnet50":
        return get_resnet50_model(trained, class_num)


class MyRes18(nn.Module):
    def __init__(self, num_classes):
        super(MyRes18, self).__init__()
        self.base = resnet18(pretrained=True)
        self.feature = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.reduce_layer = nn.Conv2d(1024, 512, 1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.feature(x)
        # x1 = self.avg_pool(x).view(bs, -1)
        # x2 = self.max_pool(x).view(bs, -1)
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.reduce_layer(x).view(bs, -1)
        logits = self.fc(x)
        return logits


def get_my_resnet18(num_classes):
    return MyRes18(num_classes=num_classes)
