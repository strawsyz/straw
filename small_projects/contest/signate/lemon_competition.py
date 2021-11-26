#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/03/11 13:21
# @Author  : strawsyz
# @File    : lemon_competition.py
# @desc:
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from albumentations import Compose, Normalize, CenterCrop, HorizontalFlip, VerticalFlip, \
    Rotate, RandomContrast, IAAAdditiveGaussianNoise
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LemonTrainDataset(Dataset):
    def __init__(self, df, root_path, transform=None):
        self.df = df
        self.labels = df['class_num']
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['id'].values[idx]
        file_path = os.path.join(self.root_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels.values[idx]
        target = torch.zeros(N_CLASSES)

        target[int(label)] = 1

        return image, label, target


class LemonTestDataset(Dataset):
    def __init__(self, df, root_path, transform=None):
        self.df = df
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['id'].values[idx]
        file_path = os.path.join(self.root_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image


def get_transforms(*, data):
    if data == 'train':
        return Compose([
            CenterCrop(height=HEIGHT, width=WIDTH, p=1.0),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=180, p=0.5),
            RandomContrast(p=0.5),
            # albumentations.RandomBrightness(limit=(0, 0.3), p=0.25),
            # albumentations.RandomSunFlare(p=0.25),
            # albumentations.RandomBrightnessContrast
            IAAAdditiveGaussianNoise(p=0.25),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            CenterCrop(height=HEIGHT, width=WIDTH, p=1.0),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


def get_resnet18_model(trained, class_num):
    model = models.resnet18(pretrained=trained)
    model.fc = nn.Linear(512, class_num)
    model.to(device)
    return model


def get_resnet34_model(trained, class_num):
    model = models.resnet34(pretrained=trained)
    model.fc = nn.Linear(512, class_num)
    model.to(device)
    return model


def get_resnet50_model(trained, class_num):
    model = models.resnet50(pretrained=trained)
    model.fc = nn.Linear(2048, class_num)
    model.to(device)
    return model


def select_model(model_name, trained, class_num):
    if model_name == "resnet18":
        return get_resnet18_model(trained, class_num)
    elif model_name == "resnet34":
        return get_resnet34_model(trained, class_num)
    elif model_name == "resnet50":
        return get_resnet50_model(trained, class_num)


def train_valid(model, n_epochs, train_part=1, valid_part=1, train=True, valid=True, model_path=None):
    best_valid_ck_score = 0.
    all_part = train_part + valid_part
    best_epoch_ck_socre = 0
    best_train_valid_loss = np.inf
    best_special_score = 0
    features = []
    for epoch in range(n_epochs):
        feature = {}
        print('EPOCH :\t {}'.format(epoch + 1))
        start_time = time.time()
        if train:
            model.train()
            train_avg_loss = 0.
            optimizer.zero_grad()
            tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
            preds = []
            train_labels = []
            for i, (images, labels, one_hot_labels) in tk0:
                images = images.to(device)
                labels = labels.to(device)
                one_hot_labels = one_hot_labels.to(device)

                y_preds = model(images)

                loss = criterion(y_preds, one_hot_labels)

                _, y_preds_labels = torch.max(y_preds, 1)
                _, labels_new = torch.max(one_hot_labels, 1)

                preds.append(y_preds_labels.to('cpu').numpy())
                train_labels.append(labels_new.to('cpu').numpy())

                # loss = criterion(y_preds, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if use_swa:
                    if epoch > swa_start:
                        swa_model.update_parameters(model)
                        swa_scheduler.step()
                    else:
                        scheduler.step()
                else:
                    scheduler.step()

                train_avg_loss += loss.item() / len(train_loader)
            print(f'train_avg_loss :\t {train_avg_loss}')
            preds = np.concatenate(preds)
            train_labels = np.concatenate(train_labels)
            train_ck_score = cohen_kappa_score(train_labels, preds, weights="quadratic")
            train_acc_score = accuracy_score(train_labels, preds)
            print(f"train_ck_score :\t {train_ck_score}")
            print(f"train_acc_score :\t {train_acc_score}")

        if valid:
            if train is False and n_epochs == 1:
                model.load_state_dict(torch.load(model_path))
            model.eval()
            valid_avg_loss = 0.
            preds = []
            valid_labels = []
            tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

            for i, (images, labels, one_hot_labels) in tk1:
                images = images.to(device)
                labels = labels.to(device)
                one_hot_labels = one_hot_labels.to(device)

                with torch.no_grad():
                    y_preds = model(images)
                loss = criterion(y_preds, one_hot_labels)

                _, y_preds_labels = torch.max(y_preds, 1)
                _, labels_new = torch.max(one_hot_labels, 1)

                preds.append(y_preds_labels.to('cpu').numpy())
                valid_labels.append(labels_new.to('cpu').numpy())

                valid_avg_loss += loss.item() / len(valid_loader)
            print(f'valid_avg_loss :\t {valid_avg_loss}')
            if use_swa is False:
                scheduler.step(valid_avg_loss)
            preds = np.concatenate(preds)
            valid_labels = np.concatenate(valid_labels)
            valid_ck_score = cohen_kappa_score(valid_labels, preds, weights="quadratic")
            valid_acc_score = accuracy_score(valid_labels, preds)
            print(f"valid_acc_score :\t {valid_acc_score}")

            train_valid_loss = (train_part * train_avg_loss + valid_part * valid_avg_loss) / all_part
            if train_valid_loss < best_train_valid_loss:
                best_train_valid_loss = train_valid_loss
                print("[BEST]train_valid_loss :\t {}".format(best_train_valid_loss))
            else:
                print("train_valid_loss :\t {}".format(train_valid_loss))

            epoch_ck_score = (train_part * train_ck_score + valid_part * valid_ck_score) / all_part
            if best_epoch_ck_socre < epoch_ck_score:
                best_epoch_ck_socre = epoch_ck_score
                print(f"[BEST]epoch_ck_socre :\t {best_epoch_ck_socre}")
            else:
                print(f"epoch_ck_score :\t {epoch_ck_score}")

            if valid_ck_score > best_valid_ck_score:
                best_valid_ck_score = valid_ck_score
                print(f'[BEST]valid_ck_score :\t {best_valid_ck_score}')
                save_path = f'{dataset_root_path}/models/{epoch + 1}-{epoch_ck_score}_{valid_ck_score}_best_valid_ck_score.pth'
                torch.save(model.state_dict(), save_path)
            else:
                print(f'valid_ck_score :\t {valid_ck_score}')
                save_path = f'{dataset_root_path}/models/{epoch + 1}-{epoch_ck_score}_{valid_ck_score}.pth'
                torch.save(model.state_dict(), save_path)

        if train_ck_score > valid_ck_score:
            print("maybe over-fitting！")
        feature["epoch"] = epoch + 1
        feature["train_avg_loss"] = train_avg_loss
        feature["train_cohen_kappa_score"] = train_ck_score
        feature["train_acc_score"] = train_acc_score
        feature["valid_avg_loss"] = valid_avg_loss
        feature["valid_cohen_kappa_score"] = valid_ck_score
        feature["valid_acc_score"] = valid_acc_score
        feature["epoch_ck_socre"] = epoch_ck_score
        features.append(feature)
        save_features(feature, "tmp.csv", init=(epoch == 0))
        end_time = time.time()
        print("use {:.1f} seconds".format(end_time - start_time))
    # Update bn statistics for the swa_model at the end
    optimizer.swap_swa_sgd()
    # optimizer.bn_update(train_loader, model)
    cpu_swa_model = swa_model.cpu()
    torch.optim.swa_utils.update_bn(train_loader, cpu_swa_model)
    # swa_model
    save_path = f'{dataset_root_path}/models/swa_model-{epoch + 1}-{epoch_ck_score}_{valid_ck_score}.pth'
    torch.save(cpu_swa_model.state_dict(), save_path)
    return model, features


def test(model, batch_size, model_path):
    start_time = time.time()
    test_df = pd.read_csv(f'{dataset_root_path}/test_images.csv')
    test_dataset = LemonTestDataset(test_df, root_path=rf"{dataset_root_path}\test_images",
                                    transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    tk_test = tqdm(enumerate(test_loader), total=len(test_loader))
    from utils.file_utils import get_filename
    model_filename = get_filename(model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    preds = []
    for i, (images) in tk_test:
        images = images.to(device)
        y_preds = model(images)

        _, y_preds_labels = torch.max(y_preds, 1)

        preds.extend(y_preds_labels.to('cpu').numpy())

    test_df['preds'] = preds
    # result_save_path = '{}/fold{}_{}_submission.csv'.format(dataset_root_path, FOLD, model_filename)
    result_save_path = '{}/{}.csv'.format(dataset_root_path, model_filename)
    test_df[['id', 'preds']].to_csv(result_save_path,
                                    index=False,
                                    header=None)
    print("save results at {}".format(result_save_path))
    print("use {:.1f} seconds".format(time.time() - start_time))
    return model


def analysis_csv_file(csv_file_path):
    train_df = pd.read_csv(csv_file_path)
    train_df.iloc[:, [1]].value_counts().plot.bar(figsize=(10, 3), rot=0)
    plt.show()


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


def create_dataset(train_part, valid_part, random_state):
    assert train_part == 1 or valid_part == 1
    N_Splits = train_part + valid_part

    train_df = pd.read_csv('{}/train_images.csv'.format(dataset_root_path))

    train_df['fold'] = 0
    kf = StratifiedKFold(n_splits=N_Splits, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(kf.split(train_df, train_df['class_num'])):
        # print('FOLD{}'.format(fold))
        train_df.loc[test_index, 'fold'] = fold
    if train_part == 1:
        df_4_train = train_df[train_df['fold'] == FOLD]
        df_4_valid = train_df[train_df['fold'] != FOLD]
    elif valid_part == 1:
        df_4_train = train_df[train_df['fold'] != FOLD]
        df_4_valid = train_df[train_df['fold'] == FOLD]

    train_dataset = LemonTrainDataset(df_4_train.reset_index(drop=True),
                                      root_path=rf"{dataset_root_path}/train_images",
                                      transform=get_transforms(data='train'))
    valid_dataset = LemonTrainDataset(df_4_valid.reset_index(drop=True),
                                      root_path=rf"{dataset_root_path}/train_images",
                                      transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader


# RandomSunFlare
import csv


def save_features(feature, csv_save_path: str, init=False):
    with open(csv_save_path, 'a+', newline='') as csvfile:
        fieldnames = ["epoch", "train_avg_loss", "train_cohen_kappa_score", "train_acc_score", "valid_avg_loss",
                      "valid_cohen_kappa_score", "valid_acc_score", "epoch_ck_socre"]
        # print(fieldnames)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if init:
            writer.writeheader()
        writer.writerow(feature)



class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == '__main__':
    N_CLASSES = 4
    train_part = 4
    valid_part = 1
    HEIGHT = 512
    WIDTH = 512
    FOLD = 0  # resnet34 3 [BEST]valid_ck_score : 0.8798086532439671
    BATCH_SIZE = 8
    n_epochs = 1000
    random_state = 0
    weight_decay = 0  # 1e-4  # 0  # 0.1
    set_seed(random_state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_root_path = None

    train_loader, valid_loader = create_dataset(train_part, valid_part, random_state)

    model = get_resnet18_model(trained=True, class_num=N_CLASSES)
    # model = get_resnet34_model(trained=True, class_num=N_CLASSES)
    # model = get_resnet50_model(trained=True, class_num=N_CLASSES)
    # 优先三个图像
    # roc图制作
    # optional
    model.to(device)
    from torchcontrib.optim import SWA
    import torchcontrib

    # training loop
    # sgd
    lr = 1e-2
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # Adam
    lr = 3e-4  # 1e-4
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=4, verbose=True, eps=1e-6)
    use_swa = False
    if use_swa:
        # swa
        swa_start = 10
        optimizer = torchcontrib.optim.SWA(optimizer=optimizer, swa_start=swa_start, swa_freq=5, swa_lr=lr)
        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(optimizer, T_max=40)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    criterion = nn.BCEWithLogitsLoss()

    model, features = train_valid(model, n_epochs, train_part=train_part, valid_part=valid_part)
