from sklearn.model_selection import train_test_split
import operator
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from base.base_dataset import BaseDataSet
from utils.utils_ import copy_attr
from configs.dataset_config import CSVDataSetConfig as default_config


class CsvDataSet(BaseDataSet):
    def __init__(self, test_model=False, config_cls=None):
        super(CsvDataSet, self).__init__(config_cls=config_cls)
        self.load_data()
        self.deduplication()
        self.data_split()
        self.test_model = test_model

    def data_split(self, test_size=0.3):
        if self.test_size is not None:
            self.num_test = int(len(self.Y) * test_size)
            self.num_train = len(self.Y) - self.num_test
        if self.test_model:
            self.X = self.X[:-self.num_test]
            self.Y = self.Y[:-self.num_test]
        else:
            if self.num_train is not None:
                if len(self) > self.num_train:
                    self.set_data_num(self.num_train)
                else:
                    self.num_train = len(self)
            else:
                self.num_train = len(self.train_data)
            if self.valid_rate is None:
                self.train_loader = DataLoader(self, batch_size=self.batch_size, shuffle=True)
                return self.train_loader
            else:
                num_valid_data = int(self.num_train * self.valid_rate)
                if num_valid_data == 0 or num_valid_data == self.num_train:
                    self.logger.error("valid datateset is None or train dataset is None")
                self.train_data, self.val_data = torch.utils.data.random_split(self,
                                                                               [self.num_train - num_valid_data,
                                                                                num_valid_data])
                self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
                self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)

    def load_data(self):
        df = pd.read_excel(self.xls_path, parse_dates=["ecg_date", "mibg_date"])
        data = df.iloc[:, [1, 2, 3, 4, 5, 6]].values
        time_deltas = (data[:, 2] - data[:, 1])
        for i, time_delta in enumerate(time_deltas):
            data[i, 1] = time_delta.days
        data = np.hstack([data[:, :2], data[:, 3:]])

        # 每行存储一个numpy矩阵
        X = []
        hm_early = []
        hm_delay = []
        wr = []
        time_delta_data = []
        file_names = []
        Y_dict = {}
        for i in data:
            file_name = i[0].split(".")[0] + ".csv"
            file_path = os.path.join(self.csv_dir, file_name)
            if not os.path.exists(file_path):
                self.logger.info("{} is not exist!".format(file_path))
                continue
            # usecols读取的列索引，header表示开始读取的行
            df = pd.read_csv(file_path, usecols=[0, 1, 2, 3, 4, 5, 6, 7], header=1, index_col=False)
            ecg_data = df.values.transpose().flatten()
            if ecg_data.shape[0] != 40000:
                print("{} has some problems on data's size".format(file_name))
            if Y_dict.get(file_name, None) is None:
                Y_dict[file_name] = []
            X.append(ecg_data)
            time_delta_data.append(i[1])
            hm_early.append([i[2]])
            hm_delay.append([i[3]])
            wr.append([i[4]])
            file_names.append([file_name])

        hm_early = np.array(hm_early)
        hm_delay = np.array(hm_delay)
        wr = np.array(wr)
        time_delta_data = np.array(time_delta_data)
        self.file_names = file_names
        self.X, self.time_delta_data, self.hm_early, self.hm_delay, self.wr = X, time_delta_data, hm_early, hm_delay, wr

    def deduplication(self):
        from collections import namedtuple

        ecg_train = namedtuple("ecg_train", "x time_delta")
        ecg = namedtuple("ecg", "x y time_delta")

        X = self.X
        Y = np.hstack([self.hm_early, self.hm_delay])
        time_delta_data = self.time_delta_data
        ecgs_data = []
        # 使用特征值作为key，来进行保存
        ecgs_dict = {}
        for x, y, time_delta in zip(X, Y, time_delta_data):

            y = y.tostring()
            res = ecgs_dict.get(y)
            if res is None:
                # 保存训练的数据，
                ecgs_dict[y] = ecg_train(x, time_delta)
            else:
                # 如果当前时差要比较值记录的更加小
                if abs(res.time_delta) > abs(time_delta):
                    ecgs_dict[y] = ecg_train(x, time_delta)
        for key in ecgs_dict:
            y = np.fromstring(key)
            x = ecgs_dict.get(key)
            ecgs_data.append(ecg(x.x, y, x.time_delta))
        # 重新组建X和Y的数据
        self.X = []
        self.Y = []
        for ecg_ in ecgs_data:
            self.X.append(ecg_.x)
            self.Y.append(ecg_.y[0])

    def get_dataloader(self, target=None):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)  # cpu
            torch.cuda.manual_seed(self.random_state)  # gpu
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)  # numpy
            random.seed(self.random_state)  # random and transforms
            torch.backends.cudnn.deterministic = True  # cudnn

        if not self.test_model:
            self.train_loader = DataLoader(self, batch_size=self.batch_size, shuffle=True)
            return self.train_loader
        else:
            self.test_loader = DataLoader(self, batch_size=self.batch_size4test, shuffle=True)
            return self.test_loader
        self.copy_attr(target)

    def copy_attr(self, target):
        target.train_loader = self.train_loader
        target.valid_loader = self.valid_loader
        target.test_loader = self.test_loader

    def load_config(self):
        if self.config_cls is not None:
            copy_attr(self.config_cls(), self)
        else:
            copy_attr(default_config(), self)

    def read_from_csv(self, csv_path):
        df = pd.read_csv(csv_path, usecols=[0, 1, 2, 3, 4, 5, 6, 7], header=1, index_col=False)
        ecg_data = df.values.transpose().flatten()
        if ecg_data.shape[0] != 40000:
            print("{} has some problems on data's size".format(csv_path))
        else:
            return ecg_data

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


if __name__ == '__main__':
    pass
