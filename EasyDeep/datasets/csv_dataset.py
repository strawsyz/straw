import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from base.base_dataset import BaseDataSet
from configs.dataset_config import CSVDataSetConfig as default_config
from utils.utils_ import copy_attr


class CsvDataSet(BaseDataSet):
    def __init__(self, config_instance=None):
        self.test_model = False
        self.test_dataloader = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_rate = 0
        self.valid_rate = 0
        self.num_test = 0
        self.num_valid = 0
        super(CsvDataSet, self).__init__(config_instance=config_instance)
        self.load_data()
        self.deduplication()

    def get_dataloader(self, target):
        self.set_seed()
        data_types = []
        data_num = []
        if self.test_rate > 0:
            self.num_test = int(self.num_data * self.test_rate)
            self.num_train = self.num_train - self.num_test
            data_types.append("test")
            data_num.append(self.num_test)
        if self.valid_rate > 0:
            self.num_valid = int(self.num_data * self.valid_rate)
            self.num_train = self.num_train - self.num_valid
            data_types.append("valid")
            data_num.append(self.num_valid)
        data_types.append("train")
        data_num.append(self.num_train)
        self.all_data = torch.utils.data.random_split(self, data_num)
        for data_type, data in zip(data_types, self.all_data):
            if data_type == "test":
                setattr(self, "{}_dataloader".format(data_type),
                        DataLoader(data, batch_size=self.batch_size4test, shuffle=True))
            else:
                setattr(self, "{}_dataloader".format(data_type),
                        DataLoader(data, batch_size=self.batch_size, shuffle=True))
        self.copy_attr(target)

    def load_data(self):
        df = pd.read_excel(self.xls_path, parse_dates=["ecg_date", "mibg_date"])
        data = df.iloc[:, [1, 2, 3, 4, 5, 6]].values
        time_deltas = (data[:, 2] - data[:, 1])
        for i, time_delta in enumerate(time_deltas):
            data[i, 1] = time_delta.days
        data = np.hstack([data[:, :2], data[:, 3:]])

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
        self.num_train = self.num_data = len(self.X)

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
        self.num_train = self.num_data = len(self.X)

    def copy_attr(self, target):
        target.train_loader = self.train_dataloader
        target.valid_loader = self.valid_dataloader
        target.test_loader = self.test_dataloader

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
        import numpy as np
        # for test on sample data
        # return np.ones(tuple(self.X[index].shape)), np.ones(tuple(self.Y[index].shape))
        # return np.ones(40000), np.ones(1)

    def get_samle_dataloader(self, num_samples, target):
        self.X, self.Y = self.X[:num_samples * 3], self.Y[:num_samples * 3]
        self.train_data, self.valid_data, self.test_data = torch.utils.data.random_split(self,
                                                                                         [num_samples,
                                                                                          num_samples,
                                                                                          num_samples])
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size4test, shuffle=True)
        self.copy_attr(target)


if __name__ == '__main__':
    from configs import CSVDataSetConfig

    dataset = CsvDataSet(CSVDataSetConfig())
    dataset.get_dataloader(dataset)

    print(len(dataset.train_loader))
    print(len(dataset.valid_loader))
    print(len(dataset.test_loader))
    print()
    pass
