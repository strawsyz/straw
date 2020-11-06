import numpy as np
import pandas as pd

from base.base_dataset import BaseDataSet
from configs.dataset_config import LoanDatasetConfig
from utils.common_utils import copy_need_attr
from utils.machine_learning_utils import normalization

"""read loan data from csv file"""
class LoanDataset(BaseDataSet, LoanDatasetConfig):
    def __init__(self):
        super(LoanDataset, self).__init__()
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def get_dataloader(self, target=None):
        self.prepare_dataset()
        self.prepare_dataloader()
        if target is None:
            return self.train_loader, self.valid_loader, self.test_loader
        else:
            copy_need_attr(self, target, ["train_loader", "test_loader", "valid_loader"])

    def prepare_dataset(self):
        """prepare dataset"""
        if not self.test_model:
            self.train_data, self.train_gt = self.read_train_data()
            self.train_num = len(self.train_data)
            if self.valid_rate is not None:
                valid_num = int(self.train_num * self.valid_rate)
                if valid_num == 0 or valid_num == self.train_num:
                    self.logger.warning("size of valid dataset or train d ataset is 0")
                self.train_data, self.train_gt, self.valid_data, self.valid_gt = split_train_valid(self.train_data,
                                                                                                   self.train_gt,
                                                                                                   self.valid_rate)
        else:
            self.test_data = self.read_test_data()

    def prepare_dataloader(self):
        if self.train_data is not None:
            self.train_loader = get_dataloader(self.train_data, self.train_gt, batch_size=self.batch_size)
        if self.valid_data is not None and len(self.valid_data) != 0:
            self.valid_loader = get_dataloader(self.valid_data, self.valid_gt,
                                               batch_size=self.batch_size_4_test)
        if self.test_data is not None:
            self.test_loader = get_dataloader(self.test_data, batch_size=self.batch_size_4_test)

    def read_train_data(self):
        df = pd.read_csv(self.train_path, index_col=None)
        for feature_name in self.feature_target_names:
            handle_func = get_handle_func("handle_{}".format(feature_name))
            if handle_func is not None:
                df[feature_name] = df[feature_name].map(handle_func)

        train_data = df.loc[:, self.feature_names[:]]
        train_data = np.array(train_data)
        train_gt_tmp = df.loc[:, self.target_names[:]]
        train_gt_tmp = np.array(train_gt_tmp)
        train_gt = []
        for i in train_gt_tmp:
            train_gt.append(i[0])
        train_gt = np.array(train_gt, dtype=np.int)
        if "normalization" in self.preprocess:
            train_data, self.range_, self.min_val = normalization(train_data)
        if "z_score" in self.preprocess:
            train_data, self.mean, self.std = z_score(train_data)
        if self.train_num is not None:
            train_data = train_data[:self.train_num]
            train_gt = train_gt[:self.train_num]
        return train_data, train_gt

    def read_test_data(self):
        df = pd.read_csv(self.test_path, index_col=None)
        for feature_name in self.feature_names:
            handle_func = get_handle_func("handle_{}".format(feature_name))
            if handle_func is not None:
                df[feature_name] = df[feature_name].map(handle_func)
        print(self.feature_names)
        # self.feature_names = ['term', 'interest_rate', 'grade', 'credit_score']
        test_data = df.loc[:, self.feature_names[:]]
        test_data = np.asarray(test_data)
        # test_data = df.loc[:, ['term', 'interest_rate', 'grade', 'credit_score']]
        return test_data


def get_dataloader(data, gt_data=None, batch_size=4):
    class DataLoader():
        def __call__(self, *args, **kwargs):
            """create a data loader"""
            batch_data = []
            if gt_data is None:
                for idx, item in enumerate(data, start=1):
                    batch_data.append(item)
                    if idx % batch_size == 0:
                        yield np.asarray(batch_data)
                        batch_data = []
            else:
                batch_gt_data = []
                for idx, (item, gt_item) in enumerate(zip(data, gt_data), start=1):
                    batch_data.append(item)
                    batch_gt_data.append(gt_item)
                    if idx % batch_size == 0:
                        yield (np.asarray(batch_data), np.asarray(batch_gt_data))
                        batch_data = []
                        batch_gt_data = []

        def __len__(self):
            return int(len((data)) / batch_size)

    return DataLoader()


def handle_term(term: str):
    return year_2_int(term)


def handle_employment_length(employment_length: str):
    return year_2_int(employment_length)


def z_score(x, axis=0, mean=None, std=None):
    xr = np.rollaxis(x, axis=axis)
    if mean is None:
        mean = np.mean(x, axis=axis)
    xr -= mean
    if std is None:
        std = np.std(x, axis=axis)
    xr /= std
    return x, mean, std


def year_2_int(year: str):
    year = year.replace("years", "").replace("year", "").strip()
    return int(year)


def handle_application_type(application_type: str):
    if application_type == "Joint App":
        return 1
    elif application_type == "Individual":
        return 2
    else:
        return 0


def handle_purpose(purpose: str):
    choices = ['small_business', 'house', 'medical', 'home_improvement', 'car', 'debt_consolidation', 'other',
               'credit_card', 'major_purchase']
    for idx, choice in enumerate(choices, start=1):
        if choice == purpose:
            return idx
    return 0


def get_handle_func(func_name):
    try:
        return eval(func_name)
    except NameError as e:
        print(e)
        return None


def handle_loan_status(loan_status: str):
    if loan_status == "ChargedOff":
        return -1, 1
    elif loan_status == "FullyPaid":
        return 1, -1
    else:
        return None


def split_train_valid(data, gt_data, valid_rate=0.2):
    num_train = len(data)
    num_valid = int(num_train * valid_rate)
    num_train = num_train - num_valid
    valid_data = data[num_train:]
    valid_gt_data = gt_data[num_train:]
    data = data[:num_train]
    gt_data = gt_data[:num_train]
    return data, gt_data, valid_data, valid_gt_data


def handle_grade(grade: str):
    grade_level_ = grade[0]
    grade_no_ = int(grade[1])
    grade_levels = ["A", "B", "C", "D", "E", "F"]
    for idx, grade_level in enumerate(grade_levels):
        if grade_level_ == grade_level:
            return idx * 10 + grade_no_
    return None
