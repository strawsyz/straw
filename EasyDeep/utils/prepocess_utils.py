#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 16:42
# @Author  : strawsyz
# @File    : prepocess_utils.py
# @desc:
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def check_data(data):
    # shape of data
    print("shape of data is {}".format(data.shape))
    # dtypes of data
    print("dtypes of data is \n{}".format(data.dtypes))
    # head 5 of data
    print(data.head(5))
    # describe data
    print("describe data : \n{}".format(data.describe()))

    print("num features which are null:\n{}".format(data.apply(lambda x: sum(x.isnull()))))

    print("=" * 20 + "analysis data" + "=" * 20)
    for feature_name in data.columns:
        feature_data = data[feature_name]
        print(
            '{} have {} kinds of values. \n'
            'The number of times each value appears is as follows：'.format(feature_name, len(
                feature_data.unique())))
        print(feature_data.value_counts())
        print()


def drop_features(data, feature_names):
    return data.drop(feature_names, axis=1, inplace=False)


def fillna(feature_data, value=0):
    feature_data.fillna(value, inplace=False)


def label_encoder(data, feature_names):
    le = LabelEncoder()
    for feature_name in feature_names:
        data[feature_name] = le.fit_transform(data[feature_name])
    return data


def one_hot(data, feature_names):
    return pd.get_dummies(data, columns=feature_names)


def preprocess(train_data, test_data, if_check_data=True, drop_feature_names=[],
               one_hot_feature_names=[], label_encoder_feature_names=[],
               fillna_model="mean"):
    if if_check_data:
        print(20 * "=" + "information about train" + 20 * "=")
        check_data(train_data)
        print(20 * "=" + "information about test" + 20 * "=")
        check_data(test_data)

    print(20 * "=" + "start preprocess" + 20 * "=")
    train_data.loc[:, 'source'] = 'train'
    test_data.loc[:, 'source'] = 'test'
    data = pd.concat([train_data, test_data], ignore_index=True)
    fill_na(data, fillna_model)
    # drop some data
    if drop_feature_names != []:
        drop_features(data, feature_names=drop_feature_names)
    # ont_hot
    if one_hot_feature_names != []:
        data = one_hot(data, feature_names=one_hot_feature_names)

    if label_encoder_feature_names != []:
        data = label_encoder(data, feature_names=label_encoder_feature_names)

    print(20 * "=" + "preprocess is over" + 20 * "=")
    train_data = data.loc[data['source'] == 'train']
    test_data = data.loc[data['source'] == 'test']
    train_data = drop_features(train_data, "source")
    test_data = drop_features(test_data, "source")
    return train_data, test_data


#  analysis
def boxplot(data, feature_names: list, show=True, save=True, save_path="boxplot.png"):
    data.boxplot(column=feature_names, return_type='axes')
    if show:
        plt.show()
    if save:
        plt.savefig(save_path)


def fill_null_mean(data):
    return data.fillna(data.mean(), inplace=False)


def fill_null_median(data):
    return data.fillna(data.median(), inplace=False)


def fill_null_mode(data):
    return data.fillna(data.mode(), inplace=False)



def handle_imbalanced_data(data, gt, imb):
    assert imb in ["SMOTETomek", "SMOTEENN", "RandomUnderSampler", "ADASYN", "SMOTE",
                   "RandomOverSampler"]
    # imbalanced data
    if imb == "SMOTETomek":
        from imblearn.combine import SMOTETomek

        data, gt = SMOTETomek(random_state=0).fit_sample(data, gt)

    elif imb == "SMOTEENN":
        from imblearn.combine import SMOTEENN

        data, gt = SMOTEENN(random_state=0).fit_sample(data, gt)

    elif imb == "RandomUnderSampler":
        from imblearn.under_sampling import RandomUnderSampler

        data, gt = RandomUnderSampler(random_state=0).fit_sample(data, gt)
    elif imb == "ADASYN":
        from imblearn.over_sampling import ADASYN

        data, gt = ADASYN(random_state=0).fit_sample(data, gt)
    elif imb == "SMOTE":
        from imblearn.over_sampling import SMOTE

        data, gt = SMOTE(random_state=0).fit_sample(data, gt)
    elif imb == "RandomOverSampler":
        from imblearn.over_sampling import RandomOverSampler
        # 使用RandomOverSampler从少数类的样本中进行随机采样来增加新的样本使各个分类均衡
        data, gt = RandomOverSampler(random_state=0).fit_sample(data, gt)

    re = sorted(Counter(np.squeeze(gt.values)).items())
    print(re)
    return data, gt


def fill_na(data, mode, exclude_column_names: list = []):
    column_names = data.columns.values.tolist()
    for column_name in column_names:
        if column_name not in exclude_column_names and sum(data[column_name].isnull()) > 0:
            if mode == "mean":
                data[column_name] = fill_null_mean(data[column_name])
            elif mode == "mode":
                data[column_name] = fill_null_mode(data[column_name])
            elif mode == "median":
                data[column_name] = fill_null_median(data[column_name])
            else:
                data[column_name] = data[column_name].fillna(mode)
