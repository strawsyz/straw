#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 16:42
# @Author  : strawsyz
# @File    : prepocess_utils.py
# @desc:
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd


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
    data.drop(feature_names, axis=1, inplace=False)


def fillna(feature_data, value=0):
    feature_data.fillna(value, inplace=False)


def label_encoder(data, feature_names):
    le = LabelEncoder()
    for feature_name in feature_names:
        data[feature_name] = le.fit_transform(data[feature_name])


def one_hot(data, feature_names):
    return pd.get_dummies(data, columns=feature_names)


def preprocess(train_csv_path, test_csv_path, only_check=True, drop_feature_names=[], one_hot_feature_names=[]):
    train = pd.read_csv(train_csv_path)
    test = pd.read_csv(test_csv_path)
    if only_check:
        print(20 * "=" + "information about train" + 20 * "=")
        check_data(train)
        print(20 * "=" + "information about test" + 20 * "=")
        check_data(test)
    else:
        if drop_feature_names is [] and one_hot_feature_names is []:
            print("no preprocess")
        else:
            # create all data with train data and test data
            print(20 * "=" + "start preprocess" + 20 * "=")
            train['source'] = 'train'
            test['source'] = 'test'
            data = pd.concat([train, test], ignore_index=True)
            if drop_feature_names is not []:
                drop_features(data, feature_names=drop_feature_names)
            if one_hot_feature_names is not []:
                data = one_hot(data, feature_names=one_hot_feature_names)
            print(20 * "=" + "preprocess is over" + 20 * "=")
            train = data.loc[data['source'] == 'train']
            test = data.loc[data['source'] == 'test']
            drop_features(train, "source")
            drop_features(test, "source")
        return train, test


#  analysis
def boxplot(data, feature_names: list, show=True, save=True, save_path="boxplot.png"):
    data.boxplot(column=feature_names, return_type='axes')
    if show:
        plt.show()
    if save:
        plt.savefig(save_path)
