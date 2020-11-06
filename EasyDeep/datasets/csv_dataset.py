#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 23:47
# @Author  : strawsyz
# @File    : csv_dataset.py
# @desc:

import pandas as pd
from matplotlib import pyplot as plt

"""read information data from csv file"""


class CsvFile:

    def __init__(self, csv_path, encoding=None):
        self.data = pd.read_csv(csv_path, encoding=encoding)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', 180)
        pd.set_option('max_colwidth', 100)

    def info_feature(self, feature_name):
        self.check_feature(feature_name)
        res = self.data[feature_name].describe()
        print(res)
        return res

    def print_info(self, save_file_path=None):
        """print information about csv file"""
        print("types of data ")
        print(self.data.dtypes)
        print("=" * 100)
        result = self.data.describe()
        print(result)
        if save_file_path is not None:
            # save
            pass
        return result

    def boxplot(self, feature_names: list, show=True, save=True, save_path="boxplot.png"):
        self.data.boxplot(column=feature_names, return_type='axes')
        if show:
            plt.show()
        if save:
            plt.savefig(save_path)

    def drop_features(self, feature_names: list):
        self.data.drop(feature_names, axis=1, inplace=True)

    def head(self, num_row):
        result = self.data.head(num_row)
        print(result)
        return result

    def check_data(self, null=True, feature=True, save_file_path=None):
        """check data in csv file"""
        if null:
            self.check_null()
        if feature:
            self.check_feature()

    def check_null(self):
        result = self.data.apply(lambda x: sum(x.isnull()))
        print(result)
        return result

    def check_feature(self, feature_names: list):
        for feature_name in feature_names:
            print('\n{}: amount of different values as follow\n'.format(feature_name))
            print(self.data[feature_name].value_counts())
            print("{} kinds of value".format(len(self.data[feature_name].unique())))

    def create_null_signal_column(self, feature_name):
        """create a new column to show the other column is null or not
        use to some feature which have many null sample"""
        new_feature_name = feature_name + "_Missing"
        self.data[new_feature_name] = self.data[feature_name].apply(lambda x: 1 if pd.isnull(x) else 0)

    def fillna(self, feature_name, value):
        self.data[feature_name].fillna(value, inplace=True)

    def feature_median(self, feature_name: str):
        return self.data[feature_name].median()

    def merge_other_values(self, feature_name: str, values: list, merge_value='other'):
        self.data[feature_name].apply(lambda x: merge_value if x not in values else x)
