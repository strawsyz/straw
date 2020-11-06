#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/14 19:24
# @Author  : Shi
# @FileName: machine_learning_utils.py
# @Description：
import operator
from collections import deque

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor


def kNN(test_x, data_set, labels, k):
    num_data_set = data_set.shape[0]
    diff_mat = np.tile(test_x, (num_data_set, 1)) - data_set
    distances = (diff_mat ** 2).sum(axis=1) ** 0.5
    indicies_sorted_distance = distances.argsort()
    label_counter = {}
    for i in range(k):
        voted_label = labels[indicies_sorted_distance[i]]
        label_counter[voted_label] = label_counter.get(voted_label, 0) + 1
    return sorted(label_counter.items(),
                  key=operator.itemgetter(1), reverse=True)[0][0]


def kNN_classify(train_data, train_gt_data, valid_data, valid_gt_data=None, k=5):
    predict_data = []
    if valid_gt_data is None:
        for idx, data in enumerate(valid_data):
            predict = kNN(data, train_data, train_gt_data, k)
            predict_data.append(predict)
    else:
        correct = 0
        for data, gt_data in zip(valid_data, valid_gt_data):
            predict = kNN(data, train_data, train_gt_data, k)
            predict_data.append(predict)
            if gt_data == predict:
                correct += 1
        print("correct : {}, number of validation data : {}".format(correct, len(valid_data)))
        print("correct rate : {}".format(correct / len(valid_data)))
    return predict_data


def rmse_cv(model, x, y, scoring="neg_mean_squared_error", cv=5):
    return cross_val_score(model, x, y, scoring=scoring, cv=cv)


def print_cv_result(grid_search):
    print("Best Estimator:\n{}".format(grid_search.best_estimator_))
    print("Best Score :{}".format(grid_search.best_score_))
    print("CV Result : \n{}".format(pd.DataFrame(grid_search.cv_results_)[
        ['params', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'std_fit_time', 'mean_score_time',
         'std_score_time']].sort_values(
        'mean_test_score', ascending=False)))


def normalization(data_set, range_=None, min_val=None):
    if min_val is None or range_ is None:
        min_val = np.min(data_set, axis=0)
        max_val = np.max(data_set, axis=0)
        range_ = max_val - min_val
    norm_data_set = (data_set - min_val) / range_
    return norm_data_set, range_, min_val


def z_score(x, mean=None, std=None, axis=0):
    if mean is None or std is None:
        mean = np.mean(x, axis=axis)
        std = np.std(x, axis=axis)
    # 旋转特定的轴到0之前 ，这段代码可能有问题
    xr = np.rollaxis(x, axis=axis)
    xr -= mean
    xr /= std
    return x, mean, std


def save_model(model, save_path):
    joblib.dump(model, save_path)
    print("save best model at {}".format(save_path))


def models_compare(X, Y):
    """compare different models to train"""
    models = [LinearRegression(), Ridge(), Lasso(alpha=0.01, max_iter=10000), RandomForestRegressor(),
              GradientBoostingRegressor(), SVR(), LinearSVR(),
              ElasticNet(alpha=0.001, max_iter=10000), SGDRegressor(max_iter=1000, tol=1e-3), BayesianRidge(),
              KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
              ExtraTreesRegressor(), XGBRegressor()]
    model_names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinearSVR", "Ela", "SGD", "Bay", "Ker", "Extra", "Xgb"]

    for name, model in zip(model_names, models):
        score = rmse_cv(model, X, Y)
        print("{}: {:6f}, {:6f}".format(name, score.mean(), score.std()))


class GridSearch:
    """use model to compare different parameters"""

    def __init__(self, model, cv=5):
        self.model = model
        self.cv = cv

    def use_grid_search(self, X, Y, param_grid, save_model=True):
        grid_search = GridSearchCV(self.model, param_grid, cv=self.cv, scoring="neg_mean_squared_error")
        grid_search.fit(X, Y)
        best_estimator = grid_search.best_estimator_
        # save model
        if save_model:
            save_path = str(best_estimator)
            self.save_model(best_estimator, save_path)
        print("best estimator is {} ,best score is {}".format(best_estimator,
                                                              np.sqrt(-grid_search.best_score_)))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        # print result of training and sort by mean_test_score
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']].sort_values(
            'mean_test_score'))

    def save_model(self, model, save_path):
        joblib.dump(model, save_path)
        print("save best model at {}".format(save_path))


class AverageWeight(BaseEstimator, RegressorMixin):
    """use different models with weights to train"""

    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def fit(self, X, Y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, Y)

    def predict(self, X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        for data in range(pred.shape[1]):
            single = [pred[model, data] * weight for model, weight in zip(range(pred.shape[0]), self.weights)]
            w.append(np.sum(single))
        return w


class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, meta_model):
        self.models = models
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=0, shuffle=True)

    def fit(self, X, Y):
        self.saved_model = [deque(maxlen=5) for _ in self.models]
        # save every result from predict
        train_result = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            for train_index, val_index in self.kf.split(X, Y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], Y[train_index])
                self.saved_model[i].append(renew_model)
                train_result[val_index, i] = renew_model.predict(X[val_index])
        # train meta model
        self.meta_model.fit(train_result, Y)
        return self

    def predict(self, X):
        whole_test = np.column_stack(
            [np.column_stack([model.predict(X)
                              for model in single_model]).mean(axis=1)
             for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)

    def get_oof(self, X, Y, test_X):
        oof = np.zeros((X.shape[0], len(self.models)))
        test_single = np.zeros((test_X.shape[0], 5))

        test_mean = np.zeros((test_X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            for j, (train_index, val_index) in enumerate(self.kf.split(X, Y)):
                clone_model = clone(model)
                # train model
                clone_model.fit(X[train_index], Y[train_index])
                # predict for validation dataset
                oof[val_index, i] = clone_model.predict(X[val_index])
                test_single[:, j] = clone_model.predict(test_X)
            # save the mean of every model
            test_mean[:, i] = test_single.mean(axis=1)
        return oof, test_mean


def grid_usage():
    X = np.random.randn(100, 10)
    Y = np.random.randn(100)

    GridSearch(Lasso()).use_grid_search(X, Y, {'alpha': [0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009],
                                               'max_iter': [10000]})
    GridSearch(Ridge()).use_grid_search(X, Y, {'alpha': [35, 40, 45, 50, 55, 60, 65, 70, 80, 90]})
    GridSearch(SVR()).use_grid_search(X, Y, {'C': [11, 12, 13, 14, 15], 'kernel': ['rbf'], 'gamma': [0.0003, 0.0004],
                                             'epsilon': [0.008, 0.009]})
    params = {'alpha': [0.2, 0.3, 0.4, 0.5], 'kernel': ['polynomial'], 'degree': [3],
              'coef0': [0.8, 1, 1.2]}
    GridSearch(KernelRidge()).use_grid_search(X, Y, params)
    GridSearch(ElasticNet()).use_grid_search(X, Y,
                                             {'alpha': [0.0005, 0.0008, 0.004, 0.005],
                                              'l1_ratio': [0.08, 0.1, 0.3, 0.5, 0.7],
                                              'max_iter': [10000]})


def average_weight_usage():
    X = np.random.randn(100, 10)
    Y = np.random.randn(100)
    # initialize models
    lasso = Lasso(alpha=0.0005, max_iter=10000)
    ridge = Ridge(alpha=60)
    svr = SVR(gamma=0.0004, kernel='rbf', C=13, epsilon=0.009)
    ker = KernelRidge(alpha=0.2, kernel='polynomial', degree=3, coef0=0.8)
    ela = ElasticNet(alpha=0.005, l1_ratio=0.08, max_iter=10000)
    bay = BayesianRidge()
    # initialize weights
    w1 = 0.02
    w2 = 0.2
    w3 = 0.25
    w4 = 0.3
    w5 = 0.03
    w6 = 0.2

    weight_avg = AverageWeight(models=[lasso, ridge, svr, ker, ela, bay], weights=[w1, w2, w3, w4, w5, w6])
    res = rmse_cv(weight_avg, X, Y)

    result = res.mean()
    return result


def stacking_usage():
    X = np.random.randn(100, 10)
    Y = np.random.randn(100)
    test_X = np.random.randn(20, 10)
    lasso = Lasso(alpha=0.0005, max_iter=10000)
    ridge = Ridge(alpha=60)
    svr = SVR(gamma=0.0004, kernel='rbf', C=13, epsilon=0.009)
    ker = KernelRidge(alpha=0.2, kernel='polynomial', degree=3, coef0=0.8)
    ela = ElasticNet(alpha=0.005, l1_ratio=0.08, max_iter=10000)
    bay = BayesianRidge()
    stack_model = stacking(models=[lasso, ridge, svr, ker, ela, bay], meta_model=ker)
    result = rmse_cv(stack_model, X, Y)
    print(result)
    print(result.mean())

    X_train_stack, X_test_stack = stack_model.get_oof(X, Y, test_X)
    x_train_add = np.hstack((X, X_train_stack))

    print(rmse_cv(stack_model, x_train_add, Y))
    print(rmse_cv(stack_model, x_train_add, Y).mean())
