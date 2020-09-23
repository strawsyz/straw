#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/14 19:24
# @Author  : Shi
# @FileName: machine_learning_utils.py
# @Description：

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from scipy.stats import skew
from collections import deque


def rmse_cv(model, x, y):
    return np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))


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


class GridSearch():
    """use model to compare different parameters"""
    def __init__(self, model,cv=5):
        self.model = model
        self.cv = cv

    def grid_get(self, X, Y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=self.cv, scoring="neg_mean_squared_error")
        grid_search.fit(X, Y)
        print(grid_search.best_estimator_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        # print result of training and sort by mean_test_score
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']].sort_values(
            'mean_test_score'))


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

    GridSearch(Lasso()).grid_get(X, Y, {'alpha': [0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009], 'max_iter': [10000]})
    GridSearch(Ridge()).grid_get(X, Y, {'alpha': [35, 40, 45, 50, 55, 60, 65, 70, 80, 90]})
    GridSearch(SVR()).grid_get(X, Y, {'C': [11, 12, 13, 14, 15], 'kernel': ['rbf'], 'gamma': [0.0003, 0.0004],
                                'epsilon': [0.008, 0.009]})
    params = {'alpha': [0.2, 0.3, 0.4, 0.5], 'kernel': ['polynomial'], 'degree': [3],
              'coef0': [0.8, 1, 1.2]}
    GridSearch(KernelRidge()).grid_get(X, Y, params)
    GridSearch(ElasticNet()).grid_get(X, Y, {'alpha': [0.0005, 0.0008, 0.004, 0.005], 'l1_ratio': [0.08, 0.1, 0.3, 0.5, 0.7],
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


