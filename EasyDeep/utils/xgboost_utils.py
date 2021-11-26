#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/6 17:10
# @Author  : strawsyz
# @File    : xgboost_utils.py
# @desc:

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import matplotlib.pylab as plt
from utils.machine_learning_utils import print_gridcv_result

""" use xgboost to train, validation and test"""


def model_fit(model, train_df, feature_names, target_names, test_df=None, use_cv=False, cv_folds=5,
              early_stopping_rounds=50):
    if use_cv:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(train_df[feature_names].values, label=train_df[target_names].values)
        print("num of estimators : {}".format(model.get_params()['n_estimators']))
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        print("cv Result is \n{}".format(cvresult))
        model.set_params(n_estimators=cvresult.shape[0])
        print("new num of estimators : {}".format(cvresult.shape[0]))

    model.fit(train_df[feature_names], train_df[target_names], eval_metric='auc')

    # use validation dataset to test
    if test_df is None:
        print("use test dataset to test")
        predictions = model.predict(train_df[feature_names])
        predprob = model.predict_proba(train_df[feature_names])[:, 1]
        print("accuracy : {}".format(metrics.accuracy_score(train_df[target_names].values, predictions)))
        print("f1 : {}".format(metrics.f1_score(train_df[target_names].values, predictions)))
        print("roc_auc_score: {}".format(metrics.roc_auc_score(train_df[target_names], predprob)))
    else:
        print("use train dataset to test")
        predictions = model.predict(test_df[feature_names])
        predprob = model.predict_proba(test_df[feature_names])[:, 1]
        print("accuracy : {}".format(metrics.accuracy_score(test_df[target_names].values, predictions)))
        print("f1 : {}".format(metrics.f1_score(test_df[target_names].values, predictions)))
        print("roc_auc_score on test dataset: {}".format(metrics.roc_auc_score(test_df[target_names], predprob)))

    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


def XGB_CV(train_data, train_gt, params: dict = None, scoring: str = "f1", cv=5, n_jobs=8, seed=0):
    estimator = XGBClassifier(learning_rate=0.1, n_estimators=20, max_depth=5,
                              min_child_weight=1, gamma=0, subsample=0.8,
                              colsample_bytree=0.8,
                              objective='binary:logistic', n_jobs=n_jobs, scale_pos_weight=1, seed=seed)
    for key in list(params):
        if len(params[key]) == 1:
            setattr(estimator, key, params[key][0])
            del params[key]
    grid_search = GridSearchCV(estimator=estimator, param_grid=params, scoring=scoring, n_jobs=n_jobs, iid=False, cv=cv)
    grid_search.fit(train_data, train_gt)
    print_gridcv_result(grid_search)
    # get the best estimator
    best_estimator = grid_search.best_estimator_
    return best_estimator


def use_xgb():
    data_path = r"csv file path"
    train = pd.read_csv(data_path)
    feature_names = ['feature_name0', 'feature_name1']
    target_names = ["target_name0"]
    xgb = XGBClassifier(
        learning_rate=0.1,
        n_estimators=150,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    model_fit(xgb, train, feature_names, target_names, use_cv=False)


def use_xgb_cv():
    # init
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 180)
    pd.set_option('max_colwidth', 100)

    train_cvs_file_path = r""
    test_csv_file_path = r""
    train = pd.read_csv(train_cvs_file_path)
    feature_names = ['feature_name0', "feature_name1"]
    target_names = ["target_name0"]
    train_data = train[feature_names]
    train_gt = train[[target_names]]
    valid_rate = 0.5
    scoring = "f1"
    cv = 3

    print("num of all data : {}".format(len(train_data)))
    num_valid = int(len(train_data) * valid_rate)
    valid_data = train_data.loc[:num_valid]
    valid_gt = train_gt.loc[:num_valid]
    train_data = train_data.loc[num_valid:]
    train_gt = train_gt.loc[num_valid:]
    print("num of train dataset : {}".format(len(train_data)))
    print("num of validation dataset : {}".format(num_valid))

    # find best model
    params = {'max_depth': [22],
              'min_child_weight': [2],
              'gamma': [0],
              'subsample': [0.85],
              'colsample_bytree': [0.8],
              'learning_rate': [0.0003],
              'n_estimators': [125],
              "scale_pos_weight": [6.5, 7, 7.5]
              }

    best_model = XGB_CV(train_data, train_gt, scoring=scoring, cv=cv, params=params)
    # validation
    print("=" * 20 + "validation" + "=" * 20)
    predict = best_model.predict(valid_data)
    print("num of 1 in validation: {}".format(np.sum(predict)))
    print("f1 Socre in validation: {}".format(metrics.f1_score(valid_gt.values, predict)))

    # test
    test = pd.read_csv(test_csv_file_path)
    test_data = test[feature_names]
    print("num of test dataset : {}".format(len(test_data)))
    predict = best_model.predict(test_data)
    return predict

