#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/1 11:29
# @Author  : strawsyz
# @File    : signate1.py
# @desc:

import os
import matplotlib.pylab as plt
import torch
import xgboost as xgb
from sklearn import metrics
from sklearn.svm import SVR, LinearSVR, LinearSVC
from torch import optim
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss
from xgboost.sklearn import XGBClassifier

from utils.machine_learning_utils import *


def year_2_int(year: str):
    year = year.replace("years", "").replace("year", "").strip()
    return int(year)


def str_2_int_4_application_type(application_type: str):
    if application_type == "Joint App":
        return 1
    elif application_type == "Individual":
        return 2
    else:
        return 0


def str_2_int_4_purpose(purpose: str):
    choices = ['small_business', 'house', 'medical', 'home_improvement', 'car', 'debt_consolidation', 'other',
               'credit_card', 'major_purchase']
    for idx, choice in enumerate(choices, start=1):
        if choice == purpose:
            return idx
    return 0


def str_2_int_4_loan_status(loan_status: str):
    if loan_status == "ChargedOff":
        return 1
    elif loan_status == "FullyPaid":
        return 0
    else:
        return None


def str_2_int_4_grade(grade: str):
    grade_level_ = grade[0]
    grade_no_ = int(grade[1])
    grade_levels = ["A", "B", "C", "D", "E", "F"]
    for idx, grade_level in enumerate(grade_levels):
        if grade_level_ == grade_level:
            return idx * 10 + grade_no_
    return None


def preprocess(train_data_path: str = r"",
               test_data_path: str = r""):
    df = pd.read_csv(train_data_path)
    df["purpose"] = df["purpose"].map(str_2_int_4_purpose)
    df["application_type"] = df["application_type"].map(str_2_int_4_application_type)
    df["term"] = df["term"].map(year_2_int)
    df["employment_length"] = df["employment_length"].map(year_2_int)
    df["grade"] = df["grade"].map(str_2_int_4_grade)
    df["loan_status"] = df["loan_status"].map(str_2_int_4_loan_status)

    # standard data
    # one hot
    df.to_csv(r"")

    df = pd.read_csv(test_data_path)
    df["purpose"] = df["purpose"].map(str_2_int_4_purpose)
    df["application_type"] = df["application_type"].map(str_2_int_4_application_type)
    df["term"] = df["term"].map(year_2_int)
    df["employment_length"] = df["employment_length"].map(year_2_int)
    df["grade"] = df["grade"].map(str_2_int_4_grade)
    df.to_csv(r"")


def read_train_data(file_path: str = r"", num: int = None):
    df = pd.read_csv(file_path)
    df["purpose"] = df["purpose"].map(str_2_int_4_purpose)
    df["application_type"] = df["application_type"].map(str_2_int_4_application_type)
    df["term"] = df["term"].map(year_2_int)
    df["employment_length"] = df["employment_length"].map(year_2_int)
    df["grade"] = df["grade"].map(str_2_int_4_grade)
    df["loan_status"] = df["loan_status"].map(str_2_int_4_loan_status)
    # print(df.info())
    # print(df.describe())
    # print(df.corr())
    # print(df.corr("kendall"))
    # print(df.corr("spearman"))

    train_data = df.loc[:, ['loan_amnt', 'term', 'interest_rate', 'grade',
                            'employment_length', 'purpose', 'credit_score', 'application_type']]
    # all_data = df.loc[:, ['term', 'interest_rate', 'grade', "loan_status"]]
    # all_data = np.asarray(all_data)
    # import random
    # random.shuffle(all_data)
    # train_data = all_data[:, :3]
    # train_gt = all_data[:, 3]

    # train_data = df.loc[:, ['term', 'interest_rate', 'grade', 'credit_score']]
    train_data = np.array(train_data)
    if num is not None:
        train_data = train_data[:num]
    train_data, range_, min_val = z_score(train_data)
    # df.loc[:, ['term', 'interest_rate', 'grade', 'credit_score']] = train_data
    df.loc[:, ['loan_amnt', 'term', 'interest_rate', 'grade',
               'employment_length', 'purpose', 'credit_score', 'application_type']] = train_data
    train_gt = df.loc[:, "loan_status"]
    train_gt = np.array(train_gt)
    if num is not None:
        train_gt = train_gt[:num]
    return train_data, train_gt, range_, min_val


def read_test_data(range_, min_val, file_path=r""):
    df = pd.read_csv(file_path)
    df["purpose"] = df["purpose"].map(str_2_int_4_purpose)
    df["application_type"] = df["application_type"].map(str_2_int_4_application_type)
    df["term"] = df["term"].map(year_2_int)
    df["employment_length"] = df["employment_length"].map(year_2_int)
    df["grade"] = df["grade"].map(str_2_int_4_grade)

    test_data = df.loc[:, ['loan_amnt', 'term', 'interest_rate', 'grade',
                           'employment_length', 'purpose', 'credit_score', 'application_type']]
    # test_data = df.loc[:, ['term', 'interest_rate', 'grade',
    #                        'credit_score']]
    test_data = np.array(test_data)
    test_data, _, _ = z_score(test_data, range_, min_val)
    return test_data


def create_model(gpu=True):
    from net_structures.FNN import AdaptiveFNN
    model = AdaptiveFNN(num_in=8, num_out=2, num_units=[8, 16, 32, 16, 8, 4, 1], activate_func="sigmoid")
    if gpu:
        model.cuda()
    return model


def save_predict_data(data, save_path, file_path=r""):
    df = pd.read_csv(file_path, header=None, index_col=0)
    save_path = os.path.join("C:\(lab\datasets\signate01", save_path)
    length = len(data)
    df.iloc[:length, 0] = data
    print("save predict data at {}".format(save_path))
    df.to_csv(save_path, header=None)


def ML_01(X, Y):
    GridSearch(Lasso()).use_grid_search(X, Y,
                                        {'alpha': [0.0000001, 0.000002, 0.0000003, 0.0000004, 0.000005],
                                         'max_iter': [10000]})
    GridSearch(Ridge()).use_grid_search(X, Y, {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5]})
    # GridSearch(SVR()).grid_get(X, Y, {'C': [11], 'kernel': ['rbf'], 'gamma': [0.0003, 0.0004],
    #                                   'epsilon': [0.008, 0.009]})
    # params = {'alpha': [0.2, 0.3, 0.4, 0.5], 'kernel': ['polynomial'], 'degree': [3],
    #           'coef0': [0.8, 1, 1.2]}
    # GridSearch(KernelRidge()).grid_get(X, Y, params)
    GridSearch(ElasticNet()).use_grid_search(X, Y,
                                             {'alpha': [0.0005, 0.0008, 0.004, 0.005],
                                              'l1_ratio': [0.08, 0.1, 0.3, 0.5, 0.7],
                                              'max_iter': [10000]})


def create_data_loader(data, batch_size):
    batch_data = []
    for idx, item in enumerate(data, start=1):
        batch_data.append(item)
        if idx % batch_size == 0:
            yield np.asarray(batch_data)
            batch_data = []


def normalization(data_set, range_=None, min_val=None, axis=0):
    if min_val is None or range_ is None:
        min_val = np.min(data_set, axis=axis)
        max_val = np.max(data_set, axis=axis)
        range_ = max_val - min_val
    norm_data_set = (data_set - min_val) / range_
    return norm_data_set, range_, min_val


def z_score(x, mean=None, std=None, axis=0):
    """ 这段代码可能有问题 """
    if mean is None or std is None:
        mean = np.mean(x, axis=axis)
        std = np.std(x, axis=axis)
    xr = np.rollaxis(x, axis=axis)
    xr -= mean
    xr /= std
    return x, mean, std


def ML_02(X, Y, test_X):
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


def ML_03(X, Y, test_data=None, test_gt=None, cv=5):
    """compare different models to train"""
    # models = [LinearRegression(), Ridge(), Lasso(alpha=0.01, max_iter=10000), RandomForestRegressor(),
    #           GradientBoostingRegressor(), SVR(), LinearSVR(),
    #           ElasticNet(alpha=0.001, max_iter=10000), SGDRegressor(max_iter=1000, tol=1e-3), BayesianRidge(),
    #           KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
    #           ExtraTreesRegressor(), XGBRegressor()]
    # model_names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinearSVR", "Ela", "SGD", "Bay", "Ker", "Extra", "Xgb"]
    # for name, model in zip(model_names, models):
    #     score = rmse_cv(model, X, Y)
    model_dict = [
        ("LinearSVC", LinearSVC()),
        ("RF", RandomForestRegressor()),
        ("LinearSVR", LinearSVR(max_iter=10000)),
        ("Extra", ExtraTreesRegressor())]

    # model_dict = [("LinearSVC", LinearSVC()), ("LR", LinearRegression()), ("Ridge", Ridge()),
    #               ("Lasso", Lasso(alpha=0.01, max_iter=10000)),
    #               ("RF", RandomForestRegressor()), ("GBR", GradientBoostingRegressor()),
    #               ("LinearSVR", LinearSVR(max_iter=10000)), ("Ela", ElasticNet(alpha=0.001, max_iter=10000)),
    #               ("SGD", SGDRegressor(max_iter=1000, tol=1e-3)), ("Bay", BayesianRidge()),
    #               ("Extra", ExtraTreesRegressor()), ("Xgb", XGBRegressor())]
    for name, model in model_dict:
        score = rmse_cv(model, X, Y, scoring="f1", cv=cv)
        print("{} 's score is {}".format(name, score))

        # model.fit(X, Y)
        # # for test_data_item in test_data:
        # predict_data = model.predict(test_data)
        # predict_data = np.where(predict_data < 0.5, 0, 1)
        # # save_predict_data(predict_data, "{}.csv".format(name))
        # if test_gt is not None:
        #     # print(predict_data)
        #     # print(test_gt)
        #     print("num of correct data is {}".format(np.sum(predict_data == test_gt)))
        #     print("num of valid data is {}".format(len(test_gt)))
        #     print(np.sum(predict_data == test_gt) / len(test_gt))
        #     save_model(model, "{}.pkl".format(name))
        # print(predict_data)
        # print("{}: {:6f}, {:6f}".format(name, score.mean(), score.std()))


def ML_04(X, Y, test_X):
    linear_svc = LinearSVC()
    rf = RandomForestRegressor()
    linear_svr = LinearSVR(max_iter=10000)
    extra = ExtraTreesRegressor()
    ker = KernelRidge(alpha=0.2, kernel='polynomial', degree=3, coef0=0.8)
    stack_model = stacking(models=[linear_svc, rf, linear_svr, extra], meta_model=ker)
    result = rmse_cv(stack_model, X, Y)
    print(result)
    print(result.mean())

    X_train_stack, X_test_stack = stack_model.get_oof(X, Y, test_X)
    x_train_add = np.hstack((X, X_train_stack))

    print(rmse_cv(stack_model, x_train_add, Y))
    print(rmse_cv(stack_model, x_train_add, Y).mean())


def FNN(X, Y, valid_data=None, valid_gt=None, display_iter=10000):
    model = create_model()
    EPOCH = 50
    batch_size = 8
    lr = 0.3

    loss_function = BCEWithLogitsLoss()
    num_train = len(X)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(model)
    for epoch in range(EPOCH):
        model.train()
        x_dataloader = create_data_loader(X, batch_size=batch_size)
        y_dataloader = create_data_loader(Y, batch_size=batch_size)
        sum_loss = 0
        for idx, (x_item, y_item) in enumerate(zip(x_dataloader, y_dataloader), start=1):
            print(x_item)
            print(y_item)
            optimizer.zero_grad()
            x_item = Variable(torch.from_numpy(x_item)).float().cuda()
            y_item = Variable(torch.from_numpy(y_item)).float().cuda()
            # y_item = torch.squeeze(y_item)
            y_item = torch.unsqueeze(y_item, dim=1)
            predict = model(x_item)
            loss = loss_function(predict, y_item)
            loss.backward()
            optimizer.step()
            if idx % display_iter == 0:
                print("predic is {},out is {},loss is {}".format(predict, y_item, loss.data))
            sum_loss += loss
        ave_loss = sum_loss / (int(num_train / batch_size))
        print("epoch is {},sum_loss is {},ave_loss is {}".format(epoch, sum_loss, ave_loss))

        # valid
        if valid_data is not None:
            model.eval()
            num_valid = len(valid_data)
            sum_loss = 0
            x_dataloader = create_data_loader(valid_data, batch_size=batch_size)
            y_dataloader = create_data_loader(valid_gt, batch_size=batch_size)
            for idx, (x_item, y_item) in enumerate(zip(x_dataloader, y_dataloader), start=1):
                x_item = Variable(torch.from_numpy(x_item)).float().cuda()
                y_item = Variable(torch.from_numpy(y_item)).float().cuda()
                y_item = torch.unsqueeze(y_item, dim=1)
                predict = model(x_item)
                loss = loss_function(predict, y_item)
                sum_loss += loss
            ave_loss = sum_loss / (int(num_valid / batch_size))
            print("Valid sum_loss is {}, ave_loss is {}".format(sum_loss, ave_loss))


import operator


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


def classify(train_data, train_gt_data, valid_data, valid_gt_data=None, k=5):
    if valid_gt_data is None:
        predict_data = []
        for idx, data in enumerate(valid_data):
            predict = kNN(data, train_data, train_gt_data, k)
            predict_data.append(predict)
            if idx % 1000:
                print(predict)
        return predict_data
    else:
        correct = 0

        for data, gt_data in zip(valid_data, valid_gt_data):
            predict = kNN(data, train_data, train_gt_data, k)
            if gt_data == predict:
                correct += 1
                print("correct")
            else:
                print("error")
        print(correct / len(valid_data))


def split_train_valid(data, gt_data, valid_rate=0.2):
    num_train = len(data)
    num_valid = int(num_train * valid_rate)
    num_train = num_train - num_valid
    valid_data = data[num_train:]
    valid_gt_data = gt_data[num_train:]
    data = data[:num_train]
    gt_data = gt_data[:num_train]
    return data, gt_data, valid_data, valid_gt_data


def use_ML_03():
    valid_rate = 0.99
    train_data, gt_data, range_, min_val = read_train_data()
    # test_data = read_test_data(range_, min_val)
    # ML_03(train_data, gt_data, test_data)

    train_data, gt_data, valid_data, valid_gt_data = split_train_valid(train_data, gt_data, valid_rate)
    ML_03(train_data, gt_data, valid_data, valid_gt_data)


def use_ML_04():
    valid_rate = 0.2
    train_data, gt_data, range_, min_val = read_train_data()
    train_data, gt_data, valid_data, valid_gt_data = split_train_valid(train_data, gt_data, valid_rate)
    ML_04(train_data, gt_data, valid_data)


def use_KNN(test=False):
    k = 11
    train_data, gt_data, range_, min_val = read_train_data()
    if test:
        valid_rate = 0.2
        train_data, gt_data, valid_data, valid_gt_data = split_train_valid(train_data, gt_data, valid_rate)
        predict = classify(train_data, gt_data, valid_data, valid_gt_data, k=k)
    else:
        test_data = read_test_data(range_, min_val)
        predict = classify(train_data, gt_data, test_data, valid_gt_data=None, k=k)
        save_predict_data(predict, save_path="k={}.csv".format(k))
    print(k)


def XGB_CV(train_data, train_gt, params: dict = None, scoring: str = "f1", cv=5):
    estimator = XGBClassifier(learning_rate=0.1, n_estimators=20, max_depth=5,
                              min_child_weight=1, gamma=0, subsample=0.8,
                              colsample_bytree=0.8,
                              objective='binary:logistic', nthread=8, scale_pos_weight=1,
                              seed=27)
    for key in list(params):
        if len(params[key]) == 1:
            setattr(estimator, key, params[key][0])
            del params[key]
    grid_search = GridSearchCV(estimator=estimator, param_grid=params, scoring=scoring, n_jobs=8, iid=False, cv=cv)
    grid_search.fit(train_data, train_gt)
    print_gridcv_result(grid_search)
    # get the best estimator
    best_estimator = grid_search.best_estimator_
    return best_estimator


def print_cv_result(grid_search):
    print("Best Estimator:\n{}".format(grid_search.best_estimator_))
    print("Best Score :{}".format(grid_search.best_score_))
    print("CV Result : \n{}".format(pd.DataFrame(grid_search.cv_results_)[
        ['params', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'std_fit_time', 'mean_score_time',
         'std_score_time']].sort_values(
        'mean_test_score', ascending=False)))


def model_fit(model, train_df, predictors, targets, useTrainCV=False, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(train_df[predictors].values, label=train_df[targets].values)
        # xgtest = xgb.DMatrix(test_df[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds, show_progress=False)
        model.set_params(n_estimators=cvresult.shape[0])

    model.fit(train_df[predictors], train_df[targets], eval_metric='auc')

    dtrain_predictions = model.predict(train_df[predictors])
    dtrain_predprob = model.predict_proba(train_df[predictors])[:, 1]
    print("accuracy : %.4g" % metrics.accuracy_score(train_df[targets].values, dtrain_predictions))
    print("f1 : %.4g" % metrics.f1_score(train_df[targets].values, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(train_df[targets], dtrain_predprob))

    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


def use_xg(data_path=r"",
           predict=['term', 'interest_rate', 'grade', 'credit_score'],
           target=["loan_status"]):
    train = pd.read_csv(data_path)
    xgb = XGBClassifier(
        learning_rate=0.1,
        n_estimators=140,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    model_fit(xgb, train, predict, target)


def use_xgb_cv():
    train = pd.read_csv(r"")
    # feature_names = ['term', 'interest_rate', 'grade', 'credit_score', 'purpose', 'loan_amnt', 'application_type',
    #                  'employment_length']
    feature_names = ['loan_amnt', 'term', 'interest_rate', 'employment_length', 'credit_score', 'grade_A1', 'grade_A2',
                     'grade_A3', 'grade_A4', 'grade_A5',
                     'grade_B1', 'grade_B2', 'grade_B3', 'grade_B4', 'grade_B5', 'grade_C1', 'grade_C2', 'grade_C3',
                     'grade_C4', 'grade_C5', 'grade_D1', 'grade_D2', 'grade_D3', 'grade_D4',
                     'grade_D5', 'grade_E1', 'grade_E2', 'grade_E3', 'grade_E4', 'grade_E5', 'grade_F3', 'grade_F5',
                     'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation',
                     'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical',
                     'purpose_moving', 'purpose_other', 'purpose_small_business',
                     'application_type_Individual', 'application_type_Joint App']
    train_data = train[feature_names]
    train_gt = train[["loan_status"]]
    print("num of all data : {}".format(len(train_data)))
    # read_train_data()
    valid_rate = 0.5
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

    best_model = XGB_CV(train_data, train_gt, scoring="f1", cv=3, params=params)
    # validation
    print("=" * 20 + "validation" + "=" * 20)
    predict = best_model.predict(valid_data)
    print("num of 1 in validation: {}".format(np.sum(predict)))
    print("f1 Socre in validation: {}".format(metrics.f1_score(valid_gt.values, predict)))

    # test
    test = pd.read_csv(r"")
    test_data = test[feature_names]
    print("num of test dataset : {}".format(len(test_data)))
    predict = best_model.predict(test_data)
    print("num of 1 : {} in test dataset".format(np.sum(predict)))

    # save predict data
    save_predict_data(predict, save_path=r"")


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 180)
    pd.set_option('max_colwidth', 100)
    use_xgb_cv()
