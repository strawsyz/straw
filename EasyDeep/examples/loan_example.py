#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/4 15:06
# @Author  : strawsyz
# @File    : loan_example.py
# @desc:
import time

import numpy as np
import torch
from torch.autograd import Variable

from configs.experiment_config import LoanConfig
from experiments.deep_experiment import DeepExperiment
from utils.log_utils import get_logger
import pandas as pd
import os

"""predict load_status"""
def save_predict_data(data, save_path, file_path=r""):
    df = pd.read_csv(file_path, header=None, index_col=0)
    save_path = os.path.join("", save_path)
    length = len(data)
    df.iloc[:length, 0] = data
    print(save_path)
    df.to_csv(save_path, header=None)


class LoanExperiment(LoanConfig, DeepExperiment):
    def __init__(self, expand_config: dict = {}):
        super(LoanExperiment, self).__init__()
        self.logger = get_logger()
        self.is_bigger_better = False
        for key in expand_config:
            setattr(self, key, expand_config[key])

    def test(self, prepare_dataset=True, prepare_net=True, save_predict_result=True, result_save_path=None,
             pretrain_model_path=None):

        if pretrain_model_path is not None:
            self.pretrain_path = pretrain_model_path
        self.before_test(prepare_dataset, prepare_net)
        self.logger.info("=" * 10 + " test start " + "=" * 10)
        pps = 0
        loss = 0
        all_predict_data = []
        with torch.no_grad():
            for i, X in enumerate(self.test_loader()):
                pps_batch, predict = self.test_one_batch(X)
                all_predict_data.extend(predict)
                pps += pps_batch
        if save_predict_result:
            save_predict_data(all_predict_data, save_path=result_save_path)
        sum = len(self.test_loader)
        pps /= sum
        loss /= sum
        self.logger.info("average predict {} sample per second".format(pps))
        self.logger.info("=" * 10 + " testing end " + "=" * 10)
        return self.result_save_path

    def test_one_batch(self, X, Y=None):
        x_item = Variable(torch.from_numpy(X)).float().cuda()
        start_time = time.time()
        out = self.net_structure(x_item)
        predict = np.where(out.cpu().numpy() < 0, -1, 1)
        print(predict)
        predict = np.squeeze(predict).tolist()
        time.sleep(0.00001)
        pps = len(X) / (time.time() - start_time)

        return pps, predict

    def train_one_batch(self, X, Y, *args, **kwargs):
        self.optimizer.zero_grad()
        x_item = Variable(torch.from_numpy(X)).float().cuda()
        y_item = Variable(torch.from_numpy(Y)).float().cuda()
        predict = self.net_structure(x_item)
        # one_arr = torch.ones(predict.shape).cuda()
        # minus_one_arr = torch.full(predict.shape, -1.0).cuda()

        # predict = torch.where(predict < 0, minus_one_arr, one_arr)
        loss = self.loss_function(predict, y_item)
        # print("loss is {}".format(loss))
        loss.backward()
        self.optimizer.step()
        return [loss.data]

    def train_one_epoch(self, epoch):
        """train one epoch"""
        train_loss = 0
        self.net_structure.train()
        other_param = self.other_param_4_batch
        tmp_idx = 0
        for idx, (sample, label) in enumerate(self.train_loader(), start=1):
            other_param = self.train_one_batch(sample, label, other_param)
            loss = other_param[0]
            train_loss += loss
            if idx % self.num_iter == 0:
                self.logger.info("EPOCH : {}\t iter：{}, loss_data : {:.6f}".format(epoch, idx, train_loss / idx))
            tmp_idx = idx
        print(tmp_idx)
        train_loss = train_loss / tmp_idx
        self.logger.info("EPOCH : {}\t train_loss : {:.6f}\t".format(epoch, train_loss))
        if self.scheduler is not None:
            self.scheduler.step()
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.logger.info("learning rate is {:.6f}".format(lr))
        return train_loss

    def valid_one_epoch(self, epoch=0):
        """validate one epoch"""
        self.net_structure.eval()
        with torch.no_grad():
            valid_loss = 0
            acc = 0
            for idx, (x, y) in enumerate(self.valid_loader(), start=1):
                batch_valid_loss, batch_acc = self.valid_one_batch(x, y)
                valid_loss += batch_valid_loss
                acc += batch_acc
            valid_loss /= idx
            acc /= idx
            print(acc)
            self.logger.info("Epoch:{}\t valid_loss:{:.6f},acc:{:.6f}".format(epoch, valid_loss, acc))
        return valid_loss

    def valid_one_batch(self, X, Y, *args, **kwargs):
        x_item = Variable(torch.from_numpy(X)).float().cuda()
        y_item = Variable(torch.from_numpy(Y)).float().cuda()
        # y_item = torch.unsqueeze(y_item, dim=1)
        self.optimizer.zero_grad()
        predict = self.net_structure(x_item)
        bin_predict = np.sum(np.where(predict.cpu().numpy() < 0, -1, 1) == Y) / (len(x_item)*2)
        # print("bin_predict is {}".format(bin_predict))
        # print(np.sum(np.where(predict.cpu().numpy() < 0.5, 0, 1) == Y) / 64)
        return self.loss_function(predict, y_item), bin_predict
