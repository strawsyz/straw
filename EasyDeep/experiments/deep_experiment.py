import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from base.base_checkpoint import BaseCheckPoint
from base.base_experiment import BaseExpriment
from configs.experiment_config import DeepConfig as config
from dataset import PolypDataset as dataset
from datasets.image_dataset import ImageDataSet as dataset
from models import FCN
from utils import file_utils
from utils import time_utils
from utils.net_utils import save, load
from utils.utils_ import copy_attr


class HistoryRecorder:
    def __init__(self, train_loss, valid_loss):
        self.train_loss = train_loss
        self.valid_loss = valid_loss


class DeepExperiment(BaseExpriment):
    def __init__(self):
        super(DeepExperiment, self).__init__()
        # 父类已经做了
        # self.load_config()
        # self.logger = Logger.get_logger()
        # if not load net from pretrained model, then 0
        self.current_epoch = 0
        self.recorder = HistoryRecorder
        self.history = {}
        self.init_params()

    def init_params(self):
        self.scheduler_step_size = 15
        self.scheduler_gamma = 0.8

    def load_config(self):
        copy_attr(config(), self)

    def prepare_data(self, test_model=False):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)  # cpu
            torch.cuda.manual_seed(self.random_state)  # gpu
            np.random.seed(self.random_state)  # numpy
            random.seed(self.random_state)  # random and transforms
            torch.backends.cudnn.deterministic = True  # cudnn

        if not test_model:
            self.train_data = dataset()
            if self.num_train is not None:
                if len(self.train_data) > self.num_train:
                    self.train_data.set_data_num(self.num_train)
                else:
                    self.num_train = len(self.train_data)
            else:
                self.num_train = len(self.train_data)
            if self.valid_rate is None:
                self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
                return self.train_loader
            else:
                num_valid_data = int(self.num_train * self.valid_rate)
                if num_valid_data == 0 or num_valid_data == self.num_train:
                    self.logger.error("valid datateset is None or train dataset is None")
                self.train_data, self.val_data = torch.utils.data.random_split(self.train_data,
                                                                               [self.num_train - num_valid_data,
                                                                                num_valid_data])
                self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
                self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)
        else:
            self.test_data = dataset(test_model=True)
            if self.num_test is not None:
                self.test_data.set_data_num(self.num_test)
            else:
                self.num_test = len(self.test_data)
            self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size4test, shuffle=True)

    def prepare_net(self):
        self.net = FCN(n_out=1)
        self.loss_function = nn.BCEWithLogitsLoss()
        if self.is_use_gpu:
            self.net = self.net.cuda()
            self.loss_function = self.loss_function.cuda()

        if self.optim_name == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step_size,
                                                         gamma=self.scheduler_gamma)
        if self.is_pretrain:
            self.load()

    def train_one_epoch(self, epoch):
        self.net.train()
        train_loss = 0
        for image, mask in self.train_loader:
            if self.is_use_gpu:
                image, mask = Variable(image.cuda()), Variable(mask.cuda())
            else:
                image, mask = Variable(image), Variable(mask)

            self.optimizer.zero_grad()
            out = self.net(image)
            loss = self.loss_function(out, mask)
            # 调用了这个函数就无法关闭进程了
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data * len(image)
        train_loss = train_loss / len(self.train_loader)
        self.logger.debug("EPOCH:{}\t train_loss:{:.6f}".format(epoch, train_loss))
        self.scheduler.step()

        self.net.eval()
        with torch.no_grad():
            valid_loss = 0
            for image, mask in self.val_loader:
                if self.is_use_gpu:
                    image, mask = Variable(image.cuda()), Variable(mask.cuda())
                else:
                    image, mask = Variable(image), Variable(mask)

                self.net.zero_grad()
                predict = self.net(image)
                valid_loss += self.loss_function(predict, mask) * len(image)
            valid_loss /= len(self.val_loader)
            self.logger.info("Epoch{}:\t valid_loss:{:.6f}".format(epoch, valid_loss))
        valid_loss = 0
        return self.recorder(train_loss, valid_loss)

    def train(self):
        self.prepare_data()
        self.prepare_net()
        self.logger.debug("================training start=================")

        for epoch in range(self.current_epoch, self.current_epoch + self.num_epoch):
            recorder = self.train_one_epoch(epoch)
            # 增加历史记录，历史记录的保存方式
            # self.history[epoch] = recorder
            # todo 增加是否保存模型的判断
            # self.save(epoch)
        self.logger.debug("================training is over=================")

    def test(self):
        self.prepare_data(test_model=True)
        self.prepare_net()

        self.net.eval()

        # todo 有必要改进
        # self.RESULT_SAVE_PATH = os.path.join(self.result_save_path, time_util.get_date())
        file_utils.make_directory(self.result_save_path)
        self.logger.info("=" * 10 + "test start" + "=" * 10)
        for i, (image, mask, image_name) in enumerate(self.test_loader):
            if self.is_use_gpu:
                image, mask = Variable(image).cuda(), Variable(mask).cuda()
            else:
                image, mask = Variable(image), Variable(mask)
            self.optimizer.zero_grad()
            out = self.net(image)
            _, _, width, height = mask.size()
            loss = self.loss_function(out, mask)
            self.debug(loss)
            predict = out.squeeze().cpu().data.numpy()
            # save predict result
            for index, pred in enumerate(predict):
                pred = pred * 255
                pred = pred.astype('uint8')
                pred = Image.fromarray(pred)
                pred = pred.resize((width, height))
                save_path = os.path.join(self.result_save_path, image_name[index])
                pred.save(save_path)
                self.logger.debug("================{}=================".format(save_path))

        self.logger.debug("================testing end=================")

    def save(self, epoch):
        self.logger.info("==============saving model data===============")
        model_save_path = os.path.join(self.model_save_path,
                                       'ep{}_{}.pkl'.format(epoch, time_utils.get_time("%H-%M-%S")))
        experiment_data = {"epoch": epoch,
                           "state_dict": self.net.state_dict(),
                           "optimizer": self.optimizer.state_dict()}
        checkpoint = BaseCheckPoint.create_checkpoint(experiment_data)
        save(checkpoint, model_save_path)
        self.logger.info("==============saving at {}===============".format(model_save_path))

    def load(self):
        model_save_path = self.pretrain_path
        if os.path.isfile(model_save_path):
            self.logger.info("==============loading model data===============")
            checkpoint = BaseCheckPoint()
            load(checkpoint, model_save_path)
            self.current_epoch = checkpoint.epoch
            self.net.load_state_dict(checkpoint.state_dict)
            self.optimizer.load_state_dict(checkpoint.optimizer)
            self.logger.info("=> loaded checkpoint '{}' (epoch {})"
                             .format(model_save_path, self.current_epoch))
        else:
            self.logger.info("=> no checkpoint found at '{}'".format(model_save_path))

    def save_history(self):
        # todo 如果一秒内保存两次历史会有覆盖的风险，但基本不会发生
        self.logger.info("=" * 10 + " saving history" + "=" * 10)
        file_utils.make_directory(self.history_save_path)
        history_save_path = os.path.join(self.history_save_path, "hitory_{}.pth".format(time_utils.timestamp()))
        torch.save(self.history, history_save_path)
        self.logger.info("=" * 10 + " saved history at {}".format(history_save_path) + "=" * 10)

    def save_history(self):
        # todo 如果一秒内保存两次历史会有覆盖的风险，但基本不会发生
        self.logger.info("=" * 10 + " saving history" + "=" * 10)
        file_utils.make_directory(self.history_save_path)
        history_save_path = os.path.join(self.history_save_path, "hitory_{}.pth".format(time_utils.timestamp()))
        torch.save(self.history, history_save_path)
        self.logger.info("=" * 10 + " saved history at {}".format(history_save_path) + "=" * 10)


if __name__ == '__main__':
    # todo win7系统下调用loss.backward()会导致程序无法关闭
    experiment = DeepExperiment()
    experiment.test()
    # experiment.save_history()
    # print("end")
