import os
import time

import torch
from PIL import Image
from torch.autograd import Variable

from configs.experiment_config import MnistConfig
from experiments.deep_experiment import DeepExperiment
from utils.log_utils import get_logger


class MnistExperiment(MnistConfig, DeepExperiment):
    def __init__(self):
        super(MnistExperiment, self).__init__()
        self.logger = get_logger()
        self.is_bigger_better = False

    def test_one_batch(self, image, mask, image_name, save_predict_result=False):
        image = self.prepare_data(image)
        mask = self.prepare_data(mask)
        start_time = time.time()
        out = self.net_structure(image)
        end_time = time.time()
        pps = len(image) / (end_time - start_time)
        _, _, width, height = mask.size()
        loss = self.loss_function(out, mask)
        self.logger.info("test_loss is {}".format(loss))
        # save predict result
        if save_predict_result:
            predict = out.squeeze().cpu().data.numpy()
            for index, pred in enumerate(predict):
                pred = pred * 255
                pred = pred.astype('uint8')
                pred = Image.fromarray(pred)
                pred = pred.resize((width, height))
                save_path = os.path.join(self.result_save_path, image_name[index])
                pred.save(save_path)
                self.logger.info("================{}=================".format(save_path))
        return pps, loss.data

    def train_one_batch(self, X, Y, *args, **kwargs):
        ones = torch.sparse.torch.eye(10)
        label = ones.index_select(0, Y)
        batch_size = len(X)
        image = X.reshape(batch_size, -1)
        image = self.prepare_data(image, "float")
        label = self.prepare_data(label, "float")
        self.optimizer.zero_grad()
        out = self.net_structure(image)
        loss = self.loss_function(out, label)
        loss.backward()
        self.optimizer.step()
        return [loss.data]

    def valid_one_batch(self,X,Y, *args, **kwargs):
        ones = torch.sparse.torch.eye(10)
        Y = ones.index_select(0, Y)
        batch_size = len(X)
        X = X.reshape(batch_size, -1)
        if self.is_use_gpu:
            X, Y = Variable(X.cuda()), Variable(Y.cuda())
        else:
            X, Y = Variable(X), Variable(Y)
        self.net_structure.zero_grad()
        predict = self.net_structure(X)
        return self.loss_function(predict, Y).data
