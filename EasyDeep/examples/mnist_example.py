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

    def test(self, prepare_dataset=True, prepare_net=True, save_predict_result=False):
        super(MnistExperiment, self).test(prepare_dataset, prepare_net)
        self.logger.info("=" * 10 + " test start " + "=" * 10)
        pps = 0
        loss = 0
        for i, (image, mask, image_name) in enumerate(self.test_loader):
            pps_batch, loss_batch = self.test_one_batch(image, mask, image_name, save_predict_result)
            pps += pps_batch
            loss += loss_batch
        pps /= len(self.test_loader)
        loss /= len(self.test_loader)
        self.logger.info("average predict {} images per second".format(pps))
        self.logger.info("average loss is {}".format(loss))
        self.logger.info("=" * 10 + " testing end " + "=" * 10)
        return self.result_save_path

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

    def train_one_epoch(self, epoch):
        self.net_structure.train()
        train_loss = 0
        num_iter = 3000
        for idx_iter, (image, label) in enumerate(self.train_loader):
            ones = torch.sparse.torch.eye(10)
            label = ones.index_select(0, label)
            batch_size = len(image)
            image = image.reshape(batch_size, -1)
            image = self.prepare_data(image, "float")
            label = self.prepare_data(label, "float")
            self.optimizer.zero_grad()
            out = self.net_structure(image)
            # out = torch.max(out, 1).indices
            loss = self.loss_function(out, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data
            if idx_iter % num_iter == 0:
                self.logger.info("loss is {:.6f}, num_iter is {}".format(train_loss / (idx_iter + 1), idx_iter))
        train_loss = train_loss / len(self.train_loader)
        self.scheduler.step()
        self.logger.info("EPOCH:{}\t train_loss:{:.6f}".format(epoch, train_loss))
        return train_loss
    def train_one_batch(self,*args,**kwargs):
        X,Y = args
        ones = torch.sparse.torch.eye(10)
        label = ones.index_select(0, Y)
        batch_size = len(X)
        image = X.reshape(batch_size, -1)
        image = self.prepare_data(image, "float")
        label = self.prepare_data(label, "float")
        self.optimizer.zero_grad()
        out = self.net_structure(image)
        # out = torch.max(out, 1).indices
        loss = self.loss_function(out, label)
        loss.backward()
        self.optimizer.step()
        return loss.data

    # def valid_one_epoch(self, epoch):
    #     self.net_structure.eval()
    #     with torch.no_grad():
    #         valid_loss = 0
    #         for image, mask in self.valid_loader:
    #             if self.is_use_gpu:
    #                 image, mask = Variable(image.cuda()), Variable(mask.cuda())
    #             else:
    #                 image, mask = Variable(image), Variable(mask)
    #             self.net_structure.zero_grad()
    #             predict = self.net_structure(image)
    #             valid_loss += self.loss_function(predict, mask)
    #         valid_loss /= len(self.valid_loader)
    #         self.logger.info("Epoch:{}\t valid_loss:{:.6f}".format(epoch, valid_loss))
    #     return valid_loss

    def valid_one_batch(self,*args,**kwargs):
        x,y = args
        # self.valid_one_batch(x, y)
        if self.is_use_gpu:
            x, y = Variable(x.cuda()), Variable(y.cuda())
        else:
            x, y = Variable(x), Variable(y)
        self.net_structure.zero_grad()
        predict = self.net_structure(x)
        return self.loss_function(predict, y)

if __name__ == '__main__':
    experiment = MnistExperiment()
    # experiment.predict_all_data()
    # experiment.sample_test()
    experiment.train(max_try_times=8)

    # configs = {"lr": [0.01, 0.001, 0.0003], "random_seed": [1, 2, 3]}
    # experiment.grid_train(configs)
    # experiment.test(save_predict_result=True)
    # experiment.estimate(use_log10=True)
# /home/straw/Downloads/models/polyp/history/history_1596127749.pth
