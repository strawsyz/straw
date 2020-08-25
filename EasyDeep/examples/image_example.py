import os
import time

import torch
from PIL import Image
from torch.autograd import Variable

from configs.experiment_config import ImageSegmentationConfig
from experiments.deep_experiment import DeepExperiment


class Experiment(DeepExperiment):
    def __init__(self, config_instance=ImageSegmentationConfig()):
        super(Experiment, self).__init__(config_instance)

    def test(self, prepare_dataset=True, prepare_net=True, save_predict_result=False):
        super(Experiment, self).test(prepare_dataset, prepare_net)
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
        # self.optimizer.zero_grad()
        start_time = time.time()
        out = self.net(image)
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
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data
            # print(loss.data)
        train_loss = train_loss / len(self.train_loader)
        self.scheduler.step()
        self.logger.info("EPOCH:{}\t train_loss:{:.6f}".format(epoch, train_loss))
        return train_loss

    def valid_one_epoch(self, epoch):
        self.net.eval()
        with torch.no_grad():
            valid_loss = 0
            for image, mask in self.valid_loader:
                if self.is_use_gpu:
                    image, mask = Variable(image.cuda()), Variable(mask.cuda())
                else:
                    image, mask = Variable(image), Variable(mask)

                self.net.zero_grad()
                predict = self.net(image)
                valid_loss += self.loss_function(predict, mask)
            valid_loss /= len(self.valid_loader)
            self.logger.info("Epoch:{}\t valid_loss:{:.6f}".format(epoch, valid_loss))
        return valid_loss

    def predict_all_data(self):
        self.prepare_net()
        # self.prepare_dataset()
        self.dataset.get_dataloader(self)
        self.net.eval()

        from utils.file_utils import make_directory
        make_directory(self.result_save_path)
        for image, image_names in self.all_dataloader:
            image = self.prepare_data(image)
            self.optimizer.zero_grad()
            out = self.net(image)
            predict = out.squeeze().cpu().data.numpy()
            for index, pred in enumerate(predict):
                pred = pred * 255
                pred = pred.astype('uint8')
                pred = Image.fromarray(pred)
                pred = pred.resize((224, 224))
                save_path = os.path.join(self.result_save_path, image_names[index])
                pred.save(save_path)
                self.logger.info("================{}=================".format(save_path))



if __name__ == '__main__':
    from configs.experiment_config import ImageSegmentationConfig

    experiment = Experiment(ImageSegmentationConfig())
    # experiment.predict_all_data()
    # experiment.sample_test()
    experiment.train(max_try_times=8)

    configs = {"lr": [0.01, 0.001, 0.0003]}

    # experiment.test(save_predict_result=True)
    # experiment.estimate(use_log10=True)
# /home/straw/Downloads/models/polyp/history/history_1596127749.pth
