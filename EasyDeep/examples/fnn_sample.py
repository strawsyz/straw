import os

import torch
from PIL import Image
from torch.autograd import Variable

from configs.experiment_config import FNNConfig as experiment_config
from experiments.deep_experiment import FNNExperiment


class Experiment(FNNExperiment):
    def __init__(self):
        super(Experiment, self).__init__(experiment_config)

    def test(self):
        super(Experiment, self).test()
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
            self.info("".format(loss))
            predict = out.squeeze().cpu().data.numpy()
            # save predict result
            for index, pred in enumerate(predict):
                pred = pred * 255
                pred = pred.astype('uint8')
                pred = Image.fromarray(pred)
                pred = pred.resize((width, height))
                save_path = os.path.join(self.result_save_path, image_name[index])
                pred.save(save_path)
                self.logger.info("================{}=================".format(save_path))

        self.logger.info("=" * 10 + "testing end" + "=" * 10)

    def list_config(self):
        configs = super(Experiment, self).list_config()
        self.logger.info("\n".join(configs))

    def train_one_epoch(self, epoch):
        train_loss = 0
        self.net.train()
        num_iter = 0
        for sample, label in self.train_loader:
            num_iter += 1
            sample = Variable(torch.from_numpy(sample)).float()
            label = Variable(torch.from_numpy(label)).float()
            if self.is_use_gpu and torch.cuda.is_available():
                sample = sample.cuda()
                label = label.cuda()

            self.optimizer.zero_grad()
            out = self.net(sample)
            loss = self.loss_function(out, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data * len(sample)
        train_loss = train_loss / len(self.train_loader)
        self.logger.info("EPOCH:{}\t train_loss:{:.6f}\t".format(epoch, train_loss))
        self.scheduler.step()

        if self.val_loader is not None:
            self.net.eval()
            with torch.no_grad():
                valid_loss = 0
                for sample, label in self.val_loader:
                    data = Variable(torch.from_numpy(data)).float()
                    y = Variable(torch.from_numpy(y)).float()
                    if self.is_use_gpu and torch.cuda.is_available():
                        data, y = data.cuda(), y.cuda()
                    self.net.zero_grad()
                    predict = self.net(data)
                    valid_loss += self.loss_function(predict, label) * len(sample)
                valid_loss /= len(self.val_loader)
                self.logger.info("Epoch{}:\t valid_loss:{:.6f}".format(epoch, valid_loss))
            return self.recorder(train_loss, valid_loss)
        else:
            return self.recorder(train_loss)


if __name__ == '__main__':
    from base.base_recorder import BaseHistory

    recoder = BaseHistory
    experiment = Experiment()
    # experiment.train()
    # experiment.save_history()
    # experiment.test()
    experiment.estimate()
