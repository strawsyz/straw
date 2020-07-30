import os

import torch
from PIL import Image
from torch.autograd import Variable

from experiments.deep_experiment import DeepExperiment

from configs.experiment_config import ImageSegmentationConfig


class Experiment(DeepExperiment):
    def __init__(self, config_instance=ImageSegmentationConfig()):
        super(Experiment, self).__init__(config_instance)

    def test(self):
        super(Experiment, self).test()
        self.logger.info("=" * 10 + "test start" + "=" * 10)
        for i, (image, mask, image_name) in enumerate(self.test_loader):
            self.test_one_batch(image, mask, image_name)
        self.logger.info("================testing end=================")

    def test_one_batch(self, image, mask, image_name):
        image = self.prepare_data(image)
        mask = self.prepare_data(mask)
        self.optimizer.zero_grad()
        out = self.net(image)
        _, _, width, height = mask.size()
        loss = self.loss_function(out, mask)
        self.logger.info("test_loss is {}".format(loss))
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


if __name__ == '__main__':
    from configs.experiment_config import ImageSegmentationConfig

    experiment = Experiment(ImageSegmentationConfig())
    # experiment.sample_test()
    experiment.train()
    # experiment.test()
    # experiment.estimate()
