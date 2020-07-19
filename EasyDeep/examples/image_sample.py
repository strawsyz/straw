import os

import torch
from PIL import Image
from torch.autograd import Variable

from experiments.deep_experiment import DeepExperiment


class Experiment(DeepExperiment):
    def __init__(self, config_instance=None):
        super(Experiment, self).__init__(config_instance)

    def test(self, test=False):
        super(Experiment, self).test(test)
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

        self.logger.info("================testing end=================")

    def list_config(self):
        self.logger.info(str(self.config_instance))

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
        self.logger.info("EPOCH:{}\t train_loss:{:.6f}".format(epoch, train_loss))
        self.scheduler.step()

        if self.val_loader is not None:
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
                    valid_loss += self.loss_function(predict, mask)
                valid_loss /= len(self.val_loader)
                self.logger.info("Epoch{}:\t valid_loss:{:.6f}".format(epoch, valid_loss))

            return self.recorder(train_loss, valid_loss)
        else:
            return self.recorder(train_loss)


if __name__ == '__main__':
    # recoder = BaseHistory
    from configs.experiment_config import ImageSegmentationConfig

    experiment = Experiment(ImageSegmentationConfig())
    # experiment.sample_test()
    experiment.train()
    # experiment.test()
    # experiment.estimate()
