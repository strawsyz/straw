import torch

from configs.experiment_config import FNNConfig as experiment_config
from experiments.deep_experiment import DeepExperiment


# 一维数据的回归任务
class Experiment(DeepExperiment):
    def __init__(self, config_instance=experiment_config()):
        super(Experiment, self).__init__(config_instance)

    def test(self):
        super(Experiment, self).test()
        self.logger.info("=" * 10 + "test start" + "=" * 10)
        for i, (data, gt) in enumerate(self.test_loader):
            self.test_one_batch(data, gt, "float")
        self.logger.info("=" * 10 + "testing end" + "=" * 10)

    def test_one_batch(self, data, gt, data_type=None):
        data = self.prepare_data(data, data_type)
        gt = self.prepare_data(gt, data_type)
        self.optimizer.zero_grad()
        out = self.net(data)
        loss = self.loss_function(out, gt)
        self.logger.info("One Batch:test_loss is {}".format(loss))

    def train_one_epoch(self, epoch):
        train_loss = 0
        self.net.train()

        for sample, label in self.train_loader:
            sample = self.prepare_data(sample, 'float')
            label = self.prepare_data(label, 'float')

            self.optimizer.zero_grad()
            out = self.net(sample)
            loss = self.loss_function(out, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data
        train_loss = train_loss / len(self.train_loader)
        self.logger.info("EPOCH:{}\t train_loss:{:.6f}\t".format(epoch, train_loss))
        self.scheduler.step()
        return train_loss

    def valid_one_epoch(self, epoch):
        self.net.eval()
        with torch.no_grad():
            valid_loss = 0
            for data, label in self.valid_loader:
                data = self.prepare_data(data, 'float')
                label = self.prepare_data(label, 'float')
                self.net.zero_grad()
                predict = self.net(data)
                valid_loss += self.loss_function(predict, label)
            valid_loss /= len(self.valid_loader)
            self.logger.info("Epoch:{}\t valid_loss:{:.6f}".format(epoch, valid_loss))

        return valid_loss
if __name__ == '__main__':
    experiment = Experiment()
    experiment.train(max_try_times=8)
    # experiment.test()
    # experiment.estimate(True)
