from torch.autograd import Variable
import os
import pickle
import time
from collections import deque
import torch
from tqdm import tqdm

from configs.experiment_config import VideoFeatureConfig
from experiments import DeepExperiment


class VideoFeatureExperiment(VideoFeatureConfig, DeepExperiment):

    def __init__(self, config_instance=None):
        # if not load net from pretrained model, then 0
        self.current_epoch = 0

        self.history = {}
        self.recorder = None
        self.net = None
        self.dataset = None

        self.num_epoch = None
        self.is_use_gpu = None
        self.history_save_dir = None
        self.history_save_path = None
        self.model_save_path = ""
        self.model_selector = None
        self.is_pretrain = None
        self.pretrain_path = None
        self.result_save_path = None
        self.selector = None

        # super().__init__(config_instance)
        if config_instance is None:
            # VideoFeatureConfig.__init__(self)
            super(VideoFeatureExperiment, self).__init__()
        else:
            super(DeepExperiment, self).__init__(config_instance)

        self.show_config()

    def get_best_accuracy(self):
        best_accuracy = 0
        for epoch_record in self.history.values():
            accuracy = epoch_record.accuracy
            if best_accuracy < accuracy:
                best_accuracy = accuracy
        return best_accuracy

    def train(self, test=False):
        if not test:
            self.prepare_dataset()
        self.prepare_net()
        self.logger.info("================training start=================")
        self.net = self.net_structure
        for epoch in range(self.current_epoch, self.current_epoch + self.num_epoch):
            start_time = time.time()
            # record = self.train_one_epoch(epoch)
            record = self.train_valid_one_epoch(epoch)

            if record is not None:
                self.history[epoch] = record
                self.save(epoch, record)

            # self.test_one_batch(epoch)
            # todo print the best score
            self.logger.info(f"Best {self.get_best_accuracy()}")
            self.logger.info("use {} seconds in the epoch".format(int(time.time() - start_time)))
        self.logger.info("================training is over=================")

    def init(self):
        # 初始化所有的内容。只要有一个没有设置就使用默认的设置内容
        if self.loss_funciton is None:
            self.loss_function = torch.nn.MSELoss()
        if self.optim_name is None:
            self.optimizer = torch.optim.SGD(lr=0.0003, weight_decay=0.0001)

    def train_one_epoch(self, epoch) -> float:
        train_loss = 0
        self.net.train()
        global data
        for sample, label in tqdm(self.train_loader):
            sample = self.prepare_data(sample)
            label = self.prepare_data(label)
            self.optimizer.zero_grad()
            out = self.net(sample)
            loss = self.loss_function(out, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data
        train_loss = train_loss / len(self.train_loader)
        self.logger.info("train loss \t {}".format(train_loss))
        return train_loss

    def train_valid_one_epoch(self, epoch):
        train_loss = self.train_one_epoch(epoch)

        if self.valid_loader is not None:
            valid_loss, accuracy = self.valid_one_epoch(epoch)
            record = self.recorder(train_loss=train_loss, valid_loss=valid_loss, accuracy=accuracy)
        else:
            record = self.recorder(train_loss=train_loss)

        if self.scheduler is not None:
            self.scheduler.step()
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.logger.info("EPOCH : {} .Learning rate is {:.10f}".format(epoch, lr))

        return record

    def valid_one_epoch(self, epoch) -> float:
        self.net.eval()
        sigmoid = torch.nn.Sigmoid()
        correct_sum = 0
        sum = 0
        with torch.no_grad():
            valid_loss = 0
            for sample, label in self.valid_loader:
                sample = self.prepare_data(sample)
                label = self.prepare_data(label)
                self.optimizer.zero_grad()
                predict = self.net(sample)
                valid_loss += self.loss_function(predict, label)
                predict_result = torch.argmax(sigmoid(predict), dim=1)
                label_result = torch.argmax(sigmoid(label), dim=1)
                correct = torch.eq(label_result, predict_result)
                correct_sum += correct.sum().float().item()
                sum += label.shape[0]

            valid_loss /= len(self.valid_loader)
            self.logger.info("Epoch:{}\t valid_loss:{:.6f}".format(epoch, valid_loss))

        correct_rate = correct_sum / sum
        self.logger.info(correct_rate)
        return valid_loss, correct_rate

    def before_test(self):
        all_loss = 0
        for i, (sample, label) in enumerate(self.valid_loader):
            batch_loss = self.test_one_batch(sample, label)
            all_loss += batch_loss
        self.logger.info("train loss \t {}".format(all_loss / 20))

    def test_one_batch(self, data, gt):
        return self.valid_one_batch(data, gt)

    def load_history(self):
        if hasattr(self, "history_save_path") and self.history_save_path is not None and self.history_save_path != "":
            if os.path.isfile(self.history_save_path):
                self.logger.info("=" * 10 + " loading history" + "=" * 10)
                with open(self.history_save_path, mode="rb+") as f:
                    self.history = pickle.load(f)
                self.logger.info("=" * 10 + " loaded history from {}".format(self.history_save_path) + "=" * 10)
            else:
                self.logger.error("{} is not a file".format(self.history_save_path))
        else:
            self.logger.error("not set history_save_path")

    def estimate(self, use_log10=False):
        # load history data
        self.load_history()
        if self.history == {}:
            self.logger.error("no history")
        else:
            # display history data
            self.estimate_history(use_log10)

    def sample_test(self, n_sample=3, epoch=3):
        cur_epoch = self.num_epoch
        self.dataset.get_samle_dataloader(num_samples=n_sample, target=self)
        # self.dataset.get_dataloader(self)
        self.num_epoch = epoch
        self.train(test=True)
        self.before_test(test=True)
        self.estimate()
        self.num_epoch = cur_epoch

    def prepare_data(self, data, data_type=None):
        data = Variable(data)
        if self.is_use_gpu and torch.cuda.is_available():
            return data.cuda()
        else:
            return data


if __name__ == '__main__':
    # config = VideoFeatureConfig()
    import sys

    sys.path.insert(0, "/workspace/straw/EasyDeep/")
    # export PYTHONPATH="${PYTHONPATH}:/workspace/straw/EasyDeep/"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    experiment = VideoFeatureExperiment()
    # experiment.test()
    # experiment.estimate()
    experiment.train()
    # experiment.save_history()
    # print("end")