from torch.autograd import Variable
import os
import pickle
import time
from collections import deque
import torch
from base.base_checkpoint import BaseCheckPoint
from base.base_experiment import BaseExperiment
from experiments import DeepExperiment
from utils import file_utils
from utils import time_utils
from utils.matplotlib_utils import lineplot
from utils.net_utils import save, load


class SimpleDeepExperiment(DeepExperiment):

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
        super(SimpleDeepExperiment, self).__init__(config_instance)

        self.show_config()

    def train(self, test=False):
        if not test:
            self.prepare_dataset()
        self.prepare_net()
        self.logger.info("================training start=================")

        for epoch in range(self.current_epoch, self.current_epoch + self.num_epoch):
            start_time = time.time()
            record = self.train_one_epoch(epoch)
            self.history[epoch] = record
            self.save(epoch, record)
            self.logger.info("use {} seconds in the epoch".format(int(time.time() - start_time)))
        self.logger.info("================training is over=================")

    def create_optimizer(self, epoch):
        super(SimpleDeepExperiment, self).create_optimizer(epoch)

    def train(self, num_epoch):
        for epoch in range(num_epoch):
            self.train_one_epoch(epoch)

    def init(self):
        # 初始化所有的内容。只要有一个没有设置就使用默认的设置内容
        if self.loss_funciton is None:
            self.loss_function = torch.nn.MSELoss()
        if self.optim_name is None:
            self.optimizer = torch.optim.SGD(lr=0.0003, weight_decay=0.0001)

    def train_one_epoch(self, epoch):
        train_loss = 0
        self.net.train()
        global data
        for sample, label in self.train_data:
            sample = Variable(torch.from_numpy(sample)).double()
            label = Variable(torch.from_numpy(label)).double()
            self.optimizer.zero_grad()
            out = self.net(sample)
            loss = self.loss_function(out, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data
        train_loss = train_loss / len(self.train_data)
        print("train loss \t {}".format(train_loss))

    def test(self):
        all_loss = 0
        for i, (data, gt) in enumerate(self.test_data):
            data = Variable(torch.from_numpy(data)).double()
            gt = Variable(torch.from_numpy(gt)).double()
            batch_loss = self.test_one_batch(data, gt)
            all_loss += batch_loss
        print("train loss \t {}".format(all_loss / 20))

    def test_one_batch(self, data, gt):
        self.optimizer.zero_grad()
        out = self.net(data)
        loss = self.loss_function(out, gt)
        return loss

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

    # def show_history(self, use_log10=False):
    #     self.logger.info("showing history")
    #     record_attrs = self.history[list(self.history.keys())[0]].__slots__
    #     epoches = [[epoch_no for epoch_no in self.history] for _ in range(len(record_attrs))]
    #     import torch
    #     if use_log10:
    #         all_records = [[torch.log10(getattr(self.history.get(epoch_no), attr_name)) for epoch_no in self.history]
    #                        for attr_name in record_attrs]
    #         record_attrs = ["log10_{}".format(attr_name) for attr_name in record_attrs]
    #         lineplot(all_records, epoches, labels=record_attrs, title="history analysis with log10")
    #     else:
    #         all_records = [[getattr(self.history.get(epoch_no), attr_name) for epoch_no in self.history]
    #                        for attr_name in record_attrs]
    #         lineplot(all_records, epoches, record_attrs, "history analysis")
    #     from utils.matplotlib_utils import show
    #     show()

    def estimate(self, use_log10=False):
        # load history data
        self.load_history()
        if self.history == {}:
            self.logger.error("no history")
        else:
            # display history data
            self.show_history(use_log10)

    def sample_test(self, n_sample=3, epoch=3):
        cur_epoch = self.num_epoch
        self.dataset.get_samle_dataloader(num_samples=n_sample, target=self)
        # self.dataset.get_dataloader(self)
        self.num_epoch = epoch
        self.train(test=True)
        self.test(test=True)
        self.estimate()
        self.num_epoch = cur_epoch

    def prepare_data(self, data, data_type=None):
        import torch
        data = Variable(data)
        if data_type == "float":
            data = data.float()
        elif data_type == "double":
            data = data.double()

        if self.is_use_gpu and torch.cuda.is_available():
            return data.cuda()
        else:
            return data


class FNNExperiment(DeepExperiment):
    def __init__(self, config):
        super(FNNExperiment, self).__init__(config_instance=config)


if __name__ == '__main__':
    experiment = DeepExperiment()
    # experiment.test()
    experiment.estimate()
    # experiment.train()
    # experiment.save_history()
    # print("end")
