import os
import pickle
import time
from collections import deque

from base.base_checkpoint import BaseCheckPoint
from base.base_experiment import BaseExperiment
from utils import file_utils
from utils import time_utils
from utils.matplotlib_utils import plot
from utils.net_utils import save, load


class DeepExperiment(BaseExperiment):

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
        super(DeepExperiment, self).__init__(config_instance)

        self.list_config()

    # def load_config(self, config_cls):
    #     if config_cls is not None:
    #         copy_attr(config_cls(), self)
    #     else:
    #         copy_attr(default_config(), self)

    def prepare_data(self, testing=False):
        if testing:
            self.dataset.test()
        else:
            self.dataset.train()
        self.dataset.get_dataloader(self)

    def prepare_net(self):
        # self.net = self.net()
        self.net.get_net(self, self.is_use_gpu)

        if self.is_pretrain:
            self.load()

    from base.base_recorder import BaseHistory
    def train_one_epoch(self, epoch) -> BaseHistory:
        raise NotImplementedError

    def train(self, test=False):
        if not test:
            self.prepare_data()
        self.prepare_net()
        self.logger.info("================training start=================")

        for epoch in range(self.current_epoch, self.current_epoch + self.num_epoch):
            start_time = time.time()
            record = self.train_one_epoch(epoch)
            self.history[epoch] = record
            self.save_history()
            if self.model_selector is None:
                self.save(epoch)
            else:
                is_need_save = self.model_selector.add_record(record)
                if is_need_save:
                    self.save(epoch)
                else:
                    self.logger.info("the eppoch's result is not good, not save")
            self.logger.info("use {} seconds in the epoch".format(int(time.time() - start_time)))
        self.logger.info("================training is over=================")

    def test(self, test=False):
        if not test:
            self.prepare_data(testing=True)
        self.prepare_net()
        self.net.eval()
        result_save_path = os.path.join(self.result_save_path, time_utils.get_date())
        file_utils.make_directory(result_save_path)
        # add methods for test

    #     todo 提取为注解，分别设置，before_test, after_test两个，train也同理

    def create_optimizer(self, epoch):
        return {"epoch": epoch,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict()}

    def save(self, epoch):
        self.logger.info("==============saving model data===============")
        model_save_path = os.path.join(self.model_save_path, time_utils.get_date())
        file_utils.make_directory(model_save_path)
        model_save_path = os.path.join(model_save_path,
                                       'ep{}_{}.pkl'.format(epoch, time_utils.get_time("%H-%M-%S")))
        experiment_data = self.create_optimizer(epoch)
        checkpoint = BaseCheckPoint.create_checkpoint(experiment_data)
        save(checkpoint, model_save_path)
        self.logger.info("==============saved at {}===============".format(model_save_path))

    def load(self):
        model_save_path = self.pretrain_path
        if os.path.isfile(model_save_path):
            self.logger.info("==============loading model data===============")
            checkpoint = BaseCheckPoint()
            load(checkpoint, model_save_path)
            self.current_epoch = checkpoint.epoch + 1
            self.net.load_state_dict(checkpoint.state_dict)
            self.optimizer.load_state_dict(checkpoint.optimizer)
            self.logger.info("=> loaded checkpoint from '{}' "
                             .format(model_save_path))
            self.load_history()
        else:
            self.logger.error("=> no checkpoint found at '{}'".format(model_save_path))

    def save_history(self):
        self.logger.info("=" * 10 + " saving history" + "=" * 10)
        file_utils.make_directory(self.history_save_dir)
        history_save_path = os.path.join(self.history_save_dir, "history.pth")
        history = {}
        for epoch_no in self.history:
            if epoch_no < self.current_epoch:
                history[epoch_no] = self.history[epoch_no]
        with open(history_save_path, "wb") as f:
            pickle.dump(self.history, f)
        self.logger.info("=" * 10 + " saved history at {}".format(history_save_path) + "=" * 10)

    def load_history(self):
        if hasattr(self, "history_save_path") and self.history_save_path is not None:
            if os.path.isfile(self.history_save_path):
                self.logger.info("=" * 10 + " loading history" + "=" * 10)
                with open(self.history_save_path, mode="rb+") as f:
                    self.history = pickle.load(f)
                self.logger.info("=" * 10 + " loaded history from {}".format(self.history_save_path) + "=" * 10)
            else:
                self.logger.error("{} is not a file".format(self.history_save_path))
        else:
            self.logger.error("not set history_save_path")

    def show_history(self):
        self.logger.info("showing history")
        epoches = deque()
        train_losses = deque()
        valid_losses = deque()
        for epoch_no in self.history:
            epoches.append(epoch_no)
            recorder = self.history.get(epoch_no)
            train_losses.append(recorder.train_loss)
            valid_losses.append(recorder.valid_loss)
        plot([epoches, epoches], [train_losses, valid_losses], ["train_loss", "valid_loss"], "loss analysis")

    def estimate(self):
        # load history data
        self.load_history()
        if self.history == {}:
            self.logger.error("no history")
        else:
            # display history data
            self.show_history()

    def sample_test(self, n_sample=3, epoch=3):
        cur_epoch = self.num_epoch
        self.dataset.get_samle_dataloader(num_samples=n_sample, target=self)
        # self.dataset.get_dataloader(self)
        self.num_epoch = epoch
        # self.train(test=True)
        self.test(test=True)
        self.estimate()
        self.num_epoch = cur_epoch


class FNNExperiment(DeepExperiment):
    def __init__(self, config):
        super(FNNExperiment, self).__init__(config_instance=config)


if __name__ == '__main__':
    # todo win7系统下调用loss.backward()会导致程序无法关闭
    experiment = DeepExperiment()
    # experiment.test()
    experiment.estimate()
    # experiment.train()
    # experiment.save_history()
    # print("end")
