import os
import pickle
import time

from torch.autograd import Variable

from base.base_checkpoint import CustomCheckPoint
from base.base_experiment import BaseExperiment
from utils import file_utils
from utils import time_utils
from utils.matplotlib_utils import lineplot
from utils.net_utils import save, load


class DeepExperiment(BaseExperiment):

    def __init__(self, config_instance=None):
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

        self.optimizer = None
        self.loss_function = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.scheduler = None
        super(DeepExperiment, self).__init__(config_instance)

        self.list_config()

    def prepare_dataset(self, testing=False):
        if testing:
            self.dataset.test()
        else:
            self.dataset.train()
        self.dataset.get_dataloader(self)

    def prepare_net(self):
        self.net.get_net(self, self.is_use_gpu)
        if self.is_pretrain:
            self.load()

    def train_one_epoch(self, epoch):
        """一个epoch中训练集进行训练"""
        raise NotImplementedError

    def valid_one_epoch(self, epoch):
        """一个epoch训练之后使用验证集数据进行验证"""
        raise NotImplementedError

    def train_valid_one_epoch(self, epoch):
        train_loss = self.train_one_epoch(epoch)

        if self.valid_loader is not None:
            valid_loss = self.valid_one_epoch(epoch)
            record = self.recorder(train_loss=train_loss, valid_loss=valid_loss)
        else:
            record = self.recorder(train_loss=train_loss)
        return record

    def train(self, max_try_times=None):
        if self.model_selector is None and max_try_times is not None:
            self.logger.warning("you have not set a model_selector!")

        self.prepare_dataset()
        self.prepare_net()
        self.logger.info("================training start=================")
        try_times = 1
        for epoch in range(self.current_epoch, self.current_epoch + self.num_epoch):
            start_time = time.time()
            record = self.train_valid_one_epoch(epoch)

            self.history[epoch] = record
            if not self.save(epoch, record):
                try_times += 1
            else:
                try_times = 0
            if max_try_times is not None and max_try_times < try_times:
                break
            self.logger.info("use {} seconds in the epoch".format(int(time.time() - start_time)))
        self.logger.info("================training is over=================")


    def test(self):
        self.prepare_dataset(testing=True)
        self.prepare_net()
        self.net.eval()
        file_utils.make_directory(self.result_save_path)
        # add methods for test
        #     todo 提取为注解，分别设置，before_test, after_test两个，train也同理

    def create_checkpoint(self, epoch=None, create4load=False):
        # 修改保存的数据之后，需要重新训练，否则无法读取之前的保存的模型
        experiment_data = {"epoch": epoch,
                           "state_dict": self.net.state_dict(),
                           "optimizer": self.optimizer.state_dict(),
                           "history_path": self.history_save_path
                           }
        if create4load:
            return CustomCheckPoint(experiment_data.keys())
        else:
            return CustomCheckPoint(experiment_data.keys())(experiment_data)

    from base.base_recorder import EpochRecord
    def save(self, epoch, record: EpochRecord):
        self.save_history()
        model_save_path = os.path.join(self.model_save_path, time_utils.get_date())
        file_utils.make_directory(model_save_path)
        file_name = 'ep{}_{}.pkl'.format(epoch, time_utils.get_time("%H-%M-%S"))
        model_save_path = os.path.join(model_save_path, file_name)
        if self.model_selector is None:
            self._save_model(epoch, model_save_path)
            return True
        else:
            is_need_save, need_reason = self.model_selector.add_record(record, model_save_path)
            if is_need_save:
                self.logger.info("save this model for {} is better".format(need_reason))
                self._save_model(epoch, model_save_path)
                return True
            else:
                self.logger.info("the eppoch's result is not good, not save")
                return False

    def _save_model(self, epoch, model_save_path):
        self.logger.info("==============saving model data===============")
        checkpoint = self.create_checkpoint(epoch)
        save(checkpoint, model_save_path)
        self.logger.info("==============saved at {}===============".format(model_save_path))

    def load(self):
        model_save_path = self.pretrain_path
        if os.path.isfile(model_save_path):
            self.logger.info("==============loading model data===============")
            self._load_model(model_save_path)
            self.logger.info("=> loaded checkpoint from '{}' ".format(model_save_path))
            self.load_history()
        else:
            self.logger.error("=> no checkpoint found at '{}'".format(model_save_path))

    def _load_model(self, model_save_path):
        check_point = self.create_checkpoint(create4load=True)
        load(check_point, model_save_path)
        self.current_epoch = check_point.epoch + 1
        self.net.load_state_dict(check_point.state_dict)
        self.optimizer.load_state_dict(check_point.optimizer)

    def save_history(self):
        self.logger.info("=" * 10 + " saving history" + "=" * 10)
        file_utils.make_directory(self.history_save_dir)
        history = {}
        # todo 应该在每次加载的时候去掉，多余的数据
        for epoch_no in self.history:
            if epoch_no < self.current_epoch:
                history[epoch_no] = self.history[epoch_no]
        with open(self.history_save_path, "wb") as f:
            pickle.dump(self.history, f)
        self.logger.info("=" * 10 + " saved history at {}".format(self.history_save_path) + "=" * 10)

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

    def show_history(self, use_log10=False):
        self.logger.info("showing history")
        record_attrs = self.history[list(self.history.keys())[0]].__slots__
        epoches = [[epoch_no for epoch_no in self.history] for _ in range(len(record_attrs))]
        import torch
        if use_log10:
            all_records = [[torch.log10(getattr(self.history.get(epoch_no), attr_name)) for epoch_no in self.history]
                           for attr_name in record_attrs]
            record_attrs = ["log10_{}".format(attr_name) for attr_name in record_attrs]
            lineplot(all_records, epoches, labels=record_attrs, title="history analysis with log10")
        else:
            all_records = [[getattr(self.history.get(epoch_no), attr_name) for epoch_no in self.history]
                           for attr_name in record_attrs]
            lineplot(all_records, epoches, record_attrs, "history analysis")
        from utils.matplotlib_utils import show
        show()

    def estimate(self, use_log10=False):
        self.load_history()
        if self.history == {}:
            self.logger.error("no history")
        else:
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
    # todo win7系统下调用loss.backward()会导致程序无法关闭
    experiment = DeepExperiment()
    # experiment.test()
    # experiment.estimate()
    experiment.train()
    # experiment.save_history()
    # print("end")
