from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from torch.autograd import Variable
import os
import pickle
import time
from collections import deque
import torch
from tqdm import tqdm
import torch.nn.functional as F

from configs.experiment_config import VideoFeatureConfig
from experiments import DeepExperiment
from utils import file_utils
from utils.other_utils import set_GPU
from utils.print_utils import red


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
        self.try_times = 0

        # super().__init__(config_instance)
        if config_instance is None:
            # VideoFeatureConfig.__init__(self)
            super(VideoFeatureExperiment, self).__init__()
        else:
            super(DeepExperiment, self).__init__(config_instance)

        self.show_config()
        # set init parameters into history
        self.init_history()

    def get_best_accuracy(self):
        best_accuracy = 0
        for epoch_record in self.experiment_record.epoch_records.values():
            accuracy = epoch_record.accuracy
            if best_accuracy < accuracy:
                best_accuracy = accuracy
        return best_accuracy

    def early_stop(self, best_accuracy, accuracy):
        if best_accuracy < accuracy:
            self.try_times = 0
        else:
            self.try_times += 1

        if self.max_try_times <= self.try_times:
            return True
        else:
            return False

    def init_history(self):
        file_utils.make_directory(self.history_save_dir)
        # experiment_record = self.experiment_record()
        # experiment_record.config_info = self.config_info
        config_dict = {}
        for attr in sorted(self.dataset_config.__dict__):
            config_dict[attr + "_dataset"] = str(getattr(self.dataset_config, attr))
        if hasattr(self, "net_config") and self.net_config is not None:
            for attr in sorted(self.net_config.__dict__):
                config_dict[attr + "_network"] = str(getattr(self.net_config, attr))
        if hasattr(self, "net") and self.net is not None:
            for attr in sorted(self.net.__dict__):
                config_dict[attr + "_network"] = str(getattr(self.net, attr))
        for attr in sorted(self.__dict__):
            config_dict[attr + "_experiment"] = str(getattr(self, attr))
        self.experiment_record.config_info = config_dict
        self.save_history()

    def save_history(self):
        self.experiment_record.save(self.history_save_path)
        self.logger.info("=" * 10 + " saved experiment history at {}".format(self.history_save_path) + "=" * 10)


    def train(self):
        self.prepare_dataset()
        self.prepare_net()
        self.net = self.net_structure

        self.logger.info("================training start=================")
        for epoch in range(self.current_epoch, self.current_epoch + self.num_epoch):
            pre_best_accuracy = self.get_best_accuracy()
            start_time = time.time()
            record = self.train_valid_one_epoch(epoch)

            if record is not None:
                self.experiment_record.epoch_records[epoch] = record
                self.save(epoch, record)

            self.logger.info(red(f"Best accuracy is {self.get_best_accuracy()}"))
            if self.early_stop(pre_best_accuracy, record.accuracy):
                self.info("Stop training for over max try times")
                break
            self.logger.info("use {} seconds in the epoch".format(int(time.time() - start_time)))
        self.logger.info("================training is over=================")
        # return self.get_best_accuracy()

    def init(self):
        # 初始化所有的内容。只要有一个没有设置就使用默认的设置内容
        if self.loss_funciton is None:
            self.loss_function = torch.nn.MSELoss()
        if self.optim_name is None:
            self.optimizer = torch.optim.SGD(lr=0.0003, weight_decay=0.0001)

    def train_one_epoch(self, epoch) -> float:
        train_loss = 0
        num_steps_per_update = 4  # accum gradient

        self.net.train()
        # global data
        for data in tqdm(self.train_loader, ncols=50):
            self.optimizer.zero_grad()

            if self.dataset_config.dataset_name == "RGBResNet":
                sample, feature, label = data
                sample = self.prepare_data(sample)
                label = self.prepare_data(label)
                feature = self.prepare_data(feature)
                out = self.net(sample, feature)
            elif self.dataset_config.dataset_name == "Video2SDataset":
                slow, fast, label = data
                slow = self.prepare_data(slow)
                fast = self.prepare_data(fast)
                label = self.prepare_data(label)
                out = self.net([slow, fast])
            else:
                sample, label = data
                sample = self.prepare_data(sample)
                label = self.prepare_data(label)
                out = self.net(sample)

            loss = self.loss_function(out, label)
            # t = sample.size(2)
            # # upsample to input size
            # per_frame_logits = F.upsample(out, t, mode='linear')
            #
            # # compute localization loss
            # loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, label)
            # # tot_loc_loss += loc_loss.data[0]
            #
            # # compute classification loss (with max-pooling along time B x C x T)
            # cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
            #                                               torch.max(label, dim=2)[0])
            # # tot_cls_loss += cls_loss.data[0]
            #
            # loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
            # tot_loss += loss.data[0]

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
            for data in self.valid_loader:
                self.optimizer.zero_grad()

                if self.dataset_config.dataset_name == "RGBResNet":
                    sample, feature, label = data
                    sample = self.prepare_data(sample)
                    label = self.prepare_data(label)
                    feature = self.prepare_data(feature)
                    out = self.net(sample, feature)
                elif self.dataset_config.dataset_name == "Video2SDataset":
                    slow, fast, label = data
                    slow = self.prepare_data(slow)
                    label = self.prepare_data(label)
                    fast = self.prepare_data(fast)
                    out = self.net([slow, fast])
                elif self.dataset_config.dataset_name == "SlowFast":
                    slow, fast, feature, label = data
                    slow = self.prepare_data(slow)
                    label = self.prepare_data(label)
                    feature = self.prepare_data(feature)
                    fast = self.prepare_data(fast)
                    out = self.net([slow, fast], feature)
                else:
                    sample, label = data
                    sample = self.prepare_data(sample)
                    label = self.prepare_data(label)
                    out = self.net(sample)

                valid_loss += self.loss_function(out, label)
                predict_result = torch.argmax(sigmoid(out), dim=1)
                label_result = torch.argmax(label, dim=1)
                correct = torch.eq(label_result, predict_result)
                correct_sum += correct.sum().float().item()
                sum += label.shape[0]

            valid_loss /= len(self.valid_loader)
            self.logger.info("Epoch:{}\t valid_loss:{:.6f}".format(epoch, valid_loss))

        correct_rate = correct_sum / sum
        self.logger.info(f"Accuracy : {correct_rate}")
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
        self.num_epoch = epoch
        self.train(test=True)
        self.before_test(test=True)
        self.estimate()
        self.num_epoch = cur_epoch

    def prepare_data(self, data, data_type=None):
        data = Variable(data).float()
        if self.is_use_gpu and torch.cuda.is_available():
            return data.cuda()
        else:
            return data



if __name__ == '__main__':
    import sys
    import numpy as np
    import random

    # torch.backends.cudnn.enabled = False
    random_state = 0
    torch.manual_seed(random_state)  # cpu
    torch.cuda.manual_seed(random_state)  # gpu
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)  # numpy
    random.seed(random_state)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(random_state)

    # sys.path.insert(0, "/workspace/straw/EasyDeep/")
    sys.path.append("/workspace/straw/EasyDeep/")
    # export PYTHONPATH="${PYTHONPATH}:/workspace/straw/EasyDeep/"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    experiment = VideoFeatureExperiment()
    experiment.train()
    print(experiment.get_best_accuracy())
