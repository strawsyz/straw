import json
import os
import time
from collections import namedtuple

import torch
from torch.autograd import Variable

from base.base_checkpoint import CustomCheckPoint
from base.base_experiment import BaseExperiment
from base.base_recorder import EpochRecord
from configs.experiment_config import DeepExperimentConfig
from utils import file_utils
from utils import time_utils
from utils.matplotlib_utils import lineplot
from utils.matplotlib_utils import save as img_save
from utils.matplotlib_utils import show
from utils.net_utils import save, load


class DeepExperiment(DeepExperimentConfig, BaseExperiment):

    def __init__(self):
        super(DeepExperiment, self).__init__()
        self.description = "experiment"
        # tag can't include a space
        self.tag = "deep"

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
        """train one epoch"""
        train_loss = 0
        self.net_structure.train()
        other_param = self.other_param_4_batch
        tmp_idx = 0
        for idx, (sample, label) in enumerate(self.train_loader, start=1):
            other_param = self.train_one_batch(sample, label, other_param)
            loss = other_param[0]
            train_loss += loss
            if idx % self.num_iter == 0:
                self.logger.info("EPOCH : {}\t iter：{}, loss_data : {:.6f}".format(epoch, idx, train_loss / idx))
            tmp_idx = idx
        print(tmp_idx)
        train_loss = train_loss / tmp_idx
        self.logger.info("EPOCH : {}\t train_loss : {:.6f}\t".format(epoch, train_loss))
        if self.scheduler is not None:
            self.scheduler.step()
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.logger.info("learning rate is {:.6f}".format(lr))
        return train_loss

    def valid_one_epoch(self, epoch):
        """validate one epoch"""
        self.net_structure.eval()
        with torch.no_grad():
            valid_loss = 0
            for idx, (x, y) in enumerate(self.valid_loader, start=1):
                valid_loss += self.valid_one_batch(x, y)
            valid_loss /= idx
            self.logger.info("Epoch:{}\t valid_loss:{:.6f}".format(epoch, valid_loss))
        return valid_loss

    def valid(self, pretrain_path=None):
        if pretrain_path is not None:
            self.pretrain_path = r"C:\(lab\models\loan\mnist-10-04-H06sZS\model\2020-10-04\ep424_19-16-16.pkl"
            self.is_pretrain = True
        assert self.is_pretrain == True
        self.prepare_net()
        self.prepare_dataset()
        self.valid_one_epoch()

    @staticmethod
    def train_one_batch(self, *args, **kwargs) -> list:
        """
        train one batch
        return a list, first element of list is loss
        """
        raise NotImplementedError

    def valid_one_batch(self, *args, **kwargs):
        """
        train one batch
        """
        raise NotImplementedError

    def train_valid_one_epoch(self, epoch):
        train_loss = self.train_one_epoch(epoch)
        if self.valid_loader is not None:
            valid_loss = self.valid_one_epoch(epoch)
            record = self.recorder(train_loss=train_loss, valid_loss=valid_loss)
        else:
            record = self.recorder(train_loss=train_loss)
        return record

    def train(self, max_try_times=None, prepare_dataset=True, prepare_net=True):
        """

        :param max_try_times: if loss don't decrease for some times, then stop training
        :return:
        """
        if self.model_selector is None and max_try_times is not None:
            self.logger.warning("you have not set a model_selector!")
        if prepare_dataset:
            self.prepare_dataset()
        if prepare_net:
            self.prepare_net()
        self.logger.info("================training start=================")
        try_times = 0
        self.best_score_models = None
        # create one record on database
        if self.use_db:
            now = int(time.time())
            time_struct = time.localtime(now)
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
            experiment_insert = 'insert into experiment(theme_id,`name`,state_time,`desc`,tag,is_delete) value({},"{}","{}","{}","{}",{})'.format(
                1,
                self.experiment_name,
                time_str,
                self.description,
                self.tag,
                0)
            self.experiment_id = self.db_utils.insert(experiment_insert)
        for epoch in range(self.current_epoch, self.current_epoch + self.num_epoch):
            start_time = time.time()
            record = self.train_valid_one_epoch(epoch)

            self.scores_history[epoch] = record
            need_save, best_score_models, model_save_path = self.save(epoch, record)
            # best score models for now
            self.best_score_models = best_score_models
            if best_score_models != {}:
                for score_name in best_score_models.keys():
                    best_score_model = best_score_models[score_name]
                    self.logger.info("{} is best score of {}, saved at {}".format(best_score_model.score, score_name,
                                                                                  best_score_model.model_path))
            if not need_save:
                try_times += 1
            else:
                try_times = 0
            if max_try_times is not None and max_try_times < try_times:
                break
            self.logger.info("use {} seconds in the epoch".format(int(time.time() - start_time)))
            if self.use_db:
                # experiment_results
                record_attrs = self.scores_history[list(self.scores_history.keys())[0]].scores
                results_4_db = {}
                for attr_name in record_attrs:
                    results_4_db[attr_name] = [float(getattr(self.scores_history.get(epoch_no), attr_name)) for epoch_no
                                               in
                                               self.scores_history]
                results_4_db = json.dumps(results_4_db)
                experiment_update = "update experiment set result = '{}' where id = {}".format(results_4_db,
                                                                                               self.experiment_id)
                self.logger.debug("sql to update result :{}".format(experiment_update))
                self.db_utils.update(experiment_update)
        self.logger.info("================training is over=================")
        self.logger.info(
            "best valid_loss is\t{}\nbest_model_path is\t{}".format(self.best_score_models["valid_loss"].score,
                                                                    best_score_models["valid_loss"].model_path))
        return self.best_score_models["valid_loss"].score if self.best_score_models is not None else None

    def before_test(self, prepare_dataset=True, prepare_net=True):
        if self.is_pretrain is False:
            self.logger.warning("need to set pretrain is True!")
            raise RuntimeError
        if prepare_dataset:
            self.prepare_dataset(testing=True)
        if prepare_net:
            self.prepare_net()

        self.net_structure.eval()

    def test(self, prepare_dataset=True, prepare_net=True, save_predict_result=False, pretrain_model_path=None):
        assert self.is_pretrain is True
        if pretrain_model_path is not None:
            self.pretrain_path = pretrain_model_path
        self.before_test(prepare_dataset, prepare_net)
        if save_predict_result:
            file_utils.make_directory(self.result_save_path)
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
        if save_predict_result:
            print("save result of prediction at {}".format(self.result_save_path))
        return self.result_save_path

    def create_checkpoint(self, epoch=None, create4load=False):
        # todo set content of experiment_data in configure file
        experiment_data = {"epoch": epoch,
                           "state_dict": self.net_structure.state_dict(),
                           "optimizer": self.optimizer.state_dict(),
                           "history_path": self.history_save_path
                           }
        if create4load:
            # for load data only need a structure, don't need data
            return CustomCheckPoint(experiment_data.keys())
        else:
            return CustomCheckPoint(experiment_data.keys())(experiment_data)

    def save(self, epoch, record: EpochRecord):
        self.save_history()
        model_save_path = os.path.join(self.model_save_path, time_utils.get_date())
        file_utils.make_directory(model_save_path)
        file_name = 'ep{}_{}.pkl'.format(epoch, time_utils.get_time("%H-%M-%S"))
        model_save_path = os.path.join(model_save_path, file_name)
        if self.model_selector is None:
            self.__save_model(epoch, model_save_path)
            return True
        else:
            is_need_save, need_reason, best_socre_models = self.model_selector.add_record(record, model_save_path)
            if is_need_save:
                self.logger.info("save this model for {} is better".format(need_reason))
                self.__save_model(epoch, model_save_path)
                return is_need_save, best_socre_models, model_save_path
            else:
                self.logger.info("the eppoch's result is not good, not save")
                return is_need_save, best_socre_models, model_save_path

    def __save_model(self, epoch, model_save_path):
        self.logger.info("==============saving model data===============")
        checkpoint = self.create_checkpoint(epoch)
        save(checkpoint, model_save_path)
        self.logger.info("==============saved at {}===============".format(model_save_path))

    def load(self):
        model_save_path = self.pretrain_path
        if os.path.isfile(model_save_path):
            self.logger.info("==============loading model data===============")
            self.__load_model(model_save_path)
            self.logger.info("=> loaded checkpoint from '{}' ".format(model_save_path))
            self.load_history()
        else:
            self.logger.error("=> no checkpoint found at '{}'".format(model_save_path))

    def __load_model(self, model_save_path):
        check_point = self.create_checkpoint(create4load=True)
        load(check_point, model_save_path)
        self.current_epoch = check_point.epoch + 1
        self.net_structure.load_state_dict(check_point.state_dict)
        self.optimizer.load_state_dict(check_point.optimizer)

    def create_experiment_record(self):
        experiment_record = self.experiment_record()
        experiment_record.config_info = self.config_info
        experiment_record.epoch_records = self.scores_history
        return experiment_record

    def save_history(self):
        self.logger.info("=" * 10 + " saving experiment history " + "=" * 10)
        file_utils.make_directory(self.history_save_dir)
        experiment_record = self.create_experiment_record()
        experiment_record.save(self.history_save_path)
        self.logger.info("=" * 10 + " saved experiment history at {}".format(self.history_save_path) + "=" * 10)

    def load_history(self, is4train=False):
        """
        load history data
        :param is4train:
        :return:
        """
        experiment_record = None
        if hasattr(self, "history_save_path") and self.history_save_path is not None and self.history_save_path != "":
            if os.path.exists(self.history_save_path):
                if os.path.isfile(self.history_save_path):
                    self.logger.info("loading history")
                    experiment_record = self.create_experiment_record()
                    experiment_record = experiment_record.load(self.history_save_path)
                    self.scores_history = experiment_record.epoch_records
                    if is4train:
                        # delete history after current_epoch
                        for epoch_no in experiment_record.epoch_records:
                            if epoch_no <= self.current_epoch:
                                self.scores_history[epoch_no] = experiment_record.epoch_records[epoch_no]
                    self.logger.info("loaded history from {}".format(self.history_save_path))
                else:
                    self.logger.warning("{} is not a file".format(self.history_save_path))
            else:
                self.logger.error("no history to load")
        else:
            self.logger.error("not set history_save_path")
        return experiment_record

    # def show_history(self, use_log10=False):
    #     self.logger.info("showing history")
    #     record_attrs = self.scores_history[list(self.scores_history.keys())[0]].scores
    #     epoches = [[epoch_no for epoch_no in self.scores_history] for _ in range(len(record_attrs))]
    #     if use_log10:
    #         all_records = [
    #             [torch.log10(getattr(self.scores_history.get(epoch_no), attr_name)) for epoch_no in self.scores_history]
    #             for attr_name in record_attrs]
    #         record_attrs = ["log10_{}".format(attr_name) for attr_name in record_attrs]
    #         lineplot(all_records, epoches, labels=record_attrs, title="history analysis with log10")
    #     else:
    #         all_records = [[getattr(self.scores_history.get(epoch_no), attr_name) for epoch_no in self.scores_history]
    #                        for attr_name in record_attrs]
    #         lineplot(all_records, epoches, record_attrs, "history analysis")
    #     show()

    def estimate_history(self, use_log10=False, is_save=True, is_show=True, save_path=None):
        self.logger.info("showing history")
        record_attrs = self.scores_history[list(self.scores_history.keys())[0]].scores
        epoches = [[epoch_no for epoch_no in self.scores_history] for _ in range(len(record_attrs))]
        if use_log10:
            all_records = [
                [torch.log10(getattr(self.scores_history.get(epoch_no), attr_name)) for epoch_no in self.scores_history]
                for attr_name in record_attrs]
            record_attrs = ["log10_{}".format(attr_name) for attr_name in record_attrs]
            lineplot(all_records, epoches, labels=record_attrs, title="history analysis with log10")
        else:
            all_records = [[getattr(self.scores_history.get(epoch_no), attr_name) for epoch_no in self.scores_history]
                           for attr_name in record_attrs]
            lineplot(all_records, epoches, record_attrs, "history analysis")
        if is_show:
            show()
        if is_save:
            if save_path is None:
                self.logger.error("please set a save_path to save the figure")
            else:
                img_save(save_path)

        # get best results
        scores_dict = {}
        score_info = namedtuple("score_info", ["score_name", "scores", "model_path"])
        model_paths = [getattr(self.scores_history.get(epoch_no), "model_path") for epoch_no in
                       self.scores_history]

        for attr_name in record_attrs:
            scores = [getattr(self.scores_history.get(epoch_no), attr_name) for epoch_no in
                      self.scores_history]
            min_score = min(scores)
            min_index = scores.index(min_score)
            best_score = {}
            for tmp_attr_name in record_attrs:
                best_score[tmp_attr_name] = float(getattr(self.scores_history.get(min_index), tmp_attr_name))
            model_path = model_paths[min_index]
            score_info(score_name=attr_name, scores=best_score, model_path=model_path)
            scores_dict[attr_name] = score_info
            self.logger.info(
                "{}'s best score is {},epoch is {},model path is {}".format(attr_name, best_score, min_index,
                                                                            model_path))
        self.logger.info("total num_epoch is {}".format(len(self.scores_history)))
        return scores_dict

    def estimate(self, save_path, use_log10=False, is_save=True, is_show=True):
        self.load_history()
        # print(experiment_record.config_info)
        if self.scores_history == {}:
            self.logger.error("no history")
        else:
            self.estimate_history(use_log10, is_save, is_show, save_path=save_path)

    def sample_test(self, n_sample=3, epoch=3):
        cur_epoch = self.num_epoch
        self.dataset.get_samle_dataloader(num_samples=n_sample, target=self)
        self.num_epoch = epoch
        self.train(prepare_dataset=False)
        self.test(prepare_dataset=False)
        self.estimate()
        self.num_epoch = cur_epoch

    def prepare_data(self, data, data_type=None):
        # handler data as numpy
        data = Variable(data)
        if data_type == "float":
            data = data.float()
        elif data_type == "double":
            data = data.double()

        if self.is_use_gpu and torch.cuda.is_available():
            return data.cuda()
        else:
            return data

    def train_new_network(self, max_try_times=None):
        assert self.is_pretrain is False
        self.train(max_try_times)

    def grid_train(experiment, configs: dict, max_try_times=4):
        """
        use different configs to train, get the best result.
        Args:
            configs:

        Returns:

        """
        best_score = None
        best_config = None
        all_configs = []
        haved_key = []

        def sort_config(configs, current_config=[]):
            if len(current_config) == len(configs):
                all_configs.append(current_config.copy())
                return

            def get_config_key():
                for key in list(configs.keys()):
                    if key not in haved_key:
                        haved_key.append(key)
                        return key

            key = get_config_key()
            for value in configs[key]:
                current_config.append("{}:{}".format(key, value))
                sort_config(configs, current_config)
                current_config.remove("{}:{}".format(key, value))
            haved_key.remove(key)

        sort_config(configs)
        experiment.logger.info("all configs \n:{}".format(all_configs))
        # todo change net or dataset
        experiment.prepare_dataset()
        experiment.prepare_net()
        for config_pattern in all_configs:
            for setting in config_pattern:
                attr, value = setting.split(":")
                setattr(experiment, attr, value)
            best_score_experiment = experiment.train(max_try_times=max_try_times, prepare_net=False,
                                                     prepare_dataset=False)
            experiment.logger.info("config is {}:valid_loss is {}".format(config_pattern, best_score_experiment))
            if best_score is None or (experiment.is_bigger_better and best_score < best_score_experiment) or (
                    not experiment.is_bigger_better and best_score > best_score_experiment):
                best_score = best_score_experiment
                best_config = config_pattern
        experiment.logger.info("best result :\n{}:{}".format(best_config, best_score))

        # todo improve the data format to return
        return best_config, best_score
