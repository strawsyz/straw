import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from configs.experiment_config import ImageSegmentationConfig
from experiments.deep_experiment import DeepExperiment


class ImageExperiment(ImageSegmentationConfig, DeepExperiment):
    def __init__(self, tag=None):
        ImageSegmentationConfig.__init__(self, tag=tag)
        # DeepExperiment.__init__(self)
        # super(ImageExperiment, self).__init__()
        self.show_config()

    def before_test(self, prepare_dataset=True, prepare_net=True, save_predict_result=False):
        """call this function before test a trained model"""
        super(ImageExperiment, self).before_test(prepare_dataset, prepare_net)

    def test_one_batch(self, image, mask, image_name, save_predict_result=False):
        image = self.prepare_data(image)
        mask = self.prepare_data(mask)
        start_time = time.time()
        out = self.net_structure(image)
        end_time = time.time()
        pps = len(image) / (end_time - start_time)
        _, _, width, height = mask.size()
        loss = self.loss_function(out, mask)
        self.logger.info("test_loss is {}".format(loss))
        num_batch, *_ = out.shape
        if save_predict_result:
            predict = out.squeeze().cpu().data.numpy()
            for index, pred in enumerate(predict):
                pred = pred * 255
                pred = pred.astype('uint8')
                pred = Image.fromarray(pred)
                # pred = pred.resize((width, height))
                save_path = os.path.join(self.result_save_path, image_name[index])
                pred.save(save_path)
                self.logger.info("================{}=================".format(save_path))
        return pps, loss.data

    def train_one_epoch(self, epoch):
        self.net_structure.train()
        train_loss = 0
        for image, mask in self.train_loader:
            if self.is_use_gpu:
                image, mask = Variable(image.cuda()), Variable(mask.cuda())
            else:
                image, mask = Variable(image), Variable(mask)

            self.optimizer.zero_grad()
            out = self.net_structure(image)
            loss = self.loss_function(out, mask)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data
        train_loss = train_loss / len(self.train_loader)
        self.scheduler.step()
        self.logger.info("EPOCH:{}\t train_loss:{:.6f}".format(epoch, train_loss))
        return train_loss

    def valid_one_epoch(self, epoch):
        self.net_structure.eval()
        with torch.no_grad():
            valid_loss = 0
            for image, mask in self.valid_loader:
                if self.is_use_gpu:
                    image, mask = Variable(image.cuda()), Variable(mask.cuda())
                else:
                    image, mask = Variable(image), Variable(mask)

                self.net_structure.zero_grad()
                predict = self.net_structure(image)
                valid_loss += self.loss_function(predict, mask)
            valid_loss /= len(self.valid_loader)
            self.logger.info("Epoch:{}\t valid_loss:{:.6f}".format(epoch, valid_loss))
        return valid_loss

    def predict_all_data(self):
        """predict all data ,include test dataset, train dataset and valid dataset"""
        self.prepare_net()
        # self.prepare_dataset()
        self.dataset.get_dataloader(self)
        self.net_structure.eval()

        from utils.file_utils import make_directory
        make_directory(self.result_save_path)
        for image, image_names in self.all_dataloader:
            image = self.prepare_data(image)
            self.optimizer.zero_grad()
            out = self.net_structure(image)
            predict = out.squeeze().cpu().data.numpy()
            for index, pred in enumerate(predict):
                pred = pred * 255
                pred = pred.astype('uint8')
                pred = Image.fromarray(pred)
                pred = pred.resize((224, 224))
                save_path = os.path.join(self.result_save_path, image_names[index])
                pred.save(save_path)
                self.logger.info("================{}=================".format(save_path))

    def test_file(self, file_path, mask_path, edge_path=None, predict_path=None):
        assert self.is_pretrain == True
        image = Image.open(file_path)
        mask = Image.open(mask_path)
        filename = os.path.basename(file_path)
        self.prepare_net()
        self.prepare_dataset()
        self.net_structure.eval()
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = image_transforms(image)
        image = self.prepare_data(image)
        if edge_path is not None and predict_path is not None:
            edge = self.prepare_data(test_transforms(Image.open(edge_path)))
            predict = self.prepare_data(test_transforms(Image.open(predict_path)))
            image = torch.cat([image, edge, predict], dim=0)
        image = torch.unsqueeze(image, dim=0)

        mask = test_transforms(mask)
        mask = self.prepare_data(mask)
        mask = torch.unsqueeze(mask, dim=0)
        start_time = time.time()
        out = self.net_structure(image)
        end_time = time.time()
        pps = len(image) / (end_time - start_time)
        _, _, width, height = mask.size()
        if self.loss_function is not None:
            loss = self.loss_function(out, mask)
            self.logger.info("test_loss is {}".format(loss))
        predict = out.squeeze(dim=1).cpu().data.numpy()
        from utils.file_utils import make_directory
        make_directory(self.result_save_path)

        for index, pred in enumerate(predict):
            pred = pred * 255
            pred = pred.astype('uint8')
            import numpy as np
            pred = np.where(pred > 1, 255, 0)

            # if len(pred.shape) == 2:
            #     pred = np.expand_dims(pred, 0)
            # pred.unsqueeze(pred)
            print(np.max(pred))
            print(np.min(pred))
            # print(pred)
            print(pred.shape)
            torch.log_softmax()
            pred = Image.fromarray(pred)
            pred = pred.resize((width, height))
            save_path = os.path.join(self.result_save_path, filename)
            pred.convert("L").save(save_path)
            self.logger.info("================{}=================".format(save_path))
        self.logger.info("{}:{}".format(filename, pps))
        return pps, loss.data

    def objective(self, trial):
        """used for optuna"""
        # Generate the model.
        # 创建包含优化的模型
        model = None
        from torch import optim
        import optuna
        # Generate the optimizers.
        # 创建可选优化器
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        # 创建可调整的学习率
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        # if self.prepare_dataset:
        #     self.prepare_dataset()
        # if self.prepare_net:
        #     self.prepare_net()
        best_loss = self.train(max_try_times=4, prepare_net=True, prepare_dataset=True)
        return best_loss
        # Get the MNIST dataset.
        # train_loader, valid_loader = get_mnist()

        # # Training of the model.
        # for epoch in range(EPOCHS):
        #     model.train()
        #     for batch_idx, (data, target) in enumerate(train_loader):
        #         # Limiting training data for faster epochs.
        #         # if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
        #         #     break
        #
        #         data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
        #
        #         optimizer.zero_grad()
        #         output = model(data)
        #         loss = F.nll_loss(output, target)
        #         loss.backward()
        #         optimizer.step()
        #
        #     # Validation of the model.
        #     model.eval()
        #     correct = 0
        #     with torch.no_grad():
        #         for batch_idx, (data, target) in enumerate(valid_loader):
        #             # Limiting validation data.
        #             # if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
        #             #     break
        #             data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
        #             output = model(data)
        #             # Get the index of the max log-probability.
        #             pred = output.argmax(dim=1, keepdim=True)
        #             correct += pred.eq(target.view_as(pred)).sum().item()
        #
        #     accuracy = correct / len(valid_loader)
        #
        #     trial.report(accuracy, epoch)
        #
        #     # Handle pruning based on the intermediate value.
        #     if trial.should_prune():
        #         raise optuna.exceptions.TrialPruned()
        #     print(accuracy)
        # return accuracy
