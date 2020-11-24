import os
import time

from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from configs.experiment_config import ImageSegmentationConfig
from experiments.deep_experiment import DeepExperiment
import torch


class ImageExperiment(ImageSegmentationConfig, DeepExperiment):
    def __init__(self):
        super(ImageExperiment, self).__init__()
        self.show_config()


    def before_test(self, prepare_dataset=True, prepare_net=True, save_predict_result=False):
        """call this function before test a trained model"""
        super(ImageExperiment, self).before_test(prepare_dataset, prepare_net)
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
        return self.result_save_path

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
            for index in range(num_batch):
                pred = predict[index]
                pred = pred * 255
                pred = pred.astype('uint8')
                pred = Image.fromarray(pred)
                pred = pred.resize((width, height))
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            pred = Image.fromarray(pred)
            pred = pred.resize((width, height))
            save_path = os.path.join(self.result_save_path, filename)
            pred.convert("L").save(save_path)
            self.logger.info("================{}=================".format(save_path))
        self.logger.info("{}:{}".format(filename, pps))
        return pps, loss.data
