import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from base.base_dataset import BaseDataSet
from configs.dataset_config import MNISTDatasetConfig
from utils.common_utils import copy_attr
from utils.log_utils import get_logger


class MNISTDataset(BaseDataSet, MNISTDatasetConfig):
    def __init__(self, config_instance=None):
        super(MNISTDataset, self).__init__(config_instance)
        self.logger = get_logger()
        self.train_dataset = datasets.MNIST(root=self.dataset_path,
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
        self.test_dataset = datasets.MNIST(root=self.dataset_path,
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)

    def get_dataloader(self, target):
        self.prepare_dataset()
        self.set_dataloader()
        copy_attr(self, target)

    def prepare_dataset(self):
        if self.set_dataset_num:
            if not self.test_model:
                self.train_dataset, _ = torch.utils.data.random_split(self.train_dataset,
                                                                      [self.train_num + self.valid_num,
                                                                       len(self.train_dataset) - (
                                                                               self.train_num + self.valid_num)])
                self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.train_dataset,
                                                                                       [self.train_num, self.valid_num])
            else:
                self.test_dataset, _ = torch.utils.data.random_split(self.test_dataset, [self.test_num, len(
                    self.test_dataset) - self.test_num])
        else:
            if not self.test_model:
                self.train_num = len(self.train_dataset)
                valid_num = int(self.train_num * self.valid_rate)
                if valid_num == 0 or valid_num == self.train_num:
                    self.logger.error("size of valid dataset or train dataset is 0")
                self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.train_dataset,
                                                                                       [self.train_num - valid_num,
                                                                                        valid_num])

    def set_dataloader(self):
        if self.train_dataset is not None:
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle)
        if self.valid_dataset is not None and len(self.valid_dataset) != 0:
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        if self.test_dataset is not None:
            self.test_loader = DataLoader(dataset=self.test_dataset,
                                          batch_size=self.batch_size_4_test, )

    def get_sample_data(self):
        images, labels = next(iter(self.train_loader))
        from utils.analysis_utils import show_images
        show_images(images, labels)
        # fig, axs = plt.subplots(1, len(images))
        # if len(images) > 1:
        #     for idx in range(len(images)):
        #         axs[idx].imshow(images[idx].squeeze(), cmap='gray')
        #         axs[idx].set_title(int(labels[idx]))
        #         axs[idx].axis("off")
        # else:
        #     axs.imshow(images[0].squeeze(), cmap='gray')
        #     axs.set_title(int(labels[0]))
        #     axs.axis("off")
        # plt.show()

    def __len__(self):
        if self.set_dataset_num:
            if self.test_model:
                return self.test_num
            else:
                return self.train_num + self.valid_num
        else:
            if self.test_model:
                return len(self.test_dataset)
            else:
                return len(self.train_dataset)


if __name__ == '__main__':
    dataset = MNISTDataset()
    dataset.get_dataloader(dataset)
    dataset.get_sample_data()
