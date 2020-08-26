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
        if not self.test_model:
            self.num_train = len(self.train_dataset)
            if self.valid_rate is not None:
                self.num_valid = self.num_train * self.valid_rate
            else:
                self.num_valid = 0

            if self.valid_rate == 0:
                self.train_loader = DataLoader(self, batch_size=self.batch_size, shuffle=True)
                return self.train_loader
            else:
                num_valid_data = int(self.num_train * self.valid_rate)
                if num_valid_data == 0 or num_valid_data == self.num_train:
                    self.logger.error("valid datateset is None or train dataset is None")
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset,
                                                                                     [self.num_train - num_valid_data,
                                                                                      num_valid_data])
                self.train_loader = DataLoader(dataset=self.train_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=self.shuffle)

                self.valid_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        else:
            self.test_loader = DataLoader(dataset=self.test_dataset,
                                          batch_size=self.batch_size_4_test,
                                          shuffle=self.shuffle)

        copy_attr(self, target)

    def get_sample_data(self):
        images, labels = next(iter(self.train_loader))

        fig, axs = plt.subplots(1, len(images))
        for idx in range(len(images)):
            axs[idx].imshow(images[idx].squeeze(), cmap='gray')
            axs[idx].set_title(int(labels[idx]))
            axs[idx].axis("off")
        plt.show()

    def __len__(self):
        if self.test_model:
            return len(self.test_dataset)
        else:
            return len(self.train_dataset)

    # def __getitem__(self, idx):
    #     return self.images[idx], self.labels[idx]


if __name__ == '__main__':
    dataset = MNISTDataset()
    dataset.get_sample_data()
