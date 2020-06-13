import os

import torchvision

ROOT_PATH = ""
TRAIN_PATH = os.path.join(ROOT_PATH, "")
TEST_PATH = os.path.join(ROOT_PATH, "")
LABEL_FILE_PAHT = os.path.join(ROOT_PATH, "")


def get_dataset():
    train_data = torchvision.datasets.ImageNet(TRAIN_PATH)
    test_data = torchvision.datasets.ImageNet(TEST_PATH)
    return train_data, test_data


if __name__ == '__main__':
    TRAIN_PATH = "C:\\Users\Administrator\Pictures\截屏"
    train_data = torchvision.datasets.ImageNet(TRAIN_PATH)
    print(train_data)
