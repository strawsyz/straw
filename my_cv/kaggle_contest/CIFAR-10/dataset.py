from torchvision import datasets

datasets_path = "/home/straw/下载/dataset/"


def get_cifar10(path, train_transform=None, test_transform=None):
    train = datasets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
    test = datasets.CIFAR10(root=path, train=False, download=True, transform=test_transform)
    return train, test


TEST_PATH = "/home/straw/Downloads/dataset/kaggle/cifar10/test"
import os

from torch.utils.data import Dataset


def cifar10_testdata(transform):
    # test_data = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform)
    test_data = []
    for image_name in os.listdir(TEST_PATH):
        # TODO 可以增加一个数据类型的判断
        tmp_path = os.path.join(TEST_PATH, image_name)
        test_data.append(tmp_path)

    # test_data = list()
    # for root, dirs, _ in os.walk(data_dir):
    #     # 遍历类别
    #     for sub_dir in dirs:
    #         img_names = os.listdir(os.path.join(root, sub_dir))
    #         img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
    #
    #         # 遍历图片
    #         for i in range(len(img_names)):
    #             img_name = img_names[i]
    #             path_img = os.path.join(root, sub_dir, img_name)
    #             label = rmb_label[sub_dir]
    #             test_data.append((path_img, int(label)))

    return test_data


from PIL import Image


class Cifar10TestDataset(Dataset):
    def __init__(self, data_dir=TEST_PATH, transform=None):
        self.data_info = self.get_img_info()
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.data_info[index]
        img = Image.open(image_path)  # 0~255
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self):
        test_data = [None for i in range(len(os.listdir(TEST_PATH[:3000])))]
        for image_name in os.listdir(TEST_PATH[:3000]):
            file_name, _ = os.path.splitext(image_name)
            # TODO 可以增加一个数据类型的判断
            tmp_path = os.path.join(TEST_PATH, image_name)
            # 添加图像的路径
            test_data[int(file_name) - 1] = tmp_path
            # test_data.append(tmp_path)
        return test_data


if __name__ == '__main__':
    print(Cifar10TestDataset()[0])
#
# import os
# import random
# from PIL import Image
# from torch.utils.data import Dataset
#
# random.seed(1)
# rmb_label = {"1": 0, "100": 1}
#
#
# class CifarDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         """
#         rmb面额分类任务的Dataset
#         :param data_dir: str, 数据集所在路径
#         :param transform: torch.transform，数据预处理
#         """
#         self.label_name = {"1": 0, "100": 1}
#         self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
#         self.transform = transform
#
#     def __getitem__(self, index):
#         path_img, label = self.data_info[index]
#         img = Image.open(path_img).convert('RGB')     # 0~255
#
#         if self.transform is not None:
#             img = self.transform(img)   # 在这里做transform，转为tensor等等
#
#         return img, label
#
#     def __len__(self):
#         return len(self.data_info)
#
#     @staticmethod
#     def get_img_info(data_dir):
#         data_info = list()
#         for root, dirs, _ in os.walk(data_dir):
#             # 遍历类别
#             for sub_dir in dirs:
#                 img_names = os.listdir(os.path.join(root, sub_dir))
#                 img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
#
#                 # 遍历图片
#                 for i in range(len(img_names)):
#                     img_name = img_names[i]
#                     path_img = os.path.join(root, sub_dir, img_name)
#                     label = rmb_label[sub_dir]
#                     data_info.append((path_img, int(label)))
#
#         return data_info
