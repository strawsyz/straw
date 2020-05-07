import os
import h5py
import numpy as np
from PIL import Image


def data_preprocess():
    """
    将mat文件中的原图像，深度图像，语义分割的标签图像提取出来
    :return: 原图像，深度图像，语义分割的标签图像的文件夹
    """
    # http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    # 原始数据保存的位置
    raw_data_path = '/home/straw/下载/dataset/NYU/nyu_depth_v2_labeled.mat'
    data = h5py.File(raw_data_path)

    image = np.array(data['images'])
    depth = np.array(data['depths'])
    label = np.array(data['labels'])

    print(image.shape)
    # (1449, 3, 640, 480)
    print(label.shape)
    # (1449, 640, 480)
    image = np.transpose(image, (0, 2, 3, 1))
    print(image.shape)
    # (1449, 640, 480, 3)
    print(depth.shape)
    # (1449, 640, 480)

    image_path = '/home/straw/下载/dataset/NYU/images/'
    depth_path = '/home/straw/下载/dataset/NYU/depths/'
    label_path = '/home/straw/下载/dataset/NYU/labels/'
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    if not os.path.exists(depth_path):
        os.mkdir(depth_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    print("==================start=====================")
    # 提取原始图像
    for i in range(image.shape[0]):
        image_index_path = os.path.join(image_path, '{}.jpg'.format(i))
        raw_image = image[i, :, :, :]
        # 保存原始图像
        Image.fromarray(raw_image).save(image_index_path)

    # 提取深度图像
    for i in range(depth.shape[0]):
        depth_image_path = os.path.join(depth_path, '{}.jpg'.format(i))
        depth_image = depth[i, :, :]
        # 先设置好数字格式，否则保存的时候会报错
        image = Image.fromarray(np.uint8(depth_image * 255))
        # 保存文件
        image.save(depth_image_path)

    for i in range(label.shape[0]):
        # 拼接标签图像文件的路径
        label_image_path = os.path.join(label_path, "{}.jpg".format(i))
        label_image = label[i, :, :]
        image = Image.fromarray(np.uint8(label_image * 255))
        # 保存文件
        image.save(label_image_path)

    print("==================end=====================")
    return image_path, depth_path, label_path


def split(data_path, train_rate=0.7):
    """分割数据集
    train_rate是占所有数据集中的比例，1-train_rate就是测试集的比例
    val_rate是验证集在训练集中的比例"""
    # todo 增加验证集的分割
    # data_paths = os.listdir(data_path)
    data_paths = []
    for file in os.listdir(data_path):
        path = os.path.join(data_path, file)
        if os.path.isfile(path):
            # 如果是文件就放入路径列表中
            data_paths.append(path)

    num_data = len(data_paths)
    num_train = int(num_data * train_rate)

    train_data_path = os.path.join(data_path, "train")
    test_data_path = os.path.join(data_path, "test")
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)
    if not os.path.exists(test_data_path):
        os.mkdir(test_data_path)

    for path in data_paths[:num_train]:
        os.rename(path,
                  os.path.join(train_data_path, path.split("/")[-1]))

    for path in data_paths[num_train:]:
        os.rename(path,
                  os.path.join(test_data_path, path.split("/")[-1]))


if __name__ == '__main__':
    # 图像预处理
    data_preprocess()
    image_path = '/home/straw/下载/dataset/NYU/images/'
    depth_path = '/home/straw/下载/dataset/NYU/depths/'
    label_path = '/home/straw/下载/dataset/NYU/labels/'
    split(image_path)
    split(depth_path)
    split(label_path)
