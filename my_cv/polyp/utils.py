import os

import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


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


def create_support_image(image_path, target_dir):
    """
    创建原图像的边缘图像
    :param image_path:
    :param target_dir:
    :return:
    """
    pass


def check_same(dir1, dir2):
    file_list1 = os.listdir(dir1)
    file_list2 = os.listdir(dir2)

    file_set1 = set(file_list1)
    # 2有，但是1没有的文件
    lack = []
    for file in file_list2:
        if file in file_set1:
            file_set1.remove(file)
        else:
            lack.append(file)
    over = list(file_set1)
    if lack != []:
        print("dir1 lack files:{}".format(lack))
    if over != []:
        print("dir2 lack files:{}".format(over))


def hist_equal(img, z_max=200):
    """
    直方图均衡化，将暗的地方变量，亮的地方变暗
    :param img:
    :param z_max:
    :return:
    """
    if len(img.shape) == 2:
        height, width = img.shape
        n_chan = 1
    elif len(img.shape) == 3:
        height, width, n_chan = img.shape
        print(img[:, :, 0].shape)
    # H, W = img.shape
    # S is the total of pixels
    n_pixle = height * width
    out = img.copy()
    sum_h = 0.
    if n_chan == 1:
        for i in range(1, 255):
            ind = np.where(img == i)
            sum_h += len(img[ind])
            z_prime = z_max / n_pixle * sum_h
            out[ind] = z_prime
    else:
        for c in range(n_chan):
            tmp_img = img[:, :, c]
            tmp_out = tmp_img.copy()
            for i in range(1, 255):
                ind = np.where(tmp_img == i)
                sum_h += len(tmp_img[ind])
                z_prime = z_max / n_pixle * sum_h
                tmp_out[ind] = z_prime
            out[:, :, c] = tmp_out
    out = out.astype(np.uint8)

    return out


def read_img_as_np(path, convert=None):
    if convert is None:
        return np.asarray(Image.open(path))
    else:
        return np.asarray(Image.open(path).convert(convert))


def img_hist(path):
    img = read_img_as_np(path)
    plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()


if __name__ == '__main__':
    path = "sample_data/Proc201506160021_1_1.png"
    raw_img = Image.open(path)
    # raw_img = raw_img.convert("L")
    # img = read_img_as_np(path, "L")
    img = np.asarray(raw_img)
    res = hist_equal(img)
    res_img = Image.fromarray(res)

    fig = plt.figure("result checking")
    ax = fig.add_subplot(221)
    ax.set_title("raw img")
    ax.imshow(raw_img, cmap='gray')
    ax.axis("off")
    # ax.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))

    ax = fig.add_subplot(222)
    ax.set_title("raw hist")
    ax.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))

    ax = fig.add_subplot(223)
    ax.set_title("res img")
    ax.imshow(res_img, cmap='gray')
    ax.axis("off")

    ax = fig.add_subplot(224)
    ax.set_title("result hist")
    ax.hist(res.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()
