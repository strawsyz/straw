import os

from torchvision import datasets

datasets_path = "/home/straw/下载/dataset/"

def get_cifar10(path, **kwargs):
    train = datasets.CIFAR10(root=path, train=True, download=True, **kwargs)
    return train
    # print(len(train))  # 50000条数据
    # print(train[0][1])  # 图像种类


def get_cifar100(path, **kwargs):
    train = datasets.CIFAR100(root=path, train=True, download=True, **kwargs)
    return train
    # print(len(train))  # 50000条数据
    # print(train[0][1])  # 图像种类


def get_mnist(path, **kwargs):
    train = datasets.MNIST(root=path, train=True, download=True, **kwargs)
    return train
    # print(len(train))  # 60000条数据
    # print(train[0][1])  # 手写数字是个Tensor


def get_stl10(path, **kwargs):
    # STL10是用于开发无监督特征学习，深度学习，自学习算法的图像识别数据集
    # split:'train'返回训练集，‘test’返回测试集，‘unlabeled’无标签数据集，‘train+unlabeled’训练集加上无标签数据，没标签的数据标记为-1
    train = datasets.STL10(root=path, split="unlabeled", download=True, **kwargs)
    return train
    # print(len(train))  # 5000条训练数据，未标记数据有10万
    # print(train[100][0])  # 3通道的图像，96x96的大小


def get_lsun(path, **kwargs):
    # 使用LSUN数据集之前需要，先下载数据，放到对应的路径中，文件总共有100多G
    #  LSun场景分类的10个场景类别。LSUN 是一个场景理解图像数据集，主要包含了卧室、固房、客厅、教室等场景图像。
    #  20对象类别，共计约100万张标记图像
    #  每个类别的图像以LMDB格式存储，然后数据库被压缩。
    train = datasets.LSUN(path, classes="train", **kwargs)
    return train
    # print(len(train))  # 6000条数据
    # print(train[0][1])  # 手写数字是个Tensor


def get_coco(path, **kwargs):
    # COCO数据集是一个大型的、丰富的物体检测，分割和字幕数据集
    # 压缩后大约25GB
    # 图像包括91类目标，328,000影像和2,500,000个label。目前为止有语义分割的最大数据集，提供的类别有80类
    # 有超过33万张图片，其中20万张有标注，整个数据集中个体的数目超过150 万个。
    train = datasets.CocoCaptions(root=path, annFile='', **kwargs)
    return train


def get_voc(path, **kwargs):
    datasets.VOCSegmentation()

def get_dataset(name, **kwargs):
    path = os.path.join(datasets_path, name)
    if name == "CIFAR10":
        return get_cifar10(path, **kwargs)
    elif name == "CIFAR100":
        return get_cifar10(path, **kwargs)
    elif name == "MNIST":
        return get_mnist(path, **kwargs)
    elif name == "STL10":
        return get_stl10(path, **kwargs)
    elif name == "LSUN":
        return get_lsun(path, **kwargs)
    elif name == "COCO":
        return get_coco(path, **kwargs)
    else:
        # 如果没有选择任何的数据
        return None
train = get_dataset("LSUN")
print(len(train))
print(len(train[0]))
print(train[0][0])
