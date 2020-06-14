import pytest


def func(x):
    return x + 1


def test_answer():
    assert func(3) == 4


def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()


def test_answer1():
    assert func(3) == 4


class TestClass(object):
    def test_one(self):
        x = "this"
        assert 'h' in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, 'check')

import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # pytest.main(["temp.py"])
    from PIL import Image

    # img = Image.open("/home/straw/Downloads/dataset/polyp/mask/Proc201801150031_1_7.pn""
    path1 = "C:\\Users\Administrator\PycharmProjects\straw\my_cv\polyp\sample_data\mask\Proc201712270044_1_15.png"
    path2 = "sample_data/mask/Proc201801150031_1_7.png"
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    # mask_transforms = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     # 将输出设置为一个通道
    #     transforms.Grayscale(),
    #     # transforms.ToTensor(),
    # ])
    # mask_transforms(img1)
    # mask_transforms(img2)
    # img1.convert
    # print(img1.size)
    # print(img1.info)
    # print(img1.split())
    # print(img2.size)
    # print(img2.info)
    # print(img2.split())
    # mask_transforms = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     # 将输出设置为一个通道
    #     transforms.Grayscale(),
    #     # transforms.ToTensor(),
    # ])
    # print(img.shape)
    # mask_transforms(img)
    img1 = np.array(img1)
    print(img1)
    fig = plt.figure()
    plt.imshow(img1)
    plt.show()

    # img1.show()

    # import numpy as np
    #
    # x = np.array([1, 2, 3])
    # weight = np.array([2, 3, 1])
    # import scipy.signal
    #
    # scipy.signal.convolve(x, weight)
    # print(scipy.signal.convolve(x, weight))
