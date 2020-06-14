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


if __name__ == '__main__':
    # pytest.main(["temp.py"])
    from PIL import Image
    from torchvision import transforms

    img = Image.open(
        "C:\\Users\Administrator\PycharmProjects\straw\my_cv\polyp\sample_data\mask\Proc201712270044_1_8.png")
    mask_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # 将输出设置为一个通道
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    img = mask_transforms(img)
    img.show()

    # import numpy as np
    #
    # x = np.array([1, 2, 3])
    # weight = np.array([2, 3, 1])
    # import scipy.signal
    #
    # scipy.signal.convolve(x, weight)
    # print(scipy.signal.convolve(x, weight))
