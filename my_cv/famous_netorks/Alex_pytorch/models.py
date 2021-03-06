import torch
from torch import nn


class Conv(nn.Sequential):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.add_module("Conv",
                        nn.Conv2d(in_n, out_n, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))


class ConvReLU(Conv):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1):
        super(ConvReLU, self).__init__(in_n, out_n, kernel_size, stride, padding)
        self.add_module("ReLU", nn.ReLU(inplace=True))


class LinearReLU(nn.Sequential):
    def __init__(self, in_n, out_n):
        super(LinearReLU, self).__init__()
        self.add_module("Linear", nn.Linear(in_n, out_n))
        self.add_module("ReLU", nn.ReLU(inplace=True))


class AlexNet(nn.Module):
    def __init__(self, n_classes=1000):
        """
        创建Alex网络结构
        :param n_classes: 图像分类的种类
        """
        super(AlexNet, self).__init__()
        self.n_features = 64
        self.features = nn.Sequential(
            ConvReLU(3, self.n_features, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvReLU(self.n_features, self.n_features * 3, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvReLU(self.n_features * 3, self.n_features * 6),
            ConvReLU(self.n_features * 6, self.n_features * 4),
            ConvReLU(self.n_features * 4, self.n_features * 4),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            LinearReLU(self.n_features * 4 * 6 * 6, 4096),
            LinearReLU(4096, 4096),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        out = self.features(x)
        # 卷积的结果压平，放入全连接层
        out = out.view(out.size(0), 256 * 6 * 6)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    # 经测试网络可以跑通
    model = AlexNet()
    print(model)
    input = torch.rand(4, 3, 224, 224)
    print(model(input).size())
