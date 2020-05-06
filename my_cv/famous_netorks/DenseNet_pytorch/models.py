from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class BNReLUConv(nn.Sequential):
    def __init__(self, in_n, out_n, **kwargs):
        super(BNReLUConv, self).__init__()
        self.add_module('BN', nn.BatchNorm2d(in_n))
        self.add_module("ReLU", nn.ReLU(inplace=True))
        self.add_module("Conv", nn.Conv2d(in_n, out_n, bias=False, **kwargs))


class DenseLayer(nn.Sequential):
    def __init__(self, in_n, growth_rate, bn_size, drop_rate=None):
        super(DenseLayer, self).__init__()
        # 继承nn.Sequential的话，记得要用add_module把层添加进去
        self.add_module("BNReLUConv_0", BNReLUConv(in_n, bn_size * growth_rate, kernel_size=1))
        self.add_module("BNReLUConv_1", BNReLUConv(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1))
        self.drop_rate = drop_rate

    def forward(self, x):
        # 调用父类的forward函数
        out = super(DenseLayer, self).forward(x)
        # todo 是否需要进行是否是训练模式的判断
        if self.drop_rate:
            # 如果有drop_out参数就需要drop_out掉
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        # 将输入和点前层的输出一起包装起来
        # x的通道数是in_n,out的通道数是growth_rate,所以最后输出的通道数是in_n + growth_rate
        return torch.cat([x, out], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, in_n, n_layers, growth_rate, bn_size, drop_rate=None):
        """
        输入是in_n个通道，输出是 in_n + n_layers * growth_rate
        """
        super(DenseBlock, self).__init__()
        for i in range(n_layers):
            layer = DenseLayer(in_n + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            self.add_module("DenseLayer_{}".format(i + 1), layer)


class Transition(nn.Sequential):
    def __init__(self, in_n, out_n):
        super(Transition, self).__init__()
        self.add_module("BNReLUConv", BNReLUConv(in_n, out_n, kernel_size=1))
        self.add_module("Pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, n_layers_each_block=(6, 12, 24, 16), n_features=64, bn_size=4, drop_rate=None,
                 n_classes=1000):
        super(DenseNet, self).__init__()

        # 输入部分，由于这个结构只出现一次，就不单独写一个类了
        self.layers = nn.Sequential(OrderedDict([
            ('Conv_0', nn.Conv2d(3, n_features, kernel_size=7, stride=2, padding=3, bias=False)),
            # 由于之后还会出现BN层所以这边不能命名为BN
            ("BN_0", nn.BatchNorm2d(num_features=n_features)),
            ("ReLU_0", nn.ReLU(inplace=True)),
            ("Pool_0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        # 添加dense块
        for i, n_layers in enumerate(n_layers_each_block):
            dense_block = DenseBlock(n_features, n_layers, growth_rate, bn_size, drop_rate)
            self.layers.add_module('DenseBlock{}'.format(i + 1), dense_block)
            # 计算下一个模块的输入通道数
            # 每一层dense_layer都会增加growth_rate个通道
            n_features = n_features + n_layers * growth_rate
            if i != len(n_layers_each_block) - 1:
                # 如果不是最后一个densor块，就添加Transition层
                transition = Transition(in_n=n_features, out_n=n_features // 2)
                self.layers.add_module('Transition{}'.format(i + 1), transition)
                # 经过Transition层之后，通道数变为原来的一半
                n_features = n_features // 2
        # 添加最后的BN层
        self.layers.add_module('BN', nn.BatchNorm2d(n_features))
        # 设置分类器
        self.classifier = nn.Linear(n_features, n_classes)

        # 官方的权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        out = F.relu(out, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(out.size(0), -1)
        # 经过分类器
        return self.classifier(out)


def densenet121(**kwargs):
    model = DenseNet(**kwargs)
    return model


def densenet161(**kwargs):
    model = DenseNet(n_features=96, growth_rate=48, n_layers_each_block=(6, 12, 36, 24), **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(n_features=64, n_layers_each_block=(6, 12, 32, 32), **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(n_features=64, n_layers_each_block=(6, 12, 48, 32), **kwargs)
    return model


if __name__ == '__main__':
    # 使用Sequential调试起来太麻烦，除非每个小模块建立之后都能经过测试
    # 进测试，可以图片可以跑通
    input = torch.rand(4, 3, 224, 224)

    model = densenet121()
    # print(model)
    print(model(input).size())
    model = densenet161()
    # print(model)
    print(model(input).size())
    model = densenet169()
    # print(model)
    print(model(input).size())
    model = densenet201()
    # print(model)
    print(model(input).size())
