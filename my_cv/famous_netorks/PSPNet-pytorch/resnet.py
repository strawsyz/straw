import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Sequential):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        # ResNet中的卷积层都没有bias
        self.add_module("Conv",
                        nn.Conv2d(in_n, out_n, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))


class ConvBN(Conv):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1):
        super(ConvBN, self).__init__(in_n, out_n, kernel_size, stride, padding)
        self.add_module("BN", nn.BatchNorm2d(out_n))


class ConvBNReLU(ConvBN):
    def __int__(self, in_n, out_n, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__(in_n, out_n, kernel_size, stride, padding)
        self.add_module("ReLU", nn.ReLU(inplace=True))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_n, out_n, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 注意传入输入参数的顺序
        self.conv_bn_relu_1 = ConvBNReLU(in_n, out_n, stride=stride)
        self.conv_bn_1 = ConvBN(out_n, out_n)
        self.downsample = downsample

    def forward(self, x):
        # 残差部分
        if self.downsample:
            # 如果有下采样，就把下采样的部分作为残差部分
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv_bn_relu_1(x)
        out = self.conv_bn_1(out)

        out += residual

        return F.relu(out, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_n, out_n, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv_bn_relu_1 = ConvBNReLU(in_n, out_n, kernel_size=1, padding=0)
        self.conv_bn_relu_2 = ConvBNReLU(out_n, out_n, stride=stride)
        self.conv_bn_1 = ConvBN(out_n, self.expansion * out_n, kernel_size=1, padding=0)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv_bn_relu_1(x)
        out = self.conv_bn_relu_2(out)
        out = self.conv_bn_1(out)
        out += residual

        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    def __init__(self, block, n_blocks, n_classes=1000, deep_base=True):
        """
        初始化ResNet
        :param block:  模块
        :param n_blocks:  每层的模块数量
        :param n_classes: 输出的类别数
        :param deep_base: 待理解
        """
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.in_n = 64
            self.conv_bn_relu_1 = ConvBNReLU(3, 64, kernel_size=7, stride=2)
        else:
            self.in_n = 128
            self.conv_bn_relu_1 = ConvBNReLU(3, 64, stride=2)
            # padding默认是1，kernel是3，stride是1
            self.conv_bn_relu_2 = ConvBNReLU(64, 64)
            self.conv_bn_relu_3 = ConvBNReLU(64, 128)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_1 = self._make_layer(block, 64, n_blocks[0])
        self.layer_2 = self._make_layer(block, 64 * 2, n_blocks[1], stride=2)
        self.layer_3 = self._make_layer(block, 64 * 4, n_blocks[2], stride=2)
        self.layer_4 = self._make_layer(block, 64 * 8, n_blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 最后接全连接层
        self.fc = nn.Linear(64 * 8 * block.expansion, n_classes)

    def _make_layer(self, block, n_chan, n_blocks, stride=1):
        """
        创建一层网络结构
        :param block: 组成层的模块
        :param n_chan: 输出的通道数
        :param n_blocks: 组成模块的数量
        :param stride: 第一个模块的移动量
        :return:
        """
        downsample = None
        if stride != 1 or self.in_n != n_chan * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_n, n_chan * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(n_chan * block.expansion),
            )

        layers = []
        # 第一个block是
        layers.append(block(self.in_n, n_chan, stride, downsample))
        # 更新in_n
        self.in_n = n_chan * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_n, n_chan))
        # 使用Sequential将所有层连接起来
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_bn_relu_1(x)
        if self.deep_base:
            # 如果不是deep_base，就还有两层
            out = self.conv_bn_relu_2(out)
            out = self.conv_bn_relu_3(out)
        out = self.maxpool(out)

        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def resnet_18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet_34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet_50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet_101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet_152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    # 经测试都可以跑通
    input = torch.rand(4, 3, 224, 224)
    model = resnet_18()
    output = model(input)
    print(output.shape)

    model = resnet_34()
    output = model(input)
    print(output.shape)

    model = resnet_50()
    output = model(input)
    print(output.shape)

    model = resnet_101()
    output = model(input)
    print(output.shape)

    model = resnet_152()
    output = model(input)
    print(output.shape)
