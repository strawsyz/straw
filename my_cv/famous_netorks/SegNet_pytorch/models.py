import torch
from torch import nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, n_chan, n_classes):
        super(SegNet, self).__init__()

        N_CHAN = 64
        self.encoder_block_1 = EncoderBlock(n_chan, N_CHAN, 2)
        self.encoder_block_2 = EncoderBlock(N_CHAN, 2 * N_CHAN, 2)
        self.encoder_block_3 = EncoderBlock(2 * N_CHAN, 4 * N_CHAN, 3)
        self.encoder_block_4 = EncoderBlock(4 * N_CHAN, 8 * N_CHAN, 3)
        self.encoder_block_5 = EncoderBlock(8 * N_CHAN, 8 * N_CHAN, 3)
        # decoder部分
        self.decoder_block_5 = DecoderBlock(8 * N_CHAN, 8 * N_CHAN, 3)
        self.decoder_block_4 = DecoderBlock(8 * N_CHAN, 4 * N_CHAN, 3)
        self.decoder_block_3 = DecoderBlock(4 * N_CHAN, 2 * N_CHAN, 3)
        self.decoder_block_2 = DecoderBlock(2 * N_CHAN, N_CHAN, 2)
        self.decoder_block_1 = DecoderBlock(N_CHAN, n_classes, 2, is_last_decoder=True)

    def forward(self, x):
        # 编码器部分
        out, indices_1 = self.encoder_block_1(x)
        out, indices_2 = self.encoder_block_2(out)
        out, indices_3 = self.encoder_block_3(out)
        out, indices_4 = self.encoder_block_4(out)
        out, indices_5 = self.encoder_block_5(out)

        # 解码器部分
        out = self.decoder_block_5(out, indices_5)
        out = self.decoder_block_4(out, indices_4)
        out = self.decoder_block_3(out, indices_3)
        out = self.decoder_block_2(out, indices_2)
        out = self.decoder_block_1(out, indices_1)

        return out


class ConvBNReLU(nn.Module):
    BATCHNROM_MONMENTUM = 0.1

    def __init__(self, in_n, out_n, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv_1 = nn.Conv2d(in_n, out_n, kernel_size, stride, padding)
        self.bn_1 = nn.BatchNorm2d(out_n, momentum=self.BATCHNROM_MONMENTUM)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        return F.relu(out)


class EncoderBlock(nn.Module):
    BATCHNROM_MONMENTUM = 0.1

    def __init__(self, in_n, out_n, n_layer, kernel_size=3, stride=1, padding=1):
        super(EncoderBlock, self).__init__()
        self.conv_bn_relu_1 = ConvBNReLU(in_n, out_n, kernel_size, stride, padding)
        self.conv_bn_relu_2 = ConvBNReLU(out_n, out_n, kernel_size, stride, padding)
        self.n_layer = 2
        if n_layer == 3:
            self.conv_bn_relu_3 = ConvBNReLU(out_n, out_n, kernel_size, stride, padding)
            self.n_layer = 3

    def forward(self, x):
        out = self.conv_bn_relu_1(x)
        out = self.conv_bn_relu_2(out)
        if self.n_layer == 3:
            out = self.conv_bn_relu_3(out)
        out, indices = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)

        return out, indices


class DecoderBlock(nn.Module):
    BATCHNROM_MONMENTUM = 0.1

    def __init__(self, in_n, out_n, n_layer, kernel_size=3, stride=1, padding=1, is_last_decoder=False):
        super(DecoderBlock, self).__init__()
        self.conv_bn_relu_1 = ConvBNReLU(in_n, in_n, kernel_size, stride, padding)
        self.n_layer = 2

        # 由于最后一个解码器是二层，不会与3层的情况冲突，就不增加代码的复杂度了
        if is_last_decoder:
            #    如果是最后一个解码器,最后一层直接卷积，不使用BN和ReLU
            self.conv_bn_relu_2 = nn.Conv2d(in_n, out_n, kernel_size, stride, padding)
        else:
            # 不是最后一个解码器,且一共有3层
            if n_layer == 3:
                self.conv_bn_relu_2 = ConvBNReLU(in_n, in_n, kernel_size, stride, padding)
            else:
                # 不是三层就是两层，如果有两层
                self.conv_bn_relu_2 = ConvBNReLU(in_n, out_n, kernel_size, stride, padding)

        if n_layer == 3:
            self.conv_bn_relu_3 = ConvBNReLU(in_n, out_n, kernel_size, stride, padding)
            self.n_layer = 3

    def forward(self, x, indices):
        # 反池化操作
        out = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        out = self.conv_bn_relu_1(out)
        out = self.conv_bn_relu_2(out)
        if self.n_layer == 3:
            out = self.conv_bn_relu_3(out)

        return out


if __name__ == "__main__":
    # 经测试可以跑通
    inputs = torch.ones(1, 3, 224, 224)
    model = SegNet(3, 12)
    print(model(inputs).size())
    print(model)
