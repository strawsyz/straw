from torch import nn


def rand_bbox(size, lam):
    """
    占据图片lam部分的内容
    :param size:
    :param lam: 且的部分所占的比例
    :return:返回左上角和右下角的坐标的位置
    """
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # 随机确定图像的中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 将数字限制在范围内
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mix(data, targets1, alpha):
    # 将数据的顺序打乱
    indices = torch.randperm(data.size(0))
    # 由于后面没有用到，就注释掉了
    # shuffled_data = data[indices]
    # 数据对应的目标也进行相同的顺序打乱方式
    shuffled_targets1 = targets1[indices]
    # shuffled_targets2 = targets2[indices]
    # shuffled_targets3 = targets3[indices]
    # 使用beta分布， 曲线中x=0.5是最高点，范围是0到1
    # 设置截取图片所占据的最大的面积的百分比
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    # 从打乱的数据中截图一部分，贴到原来的数据中
    # 一句代码就实现了，有点巧妙
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # 计算实际上截取的图像所占的最大的空间数量
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, lam]
    # targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, targets


def mixup(data, targets1, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    # 随意一个值
    lam = np.random.beta(alpha, alpha)
    # 架构打乱的数据，盖上mask放到原图像上
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, lam]
    # 输出型的数据和目标值
    return data, targets


def cutmix_criterion(preds1, targets):
    targets1, shuffled_targets1, lam = targets[0], targets[1], targets[2]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # 混合的两个种类的图片，分别计算损失，乘以权重，最后求和
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, shuffled_targets1)


def mixup_criterion(preds1, targets):
    targets1, shuffled_targets1, lam = targets[0], targets[1], targets[2]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, shuffled_targets1)
    # lam * criterion(preds2,targets3) + ( 1 - lam) * criterion( preds2, targets4) + \
    # lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)


# from implementations.lightning import FMix
import torch.nn.functional as F
from fmix import sample_mask
import torch

import math
import random

import numpy as np
from scipy.stats import beta


def fftfreqnd(h, w=None, z=None):
    """ Get bin values for discrete fourier transform of size (h, w, z)
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ Samples a fourier image with given size and frequencies decayed by decay power
    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param


def make_low_freq_image(decay, shape, ch=1):
    """ Sample a low frequency image from fourier space
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)  # .reshape((1, *shape[:-1], -1))
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask


def sample_lam(alpha, reformulate=False):
    """ Sample a lambda from symmetric beta distribution with given alpha
    :param alpha: Alpha value for beta distribution
    :param reformulate: If True, uses the reformulation of [1].
    """
    if reformulate:
        lam = beta.rvs(alpha + 1, alpha)
    else:
        lam = beta.rvs(alpha, alpha)

    return lam


def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """ Binarises a given low frequency image such that it has mean lambda.
    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param lam: Mean value of final mask
    :param in_shape: Shape of inputs
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1 - lam):
        eff_soft = min(lam, 1 - lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask


def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """ Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
    it based on this lambda
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    """
    if isinstance(shape, int):
        shape = (shape,)

    # Choose lambda
    lam = sample_lam(alpha, reformulate)

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)

    return lam, mask


def sample_and_apply(x, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """
    :param x: Image batch on which to apply fmix of shape [b, c, shape*]
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    :return: mixed input, permutation indices, lambda value of mix,
    """
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    index = np.random.permutation(x.shape[0])

    x1, x2 = x * mask, x[index] * (1 - mask)
    return x1 + x2, index, lam


class FMixBase:
    r""" FMix augmentation
        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].
    """

    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__()
        self.decay_power = decay_power
        self.reformulate = reformulate
        self.size = size
        self.alpha = alpha
        self.max_soft = max_soft
        self.index = None
        self.lam = None

    def __call__(self, x):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError


def fmix_loss(input, y1, index, lam, train=True, reformulate=False):
    r"""Criterion for fmix
    Args:
        input: If train, mixed input. If not train, standard input
        y1: Targets for first image
        index: Permutation for mixing
        lam: Lambda value of mixing
        train: If true, sum cross entropy of input with y1 and y2, weighted by lam/(1-lam). If false, cross entropy loss with y1
    """

    if train and not reformulate:
        y2 = y1[index]
        return F.cross_entropy(input, y1) * lam + F.cross_entropy(input, y2) * (1 - lam)
    else:
        return F.cross_entropy(input, y1)


class FMix(FMixBase):
    r""" FMix augmentation
        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].
        Example
        -------
        .. code-block:: python
            class FMixExp(pl.LightningModule):
                def __init__(*args, **kwargs):
                    self.fmix = Fmix(...)
                    # ...
                def training_step(self, batch, batch_idx):
                    x, y = batch
                    x = self.fmix(x)
                    feature_maps = self.forward(x)
                    logits = self.classifier(feature_maps)
                    loss = self.fmix.loss(logits, y)
                    # ...
                    return loss
    """

    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__(decay_power, alpha, size, max_soft, reformulate)

    def __call__(self, x):
        # Sample mask and 产生随机序列
        lam, mask = sample_mask(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate)
        # 返回一个0到n-1的数组,顺序是打乱的
        index = torch.randperm(x.size(0), device=x.device)
        # index = torch.randperm(x.size(0)).to(x.device)
        mask = torch.from_numpy(mask).float().to(x.device)

        # 混合图像
        x1 = mask * x
        x2 = (1 - mask) * x[index]
        self.index = index
        self.lam = lam
        return x1 + x2

    def loss(self, y_pred, y, train=True):
        return fmix_loss(y_pred, y, self.index, self.lam, train, self.reformulate)


class CoolSystem(pl.LightningModule):
    def __init__(self):
        ...

        self.fmix = FMix()

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = self.fmix(x)

        x = self.forward(x)

        loss = self.fmix.loss(x, y)
        return {'loss': loss}
