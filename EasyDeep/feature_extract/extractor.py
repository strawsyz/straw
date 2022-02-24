#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/9 13:32
# @Author  : strawsyz
# @File    : extractor.py
# @desc:
import torch

from pytorch_i3d import InceptionI3d


def get_extractor(model_name: str):
    if model_name == "resnet152":
        return get_resnet152()
    elif model_name == "i3d":
        return get_i3d_model()


# tensorflow
def get_resnet152(model_name):
    import keras
    from tensorflow.keras.models import Model  # pip install tensorflow (==2.3.0)
    base_model = keras.applications.resnet.ResNet152(include_top=True,
                                                     weights='imagenet',
                                                     input_tensor=None,
                                                     input_shape=None,
                                                     pooling=None,
                                                     classes=1000)

    # define model with output after polling layer (dim=2048)
    model = Model(base_model.input,
                  outputs=[base_model.get_layer("avg_pool").output])
    model.trainable = False
    return model


# torch
def get_i3d_model(model_path=r"C:\(lab\OtherProjects\pytorch-i3d-master\models\rgb_imagenet.pt"):
    import platform
    if platform.system() == "Linux":
        model_path = r"/workspace/datasets/rgb_imagenet.pt"

    i3d = InceptionI3d(400, in_channels=3)
    # i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(model_path))
    i3d.cuda()
    return i3d


# def get_resnet152():
#     # 这里省略掉一堆import
#     import torchvision.models as models
#
#     from resnet import resnet152 as caffe_resnet
#     # import resnet
#     # 省略掉读取图片和预处理的步骤，下面的img就是已经经过预处理之后的图片
#
#     model = caffe_resnet.resnet152(pretrained=True)
#     del model.fc
#     model.fc = lambda x: x
#     model = model.cuda()
#
#     feat = model(img)


if __name__ == '__main__':
    model = get_extractor("i3d")
    print(model)
