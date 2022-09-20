#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/4/19 20:11
# @Author  : strawsyz
# @File    : QR_generator.py
# @desc:

"""生成动态QR图像"""
from MyQR import myqr
import os


def generate_QR(image_path, save_path):
    version, level, qr_name = myqr.run(
        words="https://github.com/strawsyz",  # 可以是字符串，也可以是网址(前面要加http(s)://)
        version=1,  # 设置容错率为最高
        level='H',  # 控制纠错水平，范围是L、M、Q、H，从左到右依次升高
        picture=image_path,  # 将二维码和图片合成
        colorized=False,  # 彩色二维码
        contrast=1.0,  # 用以调节图片的对比度，1.0 表示原始图片，更小的值表示更低对比度，更大反之。默认为1.0
        brightness=1.0,  # 用来调节图片的亮度，其余用法和取值同上
        save_name=save_path,  # 保存文件的名字，格式可以是jpg,png,bmp,gif
        save_dir=os.getcwd()  # 控制位置
    )


if __name__ == '__main__':
    image_path = r"asdadaf.gif"
    save_path = r"3.gif"
    generate_QR(image_path, save_path)
