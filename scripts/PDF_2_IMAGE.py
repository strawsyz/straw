#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 13:05
# @Author  : strawsyz
# @File    : PDF_2_IMAGE.py
# @desc:


import fitz

'''
# 将PDF转化为图片
pdfPath pdf文件的路径
imgPath 图像要保存的文件夹
zoom_x x方向的缩放系数
zoom_y y方向的缩放系数
rotation_angle 旋转角度
'''


def pdf_image(pdfPath, imgPath, zoom_x, zoom_y, rotation_angle):
    # 打开PDF文件
    pdf = fitz.open(pdfPath)
    # 逐页读取PDF
    for pg in range(0, pdf.pageCount):
        page = pdf[pg]
        # 设置缩放和旋转系数
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotation_angle)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        # 开始写图像
        img_save_path = os.path.join(imgPath, str(pg) + ".png")
        print(img_save_path)
        pm.save(img_save_path)
    pdf.close()


import glob
import fitz
import os


def pic2pdf(source_dirpath, save_path, is_overwrite=False):
    doc = fitz.open()
    for img in sorted(glob.glob(source_dirpath)):  # 读取图片，确保按文件名排序
        print(img)
        imgdoc = fitz.open(img)  # 打开图片
        pdfbytes = imgdoc.convertToPDF()  # 使用图片创建单页的 PDF
        imgpdf = fitz.open("pdf", pdfbytes)
        doc.insertPDF(imgpdf)  # 将当前页插入文档
    if os.path.exists(save_path) and is_overwrite:
        os.remove(save_path)
    doc.save(save_path)  # 保存pdf文件
    doc.close()
