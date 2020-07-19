import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def img_hist(gray_img_array):
    # 不使用颜色信息
    # plt.gray()
    # 在原点左上角显示图像轮廓
    # plt.contour(gray_img_array, origin='image')
    # 保持原图的长宽比
    # plt.axis('equal')
    plt.figure()
    plt.hist(gray_img_array.flatten(), 128)
    plt.show()


if __name__ == '__main__':
    import os
    dir = "D:\Download\datasets\polyp\\06\mask"
    for file in os.listdir(dir):
        path = os.path.join(dir,file)
        img = Image.open(path)
        img = np.array(img)
        img_hist(img)
        width, height = img.shape
        # print(width * height)
        num = np.sum(img == 0) + np.sum(img == 255)
        print(width * height-num)
        print( np.sum(img==254))
