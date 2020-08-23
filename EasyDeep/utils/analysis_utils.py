from matplotlib import pyplot as plt
import cv2
from PIL import Image


def img_hist_plt(gray_image_array):
    # 不使用颜色信息
    # plt.gray()
    # 在原点左上角显示图像轮廓
    # plt.contour(gray_img_array, origin='image')
    # 保持原图的长宽比
    # plt.axis('equal')
    plt.figure()
    plt.hist(gray_image_array.flatten(), 128)
    plt.show()


def img_hist_cv2(gray_image_array):
    channels = cv2.split(gray_image_array)
    new_channels = []
    for channel in channels:
        new_channels.append(cv2.equalizeHist(channel))

    result = cv2.merge(tuple(new_channels))
    cv2.imshow("equalizeHist", result)

    cv2.waitKey(0)


def is_same_size(image_paths):
    """比较所有图片是否大小一致"""
    size = Image.open(image_paths[0]).size
    print("normal size is {}".format(size))
    for image_path in image_paths[1:]:
        image = Image.open(image_path)
        if size != image.size:
            print("{} have different shape: {}".format(image_path, image.size))
