import cv2

"""金字塔算法的函数"""


def resize(img, scale_factor):
    """通过指定因子来调整图像大小"""
    return cv2.resize(img, (int(img.shape[1] * (1 / scale_factor)), int(img.shape[0] * (1 / scale_factor))),
                      interpolation=cv2.INTER_AREA)


def pyramid(image, scale=1.5, min_size=(200, 80)):
    yield image

    while True:
        image = resize(image, scale)
        # 判断图像调整大小后是否小于最小尺寸
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        yield image
