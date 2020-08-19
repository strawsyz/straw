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


def compare_image(image_paths):
    """比较所有图片是否大小一致"""
    """允许有两种大小的图片"""
    size = Image.open(image_paths[0]).size
    print("normal size is {}".format(size))
    other_size = None
    for image_path in image_paths[1:]:
        image = Image.open(image_path)
        if size != image.size:
            if other_size is None:
                other_size = image.size
            elif other_size != image.size:
                print("{} have different shape: {}".format(image_path, image.size))


