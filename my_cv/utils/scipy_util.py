from pylab import *
from PIL import Image
import numpy as np
from scipy.ndimage import filters, measurements, morphology


# standard_deviation越大,越模糊
def gaussian_filter(img_path, standard_deviation):
    """
    高斯模糊
    :param img_path: 原图像路径
    :param standard_deviation: 越大越模糊
    :return:
    """
    origin_img = Image.open(img_path)
    print(np.shape(origin_img))
    n_chan = len(origin_img.getbands())
    origin_img = np.array(origin_img)
    for i in range(n_chan):
        origin_img[:, :, i] = filters.gaussian_filter(origin_img[:, :, i], standard_deviation)
    result_img = Image.fromarray(np.uint8(origin_img))
    # result_img.show()
    return result_img


def sobel_example(img_path):
    im = np.array(Image.open(img_path).convert('L'))

    imx = np.zeros(im.shape)
    # sobel() 函数的第二个参数表示选择 x 或者 y 方向导数，
    # 第三个参数保存输出的变量
    filters.sobel(im, 1, imx)

    imy = np.zeros(im.shape)
    filters.sobel(im, 0, imy)

    magnitude = np.sqrt(imx ** 2 + imy ** 2)
    # 正导数显示为亮的像素，负导数显示为暗的像素,。灰色区域表示导数的值接近于零。
    plt.imshow(magnitude)
    plt.show()
    # print(magnitude)

    # 使用高斯倒数滤波器
    sigma = 5  # 标准差

    imx = zeros(im.shape)
    # 第三个参数指定对每个方向计算哪种类型的导数，第二个参数为使用的标准差
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)

    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    magnitude = np.sqrt(imx ** 2 + imy ** 2)

    imshow(magnitude)
    show()


def morphology_example(img_path):
    # 形态学（或数学形态学）是度量和分析基本形状的图像处理方法的基本框架与集合。
    # 形态学通常用于处理二值图像，但是也能够用于灰度图像。

    # 载入图像，然后使用阈值化操作，以保证处理的图像为二值图像
    im = np.array(Image.open(img_path).convert('L'))
    # 通过和 1 相乘，脚本将布尔数组转换成二进制表示。
    im = 1 * (im < 240)
    # 使用 label() 函数寻找单个的物体，
    # 并且按照它们属于哪个对象将整数标签给像素赋值。
    labels, nbr_objects = measurements.label(im)
    imshow(labels)
    # nbr_objects是计算出的物体的数目
    print("Number of objects:", nbr_objects)
    # 形态学二进制开（binary open）操作更好地分离各个对象
    # binary_opening() 函数的第二个参数指定一个数组结构元素。
    # 该数组表示以一个像素为中心时，使用哪些相邻像素。
    # 在这种情况下，我们在 y 方向上使用 9 个像素
    # （上面 4 个像素、像素本身、下面 4 个像素），
    # 在 x 方向上使用 5 个像素。你可以指定任意数组为结构元素，
    # 数组中的非零元素决定使用哪些相邻像素。
    # 参数 iterations 决定执行该操作的次数。
    im_open = morphology.binary_opening(im, ones((9, 5)), iterations=2)

    labels_open, nbr_objects_open = measurements.label(im_open)
    figure()
    imshow(labels)
    print("Number of objects:", nbr_objects_open)
    show()


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """ 使用A. Chambolle（2005）在公式（11）中的计算步骤实现Rudin-Osher-Fatemi（ROF）去噪模型
      输入：含有噪声的输入图像（灰度图像）、U 的初始值、TV 正则项权值、步长、停业条件
      输出：去噪和去除纹理后的图像、纹理残留"""

    m, n = im.shape  # 噪声图像的大小

    # 初始化
    U = U_init
    Px = im  # 对偶域的x 分量
    Py = im  # 对偶域的y 分量
    error = 1

    while error > tolerance:
        Uold = U

        # 原始变量的梯度
        # roll() 函数。顾名思义，在一个坐标轴上，它循环“滚动”数组中的元素值。
        # 该函数可以非常方便地计算邻域元素的差异，比如这里的导数
        GradUx = roll(U, -1, axis=1) - U  # 变量U 梯度的x 分量
        GradUy = roll(U, -1, axis=0) - U  # 变量U 梯度的y 分量

        # 更新对偶变量
        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy
        NormNew = maximum(1, sqrt(PxNew ** 2 + PyNew ** 2))

        Px = PxNew / NormNew  # 更新x 分量（对偶）
        Py = PyNew / NormNew  # 更新y 分量（对偶）

        # 更新原始变量
        RxPx = roll(Px, 1, axis=1)  # 对x 分量进行向右x 轴平移
        RyPy = roll(Py, 1, axis=0)  # 对y 分量进行向右y 轴平移

        DivP = (Px - RxPx) + (Py - RyPy)  # 对偶域的散度
        U = im + tv_weight * DivP  # 更新原始变量

        # 更新误差
        # linalg.norm() 函数，该函数可以衡量两个数组间
        # （这个例子中是指图像矩阵 U和 Uold）的差异
        error = linalg.norm(U - Uold) / sqrt(n * m);

    return U, im - U  # 去噪后的图像和纹理残余


if __name__ == '__main__':
    path = '/home/straw/下载/dataset/my_dataset/test.jpg'
    re = gaussian_filter(path, 2)
    re.show()
