import matplotlib.pyplot as plt


def read_img(path):
    return plt.imread(path)
    # return Image.open(path)


def show_img(img):
    plt.imshow(img)
    plt.show()


def save(path):
    plt.savefig(path)


def plot(x_list, y_list, title="", style_list=None, axis='on'):
    plt.figure()
    for i in range(len(x_list)):
        if style_list is None:
            plt.plot(x_list[i], y_list[i])
        else:
            plt.plot(x_list[i], y_list[i], style_list[i])

    plt.title(title)
    plt.axis(axis)
    plt.show()


def three_D(x, y, z):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(x, y, z)
    plt.show()


def img_hist(gray_img):
    # im = array(Image.open('test.jpg').convert('L'))

    # 新建图像
    # plt.figure()
    # 不使用颜色信息
    # plt.gray()
    # 在原点左上角显示图像轮廓
    # plt.contour(im, origin='image')
    # 保持原图的长宽比
    # plt.axis('equal')
    # 不显示坐标轴
    # plt.axis('off')
    # 新建图像
    plt.figure()
    import numpy as np
    # 绘制图像的直方图
    # im.flatten()将图像转换为1维数组
    plt.hist(np.array(gray_img).flatten(), 128)
    plt.show()


if __name__ == '__main__':
    plt.figure()
    # img = read_img('test.jpg')
    import PIL.Image as Image

    img = Image.open('../study/numpy_study/test.jpg').convert('L')

    # img_hist(img)
    import numpy as np

    x = np.arange(-10, 10, 0.5)
    y = np.arange(-10, 10, 0.5)
    x, y = np.meshgrid(x, y)
    z = x ** 3 - y ** 2
    three_D(x, y, z)
