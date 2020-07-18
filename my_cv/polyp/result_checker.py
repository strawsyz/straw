import os

from PIL import Image


def read_img(file_name):
    # 特殊情况下需要转换，为了速度暂时不转换为RGB
    # img_array = np.array(Image.open(file_name))
    img = Image.open(file_name)
    # 防止一个通道的图像无法正常显示
    # img = img.convert('RGB')
    return img

def on_key_release(event):
    if event.key == 'n':
        if index[0] < len(file_names) - 1:
            index[0] += 1
            show_images(file_names[index[0]])
        else:
            print("It's the last image")
    elif event.key == "b":
        # 返回上一张图像
        if index[0] > 0:
            index[0] -= 1
            show_images(file_names[index[0]])
        else:
            print("It's the first image")


def show_images(file_name):

    fig = plt.figure('checker result')

    # fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    # 取消默认快捷键的注册
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    image_path = os.path.join(IMAGE_PATH, file_name)
    mask_path = os.path.join(MASK_PATH, file_name)
    result_path = os.path.join(RESLUT_PATH, file_name)
    ax = fig.add_subplot(131)
    ax.set_title(file_name)
    ax.imshow(read_img(image_path))
    plt.axis("off")

    ax = fig.add_subplot(132)
    ax.imshow(read_img(mask_path), cmap='gray')
    ax.set_title("mask")
    plt.axis("off")

    ax = fig.add_subplot(133)
    ax.imshow(read_img(result_path), cmap='gray')
    ax.set_title("result")
    plt.axis("off")

    # ubuntu上调用两次的plt.show()的话会报错，要用下面的函数
    fig.canvas.draw()


if __name__ == '__main__':
    """可以同时查看原图像，mask图像以及预测出来的mask图像"""
    file_names = []
    # IMAGE_PATH = '/home/straw/下载/dataset/gray2color/train/color/'
    # IMAGE_PATH = "/home/straw/.straw's back/image"
    IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/05/data"
    MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/05/mask"
    RESLUT_PATH = '/home/straw/Downloads/dataset/polyp/result/2020-06-15/'
    IMAGE_PATH = "D:\Download\datasets\polyp\\06\mask"
    MASK_PATH = "D:\Download\datasets\polyp\\06\mask"
    RESLUT_PATH = 'D:\Download\datasets\polyp\\06\mask'

    for file_name in os.listdir(RESLUT_PATH):
        file_names.append(file_name)

    from matplotlib import pyplot as plt

    # 1.    fig.canvas.draw_idle()
    # 重新绘制整个图表
    # 2.    fig.canvas.mpl_disconnect()
    # 取消已经注册的响应函数。这里是为了取消默认快捷键
    # 查询已经注册的响应函数
    # fig.canvas.callbacks.callbacks

    fig = plt.figure('checker result')

    # fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册

    # "111" means "1x1 grid, first subplot"
    index = [0]
    show_images(file_names[index[0]])

    plt.show()
