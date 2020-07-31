import os

from PIL import Image
from matplotlib import pyplot as plt


def read_img(file_name):
    img = Image.open(file_name)
    # 防止一个通道的图像无法正常显示
    img = img.convert('RGB')
    return img


def on_key_release(event):
    if event.key == 'n':
        if index[0] < len(file_names) - 1:
            index[0] += 1
            show_images(file_names[index[0]])
        else:
            print("It's the last image")
    elif event.key == "b":
        if index[0] > 0:
            index[0] -= 1
            show_images(file_names[index[0]])
        else:
            print("It's the first image")


def show_images(file_name):
    fig.suptitle(file_name)
    for dir_path, ax in zip(dir_paths, axs):
        image_path = os.path.join(dir_path, file_name)
        ax.imshow(read_img(image_path), cmap='gray')
        ax.set_title(file_name)
        ax.imshow(read_img(image_path))
        plt.axis("off")

    # ubuntu上调用两次的plt.show()的话会报错，要用下面的函数
    fig.canvas.draw()


if __name__ == '__main__':
    """比较不同文件下的同名图像"""
    file_names = []
    # MASK_PATH = "D:\Download\datasets\polyp\\06\mask"
    # EDGE_PATH = 'D:\Download\datasets\polyp\\06\edge'
    # EDGE_PATH1 = "D:\Download\datasets\polyp\\06\edge1"

    # dir_paths = [MASK_PATH, EDGE_PATH, EDGE_PATH1]
    data_path = "/home/straw/Downloads/dataset/polyp/TMP/07/data"
    mask_path = "/home/straw/Downloads/dataset/polyp/TMP/07/mask"
    predict_path = "/home/straw/Download\models\polyp\\result/2020-07-25/"
    dir_paths = [data_path, mask_path, predict_path]
    for file_name in os.listdir(dir_paths[-1]):
        file_names.append(file_name)
    fig, (axs) = plt.subplots(1, len(dir_paths))

    fig.canvas.mpl_connect("key_release_event", on_key_release)
    # 取消默认快捷键的注册
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    index = [0]
    show_images(file_names[index[0]])
    plt.show()
