import os

import numpy as np
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
        if dir_path == r"/home/straw/Download\models\polyp\result/2020-09-02/":
            image_path = os.path.join(dir_path, file_name)
            ax.imshow(read_img(image_path), cmap='gray')
            ax.set_title(file_name)
            image = read_img(image_path)
            image_array = np.asarray(image)
            image_array = np.where(image_array > 0, 255, 0)
            # np.transpose()
            image_array = image_array.astype(np.uint8)
            image = Image.fromarray(image_array)
            ax.imshow(image)
            plt.axis("off")
        else:
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
    # DATA_PATH = r"C:\(lab\datasets\polyp\TMP\07\data"
    # MASK_PATH = r"C:\(lab\datasets\polyp\TMP\07\mask"
    # RESULT_PATH = r"C:\(lab\datasets\polyp\TMP\07\predict_step_one"
    DATA_PATH = r"/home/straw/Downloads/dataset/polyp/TMP/07/data/"
    MASK_PATH = r"/home/straw/Downloads/dataset/polyp/TMP/07/mask/"
    RESULT_PATH = r"/home/straw/Download\models\polyp\result/2020-09-02/"
    dir_paths = [DATA_PATH, MASK_PATH, RESULT_PATH]
    num_files_pre_dir = [len(os.listdir(dir_path)) for dir_path in dir_paths]
    min_index = num_files_pre_dir.index(min(num_files_pre_dir))
    for file_name in os.listdir(dir_paths[min_index]):
        file_names.append(file_name)
    import random
    random.shuffle(file_names)
    fig, (axs) = plt.subplots(1, len(dir_paths))

    fig.canvas.mpl_connect("key_release_event", on_key_release)
    # 取消默认快捷键的注册
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    index = [0]
    show_images(file_names[index[0]])
    plt.show()
