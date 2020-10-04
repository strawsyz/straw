import random
import os

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.defchararray import isdigit
from utils.matplotlib_utils import save_ax


def read_img(file_name):
    img = Image.open(file_name)
    img = img.convert('RGB')
    return img


def on_key_release(event):
    global ax_save_path, file_names, index
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
    elif event.key == "s":
        ax = plt.gca()
        save_ax(ax, fig, ax_save_path, y_expand=1.08)
        print(ax)
    elif isdigit(event.key):
        idx_4_ax = int(event.key)
        if len(axs) >= idx_4_ax:
            ax = axs[idx_4_ax - 1]
            plt.sca(ax)
            print("change to {} ax".format(idx_4_ax))


def show_images(file_name):
    global fig, dir_paths, axs, RESULT_PATH
    fig.suptitle(file_name)
    for dir_path, ax in zip(dir_paths, axs):
        image_path = os.path.join(dir_path, file_name)
        ax.imshow(read_img(image_path), cmap='gray')
        ax.set_title(file_name)
        image = read_img(image_path)
        if dir_path == RESULT_PATH:
            image_array = np.asarray(image)
            image_array = np.where(image_array > 0, 255, 0)
            # np.transpose()
            image_array = image_array.astype(np.uint8)
            image = Image.fromarray(image_array)
            ax.imshow(image)
        else:
            ax.imshow(image)
        bwith = 0.5
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)

    fig.canvas.draw()


def main(result_path, ax_save_path="results/tmp.png", data_path=r"C:\(lab\datasets\polyp\TMP\07\data",
         mask_path=r"C:\(lab\datasets\polyp\TMP\07\mask"):
    """check images in three different folders"""
    dir_paths = [data_path, mask_path, result_path]
    fig, (axs) = plt.subplots(1, len(dir_paths))
    index = [0]
    num_files_pre_dir = [len(os.listdir(dir_path)) for dir_path in dir_paths]
    min_index = num_files_pre_dir.index(min(num_files_pre_dir))
    filenames = []
    for file_name in os.listdir(dir_paths[min_index]):
        filenames.append(file_name)
    random.shuffle(filenames)

    globals()["file_names"] = filenames
    globals()["RESULT_PATH"] = result_path
    globals()["DATA_PATH"] = data_path
    globals()["MASK_PATH"] = mask_path
    globals()["dir_paths"] = dir_paths
    globals()["ax_save_path"] = ax_save_path
    globals()["fig"] = fig
    globals()["axs"] = axs
    globals()["index"] = index

    fig.canvas.mpl_connect("key_release_event", on_key_release)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    show_images(file_names[index[0]])
    plt.show()
