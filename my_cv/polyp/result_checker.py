import os

import numpy as np
from PIL import Image


def read_img(file_name):
    # 特殊情况下需要转换，为了速度暂时不转换为RGB
    # img = img.convert('RGB')
    img_array = np.array(Image.open(file_name))
    return img_array


# todo 点击某个按钮进入编辑模式，然后编辑图像
# def on_mouse_release(event):
#     if event.button == 1:
#         # 画点
#         global last_point
#         global center_of_hand_point
#         global points
#         global No
#         global last_text
#         if last_text:
#             last_text.remove()
#         if No > 15:
#             last_text = plt.text(0, 0, '16 points in enough', color='red')
#         else:
#             last_text = plt.text(0, 0, 'point' + str(No))
#             points[No] = [event.xdata, event.ydata]
#             No += 1
#             if not center_of_hand_point:
#                 center_of_hand_point = [event.xdata, event.ydata]
#             ax.scatter(event.xdata, event.ydata, color='black')
#             plt.text(event.xdata, event.ydata,
#                      str(No) + ':[' + str(int(event.xdata)) + "," + str(int(event.ydata)) + "]",
#                      color='red')
#             # 画线
#             if last_point:
#                 plt.plot([event.xdata, last_point[0]], [event.ydata, last_point[1]])
#             # fig.canvas.draw()
#             last_point = [event.xdata, event.ydata]
#         fig.canvas.draw_idle()


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
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册

    image_path = os.path.join(IMAGE_PATH, file_name)
    mask_path = os.path.join(MASK_PATH, file_name)
    result_path = os.path.join(RESLUT_PATH, file_name)
    ax = fig.add_subplot(131)
    ax.set_title(file_name)
    ax.imshow(read_img(image_path))
    plt.axis("off")

    ax = fig.add_subplot(132)
    ax.imshow(read_img(mask_path))
    ax.set_title("mask")
    plt.axis("off")

    ax = fig.add_subplot(133)
    ax.imshow(read_img(result_path))
    ax.set_title("result")
    plt.axis("off")

    # ubuntu上调用两次的plt.show()的话会报错，要用下面的函数
    fig.canvas.draw()


if __name__ == '__main__':
    """可以同时查看原图像，mask图像以及预测出来的mask图像"""
    file_names = []
    # IMAGE_PATH = '/home/straw/下载/dataset/gray2color/train/color/'
    # IMAGE_PATH = "/home/straw/.straw's back/image"
    IMAGE_PATH = '/home/straw/Downloads/dataset/polyp/data/'
    MASK_PATH = '/home/straw/Downloads/dataset/polyp/mask/'
    RESLUT_PATH = '/home/straw/Downloads/dataset/polyp/result/2020-06-07/'

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
