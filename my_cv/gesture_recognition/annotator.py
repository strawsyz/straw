import os
from PIL import Image
from pylab import *

# 0 是掌心
# 1-3 是大拇指上的点
# 4-6 是食指上的点
# 7-9 是中指上的点
# 10-12 是无名指指上的点
# 13-15 是大拇指上的点

import numpy as np


def read_img(file_name):
    img = Image.open(file_name)
    img = img.convert('RGB')
    img_array = np.array(img)
    return img_array


def make_directory(dir_name):
    """判断文件夹是否存在，如果不存在就新建文件夹"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, mode=0o777)


def on_mouse_release(event):
    if event.button == 1:
        # 画点
        global last_point
        global center_of_hand_point
        global points
        global No
        global last_text
        if last_text:
            last_text.remove()
        if No > 15:
            last_text = plt.text(0, 0, '16 points in enough', color='red')
        else:
            last_text = plt.text(0, 0, 'point' + str(No))
            points[No] = [event.xdata, event.ydata]
            No += 1
            if not center_of_hand_point:
                center_of_hand_point = [event.xdata, event.ydata]
            ax.scatter(event.xdata, event.ydata, color='black')
            plt.text(event.xdata, event.ydata,
                     str(No) + ':[' + str(int(event.xdata)) + "," + str(int(event.ydata)) + "]",
                     color='red')
            # 画线
            if last_point:
                plt.plot([event.xdata, last_point[0]], [event.ydata, last_point[1]])
            # fig.canvas.draw()
            last_point = [event.xdata, event.ydata]
        fig.canvas.draw_idle()


def on_key_release(event):
    global image_path
    global last_point
    global last_text
    global center_of_hand_point
    global No
    if event.key == '0':
        last_point = center_of_hand_point
    elif event.key == 'n':
        if event.inaxes:
            event.inaxes.clear()
            if len(img_list) > 0:
                center_of_hand_point = None
                last_point = None
                No = 0
                image_path = img_list.pop()
                event.inaxes.imshow(read_img(image_path))
                fig.canvas.draw()
            else:
                info = "no Image"
                if last_text:
                    last_text.remove()
                last_text = plt.text(0, 0, info, color='red')
                fig.canvas.draw_idle()
    elif event.key == 's':
        file_name = '.'.join(os.path.basename(image_path).split('.')[:-1])
        save_path = os.path.join(LABEL_SAVE_PATH, file_name)
        global points
        if event.inaxes:
            text = event.inaxes.text(0.5, 0.5, 'event', ha='center', va='center', fontdict={'size': 20})
            text.set_text(file_name)
            fig.canvas.draw_idle()
            #            save file
            np.save(save_path, points)

    elif event.key == 'r':
        # reset all points
        last_point = None
        center_of_hand_point = None
        No = 0
        if event.inaxes:
            event.inaxes.clear()
            # if len(img_list) > 0:
            # image_path = img_list.pop()
            event.inaxes.imshow(read_img(image_path))
            fig.canvas.draw()

if __name__ == '__main__':
    # prepare files
    img_list = []
    # IMAGE_PATH = '/home/straw/下载/dataset/gray2color/train/color/'
    # IMAGE_PATH = "/home/straw/.straw's back/image"
    IMAGE_PATH = '/home/straw/下载/dataset/FreiHAND_pub_v2/evaluation/rgb'
    LABEL_SAVE_PATH = 'result/'
    last_text = None
    make_directory(LABEL_SAVE_PATH)
    for file in os.listdir(IMAGE_PATH):
        img_list.append(os.path.join(IMAGE_PATH, file))

    last_point = None
    center_of_hand_point = None
    No = 0
    image_path = img_list.pop()
    img_array = read_img(image_path)
    points = np.full((16, 2), -1)
    (height, width) = img_array.shape[:2]

    from matplotlib import pyplot as plt

    # 1.    fig.canvas.draw_idle()
    # 重新绘制整个图表
    # 2.    fig.canvas.mpl_disconnect()
    # 取消已经注册的响应函数。这里是为了取消默认快捷键
    # 查询已经注册的响应函数
    # fig.canvas.callbacks.callbacks
    print('Please click 16 points in order')

    fig = plt.figure('Please click 16 points in order')

    fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册

    # "111" means "1x1 grid, first subplot"
    ax = fig.add_subplot(111)
    ax.imshow(img_array)
    plt.axis("off")
    plt.show()

    # plt.ion()
    # while True:
    #     plt.show()
    #     y, x = plt.ginput()[0]
    #     x, y = int(x), int(y)
    #     print('you clicked:x is {0}, y is {1}'.format(x, y))
    #     new_color = [255, 255, 255]
    #     # get_neighbors(img_array,(x, y), width, height)
    #     set_color(img_array, (x, y), width, height, scale=5)
    #     # expanded_neighbors = expand(neighbors, expaned_scale, width, height)
    #
    #     plt.close()
    #     plt.figure()
    #     new_image = Image.fromarray(img_array)
    #     plt.imshow(new_image)
    #     # new_image.save('res.png')
    # plt.ioff()
