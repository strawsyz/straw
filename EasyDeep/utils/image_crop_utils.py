import os

from PIL import Image


def read_img(file_name):
    img = Image.open(file_name)
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
    fig.suptitle(file_name)
    image_path = os.path.join(images_dir, file_name)
    ax.imshow(read_img(image_path))
    ax.set_title(file_name)
    ax.imshow(read_img(image_path))
    plt.axis("off")

    # ubuntu上调用两次的plt.show()的话会报错，要用下面的函数
    fig.canvas.draw()


if __name__ == '__main__':
    """可以同时查看原图像，mask图像以及预测出来的mask图像"""
    file_names = []
    images_dir = os.path.dirname(__file__)
    images_dir = "C:\data_analysis\datasets\samples"
    dir_paths = [images_dir]

    for file_name in os.listdir(images_dir):
        file_names.append(file_name)

    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax = plt.axes()
    # fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册

    # "111" means "1x1 grid, first subplot"
    index = [0]
    show_images(file_names[index[0]])

    plt.show()
