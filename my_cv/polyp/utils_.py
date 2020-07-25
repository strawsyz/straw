import os

import h5py
from matplotlib import pyplot as plt


def data_preprocess():
    """
    将mat文件中的原图像，深度图像，语义分割的标签图像提取出来
    :return: 原图像，深度图像，语义分割的标签图像的文件夹
    """
    # http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    # 原始数据保存的位置
    raw_data_path = '/home/straw/下载/dataset/NYU/nyu_depth_v2_labeled.mat'
    data = h5py.File(raw_data_path)

    image = np.array(data['images'])
    depth = np.array(data['depths'])
    label = np.array(data['labels'])

    print(image.shape)
    # (1449, 3, 640, 480)
    print(label.shape)
    # (1449, 640, 480)
    image = np.transpose(image, (0, 2, 3, 1))
    print(image.shape)
    # (1449, 640, 480, 3)
    print(depth.shape)
    # (1449, 640, 480)

    image_path = '/home/straw/下载/dataset/NYU/images/'
    depth_path = '/home/straw/下载/dataset/NYU/depths/'
    label_path = '/home/straw/下载/dataset/NYU/labels/'
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    if not os.path.exists(depth_path):
        os.mkdir(depth_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    print("==================start=====================")
    # 提取原始图像
    for i in range(image.shape[0]):
        image_index_path = os.path.join(image_path, '{}.jpg'.format(i))
        raw_image = image[i, :, :, :]
        # 保存原始图像
        Image.fromarray(raw_image).save(image_index_path)

    # 提取深度图像
    for i in range(depth.shape[0]):
        depth_image_path = os.path.join(depth_path, '{}.jpg'.format(i))
        depth_image = depth[i, :, :]
        # 先设置好数字格式，否则保存的时候会报错
        image = Image.fromarray(np.uint8(depth_image * 255))
        # 保存文件
        image.save(depth_image_path)

    for i in range(label.shape[0]):
        # 拼接标签图像文件的路径
        label_image_path = os.path.join(label_path, "{}.jpg".format(i))
        label_image = label[i, :, :]
        image = Image.fromarray(np.uint8(label_image * 255))
        # 保存文件
        image.save(label_image_path)

    print("==================end=====================")
    return image_path, depth_path, label_path


def split(data_path, train_rate=0.7):
    """分割数据集
    train_rate是占所有数据集中的比例，1-train_rate就是测试集的比例
    val_rate是验证集在训练集中的比例"""
    # todo 增加验证集的分割
    # data_paths = os.listdir(data_path)
    data_paths = []
    for file in os.listdir(data_path):
        path = os.path.join(data_path, file)
        if os.path.isfile(path):
            # 如果是文件就放入路径列表中
            data_paths.append(path)

    num_data = len(data_paths)
    num_train = int(num_data * train_rate)

    train_data_path = os.path.join(data_path, "train")
    test_data_path = os.path.join(data_path, "test")
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)
    if not os.path.exists(test_data_path):
        os.mkdir(test_data_path)

    for path in data_paths[:num_train]:
        os.rename(path,
                  os.path.join(train_data_path, path.split("/")[-1]))

    for path in data_paths[num_train:]:
        os.rename(path,
                  os.path.join(test_data_path, path.split("/")[-1]))


def create_support_image(image_path, target_dir):
    """
    创建原图像的边缘图像
    :param image_path:
    :param target_dir:
    :return:
    """
    pass


def check_same(dir1, dir2):
    """
    判断两个文件夹中的文件是否一样
    如果不一样分别显示两个文件夹中缺少了哪些文件
    :param dir1:
    :param dir2:
    :return:
    """
    file_list1 = os.listdir(dir1)
    file_list2 = os.listdir(dir2)

    file_set1 = set(file_list1)
    # 2有，但是1没有的文件
    lack = []
    for file in file_list2:
        if file in file_set1:
            file_set1.remove(file)
        else:
            lack.append(file)
    over = list(file_set1)
    if lack != []:
        print("dir1 lack files:{}".format(lack))
    if over != []:
        print("dir2 lack files:{}".format(over))


def hist_equal_one_chan(img, height, width):
    n_pixle = height * width
    out = img.copy()
    sum_h = 0
    n_dark_pixle = len(img[np.where(img < 128)])
    n_bright_pixle = n_pixle - n_dark_pixle
    # 128
    for i in range(128):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = 127 / n_dark_pixle * sum_h
        out[ind] = z_prime
    sum_h = 0
    for i in range(255, 127, -1):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = 255 - (127 / n_bright_pixle * sum_h)
        out[ind] = z_prime
    return out


def hist_equal_one_chan2(img, height, width):
    n_pixle = height * width
    out = img.copy()
    sum_h = 0
    n_dark_pixle = len(img[np.where(img < 128)])
    n_bright_pixle = n_pixle - n_dark_pixle
    print(len(img[np.where(img == 0)]))
    print(len(img[np.where(img == 254)]))
    # 128
    for i in range(127, -1, -1):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = 128 - (128 / n_dark_pixle * sum_h)
        out[ind] = z_prime
    sum_h = 0
    for i in range(128, 256):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = 127 + (128 / n_bright_pixle * sum_h)
        out[ind] = z_prime
    pixels = [0, 1, 2, 3, 4, 250, 251, 252, 253, 254, 255]
    for pixel in pixels:
        print("pixel is : {}".format(pixel))
        print("raw:", len(img[np.where(img == pixel)]))
        print("result:", len(out[np.where(out == pixel)]))
    # print(len(img[np.where(out == 254)]))
    return out


def hist_equal(img, z_max=255):
    """
    直方图均衡化，将暗的地方变量，亮的地方变暗
    :param img:
    :param z_max: 原图像最亮的地方减去最暗的地方的值
    :return:
    """
    if len(img.shape) == 2:
        height, width = img.shape
        n_chan = 1
    elif len(img.shape) == 3:
        height, width, n_chan = img.shape
        print(img[:, :, 0].shape)
    # H, W = img.shape
    # S is the total of pixels
    n_pixle = height * width
    out = img.copy()
    sum_h = 0.
    if n_chan == 1:
        for i in range(1, 255):
            ind = np.where(img == i)
            sum_h += len(img[ind])
            z_prime = z_max / n_pixle * sum_h
            out[ind] = z_prime
    else:
        for c in range(n_chan):
            tmp_img = img[:, :, c]
            tmp_out = tmp_img.copy()
            for i in range(1, 255):
                ind = np.where(tmp_img == i)
                sum_h += len(tmp_img[ind])
                z_prime = z_max / n_pixle * sum_h
                tmp_out[ind] = z_prime
            out[:, :, c] = tmp_out
    out = out.astype(np.uint8)

    return out


def read_img_as_np(path, convert=None):
    if convert is None:
        return np.asarray(Image.open(path))
    else:
        return np.asarray(Image.open(path).convert(convert))


def img_hist(path):
    img = read_img_as_np(path)
    plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()


def convert_imgs(source_path, target_path):
    for file_name in os.listdir(source_path):
        file_path = os.path.join(source_path, file_name)
        img = Image.open(file_path)
        img = img.convert("L")
        save_path = os.path.join(target_path, file_name)
        img.save(save_path)
        # 理论上来讲经过之前的处理之后，所有的文件都被保存为png格式了。
        # 但为了保险，还是重新处理一遍后缀名
        # target_path = os.path.join(target_path, "{}.png".format(file_util.get_filename(file_path)))


def copy_images(source_path, target_path):
    """复制所有的文件"""
    for file_name in os.listdir(source_path):
        file_path = os.path.join(source_path, file_name)
        img = Image.open(file_path)
        save_path = os.path.join(target_path, file_name)
        img.save(save_path)


def convert2PNG(source_path, target_path, mask=False):
    img = Image.open(source_path)
    from utils import file_utils
    target_path = os.path.join(target_path, "{}.png".format(file_utils.get_filename(source_path)))
    if mask:
        # 如果是处理mask图像,把文件名中M去掉
        target_path = target_path.replace("M.png", ".png")
    img.save(target_path)
    print(target_path)


def get_needed_data(source_path, target_path, needed_part, ext=None):
    """
    将需要的部分图像从原图像切下来，然后保存到目标目录中
    :param source_path:
    :param target_path:
    :param needed_part: 下标是从0开始[ left, upper, right, lower]
    :param ext: 防止图片重名，用于标注
    :return:
    """
    for file_name in os.listdir(source_path):
        file_path = os.path.join(source_path, file_name)
        img = Image.open(file_path)
        cropped = img.crop(needed_part)
        # 理论上来讲经过之前的处理之后，所有的文件都被保存为png格式了。
        # 但为了保险，还是重新处理一遍后缀名
        # target_path = os.path.join(target_path, "{}.png".format(file_util.get_filename(file_path)))
        from utils import file_utils
        if ext is None:
            cropped.save(os.path.join(target_path, "{}.png".format(file_utils.get_filename(file_path))))
        else:
            cropped.save(os.path.join(target_path, "{}_{}.png".format(file_utils.get_filename(file_path), ext)))


def save_patch(source_path, target_dir, patch_box, ext, is_convert_gray=False):
    """
    将需要的部分图像从原图像切下来，然后保存到目标目录中
    :param source_path:
    :param target_dir:
    :param patch_box: 下标是从0开始[ left, upper, right, lower]
    :param ext: 防止图片重名，用于标注
    :return:
    """
    img = Image.open(source_path)
    if is_convert_gray:
        img = img.convert("L")
    from utils.file_utils import split_path
    _, _, filename, extension = split_path(source_path)
    target_path = os.path.join(target_dir, "{}_{}.{}".format(filename, ext, extension))
    print(target_path)
    cropped = img.crop(patch_box)
    cropped.save(target_path)


# 将图片切成小块增加图像数量
# 原本的图像大小1000width  800height
def create_patch(source_path, target_path, width, height, n_width, n_height):
    """
    将图像切成小块
    :param source_path:原图像的路径
    :param target_path:目标图像的保存路径
    :param width:
    :param height:
    :param n_width:在宽度上切成一定的份数
    :param n_height:在高度上切成一定的份数
    :return:
    """
    patch_width = width // n_width
    patch_height = height // n_height
    label = 0
    for start_x in range(0, width - patch_width + 1, patch_width):
        for start_y in range(0, height - patch_height + 1, patch_height):
            get_needed_data(source_path, target_path,
                            needed_part=(start_x, start_y, start_x + patch_width, start_y + patch_height), ext=label)
            label += 1


# 将图片切成小块增加图像数量
# 原本的图像大小1000width  800height
def create_patch_by_absolute_size(image_dir_path, mask_dir_path, target_image_dir_path, target_mask_dir_path,
                                  detected_size,
                                  patch_width, patch_height, max_num=20):
    import random
    random.seed(0)
    sample_mask_image = os.path.join(mask_dir_path, os.listdir(mask_dir_path)[0])
    sample_mask_image = Image.open(sample_mask_image)
    sample_mask_image = sample_mask_image.convert("L")
    sample_width, sample_height = sample_mask_image.size
    num_pixels = patch_width * patch_height
    for mask_filename in os.listdir(mask_dir_path):
        mask_filepath = os.path.join(mask_dir_path, mask_filename)
        image_filepath = os.path.join(image_dir_path, mask_filename)
        if os.path.isfile(mask_filepath) and os.path.isfile(image_filepath):
            pass
        else:
            print("{} or {} is not exist!".format(mask_filepath, image_filepath))
            continue
        # read mask file as gray image
        mask_image = Image.open(mask_filepath).convert('L')
        index = 0
        for _ in range(max_num):
            start_x, start_y = get_point(sample_width - patch_width, sample_height - patch_height)
            patch_box = [start_x, start_y, start_x + patch_width, start_y + patch_width]
            patch_image = mask_image.crop(patch_box)
            if check_mask_valid(patch_image, num_pixels, detected_size=detected_size):
                index += 1
                # 保存切割后的mask文件
                save_patch(mask_filepath, target_mask_dir_path, patch_box, index, is_convert_gray=True)
                # 将原图像对应的位置保存到目标文件夹
                save_patch(image_filepath, target_image_dir_path, patch_box, index)
                # path = os.path.join(mask_dir_path, "{}_{}".format(mask_image))


import random


def get_point(max_x, max_y):
    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)
    return start_x, start_y


from convert_utils import *


def check_mask_valid(mask_image, num_pixels, detected_size=0.3):
    mask_array = Image2np(mask_image)
    # print((mask_array==0).sum())
    print((mask_array == 255).sum() / num_pixels)
    print(mask_image.size)
    if (mask_array == 255).sum() < num_pixels * detected_size or (mask_array == 255).sum() > num_pixels * 0.9:
        print('==========================')
        return False
    else:
        return True


def analy_image(path):
    image = Image.open(path)
    print("format : {}".format(image.format))
    print("indo : {}".format(image.info))
    print("mode : {}".format(image.mode))
    print("size : {}".format(image.size))
    print("num_channel : {}".format(len(image.split())))
    return image


def binary_img(img_path, target_path, threshold=127):
    img_arr = np.array(Image.open(img_path).convert("L"))
    img_arr[img_arr < threshold] = 0
    img_arr[img_arr >= threshold] = 255
    Image.fromarray(img_arr).save(target_path)


if __name__ == '__main__':
    path = "sample_data/Proc201506160021_1_1.png"
    raw_img = Image.open(path)
    raw_img = raw_img.convert("L")
    img = np.asarray(raw_img)

    height, width = img.shape
    res = hist_equal_one_chan2(img, height, width)
    res_img = Image.fromarray(res)
    # 使用标准的算法
    res1 = hist_equal(img)
    res_img1 = Image.fromarray(res1)

    fig = plt.figure("result checking")
    ax = fig.add_subplot(321)
    ax.set_title("raw img")
    ax.imshow(raw_img, cmap='gray')
    ax.axis("off")
    # ax.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))

    ax = fig.add_subplot(322)
    ax.set_title("raw hist")
    ax.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))

    ax = fig.add_subplot(323)
    ax.set_title("res img")
    ax.imshow(res_img, cmap='gray')
    ax.axis("off")

    ax = fig.add_subplot(324)
    ax.set_title("result hist")
    ax.hist(res.ravel(), bins=255, rwidth=0.8, range=(0, 255))

    ax = fig.add_subplot(325)
    ax.set_title("res1 img")
    ax.imshow(res_img1, cmap='gray')
    ax.axis("off")

    ax = fig.add_subplot(326)
    ax.set_title("result1 hist")
    ax.hist(res1.ravel(), bins=255, rwidth=0.8, range=(0, 255))

    plt.show()

    # 保存图像数据
    raw_img.save("sample_data/raw.jpg")
    res_img.save("sample_data/res_img.jpg")
    res_img1.save("sample_data/res_img1.jpg")
