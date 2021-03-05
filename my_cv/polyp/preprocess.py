from utils_ import *

"""
预处理数据
"""

ROOT_PATH = "/home/straw/Downloads/dataset/polyp/"
# 保存用医生那里获得的元图像文件
RAW_DATA_PATH = "/home/straw/Downloads/dataset/polyp/raw/"
# 图像保存路径
IMAGE_PATH = os.path.join(ROOT_PATH, "data")
# mask保存路径
MASK_PATH = os.path.join(ROOT_PATH, "mask")


def check_files(path):
    file_names = set()
    # 先比较查看有那些图片
    for file_name in os.listdir(path):
        file_name, _ = os.path.splitext(file_name)
        file_names.add(file_name)
    return file_names


def check_image_mask(image_path, mask_path):
    """检测两个路径下的文件名是否一一对应"""
    image_names = check_files(image_path)
    counter_notexist = 0
    for image_name in image_names:
        tmp_path = os.path.join(mask_path, "{}M.bmp".format(image_name))
        if os.path.exists(tmp_path):
            print("{} is exist, {}".format(tmp_path, os.path.basename(tmp_path)))
        else:
            counter_notexist += 1
            print("{} is not exist, {}".format(tmp_path, os.path.basename(tmp_path)))
    print("{} files are not exist".format(counter_notexist))


# image_file_names = check_files(os.path.join(RAW_DATA_PATH, 'data'))
# print(image_file_names)
# print(len(image_file_names))

# 调查结果，源图像数据中有bmp和jpg两种数据类型，不存在相同文件名但是后缀名不相同的情况
# mask_file_names = check_files(os.path.join(RAW_DATA_PATH, 'mask'))
# print(mask_file_names)
# print(len(mask_file_names))

# check_image_mask(os.path.join(RAW_DATA_PATH, 'data'), os.path.join(RAW_DATA_PATH, 'mask'))
# 调查结果，每张图片都已一个对应的mask图像存在。mask图像的文件名是，源文件名+M.bmp


import file_util

# 将bmp和jpg类型的原图像转化为png图像保存到tmp文件夹中
TMP_PATH = "/home/straw/Downloads/dataset/polyp/TMP/01/data/"
# flag = True
flag = False
if flag:
    file_util.make_directory(TMP_PATH)
    RAW_IMAGE_PATH = os.path.join(RAW_DATA_PATH, "data")
    for image_name in os.listdir(RAW_IMAGE_PATH):
        source_path = os.path.join(RAW_IMAGE_PATH, image_name)
        convert2PNG(source_path, target_path=TMP_PATH)

# 将原来的mask图像（bmp类型）转化为png类型
# flag = True
flag = False
if flag:
    TMP_PATH = "/home/straw/Downloads/dataset/polyp/TMP/01/mask/"
    file_util.make_directory(TMP_PATH)
    RAW_MASK_PATH = os.path.join(RAW_DATA_PATH, "mask")
    for mask_image_name in os.listdir(RAW_MASK_PATH):
        source_path = os.path.join(RAW_MASK_PATH, mask_image_name)
        convert2PNG(source_path, target_path=TMP_PATH, mask=True)

# 将图像需要的部分截取下来保存到数据的目录中
# 主要是去掉左右两边的图像参数信息等
# flag = True
flag = False

if flag:
    TMP_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/01/data/"
    TMP_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/01/mask/"
    TARGET_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/02/data/"
    TARGET_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/02/mask/"
    file_util.make_directory(TARGET_IMAGE_PATH, needed_part=(294, 35, 1454, 1040))
    file_util.make_directory(TARGET_MASK_PATH, needed_part=(294, 35, 1454, 1040))
    get_needed_data(TMP_IMAGE_PATH, TARGET_IMAGE_PATH)
    get_needed_data(TMP_MASK_PATH, TARGET_MASK_PATH)

# 去掉图像四个角落上的黑色部分
# flag = True
flag = False
if flag:
    SOURCE_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/02/data/"
    SOURCE_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/02/mask/"
    TARGET_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/03/data/"
    TARGET_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/03/mask/"
    file_util.make_directory(TARGET_IMAGE_PATH)
    file_util.make_directory(TARGET_MASK_PATH)
    get_needed_data(SOURCE_IMAGE_PATH, TARGET_IMAGE_PATH, needed_part=(100, 100, 1100, 900))
    get_needed_data(SOURCE_MASK_PATH, TARGET_MASK_PATH, needed_part=(100, 100, 1100, 900))

# flag = True
flag = False
# 将图像切成平均的切成4份
if flag:
    SOURCE_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/03/data/"
    SOURCE_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/03/mask/"
    TARGET_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/04/data/"
    TARGET_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/04/mask/"
    file_util.make_directory(TARGET_IMAGE_PATH)
    file_util.make_directory(TARGET_MASK_PATH)
    width = 1000
    height = 800
    create_patch(SOURCE_IMAGE_PATH, TARGET_IMAGE_PATH, width, height, 2, 2)
    create_patch(SOURCE_MASK_PATH, TARGET_MASK_PATH, width, height, 2, 2)

# 由于mask图像的通道数有的是3通道，有的是1通道
# 所以将mask图像全部都统一为1通道
# flag = True
flag = False
if flag:
    SOURCE_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/04/data/"
    SOURCE_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/04/mask/"
    TARGET_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/05/data/"
    TARGET_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/05/mask/"
    file_util.make_directory(TARGET_IMAGE_PATH)
    file_util.make_directory(TARGET_MASK_PATH)
    copy_images(SOURCE_IMAGE_PATH, TARGET_IMAGE_PATH)
    convert_imgs(SOURCE_MASK_PATH, TARGET_MASK_PATH)

# flag = True
flag = False
# 将原来的图像patch化
if flag:
    SOURCE_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/03/data/"
    SOURCE_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/03/mask/"
    TARGET_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/06/data/"
    TARGET_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/06/mask/"
    import shutil

    shutil.rmtree(TARGET_IMAGE_PATH)
    shutil.rmtree(TARGET_MASK_PATH)

    file_util.make_directory(TARGET_IMAGE_PATH)
    file_util.make_directory(TARGET_MASK_PATH)

    detected_size = 0.3
    patch_width = 224
    patch_height = 224
    create_patch_by_absolute_size(SOURCE_IMAGE_PATH, SOURCE_MASK_PATH, TARGET_IMAGE_PATH, TARGET_MASK_PATH,
                                  detected_size,
                                  patch_width, patch_height)

# 使用canny生成轮廓图
from my_data_aug import edge_detector

# flag = True
flag = False
if flag:
    SOURCE_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/06/data/"
    # SOURCE_IMAGE_PATH = "D:\Download\datasets\polyp\\06\data"
    TARGET_EDGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/06/edge/"
    file_util.make_directory(TARGET_EDGE_PATH)

    for filename in os.listdir(SOURCE_IMAGE_PATH):
        source_path = os.path.join(SOURCE_IMAGE_PATH, filename)
        target_path = os.path.join(TARGET_EDGE_PATH, filename)
        print(target_path)
        edge_detector(source_path, target_path)

# 重新处理mask图像，将mask图像，变成只有黑色和白色的图像
# flag = True
flag = False
if flag:
    SOURCE_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/06/mask/"
    # SOURCE_MASK_PATH = "D:\Download\datasets\polyp\\06\mask"
    TARGET_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/06/mask/"
    # TARGET_MASK_PATH = "D:\Download\datasets\polyp\\06\mask"
    file_util.make_directory(TARGET_MASK_PATH)

    for filename in os.listdir(SOURCE_MASK_PATH):
        source_path = os.path.join(SOURCE_MASK_PATH, filename)
        target_path = os.path.join(TARGET_MASK_PATH, filename)
        binary_img(source_path, target_path)

# increase number of images
# flag = True
flag = False
if flag:
    SOURCE_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/03/data/"
    SOURCE_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/03/mask/"
    TARGET_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/07/data/"
    TARGET_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/07/mask/"
    file_util.make_directory(TARGET_IMAGE_PATH)
    file_util.make_directory(TARGET_MASK_PATH)

    detected_size = 0.3
    patch_width = 224
    patch_height = 224
    create_patch_by_absolute_size(SOURCE_IMAGE_PATH, SOURCE_MASK_PATH, TARGET_IMAGE_PATH, TARGET_MASK_PATH,
                                  detected_size,
                                  patch_width, patch_height, max_num=200)

# flag = True
flag = False
# binarization
if flag:
    SOURCE_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/07/mask/"
    TARGET_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/07/mask/"
    file_util.make_directory(TARGET_MASK_PATH)

    for filename in os.listdir(SOURCE_MASK_PATH):
        source_path = os.path.join(SOURCE_MASK_PATH, filename)
        target_path = os.path.join(TARGET_MASK_PATH, filename)
        binary_img(source_path, target_path)
    # print(len(os.listdir(SOURCE_MASK_PATH)))

# 使用canny生成轮廓图
from my_data_aug import edge_detector

# flag = True
flag = False
if flag:
    SOURCE_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/07/data/"
    TARGET_EDGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/07/edge/"
    file_util.make_directory(TARGET_EDGE_PATH)

    for filename in os.listdir(SOURCE_IMAGE_PATH):
        source_path = os.path.join(SOURCE_IMAGE_PATH, filename)
        target_path = os.path.join(TARGET_EDGE_PATH, filename)
        print(target_path)
        edge_detector(source_path, target_path)

# step 2 use predict result from step one
# copy predict result in step one to dataset fold
# flag = True
flag = False
if flag:
    SOURCE_IMAGE_PATH = "/home/straw/Download\models\polyp\\result/2020-08-02"
    TARGET_PREDICT_PATH = "/home/straw/Downloads/dataset/polyp/TMP/07/predict_step_one/"
    file_util.make_directory(TARGET_PREDICT_PATH)
    copy_images(SOURCE_IMAGE_PATH, TARGET_PREDICT_PATH)

# flag = True
flag = False
if flag:
    EDGE_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/07/edge/"
    EDGE_TARGET_PATH = "/home/straw/Downloads/dataset/polyp/TMP/07/edge_bi/"
    file_util.make_directory(EDGE_TARGET_PATH)
    for file_name in os.listdir(EDGE_IMAGE_PATH):
        source_path = os.path.join(EDGE_IMAGE_PATH, file_name)
        target_path = os.path.join(EDGE_TARGET_PATH, file_name)
        binary_img(source_path, target_path)

# 将没有切成patch的图像分为测试图像和训练图像
flag = False
test_rate = 0.2
if flag:
    SOURCE_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/03/data/"
    SOURCE_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/03/mask/"

    TARGET_TRAIN_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/train/data"
    TARGET_TEST_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/test/data"
    TARGET_TRAIN_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/train/mask"
    TARGET_TEST_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/test/mask"
    file_util.make_directory(TARGET_TRAIN_IMAGE_PATH)
    file_util.make_directory(TARGET_TEST_MASK_PATH)
    file_util.make_directory(TARGET_TRAIN_MASK_PATH)
    file_util.make_directory(TARGET_TEST_IMAGE_PATH)

    filenames = os.listdir(SOURCE_IMAGE_PATH)
    filenames.sort()
    import random
    import shutil

    random.seed(0)
    random.shuffle(filenames)
    num_test = int(len(filenames) * test_rate)  # 37
    num_train = len(filenames) - num_test  # 148
    # copy train image files
    for filename in filenames[:num_train]:
        source_image_path = os.path.join(SOURCE_IMAGE_PATH, filename)
        source_mask_path = os.path.join(SOURCE_MASK_PATH, filename)
        target_image_path = os.path.join(TARGET_TRAIN_IMAGE_PATH, filename)
        target_mask_path = os.path.join(TARGET_TRAIN_MASK_PATH, filename)
        shutil.copyfile(source_image_path, target_image_path)
        binary_img(source_mask_path, target_mask_path)
        print("train: {}".format(filename))
    # copy test image files
    for filename in filenames[num_train:]:
        source_image_path = os.path.join(SOURCE_IMAGE_PATH, filename)
        source_mask_path = os.path.join(SOURCE_MASK_PATH, filename)
        target_image_path = os.path.join(TARGET_TEST_IMAGE_PATH, filename)
        target_mask_path = os.path.join(TARGET_TEST_MASK_PATH, filename)
        shutil.copyfile(source_image_path, target_image_path)
        binary_img(source_mask_path, target_mask_path)
        print("test: {}".format(filename))

# 将图像分别切成小的patch图像
flag = False
if flag:
    SOURCE_TRAIN_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/train/data"
    SOURCE_TEST_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/test/data"
    SOURCE_TRAIN_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/train/mask"
    SOURCE_TEST_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/test/mask"

    TARGET_TRAIN_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/09/train/data"
    TARGET_TEST_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/09/test/data"
    TARGET_TRAIN_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/09/train/mask"
    TARGET_TEST_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/09/test/mask"
    file_util.make_directory(TARGET_TRAIN_IMAGE_PATH)
    file_util.make_directory(TARGET_TEST_MASK_PATH)
    file_util.make_directory(TARGET_TRAIN_MASK_PATH)
    file_util.make_directory(TARGET_TEST_IMAGE_PATH)

    detected_size = 0.3
    patch_width = 224
    patch_height = 224
    create_patch_by_absolute_size(SOURCE_TRAIN_IMAGE_PATH, SOURCE_TRAIN_MASK_PATH, TARGET_TRAIN_IMAGE_PATH,
                                  TARGET_TRAIN_MASK_PATH,
                                  detected_size,
                                  patch_width, patch_height, max_num=200)
    create_patch_by_absolute_size(SOURCE_TEST_IMAGE_PATH, SOURCE_TEST_MASK_PATH, TARGET_TEST_IMAGE_PATH,
                                  TARGET_TEST_MASK_PATH,
                                  detected_size,
                                  patch_width, patch_height, max_num=200)
    # test batches:2103   train batches:8393

# create edge images
flag = False
if flag:
    SOURCE_TRAIN_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/09/train/data"
    SOURCE_TEST_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/09/test/data"
    TARGET_TRAIN_EDGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/09/train/edge"
    TARGET_TEST_EDGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/09/test/edge"
    file_util.make_directory(TARGET_TRAIN_EDGE_PATH)
    file_util.make_directory(TARGET_TEST_EDGE_PATH)
    for filename in os.listdir(SOURCE_TRAIN_IMAGE_PATH):
        source_path = os.path.join(SOURCE_TRAIN_IMAGE_PATH, filename)
        target_path = os.path.join(TARGET_TRAIN_EDGE_PATH, filename)
        edge_detector(source_path, target_path)
        binary_img(target_path, target_path)
    for filename in os.listdir(SOURCE_TEST_IMAGE_PATH):
        source_path = os.path.join(SOURCE_TEST_IMAGE_PATH, filename)
        target_path = os.path.join(TARGET_TEST_EDGE_PATH, filename)
        edge_detector(source_path, target_path)
        binary_img(target_path, target_path)

# 将图像分别切成小的patch图像
flag = True
if flag:
    SOURCE_TRAIN_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/train/data"
    SOURCE_TEST_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/test/data"
    SOURCE_TRAIN_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/train/mask"
    SOURCE_TEST_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/08/test/mask"

    TARGET_TRAIN_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/10/train/data"
    TARGET_TEST_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/10/test/data"
    TARGET_TRAIN_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/10/train/mask"
    TARGET_TEST_MASK_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/10/test/mask"
    file_util.make_directory(TARGET_TRAIN_IMAGE_PATH)
    file_util.make_directory(TARGET_TEST_MASK_PATH)
    file_util.make_directory(TARGET_TRAIN_MASK_PATH)
    file_util.make_directory(TARGET_TEST_IMAGE_PATH)
    # create patches
    detected_size = 0.2
    patch_width = 224
    patch_height = 224
    max_num = 100
    create_patch_by_absolute_size(SOURCE_TRAIN_IMAGE_PATH, SOURCE_TRAIN_MASK_PATH, TARGET_TRAIN_IMAGE_PATH,
                                  TARGET_TRAIN_MASK_PATH,
                                  detected_size,
                                  patch_width, patch_height, max_num=max_num)
    create_patch_by_absolute_size(SOURCE_TEST_IMAGE_PATH, SOURCE_TEST_MASK_PATH, TARGET_TEST_IMAGE_PATH,
                                  TARGET_TEST_MASK_PATH,
                                  detected_size,
                                  patch_width, patch_height, max_num=max_num)

    # create edge images
    SOURCE_TRAIN_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/10/train/data"
    SOURCE_TEST_IMAGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/10/test/data"
    TARGET_TRAIN_EDGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/10/train/edge"
    TARGET_TEST_EDGE_PATH = r"/home/shi/Downloads/dataset/polyp/TMP/10/test/edge"
    file_util.make_directory(TARGET_TRAIN_EDGE_PATH)
    file_util.make_directory(TARGET_TEST_EDGE_PATH)
    for filename in os.listdir(SOURCE_TRAIN_IMAGE_PATH):
        source_path = os.path.join(SOURCE_TRAIN_IMAGE_PATH, filename)
        target_path = os.path.join(TARGET_TRAIN_EDGE_PATH, filename)
        edge_detector(source_path, target_path)
        binary_img(target_path, target_path)
    for filename in os.listdir(SOURCE_TEST_IMAGE_PATH):
        source_path = os.path.join(SOURCE_TEST_IMAGE_PATH, filename)
        target_path = os.path.join(TARGET_TEST_EDGE_PATH, filename)
        edge_detector(source_path, target_path)
        binary_img(target_path, target_path)

# 轮廓图和mask图像重叠的部分作为正确的图
# 输入原图像，从图像检测出轮廓，获得轮廓图
# 输出是一个轮廓图。真值是轮廓图和mask图像重叠的部分
# 简介：检测polyp的轮廓的网络
# 需要的数据：轮廓图。轮廓和mask图像重叠的部分
