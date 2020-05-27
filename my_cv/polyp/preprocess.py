import os

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


image_file_names = check_files(os.path.join(RAW_DATA_PATH, 'data'))
print(image_file_names)
print(len(image_file_names))

# 调查结果，源图像数据中有bmp和jpg两种数据类型，不存在相同文件名但是后缀名不相同的情况
mask_file_names = check_files(os.path.join(RAW_DATA_PATH, 'mask'))
print(mask_file_names)
print(len(mask_file_names))

check_image_mask(os.path.join(RAW_DATA_PATH, 'data'), os.path.join(RAW_DATA_PATH, 'mask'))
# 调查结果，每张图片都已一个对应的mask图像存在。mask图像的文件名是，源文件名+M.bmp


import file_util
from PIL import Image


def convert2PNG(source_path, target_path, mask=False):
    img = Image.open(source_path)
    target_path = os.path.join(target_path, "{}.png".format(file_util.get_filename(source_path)))
    if mask:
        # 如果是处理mask图像,把文件名中M去掉
        target_path = target_path.replace("M.png", ".png")
    img.save(target_path)
    print(target_path)


def get_needed_data(source_path, target_path, needed_part=(294, 35, 1454, 1040)):
    """
    将需要的部分图像从原图像切下来，然后保存到目标目录中
    :param source_path:
    :param target_path:
    :param needed_part: 下标是从0开始
    :return:
    """
    for file_name in os.listdir(source_path):
        file_path = os.path.join(source_path, file_name)
        img = Image.open(file_path)
        cropped = img.crop(needed_part)
        # 理论上来讲经过之前的处理之后，所有的文件都被保存为png格式了。
        # 但为了保险，还是重新处理一遍后缀名
        # target_path = os.path.join(target_path, "{}.png".format(file_util.get_filename(file_path)))
        cropped.save(os.path.join(target_path, "{}.png".format(file_util.get_filename(file_path))))


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
# flag = True
flag = False

if flag:
    TMP_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/01/data/"
    TMP_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP//01/mask/"
    TARGET_IMAGE_PATH = "/home/straw/Downloads/dataset/polyp/TMP/02/data/"
    TARGET_MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/02/mask/"
    file_util.make_directory(TARGET_IMAGE_PATH)
    file_util.make_directory(TARGET_MASK_PATH)
    get_needed_data(TMP_IMAGE_PATH, TARGET_IMAGE_PATH)
    get_needed_data(TMP_MASK_PATH, TARGET_MASK_PATH)