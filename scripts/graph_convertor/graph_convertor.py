import os
from PIL import Image
import zipfile
from logger import Log

"""对彩图进行图像模式转换然后按文件夹打包成zip文件
传入的路径应该是一个文件夹，文件夹下有一个或多个子文件夹
子文件夹中存放图片
对所有的彩图进行图像模式转换处理
最后按子文件夹打包到 zip文件夹中

仍然需要修改，
虽然做了转换的标记，实际上并有没转换
结果上暂时可以使用依然需要测试
"""


def image_convert(img_path, save_path):
    """
    将mode为RGB的图像转化为P格式
    使得MangaMeeya能够打开该彩图
    :param img_path: 图像路径
    :param save_path: 转换后图像的保存路径
    :return: None
    """
    try:
        empire = Image.open(img_path)
        print(empire.mode)
        if empire.mode == 'RGB':
            empire.convert('P')
            empire.save(save_path)
    except Exception as e:
        log_handle = Log(__name__).get_log()
        log_handle.warning(e, exc_info=1)

def compress_dir_2_zip(dir_path, save_path, mode=zipfile.ZIP_STORED):
    """
    压缩指定文件夹
    :param dir_path: 目标文件夹路径
    :param save_path: 压缩文件保存路径
    :return: None
    """
    # zipfile.ZIP_DEFLATED是压缩，能稍微减少一点空间
    zip = zipfile.ZipFile(save_path, "w", mode)
    for file_name in os.listdir(dir_path):
        if file_name != '.thumb' and file_name != '.ehviewer':
            zip.write(os.path.join(dir_path, file_name), file_name)
    zip.close()


def batch_image_convert(dir_path):
    """
    对彩图进行图像模式转换然后按文件夹打包成zip文件
    :param dir_path: 传入的路径应该是一个文件夹，
    文件夹下有一个或多个子文件夹，子文件夹中存放图片
    :return: None
    """
    # zip_dir = os.path.join(dir_path, 'zip')
    # os.makedirs(zip_dir, mode=0o777)
    for file_name in os.listdir(dir_path):
        img_dir = os.path.join(dir_path, file_name)
        if os.path.isdir(img_dir):
            img_dir = os.path.join(dir_path, file_name)
            is_need_compress = False
            for image_name in os.listdir(img_dir):
                if image_name.endswith('.jpg') or image_name.endswith('.png') or image_name.endswith('.jpeg'):
                    image_path = os.path.join(img_dir, image_name)
                    image_convert(image_path, image_path)
                    print(image_name + " is done")
                    is_need_compress = True
            # 压缩文件
            if is_need_compress:
                print('compressing == ', img_dir)
                compress_dir_2_zip(img_dir, os.path.join(dir_path, file_name + '.zip'))

def easy_get_path():
    import argparse
    parser = argparse.ArgumentParser()
    # 默认是当前文件夹
    parser.add_argument('-p', type=str, default=os.getcwd())
    args = parser.parse_args()
    return args.p



if __name__ == '__main__':
    # import time
    # start_time = time.time()
    path = easy_get_path()
    batch_image_convert(path)
    # end_time = time.time()
    # print('use time =====' + str(end_time - start_time))
