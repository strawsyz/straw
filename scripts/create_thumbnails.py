from PIL import Image
import os

"""处理图片，生成缩略图"""


def batch_img2thumbnail(file_dir, size, dest_dir=None):
    if dest_dir is None:
        dest_dir = os.path.join(file_dir, 'thumbnail')
    for file in os.listdir(file_dir):
        path = os.path.join(file_dir, file)
        if os.path.isdir(path):
            if path != dest_dir:
                batch_img2thumbnail(path, size, os.path.join(dest_dir, file))
            else:
                continue
        else:
            if file == 'Thumbs.db':
                continue
            img2thumbnail(path, os.path.join(dest_dir, file), size)
            # extension = file_extension(file)
            # res = valid_file(path, extension)
            # if not res:
            #     print('this file is incomplete {}'.format(path))


import file_util


def img2thumbnail(file_path, dest_path, size, is_cover=False):
    if os.path.exists(dest_path) and not is_cover:
        return
    if not os.path.exists(file_path):
        print('找不到图片，请重新检查')
        return
    file_util.make_directory(os.path.dirname(dest_path))
    # for infile in glob.glob(file_path):
    # ext = os.path.splitext(os.path.split(infile)[1])[0]
    # fPath = os.path.join(self._thumbPath, ext)
    # print(file)
    print(file_path)
    try:
        im = Image.open(file_path)
        # im.show()
        im.thumbnail(size)
        im.save(dest_path)
    except OSError:
        print('{} 文件有问题'.format(file_path))


if __name__ == '__main__':
    images_path = ""
    batch_img2thumbnail(images_path, (300, 400))
