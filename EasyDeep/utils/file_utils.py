import os

from PIL import Image


def make_directory(dir_path, mode=0o777):
    """create a directory if not exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, mode=mode)


def is_valid_jpg(jpg_file):
    try:
        with open(jpg_file, 'rb') as f:
            f.seek(-2, 2)
            buf = f.read()
            f.close()
            return buf == b'\xff\xd9'
    except Exception as e:
        raise e


def is_valid_png(png_file):
    try:
        with open(png_file, 'rb') as f:
            f.seek(-8, 2)
            buf = f.read()
            f.close()
            return buf == b'IEND\xaeB`\x82'
    except Exception as e:
        raise e


def is_valid_zip(zip_file):
    import zipfile36 as zipfile
    try:
        # test if file is complete
        zipfile.ZipFile(zip_file)
        # test unzip file
        # return z_file.testzip()  # 测试能否解压zip文件
        return True
    except zipfile.BadZipFile as e:
        raise e
    except Exception as e:
        # from logger import Log
        # logger = Log(__name__).get_log()
        # logger.error('trouble in zip_file ', exc_info=True)
        print('trouble in zip_file {}'.format(zip_file))
        raise e


def valid_file(file_path, file_type: str = None):
    file_type = get_extension(file_path) if file_type is None else file_type
    if file_type == '.jpg':
        return is_valid_jpg(file_path)
    elif file_type == '.png':
        return is_valid_png(file_path)
    elif file_type == '.zip':
        return is_valid_zip(file_path)
    else:
        raise NotImplementedError("还不支持{}格式,文件在{}".format((file_type, file_path)))


def valid_file_batch(root_path, recursive=False):
    """check if files are complete"""
    for file in os.listdir(root_path):
        path = os.path.join(root_path, file)
        if os.path.isdir(path) and recursive:
            valid_file_batch(path)
        else:
            extension = get_extension(file)
            res = valid_file(path, extension)
            if not res:
                print('this file is incomplete {}'.format(path))


def batch_img2thumbnail(file_dir, size, dest_dir=None, ignore=[]):
    if dest_dir is None:
        dest_dir = os.path.join(file_dir, 'thumbnail')
    for file in os.listdir(file_dir):
        if file in ignore:
            continue
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


def img2thumbnail(file_path, dest_path, size, is_overwrite=False):
    """
    create thumbnail images
    Args:
        file_path:
        dest_path:
        size:
        is_overwrite: if overwrite file

    Returns:

    """
    if os.path.exists(dest_path) and not is_overwrite:
        return
    # 文件后缀名检查
    ext = get_extension(file_path)
    if ext not in [".jpg", ".png"]:
        return
    if not os.path.exists(file_path):
        print("Can't find file: {}".format(file_path))
        return
    if not valid_file(file_path, ext):
        return
    make_directory(os.path.dirname(dest_path))
    try:
        im = Image.open(file_path)
        im.thumbnail(size)
        im.save(dest_path)
        print(file_path)
    except OSError:
        print('{} has some problems'.format(file_path))


def get_cur_dir(py_file):
    return os.path.dirname(os.path.abspath(py_file))


def get_extension(path):
    """get file's extension"""
    return os.path.splitext(path)[1]


def get_filename_ext(path):
    filename, ext = os.path.splitext(os.path.basename(path))
    return filename, ext


def get_filename(path):
    return get_filename_ext(path)[0]


def split_path(path):
    dir_path = os.path.dirname(path)
    full_filename = os.path.basename(path)
    filename, extension = os.path.splitext(full_filename)
    extension = extension[1:]
    return (dir_path, full_filename, filename, extension)


def create_unique_name(save_path):
    if not os.path.exists(save_path):
        return save_path
    else:
        filename, ext = get_filename_ext(save_path)
        for i in range(10 ** 7):
            new_filename = filename + str(i)
            new_save_path = new_filename + ext
            if not os.path.exists(new_save_path):
                return new_save_path
