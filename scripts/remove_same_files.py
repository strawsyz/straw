import os

"""
比较多个文件夹下的是否有名字相同的文件，
有的话就删除前面的文件
remove files which have same filename from different directory
"""


def remove_same_file_by_filename(dir_list):
    """
    删除各个文件夹下的相同文件
    :param dir_list:文件夹列表
    :return:
    """
    """需要确保全都是文件，没有文件夹"""
    id_list = []
    for dir_path in dir_list:
        file_list = os.listdir(dir_path)
        for file in file_list:
            id_ = file.split("-")[0]
            if id_ in id_list:
                print(os.path.join(dir_path, file))
            # todo 比较然后删除文件
            else:
                id_list.append(id_)
    print(len(id_list))


if __name__ == '__main__':
    path1 = ""
    path2 = ""
    path3 = ""

    remove_same_file_by_filename([path1, path2, path3])
