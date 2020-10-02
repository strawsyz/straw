import argparse
import os
import shutil


def is_empty_dir(path):
    if len(os.listdir(path)) == 0:
        return True
    else:
        flag = True
        for file_name in os.listdir(path):
            tmp_path = os.path.join(path, file_name)
            if os.path.isdir(tmp_path):
                if is_empty_dir(tmp_path):
                    deleted_dir.append(tmp_path)
                    os.rmdir(tmp_path)
                else:
                    flag = False
            else:
                flag = False
        return flag


def is_only_one_dir(path):
    if not os.path.exists(path):
        print("{} is not exist.".format(path))
        return False
    if len(os.listdir(path)) == 1:
        filename = os.listdir(path)[0]
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            return True, filepath
    return False, None


# def move_dir(source_path,target_path):


def main(path):
    for file_name in os.listdir(path):
        tmp_path = os.path.join(path, file_name)
        if os.path.isdir(tmp_path):
            if is_empty_dir(tmp_path):
                # 删除空文件夹，如果空文件下面也有空文件夹的话，就无法删除
                os.removedirs(tmp_path)
                # 为了减少依赖，所以不用shutil
                # import shutil
                # shutil.rmtree(tmp_path)  # 递归删除文件夹
                deleted_dir.append(tmp_path)
            else:
                # 并没有递归的功能
                result, moved_dir = is_only_one_dir(tmp_path)
                if result:
                    dirname1 = os.path.basename(tmp_path)
                    dirname2 = os.path.basename(moved_dir)
                    new_path = os.path.join(path, "{}--{}".format(dirname1, dirname2))
                    move_dir.append(moved_dir)
                    shutil.move(moved_dir, new_path)
    if len(deleted_dir) > 0:
        print("some empty dir is deleted")
        for i in deleted_dir:
            print(i)
    if len(move_dir) > 0:
        print("some dir is moved")
        for i in move_dir:
            print(i)


if __name__ == '__main__':
    """自动删除空文件夹"""
    parser = argparse.ArgumentParser(description='delete empty dir')
    # 默认是当前文件夹
    parser.add_argument('-p', type=str, default=os.getcwd())
    args = parser.parse_args()
    deleted_dir = []
    move_dir = []
    path = args.p
    path = r"G:\ZKR鉴赏\tmp"
    if not os.path.exists(path):
        print('dir is not exists')
    else:
        if os.path.isdir(path):
            main(path)
        else:
            print('please input dir_name')
