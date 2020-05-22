import argparse
import os

"""自动删除空文件夹"""
parser = argparse.ArgumentParser(description='delete empty dir')
# 默认是当前文件夹
parser.add_argument('-p', type=str, default=os.getcwd())
args = parser.parse_args()

deleted_dir = []


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


def main(path):
    for file_name in os.listdir(path):
        tmp_path = os.path.join(path, file_name)
        if os.path.isdir(tmp_path) and is_empty_dir(tmp_path):
            # 删除空文件夹，如果空文件下面也有空文件夹的话，就无法删除
            os.removedirs(tmp_path)
            # 为了减少依赖，所以不用shutil
            # import shutil
            # shutil.rmtree(tmp_path)  # 递归删除文件夹
            deleted_dir.append(tmp_path)
    for i in deleted_dir:
        print(i)


if __name__ == '__main__':
    path = args.p
    if not os.path.exists(path):
        print('dir is not exists')
    else:
        if os.path.isdir(path):
            main(path)
        else:
            print('please input dir_name')
