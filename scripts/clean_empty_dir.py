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


def main(path):
    for file_name in os.listdir(path):
        tmp_path = os.path.join(path, file_name)
        if os.path.isdir(tmp_path):
            if is_empty_dir(tmp_path):
                # delete empty folder
                os.removedirs(tmp_path)
                # delete dir
                # shutil.rmtree(tmp_path)
                deleted_dir.append(tmp_path)
            else:
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
    """delete empty dir. sort folder where only has one folder"""
    parser = argparse.ArgumentParser(description='delete empty dir')
    parser.add_argument('-p', type=str, default=os.getcwd())
    args = parser.parse_args()
    deleted_dir = []
    move_dir = []
    path = args.p
    if not os.path.exists(path):
        print('dir is not exists')
    else:
        if os.path.isdir(path):
            main(path)
        else:
            print('please input dir_name')
