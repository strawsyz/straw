import argparse
import os
import shutil


# 将一个文件夹中的多个文件夹中的内容都提取出来，都拿到当前文件夹中

# 经本地测试基本没有问题了
def merge_dir(root_path, depth=1, remove_empty_dir=False):
    depth -= 1
    if depth > 0:
        for file_name in os.listdir(root_path):
            file_path = os.path.join(root_path, file_name)
            if os.path.isdir(file_path):
                merge_dir(file_path, depth, remove_empty_dir)
    for file_name in os.listdir(root_path):
        file_path = os.path.join(root_path, file_name)
        if os.path.isdir(file_path):
            for file_name_in_sub_dir in os.listdir(file_path):
                sub_file_path = os.path.join(file_path, file_name_in_sub_dir)
                # 如果有空文件夹。判断是否要删除文件夹
                if os.path.isdir(sub_file_path):
                    if remove_empty_dir and os.listdir(sub_file_path) == []:
                        shutil.rmtree(sub_file_path)
                        # 删除空文件夹，跳到下一个文件或文件夹
                        continue
                # 将当前的文件移动到上面的文件中去
                new_name = "-".join([file_name, file_name_in_sub_dir])
                new_path = os.path.join(root_path, new_name)
                while os.path.exists(new_path):
                    # 防止文件冲突
                    new_path = new_path + "#"
                if os.path.isdir(sub_file_path):
                    os.renames(sub_file_path, os.path.join(root_path, new_name))
                else:
                    # 如果使用renames会使得文件从只有一个文件的文件夹中拿出来的时候，自动删除已经变为空的文件夹
                    os.rename(sub_file_path, os.path.join(root_path, new_name))

            # 如果经过处理之后当前文件夹空了，判断是否要删除当前文件夹
            if remove_empty_dir and os.path.exists(file_path) and os.listdir(file_path) == []:
                shutil.rmtree(file_path)


if __name__ == '__main__':
    # 参数
    # 循环的深度，默认是1，如果文件夹下面还有文件或者文件夹留着，就不删除
    # 但是把文件下的文件名向上提取
    # 默认重命名文件名时添加原来的文件夹
    # 循环的文件夹名
    parser = argparse.ArgumentParser(description='put files in sub_dir into current dir')
    # 主路径
    parser.add_argument('-p', type=str, default=os.getcwd())
    # 是否扔掉空文件夹
    parser.add_argument('-r', type=bool, default=False)
    # 递归的深度，还有待测试
    parser.add_argument('-d', type=int, default=1)
    args = parser.parse_args()
    merge_dir(args.p, depth=args.d,
              remove_empty_dir=args.r)
