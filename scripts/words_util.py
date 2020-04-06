import chardet
import os


def convert_encoding(data, new_coding='UTF-8'):
    encoding = chardet.detect(data)['encoding']
    if new_coding.upper() != encoding.upper():
        print(data.decode(encoding, data))
        data = data.decode(encoding, data).encode(new_coding)
    print(encoding)
    return data


# print(strin.encode('gbk').decode("Shift-JIS"))

#  用于解决日文作为文件夹名导致的乱码问题，
#  针对文件夹使用，文件夹下的文件名必须都是有乱码问题的文件，不然可能会有问题
def jp_transform(path, encoding='Shift-JIS', is_recursion=False):
    filelist = os.listdir(path)  # 该文件夹下的所有文件
    count = 0

    for file in filelist:  # 遍历所有文件 包括文件夹
        try:
            old_dir = os.path.join(path, file)  # 原来文件夹的路径
            if is_recursion and os.path.isdir(old_dir):  # 如果不是文件夹，则跳过
                jp_transform(old_dir)
            # print(old_dir)
            # filename = os.path.splitext(file)[0]  #文件名
            # filetype = ".jpg"#os.path.splitext(file)[1]   文件扩展名
            temp = file.encode('gbk').decode(encoding)
            new_dir = os.path.join(path, temp)
            # Newdir = os.path.join(path,str(count)+filetype) #新的文件路径
            os.rename(old_dir, new_dir)  # 重命名
        except Exception as e:
            print(e.args)
            continue
        count += 1


import argparse

parser = argparse.ArgumentParser(description='change word to japanese')
parser.add_argument('-p', type=str, default=os.getcwd())
parser.add_argument('-r', type=bool, default=False)
parser.add_argument('-e', type=str, default="Shift-JIS")
args = parser.parse_args()
# python temp2.py -p dir_name -r=  或者   python temp2.py -p dir_name
# 不使用递归
# python temp2.py -p dir_name -r={？}  {？}填入空格外的任何东西
# 就使用递归
if __name__ == '__main__':
    path = args.p
    if not os.path.exists(path):
        print('dir is not exists')
    else:
        # 将文件夹的名字改过来
        dir_name = os.path.basename(path)
        temp = dir_name.encode('gbk').decode(args.e)
        new_dir = os.path.join(path, temp)
        # Newdir = os.path.join(path,str(count)+filetype) #新的文件路径
        # os.rename(path,new_dir) #重命名
        if os.path.isdir(path):
            jp_transform(path, encoding=args.e, is_recursion=args.r)
        else:
            print('please input dir_name')
