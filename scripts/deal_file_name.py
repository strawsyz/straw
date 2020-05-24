import os
import re

# 处理文件名中的日语字符

# replaces_cahrs = ['。', '・', '･', '♪', '〜',
#                   '◯', 'ビ', '○', '♥',
#                   '♡',  '〇', '♯', '†','²']
replace_chars = '。・･♪〜◯ビ○♥♡〇♯†²'
pattern = r'[{}]+'.format(replace_chars)
pattern = re.compile(pattern)
def removePunctuation(text):
    return pattern.sub('', text)

def main(path):
    files = os.listdir(path)
    for file_name in files:
        new_name = removePunctuation(file_name)
        if new_name != file_name:
            new_path = os.path.join(path, new_name)
            old_path = os.path.join(path, file_name)
            os.rename(old_path, new_path)

import argparse
parser = argparse.ArgumentParser(description='delete filename which canont display in MangaMeeya')
# 默认是当前文件夹
parser.add_argument('-p', type=str, default=os.getcwd())
args = parser.parse_args()

# 使用方法
# # python temp2.py {}
# {} 填入路径名

if __name__ == '__main__':
    path = args.p
    if not os.path.exists(path):
        print('dir is not exists')
    else:
        if os.path.isdir(path):
            main(path)
        else:
            print('please input dir_name')
