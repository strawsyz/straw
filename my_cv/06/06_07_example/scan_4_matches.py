from os.path import join
from os import walk
import numpy as np
import cv2
from sys import argv

"""对所有扫描符进行单应性处理，找到可能与查询图像相匹配的图像
使用方法 python scan_4_matches.py fold_name
fold里面需要有test.jpg文件"""
# create an array of filenames
folder = argv[1]
# 读取查询的图像
query = cv2.imread(join(folder, "test.jpg"), cv2.IMREAD_GRAYSCALE)

# create files, images, descriptors globals
files = []
images = []
# 所有的描述符文件
descriptors = []
for (dirpath, dirnames, filenames) in walk(folder):
    files.extend(filenames)
    for f in files:
        if f.endswith("npy") and f != "test.npy":
            descriptors.append(f)
    print(descriptors)

# create the sift detector
sift = cv2.xfeatures2d.SIFT_create()
query_kp, query_ds = sift.detectAndCompute(query, None)

# 创建 FLANN 匹配器
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 最少匹配数
MIN_MATCH_COUNT = 10

potential_culprits = {}

print(">> 开始扫描描述符文件...")

for d in descriptors:
    print("--------- 分析 %s 是否匹配 ------------" % d)
    matches = flann.knnMatch(query_ds, np.load(join(folder, d)), k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        print("%s 是匹配的! (%d)" % (d, len(good)))
    else:
        print("%s 不匹配" % d)
    potential_culprits[d] = len(good)

# 找出匹配数最大的描述符文件文件
max_matches = None
potential_suspect = None
for culprit, matches in potential_culprits.items():
    if max_matches == None or matches > max_matches:
        max_matches = matches
        potential_suspect = culprit

print("相对匹配的描述符文件是 %s" % potential_suspect)
