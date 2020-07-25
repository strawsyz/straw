import os
import numpy as np
import shutil

# te = np.load('result/00001724.npy')
# print(len(te))
path = 'result/'
src_image_path = '/home/straw/下载/dataset/FreiHAND_pub_v2/evaluation/rgb'
dst_image_path = 'result/images/'
for file in os.listdir(path):
    # check ig npy file is valid
    if os.path.isdir(os.path.join(path, file)):
        continue
    labels = np.load(os.path.join(path, file))
    if labels[15, 1] == -1:
        print(file)
        # pass
    else:
        file_name = ".".join(file.split('.')[:-1])
        image_name = file_name + '.jpg'
        shutil.copy(os.path.join(src_image_path, image_name),
                    os.path.join(dst_image_path, image_name))
