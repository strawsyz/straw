import numpy as np
import PIL.Image as Image

te = np.load('result/00001724.npy')
print(len(te))

# te = np.array([16,2])
# te= np.full((16,2), -1)
# te[0] = [1,1]
# print(te)

# perm = np.random.permutation(1000000)
# print(perm[:10])
# image_path = 'test.jpg'
# image = np.array(Image.open(image_path))
# image.transpose(1,2,0)
#

# print(image.shape)
# height, width, _ = image.shape
#
# heat_map = np.zeros([16, height, width])
# print(len(heat_map))
#
# for i, point in enumerate(te):
#     heat_map[i,point[0],point[1]] = 1
# print(heat_map.sum())

from chainer.links import VGG16Layers
from PIL import Image

model = VGG16Layers()
img = Image.open("result/images/00000053.jpg")
feature = model.extract([img], layers=["pool2"])["pool2"]

np.set_printoptions(threshold=np.inf)
print(feature.shape)

# te = np.zeros([16, 10])
# te[5, 9] = 1
# temp_1 = te.argmax() // 10
# temp_2 = te.argmax() % 10
# print(temp_1)
# print(temp_2)
