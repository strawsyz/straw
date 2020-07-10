import scipy

#  关于 scipy.io 模块
# http://docs.scipy.org/doc/scipy/reference/io.html
# 下面的函数貌似都不能使用了，还在寻找原因

# 使用 scipy.io 模块进行读取 Matlab 的 .mat 文件格式
data = scipy.io.loadmat('test.mat')

# 保存到mat文件
data = {}
x = 1
data['x'] = x
scipy.io.savemat('test.mat', data)

from scipy.misc import imsave

im = [1, 2, 3, 4]
# 将数组直接保存为图像文件
imsave('test.jpg', im)
# 将数组直接保存为图像文件
# 著名的 Lena 测试图像：
lena = scipy.misc.lena()
