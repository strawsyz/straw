import numpy as np

#  http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
x = [1, 2, 3, 3]
# 最后一个参数表示保存整数
np.savetxt('test.txt', x, '%i')

y = np.loadtxt('test.txt')
print(y)
