import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 创建一个Axes3d对象
fig = plt.figure()
ax = Axes3D(fig)

# x，y的取值返回是从-5到+5，每隔0.25取一个点
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)

# [X,Y] = meshgrid(x,y) 将向量x和y定义的区域转换成矩阵X和Y，
# 这两个矩阵可以用来表示mesh和surf的三维空间点以及两个变量的赋值。
# 其中矩阵X的行向量是向量x的简单复制，而矩阵Y的列向量是向量y的简单复制。
x, y = np.meshgrid(x, y)
print(x)
# r=np.sqrt(x**2+y**2)
# z=np.sin(r)
z = x ** 2 + y ** 2

# plot_surface 是绘制一个平面 ax.scatter 是绘制点
surf = ax.plot_surface(x, y, z)

plt.show()
