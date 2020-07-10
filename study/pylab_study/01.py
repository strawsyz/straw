# from pylab import *
import matplotlib.pyplot as plt
from PIL import Image

im = array(Image.open('../../matplotlib_study/test.jpg'))

imshow(im)

x = [100, 100, 400, 400]
y = [200, 600, 200, 500]

# 用红色星状标记绘制点
# plt.plot(x, y, 'r*')
# # 绘制前两个点的连线
# plt.plot(x[:2], y[:2])

# 其他绘图示例
# b 蓝色 g 绿色 r红色 c青色 m品红 y黄色 k黑色 w白色
# - 实线 -- 虚线 :点线
# .点 o圆圈 s正方形 *星号 +加号 x叉号

plt.plot(x, y)  # 默认为蓝色实线
plt.plot(x, y, 'go-')  # 带有圆圈标记的绿线
plt.plot(x, y, 'ks:')  # 带有正方形标记的黑色虚线

# 添加标题
plt.title('Plotting:"test.jpg"')
# 使坐标轴不显示
plt.axis('off')
plt.show()
