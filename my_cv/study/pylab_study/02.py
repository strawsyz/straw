from PIL import Image
from pylab import *

im = array(Image.open('../../matplotlib_study/test.jpg').convert('L'))

# 新建图像
figure()
# 不使用颜色信息
gray()
# 在原点左上角显示图像轮廓
contour(im, origin='image')
# 保持原图的长宽比
axis('equal')
# 不显示坐标轴
axis('off')
# 新建图像
figure()
# 绘制图像的直方图
# im.flatten()将图像转换为1维数组
hist(im.flatten(), 128)

show()
