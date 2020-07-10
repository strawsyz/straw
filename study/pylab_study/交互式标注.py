from PIL import Image
from pylab import *

# 显示图片
# 点击图片任意位置
# 在终端打印出点击的坐标
im = array(Image.open('test.jpg'))

imshow(im)

print('Please click 3 points')
for i in range(3):
    # ginput() 函数就可以实现交互式标注
    x = ginput()
    print('you clicked:', x)
show()
