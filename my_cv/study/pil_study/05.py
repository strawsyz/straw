from PIL import Image

# 使用 seek 和 tell 方法在不同的帧之间移动
im = Image.open("test.gif")
im.seek(1)  # 跳转到第一个帧
try:
    while True:
        im.seek(im.tell() + 1)
        im.show()
except EOFError:
    pass

# 或者使用 for 循环
from PIL import ImageSequence

for frame in ImageSequence.Iterator(im):
    im.seek(frame)
    im.show()
