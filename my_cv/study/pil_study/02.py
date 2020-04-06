from PIL import Image

pil_im = Image.open('test.jpg')

# 创建缩略图
# 在设定的矩形大小的范围内，缩小图片，不改变宽高比例
pil_im.thumbnail((128, 128))

pil_im.save('test_thumbnail.jpg')

