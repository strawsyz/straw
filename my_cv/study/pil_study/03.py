from PIL import Image

pil_im = Image.open('test.jpg')

box = (100, 100, 400, 400)
# 复制图片的指定区域
region = pil_im.crop(box)

# 旋转图片
region = region.transpose(Image.ROTATE_90)
# 上下翻转
# out = pil_im.transpose(Image.FLIP_TOP_BOTTOM)
# 左右翻转
# out = pil_im.transpose(Image.FLIP_LEFT_RIGHT)

# 将图片粘贴到指定区域
pil_im.paste(region, (100, 100))

pil_im.save('test_edit.jpg')
