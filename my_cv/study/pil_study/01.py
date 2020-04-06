from PIL import Image

pil_im = Image.open('test.jpg')

# 转换为灰度图像
pil_im = pil_im.convert('L')

# 保存图片
pil_im.save('test_gray.jpg')

# 使用系统默认的图片查看程序打开图片
pil_im.show()
