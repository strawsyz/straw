from PIL import Image

pil_im = Image.open('test.jpg')
# 调整尺寸，不同于缩略图，不会保持原图片的长宽比
pil_im_resize = pil_im.resize((128, 128))
pil_im_resize.save('test_resize.jpg')
# 旋转图片
pil_im_rotate = pil_im.rotate(45)
pil_im_rotate.save('test_rotate.jpg')

