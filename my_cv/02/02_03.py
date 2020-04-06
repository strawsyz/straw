import cv2

img = cv2.imread('test.png')
print(img.item(150, 120, 0))
img.itemset((150, 120, 0), 0)
print(img.item(150, 120, 0))

print('设图片的绿色通道上的值为0')
img = cv2.imread('test.png')
img[:, :, 1] = 0
cv2.imwrite('test_no_green.png', img)

img = cv2.imread('test.png')
part = img[0:200, 0:100]
img[200:400, 300:400] = part
cv2.imwrite('test_part_move.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
cv2.imshow('win', img)
cv2.waitKey(0)
