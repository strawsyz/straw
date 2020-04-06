import cv2

img = cv2.imread('test.png')
# 显示图片
cv2.imshow('test', img)
# 按任意键，图片消失
cv2.waitKey()
# 释放所有窗口
cv2.destroyAllWindows()