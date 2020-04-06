import cv2
import numpy as np

"""使用HOGDescriptor进行人的检测"""

def is_inside(o, i):
    """判断o矩形是否包含了i矩形"""
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def draw_person(img, person):
    # 将检测到的矩形绘制出来
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


img = cv2.imread("test01.png")
# 指定cv2.HOGDescriptor作为检测人的默认检测器
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# 检测图像，该函数会返回返回一个与矩形相关的数组，与人脸检测不同，不需要先将原始图像转为灰度格式
found, w = hog.detectMultiScale(img)

found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        # 如果某些矩形包含在其他矩形里面，说明检测出现错误，
        # 应该把里面的矩形丢弃
        if ri != qi and is_inside(r, q):
            break
        else:
            found_filtered.append(r)

for person in found_filtered:
    draw_person(img, person)

cv2.imshow("people detection", img)
cv2.waitKey()
cv2.destroyAllWindows()
