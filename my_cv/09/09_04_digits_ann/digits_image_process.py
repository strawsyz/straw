import cv2
import numpy as np
import digits_ann as ANN

"""效果不怎么样，有待改进"""


def inside(r1, r2):
    """判断r1矩形是否在r2中"""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if (x1 > x2) and (y1 > y2) and (x1 + w1 < x2 + w2) and (y1 + h1 < y2 + h2):
        return True
    else:
        return False


def wrap_digit(rect, padding=5):
    """将数字放在正方形的框中，增加内边距"""
    x, y, w, h = rect
    hcenter = x + w / 2
    vcenter = y + h / 2
    # roi = None
    if h > w:
        w = h
        x = hcenter - (w / 2)
    else:
        h = w
        y = vcenter - (h / 2)
    return (int(x - padding), int(y - padding), int(w + 2 * padding), int(h + 2 * padding))


# ann, test_data = ANN.train(ANN.create_ANN(56), 50000, 5)
ann, test_data = ANN.train(ANN.create_ANN(58), 50000, 5)
font = cv2.FONT_HERSHEY_SIMPLEX

# path = "./images/MNISTsamples.png"
path = "./images/numbers.jpg"
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 做平滑处理
bw = cv2.GaussianBlur(bw, (7, 7), 0)
# 由于MNIST数据库中是黑底白字，所以要把图像也做成黑底白字
ret, thbw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY_INV)
thbw = cv2.erode(thbw, np.ones((2, 2), np.uint8), iterations=2)
# 识别图像中的轮廓
image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []

for c in cntrs:
    r = cv2.boundingRect(c)
    # r = x, y, w, h = cv2.boundingRect(c)
    a = cv2.contourArea(c)
    b = (img.shape[0] - 3) * (img.shape[1] - 3)

    is_inside = False
    # 放弃包含在其他矩形里面的矩形
    # todo 这个判断可以改进一下
    for q in rectangles:
        if inside(r, q):
            is_inside = True
            break
    if not is_inside:
        # 判断是否超过图像大小,
        # 防止将整个图像作为边框，导致其他图像全都因为包含在里面，而被抛弃
        if not a >= b:
            rectangles.append(r)

for r in rectangles:
    x, y, w, h = wrap_digit(r)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = thbw[y:y + h, x:x + w]

    try:
        # 获得预测结果
        digit_class = int(ANN.predict(ann, roi.copy())[0])
    except:
        continue
    cv2.putText(img, "%d" % digit_class, (x, y - 1), font, 1, (0, 255, 0))

cv2.imshow("thbw", thbw)
cv2.imshow("contours", img)
cv2.imwrite("sample.jpg", img)
cv2.waitKey()
cv2.destroyAllWindows()