import cv2
import numpy as np
from car_detector.detector import car_detector, bow_features
from car_detector.pyramid import pyramid
from car_detector.non_maximum import non_max_suppression_fast as nms
from car_detector.sliding_window import sliding_window
import urllib


def in_range(number, test, thresh=0.2):
    return abs(number - test) < thresh


test_image = "/home/d3athmast3r/dev/python/study/images/cars.jpg"
# img_path = "/home/d3athmast3r/dev/python/study/images/test.jpg"
# remote = "http://previews.123rf.com/images/aremac/aremac0903/aremac090300044/4545419-Lonely-car-on-an-empty-parking-lot-Stock-Photo.jpg"
# urllib.urlretrieve(test_image, img_path)

svm, extractor = car_detector()
detect = cv2.xfeatures2d.SIFT_create()

w, h = 100, 40
img = cv2.imread(test_image)

rectangles = []
counter = 1
scale_factor = 1.25
scale = 1
font = cv2.FONT_HERSHEY_PLAIN
# 循环金字塔
for resized in pyramid(img, scale_factor):
    scale = float(img.shape[1]) / float(resized.shape[1])
    print(scale)
    # 循环滑动窗口
    for (x, y, roi) in sliding_window(resized, 20, (100, 40)):

        if roi.shape[1] != w or roi.shape[0] != h:
            continue

        try:
            bf = bow_features(roi, extractor, detect)
            _, result = svm.predict(bf)
            # 通过设定svm.predict的可选参数flags，可以返回预测的评分
            # 预测结果的评分越小，置信度越高，表示被分到这一类的元素属于该类的可能性越大
            a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            print("Class: %d, Score: %f, res: %s" % (result[0][0], res[0][0], res))
            score = res[0][0]
            if result[0][0] == 1:
                if score < -1.0:
                    rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x + w) * scale), int((y + h) * scale)
                    rectangles.append([rx, ry, rx2, ry2, abs(score)])
        except Exception as e:
            raise e

        counter += 1
print(counter)
# 转为numpy数组
windows = np.array(rectangles)
# 应用非最大抑制
boxes = nms(windows, 0.25)

for (x1, y1, x2, y2, score) in boxes:
    print(x1, y1, x2, y2, score)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.putText(img, "%f" % score, (int(x1), int(y1)), font, 1, (0, 255, 0))

cv2.imshow("img", img)
cv2.waitKey(0)
