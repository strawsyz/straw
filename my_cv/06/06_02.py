import cv2
import argparse
"""
使用SIFT、SURF来检测
运行命令如下：python 06_02.py -p test.jpg -a SURF
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('-p', type=str, help='待处理图片的路径')
parser.add_argument('-a', type=str, help='使用的算法 SIFT或者SURF')
# SURF的阈值越高，能识别的特征就会越少
parser.add_argument('-r', type=int, default=4000, help='SURF的阈值，默认为4000')
args = parser.parse_args()
img = cv2.imread(args.p)
alg = args.a


def fd(algorithm):
    if algorithm == "SIFT":
        # 报错：AttributeError: module 'cv2.cv2' has no attribute 'xfeatures2d'
        # 解决方案：pip install opencv-contrib-python==3.4.1.15
        return cv2.xfeatures2d.SIFT_create()
    if algorithm == "SURF":
        return cv2.xfeatures2d.SURF_create(float(args.r))


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fd_alg = fd(alg)
# 检测和计算
# opencv中 SIFT是DoG和SIFT的组合， SURF是Hessian和SURF的组合
# 检测结果返回一组关键点，计算结果返回描述符
key_points, descriptor = fd_alg.detectAndCompute(gray, None)
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=key_points, flags=4, color=(51, 163, 236))
cv2.imshow('keypoints', img)
while True:
    if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()
