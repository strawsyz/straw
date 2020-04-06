import cv2
import numpy as np
from os.path import join

# http://12r.cs.uiuc.edu/~cogcomp/Data/Car/CarData.tar.gz
# 数据来源 http://ai.stanford.edu/~jkrause/cars/car_dataset.html
data_path = ''


def path(cls, i):
    return "%s/%s%d.pgm" % (data_path, cls, i + 1)


pos, neg = "pos-", "neg-"
# 用于提取关键点
detect = cv2.xfeatures2d.SIFT_create()
# 用于提取特征
extract = cv2.xfeatures2d.SIFT_create()
# 创建FLANN匹配器
FLANN_INDEX_FDTREE = 1
flann_params = dict(algorithm=FLANN_INDEX_FDTREE, trees=5)
flann = cv2.FlannBasedMatcher(flann_params, {})

# 创建BOW训练器
bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
# 初始化BOW提取器
extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)



def extract_sift(path):
    """根据路径读取文件，转为灰度格式，提取描述符"""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return extract.compute(im, detect.detect(im))[1]


for i in range(8):
    # 从数据集提取8个正样本和8个负样本
    bow_kmeans_trainer.add(extract_sift(path(pos, i)))
    bow_kmeans_trainer.add(extract_sift(path(neg, i)))
# bow_kmeans_trainer.cluster() 函数执行k-means分类并返回词汇
voc = bow_kmeans_trainer.cluster()
# 指定词汇，以便能从测试图像中提取描述符
extract_bow.setVocabulary(voc)


def bow_features(path):
    """根据路径加载图片，返回基于BOW描述符提取器的计算得到的描述符"""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return extract_bow.compute(im, detect.detect(im))

# 创建两个数组分别对应训练数据与标签
traindata, trainlabels = [], []
for i in range(20):
    traindata.extend(bow_features(path(pos, i)))
    trainlabels.append(1)
    traindata.extend(bow_features(path(neg, i)))
    trainlabels.append(-1)
# 创建SVM的实例
svm = cv2.ml.SVM_create()
# 将数据与标签放到numpy的数组中训练
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))


def predict(path):
    """显示predict方法的结果，并返回这些结果"""
    f = bow_features(path)
    p = svm.predict(f)
    print(path, "\t", p[1][0][0])
    return p

# 用两张图来让训练好的SVM预测
car, notcar = "", ""
car_img = cv2.imread(car)
notcar_img = cv2.imread(notcar)
car_predict = predict(car)
notcar_predict = predict(notcar)

# 最后画出结果，在相应的图片中写上预测从结果
font = cv2.FONT_HERSHEY_SIMPLEX
if (car_predict[1][0][0] == 1.0):
    cv2.putText(car_img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
if (car_predict[1][0][0] == -1.0):
    cv2.putText( notcar_img, 'Car Not Detected', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow('BOW + SVM Success', car_img)
cv2.imshow('BOW + SVM Failure', notcar_img)
cv2.waitKey()
cv2.destroyAllWindows()
