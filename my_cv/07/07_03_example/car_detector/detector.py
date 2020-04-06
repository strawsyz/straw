import cv2
import numpy as np

data_path = ""
SAMPLES = 400


def path(cls, i):
    return "%s/%s%d.pgm" % (data_path, cls, i + 1)


def get_flann_matcher():
    """获得FLANN匹配器"""
    flann_params = dict(algorithm=1, trees=5)
    return cv2.FlannBasedMatcher(flann_params, {})


def get_bow_extractor(extract, matcher):
    """返回bow训练器"""
    return cv2.BOWImgDescriptorExtractor(extract, matcher)


def get_extract_detect():
    return cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SIFT_create()


def extract_sift(fn, extractor, detector):
    """提取图像特征"""
    im = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    # SIFT检测器可以检测特征，而基于SIFT的提取器可以提取特征并返回它们
    return extractor.compute(im, detector.detect(im))[1]


def bow_features(img, extractor_bow, detector):
    """提取bow的特征"""
    return extractor_bow.compute(img, detector.detect(img))


def car_detector():
    pos, neg = "pos-", "neg-"
    detect, extract = get_extract_detect()
    matcher = get_flann_matcher()
    # extract_bow = get_bow_extractor(extract, matcher)
    print("building BOWKMeansTrainer...")
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(1000)
    extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)

    print("adding features to trainer")
    for i in range(SAMPLES):
        print(i)
        bow_kmeans_trainer.add(extract_sift(path(pos, i), extract, detect))
        bow_kmeans_trainer.add(extract_sift(path(neg, i), extract, detect))

    vocabulary = bow_kmeans_trainer.cluster()
    extract_bow.setVocabulary(vocabulary)

    train_data, train_labels = [], []
    print("adding to train data")
    for i in range(SAMPLES):
        print(i)
        train_data.extend(bow_features(cv2.imread(path(pos, i), 0), extract_bow, detect))
        train_labels.append(1)
        train_data.extend(bow_features(cv2.imread(path(neg, i), 0), extract_bow, detect))
        train_labels.append(-1)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(0.5)
    # 该参数会决定分类器的训练误差和预测从误差。
    # 其值越大，误判的可能性越小，但训练精度会降低。
    # 值太小，会导致过拟合，是预测精度降低
    svm.setC(35)
    # Kernel参数确定分类器的性质
    # SVM_LINEAR说明分类器为超平面，适用于二分类
    # SVM_RBF使用高斯函数来对数据进行分类，数据被分配到函数定义的核中
    svm.setKernel(cv2.ml.SVM_RBF)

    svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_labels))
    return svm, extract_bow


def save_svm(svm, path="svmxml"):
    svm.save(path)


def load_svm(path="svmxml"):
    return cv2.load(path)
