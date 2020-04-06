import cv2
import numpy as np

"""使用ann的方法"""

# MLP是multilayer perceptron 多层感知机
ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([9, 5, 9], dtype=np.uint8))
# 采用反向传播的方式进行训练 有 BACKPROP RPROP
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
ann.train(np.array([[1.2, 1.3, 1.9, 2.2, 2.3, 2.9, 3.0, 3.2, 3.3]], dtype=np.float32),
          cv2.ml.ROW_SAMPLE,
          np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float32))
print(ann.predict(np.array([[1.4, 1.5, 1.2, 2., 2.5, 2.8, 3., 3.1, 3.8]], dtype=np.float32)))
# 输出结果如下 第一个值是类标签，第二个是数组表示输入数据属于每个类的概率
# (5.0, array([[-0.06419383, -0.13360272, -0.1681568 , -0.18708915,  0.0970564 ,
#          0.89237726,  0.05093023,  0.17537238,  0.13388439]],
#       dtype=float32))
