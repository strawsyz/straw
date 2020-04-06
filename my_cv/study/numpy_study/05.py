from PIL import Image
from numpy import *

def pca(x):
    """
    主成分分析
    :param x:矩阵X ，其中该矩阵中存储训练数据，每一行为一张图片的所有像素
    :return:投影矩阵（按照维度的重要性排序）、方差和均值
    """
    # 获取维数
    num_data, dim = x.shape

    # 数据中心化
    mean_x = x.mean(axis=0)
    x = x - mean_x

    if dim > num_data:
        # PCA- 使用紧致技巧
        M = dot(x, x.T)  # 协方差矩阵
        e, EV = linalg.eigh(M)  # 特征值和特征向量
        tmp = dot(x.T, EV).T  # 这就是紧致技巧
        V = tmp[::-1]  # 由于最后的特征向量是我们所需要的，所以需要将其逆转
        S = sqrt(e)[::-1]  # 由于特征值是按照递增顺序排列的，所以需要将其逆转
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA- 使用SVD 方法
        U, S, V = linalg.svd(x)
        V = V[:num_data]  # 仅仅返回前nun_data 维的数据才合理

    # 返回投影矩阵、方差和均值
    return V, S, mean_x