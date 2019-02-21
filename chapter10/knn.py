import numpy as np
import operator
import pandas as pd


def classify(inX, dataSet, labels, k):
    '''
    k近邻的分类
    :param inX: 待分类的向量，[1xn]
    :param dataSet: 数据集矩阵，[mxn]
    :param labels: 数据集矩阵按列分的类别标签， [mx1]
    :param k: 分类的k值
    :return:
    '''
    # 计算待分类向量与数据集的距离
    diffMat = np.tile(inX, (dataSet.shape[0],1)) - dataSet
    sqDiffMat = diffMat**2
    # 按照行求和并开方
    sqDistances = sqDiffMat.sum(axis=1)**0.5
    # 获取从小到大排序后的下角标
    sortedSqDistancesIndexs = sqDistances.argsort()

    classCount={}
    for i in range(k):
        voteLabel = labels[sortedSqDistancesIndexs[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # operator.itemgetter(1)获取第1个域的值，reverse=True做反转操作
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回数量最多的标签
    return sortedClassCount[0][0]


if __name__ == '__main__':
    data = pd.read_excel('watermelon3.0a.xlsx', index_col='编号')
    dataSet = data.values[:-1,:-1]

    inX = data.values[-1,:-1]

    labels = data.values[:-1,-1]

    print(classify(inX, dataSet, labels, 3))

