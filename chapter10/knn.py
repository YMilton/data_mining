import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt


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


def predicts(X, k, dataSet, labels):
    '''
    批量分类数据
    :param X: 待分类的
    :param k:
    :param dataSet: KNN数据集
    :param labels: KNN数据集的标签
    :return:
    '''
    LL = []
    for line in X:
        l = classify(line, dataSet, labels, k)
        LL.append(l)
    return LL


def plot_desicion_boundary(X, y, k):
    '''
    分类边界
    :param X:  KNN学习的数据集
    :param y: KNN学习的数据集对应标签
    :param k: KNN的k值
    :return:
    '''
    x_min = X[:, 0].min() - 0.1
    x_max = X[:, 0].max() + 0.1
    y_min = X[:, 1].min() - 0.1
    y_max = X[:, 1].max() + 0.1
    # 在x, y的最大值与最小值之间做网格化
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.03), np.arange(y_min, y_max, 0.03))
    # 生成待预测的数据集
    inXs = np.vstack([xx.ravel(), yy.ravel()]).T.tolist()
    zz = predicts(inXs, k, X, y)
    # 等高线z轴数据生成，与x,y轴数据大小一致
    zz = np.array(zz).reshape(xx.shape)
    # 绘制分界线图
    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, zz, alpha=0.4)
    plt.scatter(X[:,0], X[:,1], c=y, s=20, edgecolors='k')
    plt.title('KNN( k={} )'.format(k))
    plt.show()


def test_classify():
    '''
    数据做分类预测（前16做分类训练，最后一个数据做分类预测）
    :return:
    '''
    data = pd.read_excel('watermelon3.0a.xlsx', index_col='编号')
    dataSet = data.values[:-1, :-1]

    inX = data.values[-1, :-1]
    labels = data.values[:-1, -1]
    print(classify(inX, dataSet, labels, 5))



if __name__ == '__main__':
    # 等高线绘制
    data = pd.read_excel('watermelon3.0a.xlsx', index_col='编号')
    dataSet = data.values[:, :-1]
    labels = data.values[:, -1]
    labels = [1 if label=='是' else 0 for label in labels]

    plot_desicion_boundary(dataSet, labels, 3)

