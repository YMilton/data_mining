import pandas as pd
import numpy as np


def label2digit():
    '''
    标签转换成数字
    :return:
    '''
    iris_data = pd.read_table('../data/iris.txt', header=None, sep=",")
    categories = iris_data.loc[:, 4].values
    # 把类型转换成数字
    cc = []
    for x in categories:
        if x=='Iris-setosa':
            cc.append(1)
        elif x=='Iris-versicolor':
            cc.append(2)
        elif x=='Iris-virginica':
            cc.append(3)

    iris_data.loc[:, 4] = pd.Series(cc)

    return iris_data

def sigmoid(x):
    '''
    sigmoid function definition
    :param x: x = w.T*x+b
    :return:
    '''
    y = 1.0 / (1 + np.exp(-x))
    return y


def gradDescent(X, y, lr=0.05, iters=150):
    '''
    梯度下降法
    :param X:
    :param y:
    :param lr: 学习率
    :param iters:  迭代次数
    :return:
    '''
    # 添加b项, 则X变成了[X, 1]，与[W,b]对应
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    # the number of feature
    feature_num = X.shape[1]
    # initialization
    beta = np.ones((1,feature_num)) * 0.1
    # shape [N,1]
    z = X.dot(beta.T)

    # iteration times default is150
    for i in range(iters):
        # shape[N, 1]
        p1 = np.exp(z) / (1+ np.exp(z))
        # *是各个元素相乘为[N, feature_num]，0表示列求和， keepdims保持二维特性
        first_order = -np.sum(X * (y - p1), 0, keepdims=True)

        #update
        beta  -= first_order*lr
        z = X.dot(beta.T)

    l = np.sum(y*z + np.log(1+np.exp(z)))
    print('loss function result: ', l)

    return beta


def ten_fold_cross_once(data):
    # 随机生成的数组，随机的长度，随机是否重复
    choice_labes = np.random.choice([1,2,3],2,False)
    print('random choice labels: ', choice_labes)
    # 选择随机标签的数据, 包括标签
    choice_data0 = data.values[data.values[:,-1] == choice_labes[0],:]
    choice_data1 = data.values[data.values[:,-1] == choice_labes[1],:]
    # 随机生成角标
    rand_indexs = np.random.choice(50,45,False)
    # 剩下未被筛选的角标
    other_incexs = [x for x in range(50) if x not in rand_indexs]

    # 测试集与训练集
    train_data = np.vstack((choice_data0[rand_indexs,:-1], choice_data1[rand_indexs,:-1]))
    test_data = np.vstack((choice_data0[other_incexs,:-1], choice_data1[other_incexs,:-1]))

    train_y = np.vstack((choice_data0[rand_indexs,-1:], choice_data1[rand_indexs,-1:]))
    test_y = np.vstack((choice_data0[other_incexs, -1:], choice_data1[other_incexs, -1:]))
    # set the label to [0,1]
    train_y = np.array([0 if x== train_y[0] else 1 for x in train_y]).reshape(-1,1)
    test_y = np.array([0 if x == test_y[0] else 1 for x in test_y]).reshape(-1,1)
    # logistic function
    beta = gradDescent(train_data, train_y, iters=150)

    # 测试集的计算
    test_data = np.hstack((test_data, np.ones((test_data.shape[0], 1)))) # 添加b项
    z = test_data.dot(beta.T)
    y = sigmoid(z)
    # 对率回归预测结果
    predict_y = np.around(y)
    #print(predict_y==test_y)
    accuracy_rate = np.sum(predict_y==test_y)/ test_y.shape[0]

    return accuracy_rate

def ten_fold_cross_many():
    acc_rates = []
    for i in range(10):
        rate = ten_fold_cross_once(label2digit())
        acc_rates.append(rate)
    print(acc_rates)
    print(np.mean(acc_rates))


if __name__ == '__main__':
    ten_fold_cross_many()