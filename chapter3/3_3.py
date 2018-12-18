import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
3.3 编程实现对率回归，并给出西瓜数据集3.0a上的结果
'''
def sigmoid(x):
    '''
    sigmoid function definition
    :param x:
    :return:
    '''
    y = 1.0 / (1 + np.exp(-x))
    return y


def newton(X, y):
    '''
    牛顿法
    :param X:  input data
    :param y:  label
    :return:
    '''
    N = X.shape[0]
    beta = np.ones((1,3))
    z = X.dot(beta.T)
    # log-likehood
    old_l = 0.0
    new_l = np.sum(y*z + np.log(1 + np.exp(z))) # loss function
    iters = 0
    while( np.abs(old_l - new_l) > 1e-6):
        # shape[N,1]
        p1 = np.exp(z) / (1 + np.exp(z))
        # shape[N,N]
        p = np.diag((p1*(1 - p1)).reshape(N)) # 元素全放在对角线上
        # shape[1,3]
        first_order = -np.sum(X * (y - p1), 0, keepdims=True) # keepdims保持二维特性
        # shape[3,3]
        second_order = X.T.dot(p).dot(X)

        #update
        beta -= first_order.dot(np.linalg.inv(second_order)) #inv求矩阵的逆
        z = X.dot(beta.T)
        old_l = new_l
        new_l = np.sum(y*z + np.log(1 + np.exp(z)))

        iters += 1
    print("iters: ", iters)
    print(new_l)
    return beta


def gradDescent(X, y):
    '''
    梯度下降法
    :param X:
    :param y:
    :return:
    '''
    lr = 0.05 # 学习率
    # initialization
    beta = np.ones((1,3)) * 0.1
    # shape [N,1]
    z = X.dot(beta.T)

    # iteration times 150
    for i in range(150):
        # shape[N, 1]
        p1 = np.exp(z) / (1+ np.exp(z))
        # shape[1,3]
        first_order = -np.sum(X * (y - p1), 0, keepdims=True)

        #update
        beta  -= first_order*lr
        z = X.dot(beta.T)

    l = np.sum(y*z + np.log(1+np.exp(z)))
    print(l)

    return beta


if __name__ == '__main__':
    # read the data  (编号，密度，含糖量，是否好瓜)
    workbook = pd.read_csv('../data/watermelon_3a.csv', header=None)
    # insert 3 column with ones
    workbook.insert(3,'3',1)
    # the datas
    X = workbook.values[:,1:-1]
    # the labels
    y = workbook.values[:,4].reshape(-1,1)
    # plot training data
    positive_data = workbook.values[workbook.values[:,4]==1.0, :]
    negative_data = workbook.values[workbook.values[:,4]==0.0, :]
    plt.plot(positive_data[:,1], positive_data[:,2], 'bo')# 正例
    plt.plot(negative_data[:,1], negative_data[:,2], 'r+')# 负例

    # get optimal params beta with newton method
    beta = newton(X, y)
    # 绘制直线的边界点
    newton_left = -( beta[0,0]*0.1 + beta[0,2] ) / beta[0,1]
    newton_right = -( beta[0,0]*0.9 + beta[0,2] ) / beta[0,1]
    plt.plot([0.1, 0.9], [newton_left, newton_right], 'g-')

    # get optimal params beta with gradient descent method
    beta = gradDescent(X, y)
    # 绘制直线的边界点
    grad_descent_left = -(beta[0, 0] * 0.1 + beta[0, 2]) / beta[0, 1]
    grad_descent_right = -(beta[0, 0] * 0.9 + beta[0, 2]) / beta[0, 1]
    plt.plot([0.1, 0.9], [grad_descent_left, grad_descent_right], 'y-')

    plt.xlabel('density')
    plt.ylabel('sugar rate')
    plt.title('LR')
    plt.show()