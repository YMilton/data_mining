import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
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
    while( np.abs(old_l - new_l) > 1e-5):
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
    N = X.shape[0]
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
    p1 = np.random.randint(0,255,(10,1))
    p2 = np.random.randint(0,255,(1,10))

    print('p1:', p1)
    print('p2:', p2)

    print(p1.dot(p2))




