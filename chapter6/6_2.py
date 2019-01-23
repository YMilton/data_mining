import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

'''
svm_train中的参数设置：
    -s  SVM的类型选择(svm_type)
        0 -- C-SVC (default)        使用惩罚因子(cost)的处理噪声的多分类器
        1 -- nu-SVC(多分类器)       按照错误样本比例处理噪声的多分类器
        2 -- one-class SVM      一类支持向量机
        3 -- epsilon-SVR(回归)        epsilon支持向量回归
        4 -- nu-SVR(回归) 
    -t 核函数类型(kernel_type)
        0 -- linear(线性核)    u'*v
        1 -- polynomial(多项式核)   (gamma*u'*v + coef0)^degree
        2 -- radial basis function(RBF, 径向基/高斯核) default   exp(-gamma*|u-v|^2)
        3 -- sigmoid(S型核)   tanh(gamma*u'*v + coef0)
        4 -- precomputed kernel(预计算核)
    -d  调整核函数的degree参数，default 3
    -g  调整核函数的gamma参数，default 1/num_features
    -r  调整核函数的coef0参数，default 0
    -c 调整C-SVC, epsilon-SVR和nu-SVR中的Cost参数，default 1
    -n 调整nu-SVC, one-class SVM和nu-SVR中的错误率nu参数，default 0.5
    -p 调整epsilon-SVR的loss function中的epsilon参数，default 0.1
    -m 调整内缓冲区大小，以MB为单位，default 100
    -e 调整终止判据， default 0.001
    -wi 调整C-SVC中第i个特征Cost参数
    
    -b 是否估算正确概率，取值0或1，default 0
    -h 是否使用收缩启发式算法，取值0或1，default 0
    -v 交叉校验
    -q 静默模式
'''
from svmutil import *
from svm import *


def myfun():
    df = pd.read_csv('data/watermelon_3a.csv', header=None)
    df.columns=['id', 'density', 'sugar_content', 'label']
    df.set_index(['id'])

    X = df[['density', 'sugar_content']].values
    y = df['label'].values
    model = svm_train(y, X, '-t 0')
    sv = model
    print(sv)



def svm_fun(X, y):

    y, x = [1, -1], [{1: 1, 2: 1}, {1: -1, 2: -1}]
    prob = svm_problem(y, x)
    param = svm_parameter('-t 0 -c 4 -b 1')
    model = svm_train(prob, param)
    yt, xt = [1], [{1: 1, 2: 1}]
    p_label, p_acc, p_val = svm_predict(yt, xt, model)
    print("预测标签：", p_label)


if __name__ == '__main__':
    myfun()