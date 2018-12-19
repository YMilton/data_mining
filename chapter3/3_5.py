import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def LDA1(positive, negative):
    '''
    Linear Discriminant Analysis, 线性判别分析
    J = w.T*Sb*w/w.T*Sw*w
    Sb = (mu0-mu1)*(mu0-mu1).T
    Sw = sigma0+sigma1
    w = Sw-1(mu0 - mu1)
    :return:
    '''
    # the mean of positive and negative
    mu1 = np.mean(positive, axis=0).reshape((-1,1))
    mu0 = np.mean(negative, axis=0).reshape((-1,1))
    # the cov of positive and negative
    sigma1 = np.cov(positive, rowvar=False)
    sigma0 = np.cov(negative, rowvar=False)
    Sw = sigma0 + sigma1
    # return omega
    return np.linalg.inv(Sw).dot(mu0 - mu1)



if __name__ == '__main__':
    # read the data  (编号，密度，含糖量，是否好瓜)
    workbook = pd.read_csv('../data/watermelon_3a.csv', header=None)
    # delete the row number
    data = np.array(workbook.values[:, 1:])
    # classify the positive and negative
    positive = data[data[:, -1] == 1, :-1]
    negative = data[data[:, -1] == 0, :-1]
    omega = LDA1(positive, negative)
    print(omega)
    # plot the LDA
    plt.plot(positive[:,0], positive[:,1], 'bo')
    plt.plot(negative[:,0], negative[:,1], 'r+')

    lda_left = -(omega[0]*0) / omega[1]
    lda_right = -(omega[0]*0.9) / omega[1]

    plt.plot([0,0.9], [lda_left, lda_right], 'k-')
    plt.xlabel('density')
    plt.ylabel('sugar rate')
    plt.title('LDA')

    plt.show()