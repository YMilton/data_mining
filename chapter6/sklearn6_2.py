# coding = <utf-8>

# input file
data_file_watermelon_3a = "data/watermelon_3a.csv"

########## data loading and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns

df = pd.read_csv(data_file_watermelon_3a, header=None, )
df.columns = ['id', 'density', 'sugar_content', 'label']
df.set_index(['id'])

# plt.figure(0)
# sns.FacetGrid(df, hue='label', size=5).map(plt.scatter, 'density', 'sugar_content').add_legend()
# plt.show()

X = df[['density', 'sugar_content']].values
y = df['label'].values

########## SVM training and comparison
# based on linear kernel as well as gaussian kernel
from sklearn import svm

for fig_num, kernel in enumerate(('linear', 'rbf')):
    # initial
    svc = svm.SVC(C=1000, kernel=kernel)  # classifier 1 based on linear kernel
    # train
    svc.fit(X, y)
    # get support vectors
    sv = svc.support_vectors_

    ##### draw decision zone
    plt.figure(fig_num)
    plt.clf()

    # plot point and mark out support vectors
    '''
    cmap颜色映射
    '''
    plt.scatter(X[:, 0], X[:, 1], edgecolors='k', c=y, cmap=plt.cm.Paired, zorder=10)
    plt.scatter(sv[:, 0], sv[:, 1], edgecolors='r', facecolors='none', s=80, linewidths=2, zorder=10)

    # plot the decision boundary and decision zone into a color plot
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    # meshgrid生成len(first)*len(second)的矩阵，返回的值第一个是横向重复，第二个值是纵向重复
    XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # ravel()功能是把矩阵横向追加成一行，np.c_把两个vector纵向合并，decision_function计算样本点到分割超平面的函数距离
    Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    # pcolormesh作用在于能直观表现分类边界
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['r', 'g', 'b'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
    plt.axis('tight')

plt.show()