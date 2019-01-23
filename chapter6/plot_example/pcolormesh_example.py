import matplotlib.pyplot as plt
import numpy as np

# 定义点的数量
n = 500
# 绘制点
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)
# 构造点
X, Y = np.meshgrid(x, y)
print('X: ', X)
print('Y: ', Y)
Z = np.sin(X + Y)
print('Z: ', Z)
''' 绘图
        cmap参数设置颜色，利用plt.cm.get_cmap设置样式，然后将这个样式传递给pcolormesh
        vmin, vmax控制颜色对应数值的上下限，辨析度
'''
cm = plt.cm.get_cmap('rainbow')
plt.pcolormesh(X, Y, Z, cmap=cm, vmin=-1, vmax=1)
plt.colorbar()
plt.show()