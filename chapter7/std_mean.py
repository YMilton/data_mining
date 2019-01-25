import pandas as pd
import numpy as np

def average(L):
    return np.sum(L)/len(L)

def std2(L):
    s = 0
    for i in range(len(L)):
        s+=(L[i] - average(L))**2
    return np.sqrt(s/(len(L)-1))

if __name__ == '__main__':

    data = pd.read_csv('watermelon_3a.csv', header=None)
    mat = data.values
    print(mat)
    density = mat[:8, 1]
    print("密度|好瓜的均值和标准方差")
    print(average(density), np.mean(density))
    print(std2(density), np.std(density, ddof=1))

    print(density)
    print("密度|好瓜的均值和标准方差")
    print(average(density))
    print(std2(density))


