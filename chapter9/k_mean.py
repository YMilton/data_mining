import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
    根据课本思路迭代KMean算法
'''
def KmeanMethod(data, K=3):
    '''
    根据距离中心簇，聚类
    :param data:  Dataframe
    :param K:  聚类的类别个数
    :return:
    '''
    # 随机生成簇中心脚标
    rand_indexs = np.random.randint(0, data.shape[0], K)
    # rand_indexs = [5, 11, 23]
    clusters_center = data.values[rand_indexs,:]
    # 不同的簇，创建簇的空数组
    empty_array = [[] for i in clusters_center]
    keys = [','.join(str(v) for v in c) for c in clusters_center]
    cluster_dict = dict(zip(keys, empty_array))
    for p in data.values:
        # 把每一个点与簇比较，与簇距离最小的点划分到该簇中，最小距离的簇中心点
        c_min = clusters_center[0]
        distance_min = calu_distance(p, c_min)

        for c in clusters_center:
            if calu_distance(p, c) <= distance_min:
                c_min = c
                distance_min = calu_distance(p, c)
        # 添加最小距离的点到簇
        cluster_dict[','.join(str(v) for v in c_min)].append(list(p))

    return cluster_dict



def KmeanIteration(data, K=3):
    # 随机生成簇中心脚标
    rand_indexs = np.random.randint(0, data.shape[0], K)
    clusters_center = data.values[rand_indexs,:]
    # 不同的簇，创建簇的空数组
    empty_array = [[] for i in clusters_center]
    keys = [','.join(str(v) for v in c) for c in clusters_center]
    cluster_dict = dict(zip(keys, empty_array))

    key_len = ''

    i = 0
    while(True):
        for p in data.values:
            # 把每一个点与簇比较，与簇距离最小的点划分到该簇中，最小距离的簇中心点
            c_min = clusters_center[0]
            distance_min = calu_distance(p, c_min)

            for c in clusters_center:
                if calu_distance(p, c) <= distance_min:
                    c_min = c
                    distance_min = calu_distance(p, c)
            # 添加最小距离的点到簇
            cluster_dict[','.join(str(v) for v in c_min)].append(list(p))

        print('iteration times: ', i)
        cluster_mean = []

        key_len_splice = ''
        for k, v in cluster_dict.items():
            print(k, v)
            # 求平均值并保留三位有效数
            tmp_mean = np.round(np.mean(np.array(v), axis=0), 3)
            cluster_mean.append(list(tmp_mean))
            key_len_splice += str(len(v))

        display_cluster(cluster_dict)

        # 更新簇中心点
        clusters_center = cluster_mean
        keys = [','.join(str(v) for v in c) for c in clusters_center]
        cluster_dict.clear()
        empty_array = [[] for i in clusters_center]
        cluster_dict = dict(zip(keys, empty_array))

        # 迭代终止条件
        if key_len != key_len_splice:
            key_len = key_len_splice
        else:
            break

        i = i+1



def calu_distance(point1, point2):
    '''
    计算两点的距离，返回欧式距离
    :return:
    '''
    point1 = np.array(point1).reshape(-1,1)
    point2 = np.array(point2).reshape(-1,1)

    return np.sqrt(np.sum((point1 - point2)**2))



def display_cluster(cluster_dict):
    '''
    显示kmean分类的图
    :param cluster_dict:
    :return:
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    colors = ['r','g','b','k','c','m','y']
    markers = ['*', '.', '1', 'h', 'v', 'p', 's']
    i = 0
    for k, v in cluster_dict.items():
        v = np.array(v)
        plt.plot(v[:,0], v[:,1], colors[i]+markers[i])
        center = k.split(',')

        plt.scatter(float(center[0]), float(center[1]), marker=markers[i], s=200, color=colors[i])
        i = i + 1
    plt.xlabel('密度')
    plt.ylabel('含糖量')
    plt.show()


if __name__ == '__main__':
    data = pd.read_excel('watermelon4.0.xlsx', index_col='编号')
    #cluster_dict = KmeanMethod(data, K=3)
    KmeanIteration(data, K=3)