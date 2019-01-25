import pandas as pd
import numpy as np


def cal_naive_bayes_classifier(df, rownum=0):
    if rownum>df.shape[0]-1:
        print('Input row is larger than dataframe size!')
        return
    # 先验概率的计算P(c)
    col = df.loc[:, '好瓜']
    prior_prob_true = sum(col == '是') / len(col)
    prior_prob_false = sum(col == '否') / len(col)
    # 条件概率的计算P(xi|c)
    # iloc，即index locate 用index索引进行定位，所以参数是整型，如：df.iloc[10:20, 3:5]
    # loc，则可以使用column名和index名进行定位
    test1 = df.iloc[rownum, :-1]
    Hnb_t, Hnb_f = prior_prob_true, prior_prob_false
    for i in range(len(test1)):
        col_property = df.loc[:, test1.index[i]]
        col_p_t = col_property[col == '是']
        col_p_f = col_property[col == '否']

        if type(test1[i]) == str and len(test1[i]) > 1:
            #print(test1[i])
            #print(sum(col_p_t == test1[i]) / len(col_p_t), sum(col_p_f == test1[i]) / len(col_p_f))
            Hnb_t *= sum(col_p_t == test1[i]) / len(col_p_t)
            Hnb_f *= sum(col_p_f == test1[i]) / len(col_p_f)
        else:
            #print(probability_density_function(test1[i], col_p_t), probability_density_function(test1[i], col_p_f))
            Hnb_t *= probability_density_function(test1[i], col_p_t)
            Hnb_f *= probability_density_function(test1[i], col_p_f)

    #print(Hnb_t, Hnb_f)
    if Hnb_t > Hnb_f:
        return "好瓜"
    else:
        return "坏瓜"


def probability_density_function(xi, col_property):
    '''
    概率密度函数
    :param xi:
    :param col_property:
    :return:
    '''
    sigma = np.std(col_property, ddof=1)
    mu = np.mean(col_property)
    return 1/(np.sqrt(2*np.pi) * sigma) * np.exp(-(xi - mu )**2/(2*sigma**2))


'''
    试编程实现拉普拉斯修正的朴素贝叶斯分类器，并以西瓜数据集3.0为训练集，对p.151“测1”样本进行判别。
'''
if __name__ == '__main__':
    df = pd.read_csv('watermelon_3.csv',index_col='编号')

    print(cal_naive_bayes_classifier(df, rownum=0))