import pandas as pd

def condition_col_prob(df, col_property_name, p_n_name):
    col = df.loc[:, '好瓜']

    col1 = df.loc[:, col_property_name]
    positive = col1[col == p_n_name]
    classify = list(set(positive))
    for i in range(len(classify)):
        print(classify[i],'|',p_n_name, str(sum(positive == classify[i])), '/', str(len(positive)) ,
              '\t',sum(positive == classify[i]) / len(positive))


'''
    试使用极大似然法估算西瓜数据集3.0中前3个属性的类条件概率
'''
if __name__ == '__main__':
    df = pd.read_csv('watermelon_3.csv',index_col='编号')

    cols = df.columns.values[:6]
    for i in range(len(cols)):
        print(cols[i])
        condition_col_prob(df, cols[i],'是')
        condition_col_prob(df, cols[i],'否')
        print()