import numpy as np

D = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [2, 1, 2, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1],
    [1, 1, 2, 1, 1, 1, 1],
    [3, 1, 1, 1, 1, 1, 1],
    [1, 2, 1, 1, 2, 2, 1],
    [2, 2, 1, 2, 2, 2, 1],
    [2, 2, 1, 1, 2, 1, 1],
    [2, 2, 2, 2, 2, 1, 0],
    [1, 3, 3, 1, 3, 2, 0],
    [3, 3, 3, 3, 3, 1, 0],
    [3, 1, 1, 3, 3, 2, 0],
    [1, 2, 1, 2, 1, 1, 0],
    [3, 2, 2, 2, 1, 1, 0],
    [2, 2, 1, 1, 2, 2, 0],
    [3, 1, 1, 3, 3, 1, 0],
    [1, 1, 2, 2, 2, 1, 0]])

test = [1, 1, 1, 1, 1, 1]  # the predict sample
m, n = D.shape[0], D.shape[1] - 1  # number of instances,attributes
N_i = [len(np.unique(D[:, i])) for i in range(len(D[0, :-1]))]
D_cx = [{}, {}]  # preserve p(c|xi)

for j in range(len(N_i)):
    D_cx[0][j], D_cx[1][j] = [0] * N_i[j], [0] * N_i[j]
    for i in range(m):
        if D[i, -1] == 1:
            if D[i, j] == 1:
                D_cx[0][j][0] += 1
            elif D[i, j] == 2:
                D_cx[0][j][1] += 1
            else:
                D_cx[0][j][2] += 1
        else:
            if D[i, j] == 1:
                D_cx[1][j][0] += 1
            elif D[i, j] == 2:
                D_cx[1][j][1] += 1
            else:
                D_cx[1][j][2] += 1

p0, p1 = 0, 0  # likelihood of class positive and negative
for i in range(n):  # set every attribute i to be a Super-Parent
    for key in range(N_i[i]):  # loop for every value in attribute i
        pcx0, pcx1 = 0, 0  # initiate p(c,xi) for class positive and negative
        pcxx0, pcxx1 = 1, 1  # initiate p(xj|c,xi) for class positive and negative
        pcx0 += (D_cx[0][i][key] + 1) / (17 + N_i[i])  # each value's p(c,xi) in attribute i of class positive
        pcx1 += (D_cx[1][i][key] + 1) / (17 + N_i[i])  # each value's p(c,xi) in attribute i of class negative
        count0, count1 = 0, 0  # initiate D(xj,c,xi) for class positive and negative
        for j in range(n):  # multi-product for p(xj|c,xi)
            # we need to ensure j to differ from i because i is their 'parent'
            if j != i:
                for k in range(8):  # D(xj,c,xi) for positive class
                    if D[k, i] == key + 1 and test[j] == D[k, j]:
                        count0 += 1
                for k in range(8, 17):  # D(xj,c,xi) for negative class
                    if D[k, i] == key + 1 and test[j] == D[k, j]:
                        count1 += 1
                # actually multi-product here
                pcxx0 *= (count0 + 1) / (D_cx[0][i][key] + N_i[j])
                pcxx1 *= (count1 + 1) / (D_cx[1][i][key] + N_i[j])
        p0 += pcx0 * pcxx0
        p1 += pcx1 * pcxx1
print(p0, p1)
