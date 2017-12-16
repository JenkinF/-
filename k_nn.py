import numpy as npy

'''
k-近邻算法
'''


class Knn(object):
    def __init__(self):
        pass

    def test(self, k, test_data, train_data, labels):
        test_data_arr = npy.array(test_data, dtype=int)
        train_data_arr = npy.array(train_data, dtype=int)
        train_data_size = train_data_arr.shape[0]
        test_arr = npy.tile(test_data_arr, (train_data_size, 1))

        distance = ((test_arr - train_data_arr) ** 2).sum(axis=1) ** 0.5
        sort = distance.argsort()
        count = {}
        for i in range(k):
            this_label = labels[sort[i]]
            count[this_label] = count.get(this_label, 0) + 1
        return sorted(count, key=lambda x: count[x])[0]
