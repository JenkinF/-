import numpy as npy
from bloom import BloomFilter

'''
朴素贝叶斯算法
'''


class Bayes(object):
    def __init__(self):
        # {"类别":对应的概率}
        self.label_prob = dict()
        # {"类别": [各特征向量]}
        self.label_vector = dict()

    '''计算各类别概率'''

    def calculation_label_prob(self, labels):
        labels_total = len(labels)
        bl = BloomFilter(0.001, 1000000)
        for la in labels:
            if not bl.is_element_exist(la):
                self.label_prob[la] = labels.count(la) / labels_total
                bl.insert_element(la)

    '''
        生成label_vector
        格式：{"类别": [各特征向量]}
    '''

    def compose_label_vector(self, train_data, labels):
        for index, la in enumerate(labels):
            if la not in self.label_vector: self.label_vector[la] = []
            self.label_vector[la].append(train_data[index])
        pass

    '''
        训练数据
        train_data——[[数据1特征向量], [数据2特征向量], [数据3特征向量]……]
        labels——[类别]
        train_data和labels一一对应
    '''

    def train(self, train_data, labels):
        if train_data and labels and len(train_data) != len(labels): raise ValueError("数据有误！")
        self.calculation_label_prob(labels)
        self.compose_label_vector(train_data, labels)

    '''
        测试数据
        test_data——[1,0,1……] 一维列表
        返回类别
    '''

    def test(self, test_data):
        if len(self.label_prob) * len(self.label_vector) == 0:
            raise ValueError("需要先进行训练")
        result = dict()
        for label, vector in self.label_vector.items():
            # 当前类别的概率
            this_label_prob = self.label_prob[label]
            # 当前类别的特征向量
            this_label_vector = vector
            # 转置特征向量
            tran_vector = npy.array(this_label_vector).T

            p = 1
            for index, td in enumerate(test_data):
                plist = list(tran_vector[index])
                p *= plist.count(td) / len(plist)
            result[label] = p * this_label_prob
        return sorted(result, key=lambda x: result[x], reverse=True)[0]
