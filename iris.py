import pandas as pda
from bayes import Bayes
from k_nn import Knn
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

'''数据加载'''
data = pda.read_csv("./iris.csv")
'''标准化'''
data_standard = preprocessing.scale(data.iloc[:, :-1])
'''切分数据集，处理过拟合'''
train_data, test_data, train_labels, test_labels = train_test_split(data_standard,
                                                                    data.as_matrix()[:, -1],
                                                                    test_size=0.2,
                                                                    random_state=int(time.time()))
'''
贝叶斯算法识别
'''
print("---------------------------贝叶斯----------------------------------")
start = time.clock()
by = Bayes()
by.train(list(train_data), list(train_labels))
test_data_size = test_data.shape[0]
error_count = 0
for index, td in enumerate(list(test_data)):
    this_label = by.test(td)
    print("预测类别：{0}，真实类别：{1}".format(this_label, test_labels[index]))
    if this_label != test_labels[index]:
        error_count += 1
end = time.clock()
error_rate = (error_count / test_data_size) * 100
time_consuming = end - start
print("错误率为：{0:.2f}%".format(error_rate))
print("耗时：{0:.4f}s".format(time_consuming))

'''
k-近邻算法识别
'''
print("---------------------------knn----------------------------------")
start = time.clock()
knn = Knn()
test_data_size = test_data.shape[0]
error_count = 0
for index, td in enumerate(list(test_data)):
    this_label = knn.test(3, td, list(train_data), list(train_labels))
    print("预测类别：{0}，真实类别：{1}".format(this_label, test_labels[index]))
    if this_label != test_labels[index]:
        error_count += 1
end = time.clock()
error_rate = (error_count / test_data_size) * 100
time_consuming = end - start

print("错误率为：{0:.2f}%".format(error_rate))
print("耗时：{0:.4f}s".format(time_consuming))

'''
发现：
    朴素贝叶斯算法的耗时是knn算法耗时的10多倍
    选取不同的随机种子，错误率不同。
结论：
    就该问题而言，
    如果主要考虑时间，则选取knn算法
    错误率主要取决于训练集和测试集的采样方式
'''
