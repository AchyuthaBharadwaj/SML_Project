# import numpy as np
# import csv
# import os
# from sklearn.naive_bayes import GaussianNB
#
# train_prefix = "temp/final/train/"
# test_prefix = "temp/final/test/"
#
# input = []
# label = []
#
# for file in os.listdir(train_prefix):
#     data = open(train_prefix + file,"r")
#     data_read = csv.reader(data)
#     for lines in data_read:
#         input.append([float(elem) for elem in lines[0:-1]])
#         label.append(float(lines[-1]))
#
# X = np.array(input)
# Y = np.array(label)
#
# clf = GaussianNB()
# clf.fit(X,Y)
#
# test_X = []
# test_Y = []
# for file in os.listdir(test_prefix):
#     data = open(test_prefix + file, "r")
#     data_read = csv.reader(data)
#     for lines in data_read:
#         test_X.append([float(elem) for elem in lines[0:-1]])
#         test_Y.append(float(lines[-1]))
#
# test_X = np.array(test_X)
# test_Y = np.array(test_Y)
# i = 0
# count = 0
# for elem in test_X:
#     pre = clf.predict([elem])
#     print(pre, test_Y[i])
#     if pre == test_Y[i]:
#         count +=1
#     i += 1
# print("NB Classifier accuracy:", (count/len(test_Y))*100)


import numpy as np
import csv

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import os

train_prefix = "temp_LSTM/final/train_5/"
test_prefix = "temp/final/test/"

input = []
label = []

for file in os.listdir(train_prefix):
    data = open(train_prefix + file,"r")
    data_read = csv.reader(data)
    for lines in data_read:
        input.append([float(elem) for elem in lines[0:-1]])
        label.append(float(lines[-1]))

X = np.array(input)
Y = np.array(label)

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2)
std_clf = make_pipeline(StandardScaler(), GaussianNB())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)

print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))