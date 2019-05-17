# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:30:44 2018

@author: Administrator
"""

import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import loadData

image_train, label_train = loadData.readfile()
image_test,label_test = loadData.read_test_file()
# 从训练集选1000个样本
train_image = loadData.get_image(image_train, 1000)
train_label = loadData.get_label(label_train, 1000)
# 从测试集选100个样本
test_image = loadData.get_image(image_test, 100)
test_label = loadData.get_label(label_test, 100)
#kernel、degree就是模型的参数
#kernel是核方法，常用的核方法有：‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
#这个例子中改变degree的大小或者kernel的类型，准确率就会变

#svc = svm.SVC(kernel='linear')
#svc = svm.SVC(kernel='poly',degree=2)
svc = svm.SVC(kernel='poly',degree=2)   #设置参数

svc.fit(train_image, train_label)       #训练集

res = svc.predict(test_image)           #预测

wrongNum = np.sum(res != test_label)#得出错误个数
num = len(test_image)#训练图片的总数
acc = 1-wrongNum/float(num)
print ("准确率:", acc)#得出正确率