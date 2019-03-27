# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       Stephen.Lau
   date：          2019/3/22
-------------------------------------------------
   Change Activity:
                   2019/3/22:
-------------------------------------------------
"""

"""使用Bert-encode+LogisticRegression进行分类"""
import gensim
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import jieba
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import tensorflow as tf
from bert_serving.client import BertClient

tf.flags.DEFINE_string('positive_data_file', './weibo60000/pos60000_utf8.txt', 'Data source for the positive data')
tf.flags.DEFINE_string('negative_data_file', './weibo60000/neg60000_utf8.txt', 'Data source for the negative data')
FLAGS = tf.flags.FLAGS

"""从文件中读取数据和标签"""
def load_data_and_label(pos_filename, neg_filename):
    """读取积极类别的数据"""
    positive_texts = open(pos_filename, 'r', encoding='utf-8').readlines()
    # print(positive_texts)
    # positive_texts = open(positive_filename, 'rb').readlines()
    positive_texts = [line for line in positive_texts]
    print('积极句子数目：', len(positive_texts))
    # print(len(positive_texts))
    """读取消极类别的数据"""
    negative_texts = open(neg_filename, 'r', encoding='utf-8').readlines()
    # negative_texts = open(positive_filename, 'rb').readlines()
    negative_texts = [line for line in negative_texts]
    print('消极句子数目：', len(negative_texts))

    """拼接"""
    x_text = positive_texts + negative_texts
    # print(x_text)
    print('全部句子数目：', len(x_text))

    """生成标签"""
    positive_labels = [1 for _ in negative_texts]
    negative_labels = [0 for _ in negative_texts]
    y = np.concatenate([positive_labels, negative_labels], 0)
    print('标签数目：', len(y))
    # print(y)
    # for mat in y:
    #     print(mat)
    return [x_text, y]

x_text, y = load_data_and_label(FLAGS.positive_data_file, FLAGS.negative_data_file)

# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(x_text)]
# model = Doc2Vec(documents, size=100, window=8, min_count=100, workers=8)

bc = BertClient(ip='192.168.2.17')
model = bc.encode(x_text)

#生成文本向量
print(model[1])
# print(type(model.docvecs[1]))
# print(type(model.docvecs))


#使用逻辑回归进行预测
def LR():
    clf = LogisticRegression()
    return clf
def getRecognitionRate(testPre, testClass):
    testNum = len(testPre)
    rightNum = 0
    for i in range(0, testNum):
        if testClass[i] == testPre[i]:
            rightNum += 1
    return float(rightNum) / float(testNum)

def getData():
    #生成pandas
    # tigs = []
    data_dict = {}

    for i in range(len(model)):
        data_dict['p' + str(i)] = model[i]
    # print(tigs)
    print(data_dict)
    data = pd.DataFrame(data_dict)
    data = data.T
    # data['class0'] = tigs
    X_train1, X_test1, y_train1, y_test1 = train_test_split(data, y, test_size=0.4, random_state=0)
    return X_train1, y_train1, X_test1, y_test1

T = getData()
trainMatrix, trainClass, testMatrix, testClass = T[0], T[1], T[2], T[3]
clf_LR=LR()
clf_LR.fit(trainMatrix, trainClass)
print('Logistic Regression recognition rate: ', getRecognitionRate(clf_LR.predict(testMatrix), testClass))