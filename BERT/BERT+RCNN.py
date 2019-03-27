# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     BERT+AV
   Description :
   Author :       Stephen.Lau
   date：          2019/3/25
-------------------------------------------------
   Change Activity:
                   2019/3/25:
-------------------------------------------------
"""
from kashgari.embeddings import BERTEmbedding
from models import RCNNModel
import jieba
from tqdm import tqdm
import keras
#每1000次更新一次
# tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs/rcnn_bert', update_freq=1000)

def read_neg_data(dataset_path):
    x_list = []
    y_list = []
    lines = open(dataset_path, 'r', encoding='utf-8').readlines()
    for line in tqdm(lines):
        line = line.strip()
        if len(line) > 1:
            label = '0'
            y_list.append(label)
            seg_text = list(jieba.cut(line))
            # print(seg_text)
            x_list.append(seg_text)
        else:
            continue
    return x_list, y_list

def read_pos_data(dataset_path):
    x_list = []
    y_list = []
    lines = open(dataset_path, 'r', encoding='utf-8').readlines()
    for line in tqdm(lines):
        line = line.strip()
        if len(line) > 1:
            label = '1'
            y_list.append(label)
            seg_text = list(jieba.cut(line))
            # print(seg_text)
            x_list.append(seg_text)
        else:
            continue
    return x_list, y_list

def concate_data(pos_x, pos_y, neg_x, neg_y):
    data_x = []
    data_y = []
    for i in range(len(pos_x)):
        data_x.append(pos_x[i])
        data_y.append(pos_y[i])
    for j in range(len(neg_x)):
        data_x.append(neg_x[j])
        data_y.append(neg_y[j])
    return data_x, data_y


def train():
    pos_data_path = '../dataset/weibo60000/pos60000_utf8.txt_updated'
    pos_x, pos_y = read_pos_data(pos_data_path)
    print(len(pos_x))
    print(len(pos_y))
    # print(pos_y)

    neg_data_path = '../dataset/weibo60000/neg60000_utf8.txt_updated'
    neg_x, neg_y = read_neg_data(neg_data_path)
    print(len(neg_x))
    print(len(neg_y))
    # print(neg_y)

    train_pos_x = pos_x[:41025]
    train_pos_y = pos_y[:41025]
    val_pos_x = pos_x[41025:52746]
    val_pos_y = pos_y[41025:52746]
    test_pos_x = pos_x[52746:]
    test_pos_y = pos_y[52746:]

    train_neg_x = neg_x[:41165]
    train_neg_y = neg_y[:41165]
    val_neg_x = neg_x[41165:52926]
    val_neg_y = neg_y[41165:52926]
    test_neg_x = neg_x[52926:]
    test_neg_y = neg_y[52926:]

    train_x, train_y = concate_data(train_pos_x, train_pos_y, train_neg_x, train_neg_y)
    val_x, val_y = concate_data(val_pos_x, val_pos_y, val_neg_x, val_neg_y)
    test_x, test_y = concate_data(test_pos_x, test_pos_y, test_neg_x, test_neg_y)

    print('The number of train-set:', len(train_x))
    # print(len(train_y))
    print('The number of val-set:', len(val_x))
    # print(len(val_y))
    print('The number of test-set:', len(test_x))
    # print(len(test_y))

    embedding = BERTEmbedding('../dataset/chinese_L-12_H-768_A-12', sequence_length=100)
    print('embedding_size', embedding.embedding_size)
    # print(embedding.model.output

    model = RCNNModel(embedding)
    model.fit(train_x, train_y, val_x, val_y, batch_size=128, epochs=20, fit_kwargs={'callbacks': [tf_board_callback]})
    model.evaluate(test_x, test_y)
    model.save('./model/rcnn_bert_model')

if __name__ == '__main__':
    train()

