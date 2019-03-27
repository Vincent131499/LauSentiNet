# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     CNNLSTM+Random+Sentiment
   Description :
   Author :       Stephen.Lau
   date：          2019/3/25
-------------------------------------------------
   Change Activity:
                   2019/3/25:
-------------------------------------------------
"""
from keras.preprocessing import sequence
import numpy as np
import jieba
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, LSTM
import keras
tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1000, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

#读取数据参数设置
# tf.flags.DEFINE_float('dev_sample_percentage', .1, 'Percentage of the training data to use for validation')
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string('positive_data_file', '../dataset/weibo60000/pos60000_utf8.txt_updated', 'Data source for the positive data')
tf.flags.DEFINE_string('negative_data_file', '../dataset/weibo60000/neg60000_utf8.txt_updated', 'Data source for the negative data')
tf.flags.DEFINE_string('glove_dir', '../dataset/glove.6B.100d.txt', 'Data source for the pretrained glove word vector')
tf.flags.DEFINE_integer('embedding_size', '100', '随机初始化的词嵌入矩阵的维度')
tf.flags.DEFINE_integer('filters', '64', 'CNN的卷积核的数目')
tf.flags.DEFINE_integer('kernel_size', '3', 'CNN卷积核的大小')
tf.flags.DEFINE_integer('pool_size', '4', '池化窗口大小')
tf.flags.DEFINE_integer('lstm_output_size', '70', '单向LSTM网络的单元数目')
tf.flags.DEFINE_integer('epochs', '5', 'The number of epoch')
tf.flags.DEFINE_integer('batch_size', '64', '批量大小')


# FLAGS = tf.flags.FLAGS
FLAGS = tf.flags.FLAGS

"""从文件中读取数据和标签"""
def load_data_and_label(pos_filename, neg_filename):
    """读取积极类别的数据"""
    positive_texts = open(pos_filename, 'r', encoding='utf-8').readlines()
    # print(positive_texts)
    # positive_texts = open(positive_filename, 'rb').readlines()
    positive_texts = [' '.join(list(jieba.cut(line.strip()))) for line in positive_texts]
    print('积极句子数目：', len(positive_texts))
    # print(len(positive_texts))
    """读取消极类别的数据"""
    negative_texts = open(neg_filename, 'r', encoding='utf-8').readlines()
    # negative_texts = open(positive_filename, 'rb').readlines()
    negative_texts = [' '.join(list(jieba.cut(line.strip()))) for line in negative_texts]
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

def construct_dataset():
    print('加载数据......')
    # positive_filename = './data/rt-polaritydata/rt-polarity.pos'
    # negative_filename = './data/rt-polaritydata/rt-polarity.neg'
    # positive_filename = './data/rt-polarity.pos'
    # negative_filename = './data/rt-polarity.neg'
    x_text, y = load_data_and_label(FLAGS.positive_data_file, FLAGS.negative_data_file)

    """建立词汇表"""
    max_sentence_length = max([len(text.split(' ')) for text in x_text])
    print('最长句子长度：', max_sentence_length)


    #tf.contrib.learn.preprocessing.VocabularyProcessor:生成词汇表，每一个文档/句子的长度<=max_sentnce_length,记录的是单词的位置信息
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    # #x:每一个句子中的单词对应词汇表的位置,word2id
    # x = np.array(list(vocab_processor.fit_transform(x_text)))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_text)
    sequences = tokenizer.texts_to_sequences(x_text)

    word_index = tokenizer.word_index
    print('词表大小：', len(word_index))

    x = pad_sequences(sequences, maxlen=max_sentence_length)

    print('词汇表建立完毕！')
    print('len(x):',len(x))
    print('x:',x)
    print('x.shape:', x.shape)
    print('type(x):', type(x))

    """随机模糊数据，即打乱各个元素的顺序，重新洗牌"""
    np.random.seed(10)
    #np.range()返回的是range object，而np.nrange()返回的是numpy.ndarray()
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    """划分训练集/测试集，此处直接切分"""
    #此处加负号表示是从列表的后面开始查找对应位置
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    # print('划分索引：', dev_sample_index)
    # x_train, x_dev = x_shuffled[:dev_sample_index], x[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    """使用sklearn中的cross-validation划分数据集"""
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1, random_state=10)

    print('数据集构造完毕，信息如下：')
    print('训练集样本数目：', len(x_train))
    print('训练集标签数目：', len(y_train))
    print('开发集样本数目：', len(x_dev))
    print('开发集标签数目：', len(y_dev))
    # print(type(y_dev))

    del x, y, x_shuffled, y_shuffled

    # print('词汇表 Size：', len(vocab_processor.vocabulary_))
    # print(vocab_processor.vocabulary_)

    print('x的数据类型：', type(x_train[1][1]))
    print('y的数据类型：', type(y_train[1]))

    # return x_train, x_dev, y_train, y_dev, vocab_processor
    return x_train, y_train, x_dev, y_dev, word_index

if __name__ == '__main__':
    print('Load dataset...')
    x_train, y_train, x_dev, y_dev, word_index = construct_dataset()
    Max_Sequence_Length = x_train.shape[1]
    print('Max_Sequence_Length: ', Max_Sequence_Length) #202
    print('x_train.shape: ', np.shape(x_train))
    print('y_dev.shape: ', np.shape(y_dev))

    model = Sequential()

    model = Sequential()
    model.add(Embedding(len(word_index)+1, FLAGS.embedding_size, input_length=Max_Sequence_Length))
    model.add(Dropout(0.25))
    model.add(Conv1D(FLAGS.filters,
                     FLAGS.kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=FLAGS.pool_size))
    model.add(LSTM(FLAGS.lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              validation_data=(x_dev, y_dev),
              callbacks=[tf_board_callback])
    score, acc = model.evaluate(x_dev, y_dev, batch_size=FLAGS.batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


