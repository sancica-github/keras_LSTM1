# coding=utf-8
import numpy as np
import os
import sys
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation

BASE_DIR = "/home/sancica/Desktop"
# glove.6B是词向量集
GLOVE_DIR = BASE_DIR + "/glove.6B/"
TEXT_DATA_DIR = BASE_DIR + "/20_newsgroup"
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPILT = 0.2
batch_size = 32

# 首先将words 嵌入到 vector模式

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print("Found %s word vectors." % len(embeddings_index))

# 准备转换我们的文件和标记
print("Processing text dataset")
texts = []
labels_index = {}
labels = []
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                # 元组比较大小，参照字符串
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    # 读取文件编码不为utf-8时，要致命encoding
                    f = open(fpath, encoding='latin-1')
                texts.append(f.read())
                f.close()
                labels.append(label_id)
print("Found %s texts." % len(texts))

# Tokenizer所有本文，把texts中的str值先tokenizer，然后映射到相应的index。
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))
# 经过以上步骤，所有的单词被转换为数字

# 将一个序列数组转换为一个2维数组，第二个维度的长度为maxlen或者为最长的序列长度，pad：填充步骤
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# 对类别进行one-hot编码
labels = to_categorical(np.asarray(labels))

# 打乱样本，将数据切割
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPILT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]

x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    # 找到在glove词汇中的word的向量表示
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

# LSTM训练
embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            dropout=0.2)

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.add(Dense(len(labels_index), activation='softmax'))
model.layers[1].trainable = False

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=["accuracy"])
print("Training...")
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=5, validation_data=(x_val, y_val))

score, acc = model.evaluate(x_val, y_val, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)
