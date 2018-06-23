# coding=utf-8
import os, time
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation

BASE_DIR = "/home/sancica/Desktop"
TEXT_DATA_DIR = BASE_DIR + "/VulDeePecker/"
GLOVE_DIR = BASE_DIR + "/glove.6B/"
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
VALIDATION_SPIT = 0.2
EMBEDDING_DIM = 150
batch_size = 32

# 采用GLOVE的词向量集
embeddings_index = {}
f = open("./word2vector.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# 获取文件数据和标签
print("Processing text dataset...")
texts = []
labels = []
for name in os.listdir(TEXT_DATA_DIR):
    fpath = os.path.join(TEXT_DATA_DIR, name)
    if not os.path.isdir(fpath):
        program = []
        with open(fpath)as fr:
            lines = fr.readlines()
            for row in range(len(lines)):
                if "-------" not in lines[row]:
                    program.append(lines[row].strip("\n"))
                else:
                    labels.append(int(program[-1]))
                    program.pop(0)
                    program.pop(-1)
                    texts.append(" ".join(program))
                    program = []
    break
print("Found %s programs." % len(texts))

# print(texts)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))

# 填充数据, 最短数据长度2， 最长2698， 平均48
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# 转为one-hot模式
print(labels)
labels = to_categorical(np.asarray(labels))

# 打乱样本，将数据进行切割
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPIT * data.shape[0])

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

embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            dropout=0.2)

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.add(Dense(2, activation='softmax'))
model.layers[1].trainable = False

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=["accuracy"])
print("Training...")
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=5, validation_data=(x_val, y_val))

score, acc = model.evaluate(x_val, y_val, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)
