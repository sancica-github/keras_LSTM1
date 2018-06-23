from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# 创建模型
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))

# 对模型进行编译,参数分别为优化器，损失函数，指标列表，这里指的是精度

model.compile(optimizer='rmsprop', loss="binary_crossentropy", metrics=['accuracy'])


# 隨機生成1000行，100列的张量
data = np.random.random((1000, 100))
data_Tr = data[:900]
data_Te = data[900:]
# 2为最小值，如果只有low，则low为最大值
labels = np.random.randint(2, size=(1000, 1))
labels_Tr = labels[:900]
labels_Te = labels[900:]

# 模型训练，用fit函数，epochs表示轮的次数, batch_size:每一次梯度更新时候的需要的样本数量
model.fit(data_Tr, labels_Tr, epochs=10, batch_size=32)

score = model.evaluate(data_Te, labels_Te, batch_size=32)
print(score)


