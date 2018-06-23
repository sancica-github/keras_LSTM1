# coding=utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
# 上面使用了Dropout防止过拟合（正则化方法），SGD随机梯度下降算法

# 生成实验数据
import numpy as np

x_train = np.random.random((1000, 20))
# print(type(x_train), x_train)
# 使用to_categorical函数将类别转换为binary class matrices
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# print(type(y_train), y_train)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

# 建立模型
model = Sequential()
# 在第一层，必须制定输入数据的shape
model.add(Dense(64, activation='relu', input_dim=20))
# 以一定的几率drop输入单元units
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 优化器
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)

print(score)