# coding=utf-8

from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import  to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print('x_train shape:{} , y_train shape:{}'.format(x_train.shape, y_train.shape))
print('y_train data:{}'.format(y_train))
print('x_train data:{}'.format(x_train))

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=20)
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)



