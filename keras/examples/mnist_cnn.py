# coding=utf-8
from keras.layers import Dense,Dropout,Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import keras.backend as k
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if k.image_data_format() == 'channels_first':
    x_train = x_train.reshape(60000, 1, 28,28)
    x_test = x_test.reshape(10000, 1, 28,28)
    input_shape = (1,28,28)
else:
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    input_shape = (28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test,10)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))
model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=20)
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)