# coding=utf-8
from keras.layers import  Dense, Input
from keras.models import Model
import numpy as np

inputs = Input(shape=(784,))

x = Dense(32, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs= inputs, outputs = predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
x_train = np.random.random((1000, 784))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train)


