# coding=utf-8
from keras.layers import LSTM,Dense
from keras.models import Sequential
import numpy as np

data_dim = 16
time_steps = 8
num_classes = 10

model = Sequential()
model.add(LSTM(32,return_sequences=True, input_shape=(time_steps, data_dim)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
x_train = np.random.random((1000, time_steps, data_dim))
y_train = np.random.random((1000, num_classes))
x_test = np.random.random((100, time_steps, data_dim))
y_test = np.random.random((100, num_classes))
print(y_test[0])

model.fit(x_train, y_train, epochs=20, batch_size=100)
score = model.evaluate(x_test, y_test, batch_size=20)

print('Test loss:', score[0])
print('Test accuracy:', score[1])