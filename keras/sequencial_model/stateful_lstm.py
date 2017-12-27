# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import numpy as np

data_dim = 16
time_steps = 8
batch_size = 32
num_classes = 10

model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True, batch_input_shape=(batch_size, time_steps, data_dim)))

model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_train = np.random.random((batch_size*10, time_steps, data_dim))
y_train = np.random.random((batch_size*10, num_classes))
x_test = np.random.random((batch_size*10, time_steps, data_dim))
y_test = np.random.random((batch_size*10, num_classes))
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

print("Test loss:", score[0])
print("Test accuracy:", score[1])