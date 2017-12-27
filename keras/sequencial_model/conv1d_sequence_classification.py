# coding=utf-8
from keras.layers import MaxPool1D,GlobalAveragePooling1D, Conv1D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
import numpy as np

seq_length= 20
x_train = np.random.random((10000, seq_length, 10))
y_train = np.random.randint(2, size=(10000,1))
x_test = np.random.random((100, seq_length, 10))
y_test = np.random.randint(2,size=(100,1))
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 10)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPool1D(3))
model.add(Conv1D(128,3, activation='relu'))
model.add(Conv1D(128,3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=10, epochs=100)
score = model.evaluate(x_test, y_test, batch_size=10)
print("Test loss:", score[0])
print("Test accuracy:", score[1])