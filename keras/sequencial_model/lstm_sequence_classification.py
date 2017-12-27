# coding=utf-8
import numpy as np
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential

x_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000,1))
x_test = np.random.random((100, 10))
y_test = np.random.randint(2,size=(100,1))

model = Sequential()
model.add(Embedding(20, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=20, epochs=100)
score = model.evaluate(x_test, y_test, batch_size=20)

print("Test loss:", score[0])
print("Test accuracy:", score[1])