# coding=utf-8
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
import keras
import numpy as np

# Generage fake data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
print(y_train)
x_test = np.random.random((100,20))
y_test = np.random.randint(2, size=(100,1))

# build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10)

score = model.evaluate(x_test, y_test, batch_size=128)
print('Test loss:', score[0])
print('Test accuracy', score[1])