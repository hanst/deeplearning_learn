# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import keras

# Generate dummy data
x_train = np.random.random((100000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100000,1)), num_classes=10)
x_test = np.random.random((100,20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100,1)), num_classes=10)

# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1280, epochs=100)
score = model.evaluate(x_test, y_test, batch_size=128)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


