# coding=utf-8
# VGG like CNN model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import keras

# Generate fake data
x_train = np.random.random((100,100,100,3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100,1)), num_classes=10)
x_test= np.random.random((20,100,100,3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20,1)), num_classes=10)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-4, nesterov=True, momentum=0.9)
model.compile(optimizer=sgd,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=10, epochs=20)
score = model.evaluate(x_test, y_test, batch_size=10)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
