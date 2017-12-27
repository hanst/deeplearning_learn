# coding=utf-8
from keras.models import Sequential
from keras.layers import Activation,Dense

# model = Sequential()
# model.add(Dense(32, input_shape=784))
# model.add(Activation('relu'))
# model.add(Dense(10))
# model.add(Activation('softmax'))

model = Sequential([Dense(32, input_shape=(784,)), Activation('relu'), Dense(10), Activation('softmax')])

model = Sequential()
model.add(Dense(32, input_dim=784))

model = Sequential()
model.add(Dense(32, input_shape=(784,)))

# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

# For a mean square error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K
def mean_metric(y_true, y_pred):
    return K.mean(y_pred)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_metric])


# For a single-input model with 2 classes (binary classification):
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

# Generate sample data
import numpy as np
data = np.random.rand(1000,100)
labels = np.random.randint(2, size=(1000,1))
# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, batch_size=32, epochs=20)

# For a single-input model with 10 classes (categorical classification):
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

import keras
# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

data = np.random.random((1000,100))
labels = np.random.randint(10, size=(1000,1))
model.fit(data, one_hot_labels, batch_size=50,epochs=20)
