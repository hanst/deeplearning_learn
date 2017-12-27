# coding=utf-8
# https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650734699&idx=2&sn=7aea11805597957bf5ffbafba37a27aa&chksm=871ac415b06d4d03af99c489951afbf49a4209b1b96eb58fa4925723ea40e67b88ab04fcb432&scene=38#wechat_redirect

from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.models import Model, Input
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print('x_train shape:{}, y_train shape:{}'
      '\nx_test shape:{}, y_test shape:{}'
      .format(x_train.shape, y_train.shape,
              x_test.shape, y_test.shape))

input_shape = x_train[0, :, :, :].shape
model_input = Input(shape=input_shape)


def conv_pool_cnn(model_input):
    x = Conv2D(96, (3, 3), padding='same', activation='relu')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((3, 3), strides=2)(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((3, 3), strides=2)(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='relu')(x)

    model = Model(inputs=model_input, outputs=x, name='conv_pool_cnn')
    return model

def all_cnn(model_input):
    x = Conv2D(96, (3, 3), padding='same', activation='relu')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(192, (1,1), padding='same', activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='relu')(x)

    model = Model(inputs=model_input, outputs=x, name='all_cnn')
    return model

def nih_cnn(model_input):
    x = Conv2D(32, (5, 5), padding='valid', activation='relu')(model_input)
    x = Conv2D(32, (1,1), activation='relu')(x)
    x = Conv2D(32, (1,1), activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
    x = Conv2D(64, (1,1), activation='relu')(x)
    x = Conv2D(192, (1,1), activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(128, (3, 3), padding='valid', activation='relu')(x)
    x = Conv2D(32, (1,1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='relu')(x)

    model = Model(inputs=model_input, outputs=x, name='nil_cnn')
    return model


def compile_and_train(model: Model, num_epochs):
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True,
                                 mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=num_epochs, verbose=1,
                        callbacks=[checkpoint, tensor_board], validation_split=0.2)
    return history

def evaluate_error(model: Model):
    score = model.evaluate(x_test, y_test, batch_size=32)
    print('Test loss:', score[0])
    print('Test accuracy:{}'.format(score[1]))


conv_pool_cnn_model = conv_pool_cnn(model_input)
_ = compile_and_train(conv_pool_cnn_model, num_epochs=20)
evaluate_error(conv_pool_cnn_model)

all_cnn_model = conv_pool_cnn(model_input)
_ = compile_and_train(all_cnn_model, num_epochs=20)
evaluate_error(all_cnn_model)

nil_model = nih_cnn(model_input)
_ = compile_and_train(nil_model, num_epochs=20)
evaluate_error(nil_model)

conv_pool_cnn_model.load_weights('weights/conv_pool_cnn.29-0.10.hdf5')
all_cnn_model.load_weights('weights/all_cnn.30-0.08.hdf5')
nil_model.load_weights('weights/nil_cnn.30-0.93.hdf5')

models = [conv_pool_cnn_model, all_cnn_model, nil_model]

def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = Average(outputs)
    model = Model(model_input, y , name='ensemble')
    return model

model_ensemble = ensemble(models, model_input)
evaluate_error(model_ensemble)



