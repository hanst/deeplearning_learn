# coding=utf-8
from keras.layers import Conv2D, Input
import keras

x = Input(shape=(256,256,3))
y = Conv2D(3, (3,3), padding='SAME')(x)

output = keras.layers.add([x, y])