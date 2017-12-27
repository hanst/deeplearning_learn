# coding=utf-8
from keras.layers import MaxPool2D, Conv2D, Input
import keras

input_image = Input((256,256,3))

tower1 = Conv2D(64, (1,1), padding='SAME', activation='relu')(input_image)
tower1 = Conv2D(64, (3,3), padding='SAME', activation='relu')(tower1)

tower2 = Conv2D(64,(1,1), padding='SAME', activation='relu')(input_image)
tower2 = Conv2D(64, (5,5), padding='SAME', activation='relu')(tower2)

tower3 = MaxPool2D((3,3), strides=(1,1), padding='SAME')(input_image)
tower3 = Conv2D(64, (3,3), padding='SAME', activation='relu')(tower3)

output = keras.layers.concatenate([tower1, tower2, tower3], axis=1)
