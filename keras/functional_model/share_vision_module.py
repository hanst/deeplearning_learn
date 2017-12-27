# coding=utf-8
from keras.layers import  Dense, Input, Conv2D, MaxPool2D, Flatten
from keras.models import Model
import keras

def build_model():
    digit_input = Input((256,256,3))
    x = Conv2D(64, (3,3))(digit_input)
    x = Conv2D(64,(3,3))(x)
    x = MaxPool2D((2,2))(x)
    out = Flatten()(x)

    vision_model = Model(inputs=digit_input, outputs=out)

    digit_a = Input((256,256,3))
    digit_b = Input((256,256,3))

    vision_a = vision_model(digit_a)
    vision_b = vision_model(digit_b)

    concatenated = keras.layers.concatenate([vision_a, vision_b])
    sim = Dense(1, activation='sigmoid')(concatenated)

    classification_model = Model([digit_a, digit_b], sim)
    return classification_model


