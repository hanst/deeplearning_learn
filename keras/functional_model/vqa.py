# coding=utf-8
from keras.layers import Conv2D, Dense, Input, MaxPool2D, Flatten, LSTM, Embedding
from keras.models import Sequential,Model
import keras

vision_model = Sequential()
vision_model.add(Conv2D(64,(3,3), activation='relu', padding='same', input_shape=(224,224,3)))
vision_model.add(Conv2D(64,(3,3), activation='relu'))
vision_model.add(MaxPool2D((2,2)))
vision_model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
vision_model.add((Conv2D(128,(3,3),activation='relu')))
vision_model.add(MaxPool2D((2,2)))
vision_model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
vision_model.add(Conv2D(256,(3,3)))
vision_model.add(Conv2D(256,(3,3)))
vision_model.add(MaxPool2D((2,2)))
vision_model.add(Flatten())

image_input = Input(shape=(224,224,3))
encode_image = vision_model(image_input)

question_input = Input(shape=(100,), dtype='int32')
embeded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embeded_question)

merged = keras.layers.concatenate([encode_image, encoded_question])

output = Dense(256, activation='softmax')(merged)
vqa_model = Model([image_input, question_input], outputs=output)
