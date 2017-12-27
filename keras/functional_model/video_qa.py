# coding=utf-8
from keras.layers import TimeDistributed, Input, Conv2D, MaxPool2D, Flatten, LSTM, Embedding, Dense
from keras.models import Sequential, Model
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

video_input = Input((100,224,224,3))
encoded_sequence = TimeDistributed(vision_model)(video_input)
encoded_video = LSTM(256)(encoded_sequence)


video_question_input = Input(shape=(100,), dtype='int32')
embeded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(video_question_input)
encoded_question = LSTM(256)(embeded_question)

question_encoder = Model(inputs=video_question_input, outputs=encoded_question)

encoded_question_input = question_encoder(video_question_input)

merged = keras.layers.concatenate([encoded_question_input, encoded_video])
output = Dense(1000, activation='softmax')(merged)

video_qa_model = Model(inputs = [video_question_input, video_input], outputs=output)

print(video_qa_model.to_json())
