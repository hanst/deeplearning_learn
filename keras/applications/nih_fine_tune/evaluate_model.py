# coding=utf-8

from keras.models import load_model
import nih_image_input



train_images, train_labels,test_images, test_labels = nih_image_input.read_data_sets()

model = load_model('weights/nih_nih_disease_model.01-11.69.hdf5')
score = model.evaluate(test_images, test_labels, batch_size=32)
print(score)

model = load_model('weights/nih_nih_disease_model.03-11.69.hdf5')
score = model.evaluate(test_images, test_labels, batch_size=32)
print(score)

model = load_model('weights/nih_nih_disease_model.04-11.69.hdf5')
score = model.evaluate(test_images, test_labels, batch_size=32)
print(score)