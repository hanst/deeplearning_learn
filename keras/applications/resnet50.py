# coding=utf-8

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import keras.applications.resnet50 as res
model = ResNet50(weights ='imagenet')

image_path='images/liuyifei.jpg'
img = image.load_img(image_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x , axis=0)
x = res.preprocess_input(x)

preds = model.predict(x)
print('Predicted:', res.decode_predictions(preds, top=3)[0])

image_path='images/cat1.jpg'
img = image.load_img(image_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x , axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

image_path='images/cat2.jpg'
img = image.load_img(image_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x , axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

image_path='images/猎豹.jpeg'
img = image.load_img(image_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x , axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
