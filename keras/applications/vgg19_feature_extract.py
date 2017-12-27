# coding=utf-8
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.preprocessing import image
import numpy as np

vgg19 = VGG19(weights='imagenet')
model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block4_pool').output)

img = image.load_img('images/cat1.jpg', target_size=(224,224))
img_array = image.img_to_array(img)
print(img_array.shape)
x = np.expand_dims(img_array, axis=0)
x = preprocess_input(x, mode='tf')
block4_pool_features = model.predict(x)
print(block4_pool_features)


