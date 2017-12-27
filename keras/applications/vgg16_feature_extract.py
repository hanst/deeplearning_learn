# coding=utf-8
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np
import keras.applications.resnet50 as res




# input_shape: optional shape tuple, only to be specified
#             if `include_top` is False (otherwise the input shape
#             has to be `(224, 224, 3)` (with `channels_last` data format)
#             or `(3, 224, 224)` (with `channels_first` data format).
#             It should have exactly 3 input channels,
#             and width and height should be no smaller than 48.
#             E.g. `(200, 200, 3)` would be one valid value.
model = VGG16(weights='imagenet', include_top=False)

img_path = 'images/猎豹.jpeg'
img = image.load_img(img_path, target_size=(256, 256))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
print('Predicted:', features)